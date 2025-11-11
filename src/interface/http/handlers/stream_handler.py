import asyncio
import logging
import time
from typing import Dict, Optional

import cv2
from fastapi import WebSocket

from src._lib.base64_decoder import decode_base64_frame
from src.application.use_cases.predict_person import predict_person_on_stream

async def handle_start_tracking(session_id: str, data: dict, manager):  # Add manager parameter
    """Start continuous video tracking"""
    person_embed = data["person_embed"]
    video_source = data.get("video_source", 0)
    fps_limit = data.get("fps_limit", 10)

    # Store session info
    manager.tracking_sessions[session_id] = {
        "person_embed": person_embed,
        "video_source": video_source,
        "fps_limit": fps_limit,
        "active": True
    }

    # Cancel existing task if any
    if session_id in manager.processing_tasks:
        manager.processing_tasks[session_id].cancel()

    # Start video processing task
    task = asyncio.create_task(process_video_feed(session_id, manager))
    manager.processing_tasks[session_id] = task

    await manager.send_status(session_id, "tracking_started", f"Started tracking on source: {video_source}")

async def handle_single_frame(session_id: str, data: dict, manager):  # Add manager parameter
    """Process a single frame"""
    try:
        frame_data = data["frame"]  # base64 encoded
        person_embed = data["person_embed"]

        # Decode frame
        frame = decode_base64_frame(frame_data)
        if frame is None:
            await manager.send_status(session_id, "error", "Failed to decode frame")
            return

        # Process frame
        matches = predict_person_on_stream(person_embed, frame)

        frame_info = {
            "width": frame.shape[1],
            "height": frame.shape[0],
            "processed_at": time.time()
        }

        await manager.send_matches(session_id, matches, frame_info)

    except Exception as e:
        logging.error(f"Error processing single frame for {session_id}: {e}")
        await manager.send_status(session_id, "error", str(e))

async def handle_stop_tracking(session_id: str, manager):  # Add manager parameter
    """Stop tracking session"""
    if session_id in manager.tracking_sessions:
        manager.tracking_sessions[session_id]["active"] = False

    if session_id in manager.processing_tasks:
        manager.processing_tasks[session_id].cancel()
        del manager.processing_tasks[session_id]

    await manager.send_status(session_id, "tracking_stopped")

async def process_video_feed(session_id: str, manager):
    """Process video feed and send matches in real-time"""
    session = manager.tracking_sessions.get(session_id)
    if not session:
        return

    video_source = session["video_source"]
    person_embed = session["person_embed"]
    fps_limit = session["fps_limit"]
    frame_interval = 1.0 / fps_limit  # Time between frames

    cap = cv2.VideoCapture(video_source)

    try:
        await manager.send_status(session_id, "video_opened", f"Opened video source: {video_source}")

        last_process_time = 0
        frame_count = 0

        while session.get("active", False):
            current_time = time.time()

            # Rate limiting
            if current_time - last_process_time < frame_interval:
                await asyncio.sleep(0.01)  # Short sleep
                continue

            ret, frame = cap.read()
            if not ret:
                await manager.send_status(session_id, "warning", "No more frames from video source")
                break

            frame_count += 1

            # Process every Nth frame to reduce load
            if frame_count % 3 == 0:  # Process every 3rd frame
                try:
                    matches = predict_person_on_stream(person_embed, frame)

                    if matches:
                        frame_info = {
                            "frame_number": frame_count,
                            "width": frame.shape[1],
                            "height": frame.shape[0]
                        }
                        await manager.send_matches(session_id, matches, frame_info)

                except Exception as e:
                    logging.error(f"Error processing frame {frame_count} for {session_id}: {e}")
                    await manager.send_status(session_id, "error", f"Processing error: {str(e)}")

            last_process_time = current_time

    except Exception as e:
        logging.error(f"Error in video processing for {session_id}: {e}")
        await manager.send_status(session_id, "error", f"Video processing error: {str(e)}")
    finally:
        cap.release()
        await manager.send_status(session_id, "video_closed")
