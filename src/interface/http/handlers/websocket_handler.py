import asyncio
import logging
import time
from fastapi import WebSocket
from typing import Dict, Optional

class ConnectionHandler:
    """Manage WebSocket connections for real-time tracking"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.tracking_sessions: Dict[str, dict] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logging.info(f"WebSocket connected: {session_id}")

        # Send connection confirmation
        await self.send_status(session_id, "connected", f"Session {session_id} established")

    def disconnect(self, session_id: str):
        # Cancel any running tasks
        if session_id in self.processing_tasks:
            self.processing_tasks[session_id].cancel()
            del self.processing_tasks[session_id]

        # Clean up session data
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.tracking_sessions:
            del self.tracking_sessions[session_id]

        logging.info(f"WebSocket disconnected: {session_id}")

    async def send_matches(self, session_id: str, matches: list, frame_info: Optional[dict] = None):
        """Send bounding box matches to frontend"""
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                response = {
                    "type": "matches",
                    "session_id": session_id,
                    "matches": matches,
                    "timestamp": time.time(),
                    "frame_info": frame_info or {}
                }
                await websocket.send_json(response)
                logging.debug(f"Sent {len(matches)} matches to {session_id}")
            except Exception as e:
                logging.error(f"Error sending matches to {session_id}: {e}")
                self.disconnect(session_id)

    async def send_status(self, session_id: str, status: str, message: str = ""):
        """Send status updates to frontend"""
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                response = {
                    "type": "status",
                    "session_id": session_id,
                    "status": status,
                    "message": message,
                    "timestamp": time.time()
                }
                await websocket.send_json(response)
                logging.debug(f"Sent status '{status}' to {session_id}")
            except Exception as e:
                logging.error(f"Error sending status to {session_id}: {e}")
                self.disconnect(session_id)

    async def send_error(self, session_id: str, error_message: str):
        """Send error messages to frontend"""
        await self.send_status(session_id, "error", error_message)

    def get_active_sessions(self) -> list:
        """Get list of active session IDs"""
        return list(self.active_connections.keys())

    def is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        return session_id in self.active_connections
