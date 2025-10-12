import asyncio
from asyncio import Queue
from typing import List, Union

import cv2


class VideoProcessor:
    """Handle video processing with async queues"""

    def __init__(self, max_queue_size: int = 10):
        self.max_queue_size = max_queue_size

    async def process_video_stream(self, video_source: Union[str, int]) -> List[tuple]:
        """
        Process video stream asynchronously

        Args:
            video_source: Video file path or camera index (0 for webcam)

        Returns:
            List of FrameResult objects with detections
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")

        try:
            frame_queue = Queue(maxsize=self.max_queue_size)
            result_queue = Queue()

            producer_task = asyncio.create_task(
                self._frame_producer(frame_queue, cap)
            )
            consumer_task = asyncio.create_task(
                self._frame_consumer(frame_queue, result_queue)
            )

            await producer_task
            await frame_queue.join()
            await consumer_task

            results = []
            while not result_queue.empty():
                results.append(await result_queue.get())

            return results

        finally:
            cap.release()

    async def _frame_producer(self, queue: Queue, cap: cv2.VideoCapture):
        """Produce frames from video source"""
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                await queue.put(None)
                break

            if queue.qsize() < self.max_queue_size:
                await queue.put((frame_count, frame))

            frame_count += 1
            await asyncio.sleep(0.01)

    async def _frame_consumer(self, frame_queue: Queue, result_queue: Queue):
        """Process frames from queue"""
        while True:
            item = await frame_queue.get()

            if item is None:
                frame_queue.task_done()
                break

            frame_count, frame = item
            ##TODO - fix to draw on screen
            frame_results = predict([frame])

            if frame_results and frame_results[0].detections: # type: ignore
                frame_results[0].frame_number = frame_count # type: ignore
                await result_queue.put(frame_results[0])

            frame_queue.task_done()

    def process_video_sync(self, video_source: Union[str, int]) -> List[tuple]:
        """Synchronous video processing (for simpler use cases)"""
        return asyncio.run(self.process_video_stream(video_source))
