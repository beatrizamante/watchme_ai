import asyncio
import logging
from fastapi import WebSocket
from typing import Dict

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

    def disconnect(self, session_id: str):
        if session_id in self.processing_tasks:
            self.processing_tasks[session_id].cancel()
            del self.processing_tasks[session_id]

        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.tracking_sessions:
            del self.tracking_sessions[session_id]

        logging.info(f"WebSocket disconnected: {session_id}")
