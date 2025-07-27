"""Chat-related Pydantic schemas."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message."""
    message: str
    session_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Response model for chat messages."""
    id: str
    content: str
    created_at: datetime


class ChatSession(BaseModel):
    """Basic chat session model."""
    id: str
    created_at: datetime
