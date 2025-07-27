"""Common Pydantic schemas and types."""

from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool = False
    error: str
    timestamp: datetime


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
