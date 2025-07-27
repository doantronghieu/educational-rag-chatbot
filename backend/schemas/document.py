"""Document-related Pydantic schemas."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentUpload(BaseModel):
    """Basic document upload model."""
    filename: str
    size: int


class DocumentResponse(BaseModel):
    """Basic document response model."""
    id: str
    filename: str
    status: str
    created_at: datetime


class DocumentStatus(BaseModel):
    """Document processing status."""
    document_id: str
    status: str
    error_message: Optional[str] = None
