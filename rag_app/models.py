"""
Pydantic models for request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    doc_id: str = Field(..., description="Unique document identifier")
    
    @validator("doc_id")
    def validate_doc_id(cls, v):
        """Validate doc_id format."""
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', v):
            raise ValueError("doc_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    doc_id: str
    pages_count: int
    chunks_count: int
    processing_time: float
    message: str


class QueryRequest(BaseModel):
    """Request model for document querying."""
    doc_id: str = Field(..., description="Document identifier to query")
    question: str = Field(..., description="Question to ask about the document")
    k: Optional[int] = Field(default=5, description="Number of top results to return")
    
    @validator("k")
    def validate_k(cls, v):
        """Validate k parameter."""
        if v is not None and v <= 0:
            raise ValueError("k must be a positive integer")
        return v


class Citation(BaseModel):
    """Citation model for answer references."""
    doc_id: str
    page: int
    chunk_id: str
    char_start: int
    char_end: int


class Snippet(BaseModel):
    """Snippet model for retrieved text."""
    page: int
    text: str


class QueryResponse(BaseModel):
    """Response model for document querying."""
    answer: str
    citations: List[Citation]
    snippets: List[Snippet]
    found: bool
    confidence: Optional[float] = None
    processing_time: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "ok"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
    status_code: int
