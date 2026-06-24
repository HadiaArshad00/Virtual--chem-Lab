"""
Virtual Chemistry Lab API - Response Schemas
Common response schemas for the API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    request_id: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    total: int
    page: int
    per_page: int
    total_pages: int
    data: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: float
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
