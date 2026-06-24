"""
Virtual Chemistry Lab API - User Schemas
Pydantic models for user management.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr


class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    name: Optional[str] = None
    api_key: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    email: str
    name: Optional[str] = None
    is_active: bool
    is_admin: bool
    usage_count: int
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    """Schema for updating user."""
    name: Optional[str] = None
    is_active: Optional[bool] = None


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    api_key: str
    created_at: datetime
    message: str = "Keep this key secure. It cannot be retrieved again."
