"""
Virtual Chemistry Lab API - Dependencies
FastAPI dependencies for authentication and database access.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.core.utils.exceptions import AuthenticationError, AuthorizationError
from app.core.utils.validators import validate_api_key


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_db():
    """Dependency that provides a database session.

    Yields:
        AsyncSession: Database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_current_user(
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Get current user from API key.

    Args:
        api_key: API key from header.
        db: Database session.

    Returns:
        User object.

    Raises:
        AuthenticationError: If API key is invalid.
    """
    if not api_key:
        raise AuthenticationError("API key required. Provide X-API-Key header.")

    if not validate_api_key(api_key):
        raise AuthenticationError("Invalid API key format")

    result = await db.execute(select(User).where(User.api_key == api_key))
    user = result.scalar_one_or_none()

    if user is None:
        raise AuthenticationError("Invalid API key")

    if not user.is_active:
        raise AuthorizationError("User account is deactivated")

    # Increment usage count
    user.usage_count += 1
    await db.commit()

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user.

    Args:
        current_user: Current user from dependency.

    Returns:
        User object.
    """
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current admin user.

    Args:
        current_user: Current user from dependency.

    Returns:
        User object if admin.

    Raises:
        AuthorizationError: If user is not admin.
    """
    if not current_user.is_admin:
        raise AuthorizationError("Admin access required")
    return current_user
