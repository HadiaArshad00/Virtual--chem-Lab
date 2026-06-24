"""
Virtual Chemistry Lab API - Database Base
SQLAlchemy declarative base and common model mixins.
"""

from datetime import datetime
from typing import Any
from sqlalchemy import Column, Integer, DateTime, String
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at timestamps."""

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class IDMixin:
    """Mixin that adds an auto-incrementing primary key."""

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
