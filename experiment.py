"""
Virtual Chemistry Lab API - Experiment Model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class Experiment(Base):
    """Experiment model for tracking calculations and simulations."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    type = Column(String(100), nullable=False, index=True)
    status = Column(String(50), default="pending", nullable=False, index=True)

    # Input parameters
    parameters = Column(JSON, default=dict, nullable=False)

    # Results
    results = Column(JSON, default=dict, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    version = Column(String(20), default="1.0.0", nullable=False)
    parent_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    engine_used = Column(String(100), nullable=True)
    calculation_time = Column(Float, nullable=True)

    # Relationships
    user = relationship("User", back_populates="experiments")
    parent = relationship("Experiment", remote_side=[id], backref="children")

    def __repr__(self):
        return f"<Experiment(id={self.id}, type={self.type}, status={self.status})>"

    def to_dict(self):
        """Convert experiment to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "status": self.status,
            "parameters": self.parameters,
            "results": self.results,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "version": self.version,
            "parent_id": self.parent_id,
            "engine_used": self.engine_used,
            "calculation_time": self.calculation_time,
        }
