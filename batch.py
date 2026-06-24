"""
Virtual Chemistry Lab API - Batch Job Model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class BatchJob(Base):
    """Batch job model for tracking multiple experiments."""

    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(50), default="pending", nullable=False, index=True)

    # Job info
    total_experiments = Column(Integer, nullable=False)
    completed_experiments = Column(Integer, default=0, nullable=False)
    failed_experiments = Column(Integer, default=0, nullable=False)

    # Input data
    experiments_data = Column(JSON, default=list, nullable=False)

    # Results
    results = Column(JSON, default=list, nullable=True)
    error_message = Column(String(1000), nullable=True)

    # Timing
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="batch_jobs")

    def __repr__(self):
        return f"<BatchJob(id={self.id}, status={self.status}, {self.completed_experiments}/{self.total_experiments})>"

    def to_dict(self):
        """Convert batch job to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "status": self.status,
            "total_experiments": self.total_experiments,
            "completed_experiments": self.completed_experiments,
            "failed_experiments": self.failed_experiments,
            "experiments_data": self.experiments_data,
            "results": self.results,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
