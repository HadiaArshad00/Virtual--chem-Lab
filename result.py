"""
Virtual Chemistry Lab API - Result Model
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class Result(Base):
    """Result model for storing detailed calculation outputs."""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)

    # Output data
    output_data = Column(JSON, default=dict, nullable=False)

    # Engine info
    engine_used = Column(String(100), nullable=False)
    calculation_time = Column(Float, nullable=False)

    # Quality metrics
    warnings = Column(JSON, default=list, nullable=True)
    citations = Column(JSON, default=list, nullable=True)

    # Error info
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    experiment = relationship("Experiment", back_populates="results")

    def __repr__(self):
        return f"<Result(id={self.id}, experiment_id={self.experiment_id}, engine={self.engine_used})>"

    def to_dict(self):
        """Convert result to dictionary."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "output_data": self.output_data,
            "engine_used": self.engine_used,
            "calculation_time": self.calculation_time,
            "warnings": self.warnings,
            "citations": self.citations,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
