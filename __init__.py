"""
Virtual Chemistry Lab API - Models
"""

from sqlalchemy.orm import relationship

from app.models.user import User
from app.models.experiment import Experiment
from app.models.result import Result
from app.models.batch import BatchJob

# Add relationships
User.experiments = relationship("Experiment", order_by=Experiment.id, back_populates="user")
User.batch_jobs = relationship("BatchJob", order_by=BatchJob.id, back_populates="user")
Experiment.results = relationship("Result", order_by=Result.id, back_populates="experiment")
