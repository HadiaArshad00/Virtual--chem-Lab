"""
Virtual Chemistry Lab API - Experiment Schemas
Pydantic models for experiment requests and responses.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    """Schema for creating a new experiment."""
    type: str = Field(..., description="Type of calculation (e.g., dft, kinetics, spectra)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Calculation parameters")
    parent_id: Optional[int] = Field(None, description="Parent experiment ID for forking")

    model_config = {"json_schema_extra": {
        "example": {
            "type": "dft",
            "parameters": {
                "smiles": "CCO",
                "method": "geometry_optimization",
                "functional": "PBE0",
                "basis_set": "def2-SVP",
            },
        }
    }}


class ExperimentResponse(BaseModel):
    """Schema for experiment response."""
    id: int
    user_id: int
    type: str
    status: str
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    version: str
    parent_id: Optional[int] = None
    engine_used: Optional[str] = None
    calculation_time: Optional[float] = None

    model_config = {"from_attributes": True}


class ExperimentStatus(BaseModel):
    """Schema for experiment status."""
    id: int
    status: str
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ExperimentList(BaseModel):
    """Schema for list of experiments."""
    total: int
    experiments: List[ExperimentResponse]
    page: int = 1
    per_page: int = 20


class ExperimentFork(BaseModel):
    """Schema for forking an experiment."""
    new_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Override parameters")
