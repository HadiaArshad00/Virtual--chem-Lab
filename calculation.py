"""
Virtual Chemistry Lab API - Calculation Schemas
Pydantic models for calculation requests and responses.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CalculationRequest(BaseModel):
    """Base schema for calculation requests."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    method: str = Field(..., description="Calculation method")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")

    model_config = {"json_schema_extra": {
        "example": {
            "smiles": "CCO",
            "method": "descriptors",
            "parameters": {},
        }
    }}


class DFTRequest(BaseModel):
    """Schema for DFT calculation request."""
    smiles: str = Field(..., description="SMILES string")
    calculation_type: str = Field(..., description="Type: geometry_optimization, single_point, ir_spectra, nmr")
    functional: str = Field(default="PBE0", description="DFT functional")
    basis_set: str = Field(default="def2-SVP", description="Basis set")
    charge: int = Field(default=0, description="Molecular charge")
    multiplicity: int = Field(default=1, description="Spin multiplicity")


class KineticsRequest(BaseModel):
    """Schema for kinetics calculation request."""
    calculation_type: str = Field(..., description="Type: arrhenius, eyring, activation_parameters, half_life")
    A: Optional[float] = Field(None, description="Pre-exponential factor")
    Ea: Optional[float] = Field(None, description="Activation energy")
    delta_G: Optional[float] = Field(None, description="Gibbs free energy of activation")
    temperatures: Optional[List[float]] = Field(default_factory=list, description="Temperatures in K")
    rate_constants: Optional[List[float]] = Field(default_factory=list, description="Rate constants")
    reaction_order: Optional[int] = Field(1, description="Reaction order")


class SpectraRequest(BaseModel):
    """Schema for spectra calculation request."""
    smiles: str = Field(..., description="SMILES string")
    spectra_type: str = Field(..., description="Type: ir, nmr, mass_spec, uv_vis")
    nuclei: Optional[str] = Field("1H", description="For NMR: 1H or 13C")
    solvent: Optional[str] = Field("CDCl3", description="For NMR: solvent")
    ionization: Optional[str] = Field("EI", description="For MS: ionization method")


class DockingRequest(BaseModel):
    """Schema for docking calculation request."""
    ligand_smiles: str = Field(..., description="Ligand SMILES")
    receptor_pdbqt: str = Field(..., description="Receptor structure in PDBQT format")
    center_x: float = Field(default=0.0, description="Grid center X")
    center_y: float = Field(default=0.0, description="Grid center Y")
    center_z: float = Field(default=0.0, description="Grid center Z")
    size_x: float = Field(default=20.0, description="Grid size X")
    size_y: float = Field(default=20.0, description="Grid size Y")
    size_z: float = Field(default=20.0, description="Grid size Z")
    num_poses: int = Field(default=10, description="Number of poses")


class DynamicsRequest(BaseModel):
    """Schema for molecular dynamics request."""
    smiles: str = Field(..., description="SMILES string")
    num_steps: int = Field(default=10000, description="Number of MD steps")
    timestep: float = Field(default=1.0, description="Timestep in fs")
    temperature: float = Field(default=300.0, description="Temperature in K")
    ensemble: str = Field(default="NVT", description="Ensemble: NVT, NPT, NVE")


class ElectrochemRequest(BaseModel):
    """Schema for electrochemistry request."""
    E_start: float = Field(default=-0.5, description="Start potential in V")
    E_end: float = Field(default=0.5, description="End potential in V")
    scan_rate: float = Field(default=0.1, description="Scan rate in V/s")
    concentration: float = Field(default=1.0, description="Concentration in mM")
    n_electrons: int = Field(default=1, description="Number of electrons")


class CrystallizationRequest(BaseModel):
    """Schema for crystallization prediction request."""
    smiles: str = Field(..., description="SMILES string")
    temperature: float = Field(default=298.15, description="Temperature in K")
    solvent: str = Field(default="water", description="Solvent")
    concentration: float = Field(default=1.0, description="Concentration in mg/mL")
    method: str = Field(default="cooling", description="Crystallization method")


class YieldPredictionRequest(BaseModel):
    """Schema for yield prediction request."""
    reactants_smiles: List[str] = Field(..., description="List of reactant SMILES")
    conditions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Reaction conditions")


class PKaPredictionRequest(BaseModel):
    """Schema for pKa prediction request."""
    smiles: str = Field(..., description="SMILES string")
    site: Optional[int] = Field(None, description="Specific ionizable site")


class LogPPredictionRequest(BaseModel):
    """Schema for LogP prediction request."""
    smiles: str = Field(..., description="SMILES string")


class SolventRecommendationRequest(BaseModel):
    """Schema for solvent recommendation request."""
    reaction_type: Optional[str] = Field(None, description="Type of reaction")
    desired_properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Desired solvent properties")
    top_n: int = Field(default=5, description="Number of recommendations")


class CalculationResponse(BaseModel):
    """Schema for calculation response."""
    success: bool
    data: Dict[str, Any]
    engine_used: str
    calculation_time: float
    warnings: List[str]
    citations: List[Dict[str, Any]]
    error_message: Optional[str] = None
