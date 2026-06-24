"""
Virtual Chemistry Lab API - DFT Calculator
Density Functional Theory calculations with Psi4 integration and RDKit fallback.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.engines.base import CalculationResult
from app.core.engines.psi4_engine import Psi4Engine
from app.core.engines.rdkit_engine import RDKitEngine
from app.core.utils.exceptions import CalculationError, EngineNotAvailableError
from app.core.utils.citations import CitationManager
from app.config import settings


class DFTCalculator:
    """DFT calculation handler with multi-engine support.

    Primary: Psi4 for quantum chemistry
    Fallback: RDKit MMFF94 for geometry optimization
    """

    def __init__(self):
        self.psi4_engine = Psi4Engine()
        self.rdkit_engine = RDKitEngine()

    async def optimize_geometry(
        self,
        smiles: str,
        functional: str = "PBE0",
        basis_set: str = "def2-SVP",
        charge: int = 0,
        multiplicity: int = 1,
    ) -> CalculationResult:
        """Optimize molecular geometry using DFT.

        Args:
            smiles: SMILES string of the molecule.
            functional: DFT functional (default: PBE0).
            basis_set: Basis set (default: def2-SVP).
            charge: Molecular charge.
            multiplicity: Spin multiplicity.

        Returns:
            CalculationResult with optimized geometry.
        """
        start_time = time.time()

        # Try Psi4 first
        if self.psi4_engine.is_available:
            try:
                result = await self.psi4_engine.calculate({
                    "method": "geometry_optimization",
                    "smiles": smiles,
                    "functional": functional,
                    "basis_set": basis_set,
                    "charge": charge,
                    "multiplicity": multiplicity,
                })
                if result.success:
                    return result
            except Exception as e:
                pass  # Fall through to fallback

        # Fallback to RDKit MMFF94
        if self.rdkit_engine.is_available:
            warnings = ["Psi4 not available, using RDKit MMFF94 fallback"]
            try:
                result = await self.rdkit_engine.calculate({
                    "method": "structure_optimization",
                    "smiles": smiles,
                })
                if result.success:
                    result.warnings = warnings
                    result.citations = (
                        CitationManager.get_citations("rdkit") + 
                        CitationManager.get_citations("mmff94")
                    )
                    return result
            except Exception as e:
                return CalculationResult(
                    success=False,
                    data={},
                    engine_used="fallback",
                    calculation_time=time.time() - start_time,
                    warnings=["Both Psi4 and RDKit failed"],
                    citations=CitationManager.get_citations("dft"),
                    error_message=f"All engines failed: {str(e)}",
                )

        return CalculationResult(
            success=False,
            data={},
            engine_used="none",
            calculation_time=time.time() - start_time,
            warnings=["No calculation engine available"],
            citations=CitationManager.get_citations("dft"),
            error_message="No calculation engine is available. Install Psi4 or RDKit.",
        )

    async def single_point_energy(
        self,
        smiles: str,
        functional: str = "B3LYP",
        basis_set: str = "6-31G*",
        charge: int = 0,
        multiplicity: int = 1,
    ) -> CalculationResult:
        """Calculate single point energy.

        Args:
            smiles: SMILES string.
            functional: DFT functional (default: B3LYP).
            basis_set: Basis set (default: 6-31G*).
            charge: Molecular charge.
            multiplicity: Spin multiplicity.

        Returns:
            CalculationResult with energy data.
        """
        if self.psi4_engine.is_available:
            return await self.psi4_engine.calculate({
                "method": "single_point_energy",
                "smiles": smiles,
                "functional": functional,
                "basis_set": basis_set,
                "charge": charge,
                "multiplicity": multiplicity,
            })

        # Fallback: return RDKit energy estimate
        start_time = time.time()
        if self.rdkit_engine.is_available:
            result = await self.rdkit_engine.calculate({
                "method": "structure_optimization",
                "smiles": smiles,
            })
            if result.success:
                result.warnings = ["Psi4 not available. Using RDKit MMFF94 energy as estimate."]
                result.data["note"] = "This is a force field energy, not a quantum mechanical energy."
                result.data["energy_unit"] = "kcal/mol (MMFF94)"
                return result

        return CalculationResult(
            success=False,
            data={},
            engine_used="none",
            calculation_time=time.time() - start_time,
            warnings=["No engine available for single point energy"],
            citations=CitationManager.get_citations("dft"),
            error_message="Psi4 not available and RDKit fallback failed",
        )

    async def calculate_ir_spectra(
        self,
        smiles: str,
        functional: str = "PBE0",
        basis_set: str = "def2-SVP",
    ) -> CalculationResult:
        """Calculate IR vibrational frequencies and intensities.

        Args:
            smiles: SMILES string.
            functional: DFT functional.
            basis_set: Basis set.

        Returns:
            CalculationResult with IR peak data.
        """
        if self.psi4_engine.is_available:
            return await self.psi4_engine.calculate({
                "method": "ir_spectra",
                "smiles": smiles,
                "functional": functional,
                "basis_set": basis_set,
            })

        # Fallback: harmonic oscillator approximation using RDKit
        start_time = time.time()
        if self.rdkit_engine.is_available:
            result = await self.rdkit_engine.calculate({
                "method": "structure_optimization",
                "smiles": smiles,
            })
            if result.success:
                # Generate approximate IR data from bond types
                mol_data = result.data
                warnings = ["Psi4 not available. Using harmonic oscillator approximation."]

                # Approximate frequencies based on common bond types
                peaks = self._approximate_ir_peaks(smiles)

                return CalculationResult(
                    success=True,
                    data={
                        "method": "ir_spectra_approximate",
                        "peaks": peaks,
                        "num_vibrational_modes": len(peaks),
                        "note": "Approximate IR spectra using harmonic oscillator model. For accurate results, install Psi4.",
                    },
                    engine_used="rdkit_fallback",
                    calculation_time=time.time() - start_time,
                    warnings=warnings,
                    citations=CitationManager.get_citations("rdkit"),
                )

        return CalculationResult(
            success=False,
            data={},
            engine_used="none",
            calculation_time=time.time() - start_time,
            warnings=["No engine available for IR spectra"],
            citations=CitationManager.get_citations("dft"),
            error_message="Psi4 not available for IR spectra calculation",
        )

    def _approximate_ir_peaks(self, smiles: str) -> List[Dict[str, Any]]:
        """Generate approximate IR peaks from bond types."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Common IR frequency ranges (simplified)
        bond_frequencies = {
            "O-H": [(3200, 3600, "strong", "broad")],
            "N-H": [(3300, 3500, "medium", "sharp")],
            "C-H": [(2850, 3000, "strong", "sharp")],
            "C=O": [(1650, 1750, "strong", "sharp")],
            "C=C": [(1620, 1680, "medium", "variable")],
            "C≡C": [(2100, 2260, "weak", "sharp")],
            "C-O": [(1000, 1300, "strong", "sharp")],
            "C-N": [(1020, 1250, "medium", "sharp")],
            "NO2": [(1500, 1570, "strong", "sharp"), (1380, 1400, "strong", "sharp")],
        }

        peaks = []

        # Check for functional groups using SMARTS
        patterns = {
            "O-H": "[OH]",
            "N-H": "[NH]",
            "C=O": "C=O",
            "C≡C": "C#C",
            "NO2": "[N+](=O)[O-]",
        }

        for bond_type, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                for freq_range in bond_frequencies.get(bond_type, []):
                    peaks.append({
                        "frequency": round((freq_range[0] + freq_range[1]) / 2, 1),
                        "range": [freq_range[0], freq_range[1]],
                        "intensity": freq_range[2],
                        "shape": freq_range[3],
                        "assignment": bond_type,
                    })

        # Always add C-H stretch if molecule has carbons
        if mol.GetNumAtoms() > 0:
            has_ch = False
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # Carbon
                    has_ch = True
                    break
            if has_ch and not any(p["assignment"] == "C-H" for p in peaks):
                peaks.append({
                    "frequency": 2925,
                    "range": [2850, 3000],
                    "intensity": "strong",
                    "shape": "sharp",
                    "assignment": "C-H",
                })

        peaks.sort(key=lambda x: x["frequency"], reverse=True)
        return peaks

    async def calculate_nmr(
        self,
        smiles: str,
        nuclei: str = "1H",
        functional: str = "PBE0",
        basis_set: str = "def2-TZVP",
    ) -> CalculationResult:
        """Calculate NMR chemical shifts.

        Args:
            smiles: SMILES string.
            nuclei: Nuclei to calculate (1H, 13C).
            functional: DFT functional.
            basis_set: Basis set for NMR (needs larger basis).

        Returns:
            CalculationResult with NMR chemical shifts.
        """
        if self.psi4_engine.is_available:
            return await self.psi4_engine.calculate({
                "method": "nmr_shielding",
                "smiles": smiles,
                "functional": functional,
                "basis_set": basis_set,
            })

        # Fallback: empirical NMR prediction
        start_time = time.time()
        if self.rdkit_engine.is_available:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return CalculationResult(
                    success=False,
                    data={},
                    engine_used="none",
                    calculation_time=0.0,
                    warnings=[],
                    citations=CitationManager.get_citations("nmr"),
                    error_message="Invalid SMILES",
                )

            shifts = []

            if nuclei == "1H":
                # Empirical 1H NMR shifts
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 1:  # Hydrogen
                        # Get attached atom
                        neighbors = atom.GetNeighbors()
                        if neighbors:
                            attached = neighbors[0]
                            attached_symbol = attached.GetSymbol()

                            # Base shift based on attached atom
                            base_shifts = {
                                "C": 1.0, "N": 3.0, "O": 4.5, "S": 2.5,
                                "F": 4.5, "Cl": 3.5, "Br": 2.8, "I": 2.2,
                            }
                            shift = base_shifts.get(attached_symbol, 1.0)

                            # Adjust for environment
                            if attached.GetIsAromatic():
                                shift += 1.5

                            shifts.append({
                                "atom_idx": atom.GetIdx(),
                                "attached_to": attached_symbol,
                                "chemical_shift": round(shift, 2),
                                "unit": "ppm",
                            })

            elif nuclei == "13C":
                # Empirical 13C NMR shifts
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 6:  # Carbon
                        # Base shift
                        shift = 25.0

                        # Adjust for hybridization
                        if atom.GetIsAromatic():
                            shift = 128.0
                        elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                            shift = 75.0
                        elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                            shift = 135.0

                        # Adjust for attached electronegative atoms
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetAtomicNum() in [7, 8, 9, 17, 35]:
                                shift += 20.0

                        shifts.append({
                            "atom_idx": atom.GetIdx(),
                            "chemical_shift": round(shift, 2),
                            "unit": "ppm",
                        })

            return CalculationResult(
                success=True,
                data={
                    "method": "nmr_empirical",
                    "nuclei": nuclei,
                    "shifts": shifts,
                    "num_signals": len(shifts),
                    "note": "Empirical NMR prediction. For accurate GIAO calculations, install Psi4.",
                },
                engine_used="rdkit_fallback",
                calculation_time=time.time() - start_time,
                warnings=["Psi4 not available. Using empirical NMR prediction."],
                citations=CitationManager.get_citations("nmr"),
            )

        return CalculationResult(
            success=False,
            data={},
            engine_used="none",
            calculation_time=time.time() - start_time,
            warnings=["No engine available for NMR"],
            citations=CitationManager.get_citations("nmr"),
            error_message="Psi4 not available for NMR calculation",
        )

    def recommend_basis_set(self, num_atoms: int, accuracy: str = "normal") -> str:
        """Recommend basis set based on molecule size and accuracy.

        Args:
            num_atoms: Number of atoms in molecule.
            accuracy: Desired accuracy level (quick, normal, high).

        Returns:
            Recommended basis set name.
        """
        if accuracy == "quick":
            if num_atoms <= 20:
                return "6-31G"
            else:
                return "STO-3G"
        elif accuracy == "normal":
            if num_atoms <= 20:
                return "def2-SVP"
            elif num_atoms <= 50:
                return "6-31G*"
            else:
                return "def2-SVP"
        elif accuracy == "high":
            if num_atoms <= 20:
                return "def2-TZVP"
            elif num_atoms <= 50:
                return "def2-SVP"
            else:
                return "6-31G*"
        else:
            return "def2-SVP"
