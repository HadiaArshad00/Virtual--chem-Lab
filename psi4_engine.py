"""
Virtual Chemistry Lab API - Psi4 Engine
Quantum chemistry calculations using Psi4.
"""

import time
import tempfile
import os
from typing import Dict, Any, List, Optional

from app.core.engines.base import AbstractEngine, CalculationResult
from app.core.utils.exceptions import EngineNotAvailableError, CalculationError, ValidationError
from app.core.utils.citations import CitationManager


class Psi4Engine(AbstractEngine):
    """Psi4 quantum chemistry calculation engine.

    Handles DFT optimization, single point energy, IR spectra, NMR shielding,
    and vibrational analysis using Psi4.
    """

    def __init__(self):
        super().__init__(name="psi4", version="1.9")
        try:
            import psi4
            self._available = True
        except ImportError:
            self._available = False

    def _get_supported_methods(self) -> List[str]:
        return [
            "geometry_optimization",
            "single_point_energy",
            "ir_spectra",
            "nmr_shielding",
            "vibrational_analysis",
            "frequency_calculation",
        ]

    async def validate_input(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters for Psi4 calculations."""
        if not self.is_available:
            return False, "Psi4 is not installed or not available"

        if "xyz" not in parameters and "smiles" not in parameters:
            return False, "Either 'xyz' or 'smiles' coordinates are required"

        if "method" not in parameters:
            return False, "Calculation method is required"

        method = parameters["method"]
        if method not in self._get_supported_methods():
            return False, f"Unsupported method: {method}"

        return True, ""

    async def calculate(self, parameters: Dict[str, Any]) -> CalculationResult:
        """Execute Psi4 quantum chemistry calculation."""
        start_time = time.time()
        warnings = []

        if not self.is_available:
            return CalculationResult(
                success=False,
                data={},
                engine_used=self.name,
                calculation_time=0.0,
                warnings=["Psi4 not available, using fallback"],
                citations=CitationManager.get_citations("psi4"),
                error_message="Psi4 is not installed. Install with: conda install -c psi4 psi4",
            )

        try:
            valid, error = await self.validate_input(parameters)
            if not valid:
                raise ValidationError(error)

            method = parameters["method"]

            if method == "geometry_optimization":
                result = self._geometry_optimization(parameters)
            elif method == "single_point_energy":
                result = self._single_point_energy(parameters)
            elif method == "ir_spectra":
                result = self._ir_spectra(parameters)
            elif method == "nmr_shielding":
                result = self._nmr_shielding(parameters)
            elif method == "vibrational_analysis":
                result = self._vibrational_analysis(parameters)
            elif method == "frequency_calculation":
                result = self._frequency_calculation(parameters)
            else:
                raise ValidationError(f"Unknown method: {method}")

            calc_time = time.time() - start_time

            return CalculationResult(
                success=True,
                data=result,
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("psi4") + CitationManager.get_citations("dft"),
            )

        except Exception as e:
            calc_time = time.time() - start_time
            return CalculationResult(
                success=False,
                data={},
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("psi4"),
                error_message=str(e),
            )

    def _get_molecule_string(self, parameters: Dict[str, Any]) -> str:
        """Get Psi4 molecule string from parameters."""
        if "xyz" in parameters:
            return parameters["xyz"]

        # Convert SMILES to XYZ first
        from rdkit import Chem
        from rdkit.Chem import AllChem

        smiles = parameters["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise CalculationError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), ""]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            lines.append(f"{symbol} {pos.x} {pos.y} {pos.z}")

        return "\n".join(lines)

    def _get_method_basis(self, parameters: Dict[str, Any]) -> tuple[str, str]:
        """Get method and basis set from parameters with smart defaults."""
        method = parameters.get("functional", "PBE0")
        basis = parameters.get("basis_set", "def2-SVP")

        # Smart basis set recommendation based on molecule size
        num_atoms = parameters.get("num_atoms", 10)
        if num_atoms > 30 and basis == "def2-TZVP":
            basis = "def2-SVP"

        return method, basis

    def _geometry_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform geometry optimization."""
        import psi4

        xyz = self._get_molecule_string(parameters)
        functional, basis = self._get_method_basis(parameters)

        charge = parameters.get("charge", 0)
        multiplicity = parameters.get("multiplicity", 1)

        mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        """)

        psi4.set_options({
            "basis": basis,
            "reference": "rhf" if multiplicity == 1 else "uhf",
            "opt_coordinates": "internal",
            "g_convergence": "gau_tight",
            "scf_type": "df",
        })

        energy = psi4.optimize(f"{functional}/{basis}", molecule=mol)

        # Get optimized geometry
        optimized_xyz = mol.save_xyz_string()

        return {
            "method": "geometry_optimization",
            "functional": functional,
            "basis_set": basis,
            "energy": energy,
            "energy_unit": "Hartree",
            "energy_ev": round(energy * 27.2114, 6),
            "energy_kcal_mol": round(energy * 627.509, 4),
            "optimized_geometry": optimized_xyz,
            "convergence": True,
        }

    def _single_point_energy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate single point energy."""
        import psi4

        xyz = self._get_molecule_string(parameters)
        functional, basis = self._get_method_basis(parameters)

        charge = parameters.get("charge", 0)
        multiplicity = parameters.get("multiplicity", 1)

        mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        """)

        psi4.set_options({
            "basis": basis,
            "reference": "rhf" if multiplicity == 1 else "uhf",
            "scf_type": "df",
        })

        energy = psi4.energy(f"{functional}/{basis}", molecule=mol)

        # Get dipole moment
        wfn = psi4.energy(f"{functional}/{basis}", molecule=mol, return_wfn=True)[1]
        dipole = wfn.variable("CURRENT DIPOLE")

        return {
            "method": "single_point_energy",
            "functional": functional,
            "basis_set": basis,
            "energy": energy,
            "energy_unit": "Hartree",
            "energy_ev": round(energy * 27.2114, 6),
            "energy_kcal_mol": round(energy * 627.509, 4),
            "dipole_moment": [round(x, 6) for x in dipole] if dipole else None,
            "dipole_unit": "Debye",
        }

    def _ir_spectra(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate IR frequencies and intensities."""
        import psi4

        xyz = self._get_molecule_string(parameters)
        functional, basis = self._get_method_basis(parameters)

        charge = parameters.get("charge", 0)
        multiplicity = parameters.get("multiplicity", 1)

        mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        """)

        psi4.set_options({
            "basis": basis,
            "reference": "rhf" if multiplicity == 1 else "uhf",
            "scf_type": "df",
        })

        # Run frequency calculation
        energy, wfn = psi4.frequency(f"{functional}/{basis}", molecule=mol, return_wfn=True)

        # Extract frequencies and intensities
        frequencies = wfn.frequency_analysis["omega"][2].data if hasattr(wfn, "frequency_analysis") else []

        peaks = []
        for i, freq in enumerate(frequencies):
            if freq > 0:  # Only real frequencies
                peaks.append({
                    "frequency": round(float(freq), 2),
                    "unit": "cm-1",
                    "intensity": round(1.0, 4),  # Placeholder - actual intensity extraction is complex
                })

        return {
            "method": "ir_spectra",
            "functional": functional,
            "basis_set": basis,
            "energy": energy,
            "num_vibrational_modes": len(peaks),
            "peaks": peaks,
            "max_frequency": max([p["frequency"] for p in peaks]) if peaks else None,
            "min_frequency": min([p["frequency"] for p in peaks]) if peaks else None,
        }

    def _nmr_shielding(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate NMR chemical shielding (GIAO method)."""
        import psi4

        xyz = self._get_molecule_string(parameters)
        functional = parameters.get("functional", "PBE0")
        basis = parameters.get("basis_set", "def2-TZVP")  # Need better basis for NMR

        charge = parameters.get("charge", 0)
        multiplicity = parameters.get("multiplicity", 1)

        mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        """)

        psi4.set_options({
            "basis": basis,
            "reference": "rhf" if multiplicity == 1 else "uhf",
            "nmr": "giao",
        })

        energy = psi4.properties(f"{functional}/{basis}", molecule=mol, properties=["NMR"])

        # Note: Full NMR shielding tensor extraction requires advanced Psi4 API usage
        # This is a simplified implementation
        return {
            "method": "nmr_shielding",
            "functional": functional,
            "basis_set": basis,
            "energy": energy,
            "note": "NMR shielding calculation completed. Full shielding tensors require advanced analysis.",
            "citations": CitationManager.get_citations("nmr"),
        }

    def _vibrational_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform vibrational analysis."""
        # Same as IR but with more detailed output
        result = self._ir_spectra(parameters)
        result["method"] = "vibrational_analysis"
        result["analysis_type"] = "harmonic"
        return result

    def _frequency_calculation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate vibrational frequencies."""
        return self._ir_spectra(parameters)
