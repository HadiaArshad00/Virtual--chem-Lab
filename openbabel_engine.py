"""
Virtual Chemistry Lab API - Open Babel Engine
Format conversion and molecular operations using Open Babel.
"""

import time
from typing import Dict, Any, List, Optional

from app.core.engines.base import AbstractEngine, CalculationResult
from app.core.utils.exceptions import ValidationError, CalculationError, FileFormatError
from app.core.utils.citations import CitationManager


class OpenBabelEngine(AbstractEngine):
    """Open Babel molecular format conversion engine.

    Handles format conversion, energy calculation, and basic molecular operations.
    """

    def __init__(self):
        super().__init__(name="openbabel", version="3.1")
        try:
            from openbabel import openbabel
            self._available = True
        except ImportError:
            self._available = False

    def _get_supported_methods(self) -> List[str]:
        return [
            "format_conversion",
            "energy_minimization",
            "add_hydrogens",
            "remove_hydrogens",
            "get_formula",
            "get_inchi",
        ]

    async def validate_input(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters."""
        if not self.is_available:
            return False, "Open Babel is not installed"

        if "input_data" not in parameters and "smiles" not in parameters:
            return False, "Either 'input_data' or 'smiles' is required"

        return True, ""

    async def calculate(self, parameters: Dict[str, Any]) -> CalculationResult:
        """Execute Open Babel calculation."""
        start_time = time.time()
        warnings = []

        if not self.is_available:
            return CalculationResult(
                success=False,
                data={},
                engine_used=self.name,
                calculation_time=0.0,
                warnings=["Open Babel not available"],
                citations=CitationManager.get_citations("openbabel"),
                error_message="Open Babel is not installed",
            )

        try:
            valid, error = await self.validate_input(parameters)
            if not valid:
                raise ValidationError(error)

            method = parameters.get("method", "format_conversion")

            if method == "format_conversion":
                result = self._convert_format(parameters)
            elif method == "energy_minimization":
                result = self._energy_minimization(parameters)
            elif method == "add_hydrogens":
                result = self._add_hydrogens(parameters)
            elif method == "remove_hydrogens":
                result = self._remove_hydrogens(parameters)
            elif method == "get_formula":
                result = self._get_formula(parameters)
            elif method == "get_inchi":
                result = self._get_inchi(parameters)
            else:
                raise ValidationError(f"Unknown method: {method}")

            calc_time = time.time() - start_time

            return CalculationResult(
                success=True,
                data=result,
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("openbabel"),
            )

        except Exception as e:
            calc_time = time.time() - start_time
            return CalculationResult(
                success=False,
                data={},
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("openbabel"),
                error_message=str(e),
            )

    def _convert_format(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert between molecular formats."""
        from openbabel import openbabel

        input_data = parameters.get("input_data", parameters.get("smiles", ""))
        from_format = parameters.get("from_format", "smi")
        to_format = parameters.get("to_format", "sdf")

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(from_format, to_format)

        mol = openbabel.OBMol()
        obConversion.ReadString(mol, input_data)

        if mol.NumAtoms() == 0:
            raise FileFormatError("Failed to parse molecule")

        output = obConversion.WriteString(mol)

        return {
            "from_format": from_format,
            "to_format": to_format,
            "num_atoms": mol.NumAtoms(),
            "num_bonds": mol.NumBonds(),
            "molecular_weight": mol.GetMolWt(),
            "formula": mol.GetFormula(),
            "output": output,
        }

    def _energy_minimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform energy minimization using Open Babel force field."""
        from openbabel import openbabel

        input_data = parameters.get("input_data", parameters.get("smiles", ""))
        from_format = parameters.get("from_format", "smi")
        force_field = parameters.get("force_field", "mmff94")
        steps = parameters.get("steps", 500)

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(from_format, "xyz")

        mol = openbabel.OBMol()
        obConversion.ReadString(mol, input_data)

        if mol.NumAtoms() == 0:
            raise FileFormatError("Failed to parse molecule")

        # Add hydrogens
        mol.AddHydrogens()

        # Setup force field
        ff = openbabel.OBForceField.FindForceField(force_field)
        if not ff.Setup(mol):
            raise CalculationError(f"Failed to setup force field: {force_field}")

        # Minimize
        ff.ConjugateGradients(steps)
        ff.GetCoordinates(mol)

        energy = ff.Energy()

        # Get optimized coordinates
        obConversion.SetOutFormat("xyz")
        xyz_output = obConversion.WriteString(mol)

        return {
            "force_field": force_field,
            "energy": round(energy, 6),
            "energy_unit": "kcal/mol",
            "num_atoms": mol.NumAtoms(),
            "optimized_xyz": xyz_output,
        }

    def _add_hydrogens(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add hydrogens to molecule."""
        from openbabel import openbabel

        input_data = parameters.get("input_data", parameters.get("smiles", ""))
        from_format = parameters.get("from_format", "smi")
        to_format = parameters.get("to_format", "smi")

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(from_format, to_format)

        mol = openbabel.OBMol()
        obConversion.ReadString(mol, input_data)

        if mol.NumAtoms() == 0:
            raise FileFormatError("Failed to parse molecule")

        mol.AddHydrogens()

        output = obConversion.WriteString(mol)

        return {
            "num_atoms_before": mol.NumAtoms() - mol.NumHvyAtoms(),
            "num_atoms_after": mol.NumAtoms(),
            "num_hydrogens": mol.NumHvyAtoms(),
            "output": output,
        }

    def _remove_hydrogens(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Remove hydrogens from molecule."""
        from openbabel import openbabel

        input_data = parameters.get("input_data", parameters.get("smiles", ""))
        from_format = parameters.get("from_format", "smi")
        to_format = parameters.get("to_format", "smi")

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(from_format, to_format)

        mol = openbabel.OBMol()
        obConversion.ReadString(mol, input_data)

        if mol.NumAtoms() == 0:
            raise FileFormatError("Failed to parse molecule")

        original_atoms = mol.NumAtoms()
        mol.DeleteHydrogens()

        output = obConversion.WriteString(mol)

        return {
            "num_atoms_before": original_atoms,
            "num_atoms_after": mol.NumAtoms(),
            "num_hydrogens_removed": original_atoms - mol.NumAtoms(),
            "output": output,
        }

    def _get_formula(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get molecular formula."""
        from openbabel import openbabel

        input_data = parameters.get("input_data", parameters.get("smiles", ""))
        from_format = parameters.get("from_format", "smi")

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(from_format, "smi")

        mol = openbabel.OBMol()
        obConversion.ReadString(mol, input_data)

        if mol.NumAtoms() == 0:
            raise FileFormatError("Failed to parse molecule")

        return {
            "formula": mol.GetFormula(),
            "molecular_weight": round(mol.GetMolWt(), 4),
            "exact_mass": round(mol.GetExactMass(), 6),
            "num_atoms": mol.NumAtoms(),
            "num_bonds": mol.NumBonds(),
        }

    def _get_inchi(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get InChI and InChIKey."""
        from openbabel import openbabel

        input_data = parameters.get("input_data", parameters.get("smiles", ""))
        from_format = parameters.get("from_format", "smi")

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(from_format, "inchi")

        mol = openbabel.OBMol()
        obConversion.ReadString(mol, input_data)

        if mol.NumAtoms() == 0:
            raise FileFormatError("Failed to parse molecule")

        inchi = obConversion.WriteString(mol).strip()

        # Get InChIKey
        obConversion.SetOutFormat("inchikey")
        inchikey = obConversion.WriteString(mol).strip()

        return {
            "inchi": inchi,
            "inchikey": inchikey,
            "formula": mol.GetFormula(),
        }
