"""
Virtual Chemistry Lab API - File Format Converters
Utilities for converting between molecular file formats.
"""

from typing import Optional
from app.core.utils.exceptions import FileFormatError


class MoleculeConverter:
    """Converter for molecular file formats."""

    SUPPORTED_FORMATS = {"smiles", "sdf", "mol", "mol2", "pdb", "xyz", "cif", "inchi", "json"}

    @classmethod
    def convert(cls, input_data: str, from_format: str, to_format: str) -> str:
        """Convert molecule from one format to another.

        Args:
            input_data: The molecule data in source format.
            from_format: Source format.
            to_format: Target format.

        Returns:
            Converted molecule data.

        Raises:
            FileFormatError: If format conversion fails.
        """
        from_format = from_format.lower().strip()
        to_format = to_format.lower().strip()

        if from_format not in cls.SUPPORTED_FORMATS:
            raise FileFormatError(f"Unsupported source format: {from_format}", format=from_format)
        if to_format not in cls.SUPPORTED_FORMATS:
            raise FileFormatError(f"Unsupported target format: {to_format}", format=to_format)

        if from_format == to_format:
            return input_data

        # Try Open Babel first if available
        try:
            return cls._convert_with_openbabel(input_data, from_format, to_format)
        except Exception:
            pass

        # Fallback to RDKit
        try:
            return cls._convert_with_rdkit(input_data, from_format, to_format)
        except Exception as e:
            raise FileFormatError(
                f"Conversion from {from_format} to {to_format} failed: {str(e)}",
                format=f"{from_format}->{to_format}",
            )

    @classmethod
    def _convert_with_openbabel(cls, input_data: str, from_format: str, to_format: str) -> str:
        """Convert using Open Babel."""
        try:
            from openbabel import openbabel

            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats(from_format, to_format)

            mol = openbabel.OBMol()
            obConversion.ReadString(mol, input_data)

            if mol.NumAtoms() == 0:
                raise ValueError("Failed to parse molecule")

            return obConversion.WriteString(mol)
        except ImportError:
            raise FileFormatError("Open Babel not available")

    @classmethod
    def _convert_with_rdkit(cls, input_data: str, from_format: str, to_format: str) -> str:
        """Convert using RDKit."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Parse input
        if from_format == "smiles":
            mol = Chem.MolFromSmiles(input_data)
        elif from_format in ("sdf", "mol"):
            mol = Chem.MolFromMolBlock(input_data)
        elif from_format == "pdb":
            mol = Chem.MolFromPDBBlock(input_data)
        elif from_format == "inchi":
            mol = Chem.inchi.MolFromInchi(input_data)
        else:
            raise FileFormatError(f"RDKit cannot read format: {from_format}")

        if mol is None:
            raise FileFormatError(f"Failed to parse molecule from {from_format}")

        # Add hydrogens for 3D formats
        if to_format in ("pdb", "xyz", "sdf", "mol"):
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)

        # Write output
        if to_format == "smiles":
            return Chem.MolToSmiles(mol)
        elif to_format == "sdf":
            return Chem.MolToMolBlock(mol)
        elif to_format == "mol":
            return Chem.MolToMolBlock(mol)
        elif to_format == "pdb":
            return Chem.MolToPDBBlock(mol)
        elif to_format == "inchi":
            return Chem.inchi.MolToInchi(mol)
        elif to_format == "json":
            import json
            atoms = []
            for atom in mol.GetAtoms():
                atoms.append({
                    "idx": atom.GetIdx(),
                    "atomic_num": atom.GetAtomicNum(),
                    "symbol": atom.GetSymbol(),
                    "formal_charge": atom.GetFormalCharge(),
                })
            bonds = []
            for bond in mol.GetBonds():
                bonds.append({
                    "begin_idx": bond.GetBeginAtomIdx(),
                    "end_idx": bond.GetEndAtomIdx(),
                    "bond_type": str(bond.GetBondType()),
                })
            return json.dumps({"atoms": atoms, "bonds": bonds, "smiles": Chem.MolToSmiles(mol)})
        else:
            raise FileFormatError(f"RDKit cannot write format: {to_format}")

    @classmethod
    def generate_xyz(cls, smiles: str, optimize: bool = True) -> str:
        """Generate XYZ coordinates from SMILES.

        Args:
            smiles: SMILES string.
            optimize: Whether to optimize geometry.

        Returns:
            XYZ format string.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FileFormatError("Invalid SMILES string")

        mol = Chem.AddHs(mol)

        if optimize:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)

        conf = mol.GetConformer()

        lines = [str(mol.GetNumAtoms()), ""]
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol():<2} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

        return "\n".join(lines)

    @classmethod
    def smiles_to_inchi(cls, smiles: str) -> str:
        """Convert SMILES to InChI.

        Args:
            smiles: SMILES string.

        Returns:
            InChI string.
        """
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise FileFormatError("Invalid SMILES string")
        return Chem.inchi.MolToInchi(mol)

    @classmethod
    def inchi_to_smiles(cls, inchi: str) -> str:
        """Convert InChI to SMILES.

        Args:
            inchi: InChI string.

        Returns:
            SMILES string.
        """
        from rdkit import Chem
        mol = Chem.inchi.MolFromInchi(inchi)
        if mol is None:
            raise FileFormatError("Invalid InChI string")
        return Chem.MolToSmiles(mol)
