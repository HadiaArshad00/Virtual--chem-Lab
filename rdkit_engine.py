"""
Virtual Chemistry Lab API - RDKit Engine
Molecular operations, fingerprints, descriptors, and format conversion.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from app.core.engines.base import AbstractEngine, CalculationResult
from app.core.utils.exceptions import ValidationError, CalculationError
from app.core.utils.citations import CitationManager
from app.core.utils.validators import validate_smiles_strict


class RDKitEngine(AbstractEngine):
    """RDKit-based molecular calculation engine.

    Handles SMILES validation, molecular descriptors, fingerprints,
    structure optimization, substructure matching, and format conversion.
    """

    def __init__(self):
        super().__init__(name="rdkit", version="2024.03")
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
            from rdkit.Chem import MACCSkeys
            from rdkit import DataStructs
            self._available = True
        except ImportError:
            self._available = False

    def _get_supported_methods(self) -> List[str]:
        return [
            "smiles_validation",
            "molecular_weight",
            "logp",
            "tpsa",
            "descriptors",
            "morgan_fingerprint",
            "maccs_fingerprint",
            "structure_optimization",
            "substructure_match",
            "format_conversion",
            "conformer_generation",
            "similarity",
        ]

    async def validate_input(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters for RDKit calculations."""
        if "smiles" not in parameters and "molecule" not in parameters:
            return False, "Either 'smiles' or 'molecule' is required"

        if "smiles" in parameters:
            try:
                validate_smiles_strict(parameters["smiles"])
            except Exception as e:
                return False, str(e)

        return True, ""

    async def calculate(self, parameters: Dict[str, Any]) -> CalculationResult:
        """Execute RDKit-based calculation."""
        start_time = time.time()
        warnings = []

        try:
            valid, error = await self.validate_input(parameters)
            if not valid:
                raise ValidationError(error)

            method = parameters.get("method", "descriptors")

            if method == "smiles_validation":
                result = self._validate_smiles(parameters["smiles"])
            elif method == "molecular_weight":
                result = self._calculate_molecular_weight(parameters["smiles"])
            elif method == "logp":
                result = self._calculate_logp(parameters["smiles"])
            elif method == "tpsa":
                result = self._calculate_tpsa(parameters["smiles"])
            elif method == "descriptors":
                result = self._calculate_all_descriptors(parameters["smiles"])
            elif method == "morgan_fingerprint":
                result = self._generate_morgan_fingerprint(
                    parameters["smiles"],
                    radius=parameters.get("radius", 2),
                    n_bits=parameters.get("n_bits", 2048),
                )
            elif method == "maccs_fingerprint":
                result = self._generate_maccs_fingerprint(parameters["smiles"])
            elif method == "structure_optimization":
                result = self._optimize_structure(parameters["smiles"])
            elif method == "substructure_match":
                result = self._substructure_match(
                    parameters["smiles"],
                    parameters.get("query_smiles", "")
                )
            elif method == "format_conversion":
                result = self._convert_format(
                    parameters["smiles"],
                    parameters.get("target_format", "sdf")
                )
            elif method == "conformer_generation":
                result = self._generate_conformers(
                    parameters["smiles"],
                    num_conformers=parameters.get("num_conformers", 10)
                )
            elif method == "similarity":
                result = self._calculate_similarity(
                    parameters["smiles"],
                    parameters.get("target_smiles", "")
                )
            else:
                raise ValidationError(f"Unknown method: {method}")

            calc_time = time.time() - start_time

            return CalculationResult(
                success=True,
                data=result,
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("rdkit"),
            )

        except Exception as e:
            calc_time = time.time() - start_time
            return CalculationResult(
                success=False,
                data={},
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("rdkit"),
                error_message=str(e),
            )

    def _get_mol(self, smiles: str):
        """Get RDKit molecule object from SMILES."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise CalculationError(f"Failed to parse SMILES: {smiles}")
        return mol

    def _validate_smiles(self, smiles: str) -> Dict[str, Any]:
        """Validate SMILES string."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        is_valid = mol is not None
        return {
            "smiles": smiles,
            "valid": is_valid,
            "num_atoms": mol.GetNumAtoms() if is_valid else 0,
            "num_bonds": mol.GetNumBonds() if is_valid else 0,
        }

    def _calculate_molecular_weight(self, smiles: str) -> Dict[str, Any]:
        """Calculate molecular weight."""
        from rdkit.Chem import Descriptors
        mol = self._get_mol(smiles)
        mw = Descriptors.MolWt(mol)
        return {
            "smiles": smiles,
            "molecular_weight": round(mw, 4),
            "exact_molecular_weight": round(Descriptors.ExactMolWt(mol), 6),
        }

    def _calculate_logp(self, smiles: str) -> Dict[str, Any]:
        """Calculate LogP (octanol-water partition coefficient)."""
        from rdkit.Chem import Descriptors
        mol = self._get_mol(smiles)
        logp = Descriptors.MolLogP(mol)
        return {
            "smiles": smiles,
            "logp": round(logp, 4),
            "interpretation": self._interpret_logp(logp),
        }

    def _interpret_logp(self, logp: float) -> str:
        """Interpret LogP value."""
        if logp < 0:
            return "Very hydrophilic (poor membrane permeability)"
        elif logp < 1:
            return "Hydrophilic"
        elif logp < 3:
            return "Moderately lipophilic (good oral absorption)"
        elif logp < 5:
            return "Lipophilic"
        else:
            return "Very lipophilic (poor solubility)"

    def _calculate_tpsa(self, smiles: str) -> Dict[str, Any]:
        """Calculate topological polar surface area."""
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = self._get_mol(smiles)
        tpsa = Descriptors.TPSA(mol)
        return {
            "smiles": smiles,
            "tpsa": round(tpsa, 2),
            "tpsa_angstrom2": round(tpsa, 2),
            "interpretation": self._interpret_tpsa(tpsa),
        }

    def _interpret_tpsa(self, tpsa: float) -> str:
        """Interpret TPSA value."""
        if tpsa < 60:
            return "Good blood-brain barrier penetration"
        elif tpsa < 90:
            return "Moderate BBB penetration"
        elif tpsa < 140:
            return "Poor BBB penetration (good oral absorption)"
        else:
            return "Very poor BBB penetration"

    def _calculate_all_descriptors(self, smiles: str) -> Dict[str, Any]:
        """Calculate comprehensive molecular descriptors."""
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
        mol = self._get_mol(smiles)

        descriptors = {
            "smiles": smiles,
            "molecular_weight": round(Descriptors.MolWt(mol), 4),
            "exact_molecular_weight": round(Descriptors.ExactMolWt(mol), 6),
            "logp": round(Descriptors.MolLogP(mol), 4),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "fraction_sp3": round(Descriptors.FractionCSP3(mol), 4),
            "molar_refractivity": round(Descriptors.MolMR(mol), 4),
            "heavy_atom_count": mol.GetNumHeavyAtoms(),
            "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
            "lipinski_violations": Lipinski.NumHBD(mol) > 5 or Lipinski.NumHBA(mol) > 10 or Descriptors.MolWt(mol) > 500 or Descriptors.MolLogP(mol) > 5,
        }

        # Lipinski's Rule of Five
        lipinski = {
            "mw_ok": Descriptors.MolWt(mol) <= 500,
            "logp_ok": Descriptors.MolLogP(mol) <= 5,
            "hbd_ok": Lipinski.NumHBD(mol) <= 5,
            "hba_ok": Lipinski.NumHBA(mol) <= 10,
            "passes_rule_of_five": all([
                Descriptors.MolWt(mol) <= 500,
                Descriptors.MolLogP(mol) <= 5,
                Lipinski.NumHBD(mol) <= 5,
                Lipinski.NumHBA(mol) <= 10,
            ]),
        }
        descriptors["lipinski"] = lipinski

        return descriptors

        """Generate Morgan (circular) fingerprint."""
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        mol = self._get_mol(smiles)
        mol = Chem.AddHs(mol) if mol.GetNumAtoms() < 50 else mol

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        fp_array = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_array)

        # Get non-zero indices for efficiency
        on_bits = list(fp.GetOnBits())

        return {
            "smiles": smiles,
            "fingerprint_type": "Morgan",
            "radius": radius,
            "n_bits": n_bits,
            "num_on_bits": len(on_bits),
            "on_bits": on_bits[:100],  # Limit for response size
            "density": round(len(on_bits) / n_bits, 4),
            "fp_hex": fp.ToHex()[:64] + "..." if len(fp.ToHex()) > 64 else fp.ToHex(),
        }

        """Generate MACCS fingerprint."""
        from rdkit.Chem import MACCSkeys
        from rdkit import DataStructs
        mol = self._get_mol(smiles)

        fp = MACCSkeys.GenMACCSKeys(mol)
        on_bits = list(fp.GetOnBits())

        return {
            "smiles": smiles,
            "fingerprint_type": "MACCS",
            "n_bits": 167,
            "num_on_bits": len(on_bits),
            "on_bits": on_bits,
            "density": round(len(on_bits) / 167, 4),
        }

    def _optimize_structure(self, smiles: str) -> Dict[str, Any]:
        """Optimize molecular structure using MMFF94."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = self._get_mol(smiles)
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # Optimize with MMFF94
        result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")

        # Get energy
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        energy = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()

        # Get conformer
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append({
                "atom": mol.GetAtomWithIdx(i).GetSymbol(),
                "x": round(pos.x, 6),
                "y": round(pos.y, 6),
                "z": round(pos.z, 6),
            })

        return {
            "smiles": smiles,
            "energy": round(energy, 4),
            "energy_unit": "kcal/mol",
            "convergence": result == 0,
            "num_atoms": mol.GetNumAtoms(),
            "coordinates": coords,
            "xyz": self._mol_to_xyz(mol, conf),
        }

    def _mol_to_xyz(self, mol, conf) -> str:
        """Convert molecule to XYZ format."""
        lines = [str(mol.GetNumAtoms()), ""]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            lines.append(f"{symbol:<2} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        return "\n".join(lines)

    def _substructure_match(self, smiles: str, query_smiles: str) -> Dict[str, Any]:
        """Check for substructure match."""
        from rdkit import Chem
        mol = self._get_mol(smiles)
        query = Chem.MolFromSmiles(query_smiles)

        if query is None:
            raise ValidationError("Invalid query SMILES")

        match = mol.HasSubstructMatch(query)
        matches = mol.GetSubstructMatches(query) if match else []

        return {
            "smiles": smiles,
            "query_smiles": query_smiles,
            "has_match": match,
            "num_matches": len(matches),
            "matches": [list(m) for m in matches],
        }

    def _convert_format(self, smiles: str, target_format: str) -> Dict[str, Any]:
        """Convert molecule to different format."""
        from rdkit import Chem
        mol = self._get_mol(smiles)

        result = {"smiles": smiles, "target_format": target_format}

        if target_format == "sdf":
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            result["data"] = Chem.MolToMolBlock(mol)
        elif target_format == "inchi":
            result["data"] = Chem.inchi.MolToInchi(mol)
        elif target_format == "inchi_key":
            result["data"] = Chem.inchi.MolToInchiKey(mol)
        elif target_format == "pdb":
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            result["data"] = Chem.MolToPDBBlock(mol)
        elif target_format == "json":
            import json
            atoms = []
            for atom in mol.GetAtoms():
                atoms.append({
                    "idx": atom.GetIdx(),
                    "symbol": atom.GetSymbol(),
                    "atomic_num": atom.GetAtomicNum(),
                    "formal_charge": atom.GetFormalCharge(),
                    "hybridization": str(atom.GetHybridization()),
                })
            bonds = []
            for bond in mol.GetBonds():
                bonds.append({
                    "begin": bond.GetBeginAtomIdx(),
                    "end": bond.GetEndAtomIdx(),
                    "type": str(bond.GetBondType()),
                    "is_aromatic": bond.GetIsAromatic(),
                })
            result["data"] = json.dumps({"atoms": atoms, "bonds": bonds})
        else:
            raise ValidationError(f"Unsupported target format: {target_format}")

        return result

    def _generate_conformers(self, smiles: str, num_conformers: int = 10) -> Dict[str, Any]:
        """Generate multiple conformers."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = self._get_mol(smiles)
        mol = Chem.AddHs(mol)

        AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=0xf00d)
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94")

        conformers = []
        for conf_id in range(mol.GetNumConformers()):
            conf = mol.GetConformer(conf_id)
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            energy = ff.CalcEnergy()

            conformers.append({
                "conf_id": conf_id,
                "energy": round(energy, 4),
                "energy_unit": "kcal/mol",
            })

        # Sort by energy
        conformers.sort(key=lambda x: x["energy"])

        return {
            "smiles": smiles,
            "num_conformers": len(conformers),
            "conformers": conformers,
            "lowest_energy": conformers[0]["energy"] if conformers else None,
        }

    def _calculate_similarity(self, smiles1: str, smiles2: str) -> Dict[str, Any]:
        """Calculate Tanimoto similarity between two molecules."""
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        mol1 = self._get_mol(smiles1)
        mol2 = self._get_mol(smiles2)

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)

        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        dice = DataStructs.DiceSimilarity(fp1, fp2)
        cosine = DataStructs.CosineSimilarity(fp1, fp2)

        return {
            "smiles1": smiles1,
            "smiles2": smiles2,
            "tanimoto_similarity": round(tanimoto, 6),
            "dice_similarity": round(dice, 6),
            "cosine_similarity": round(cosine, 6),
            "interpretation": self._interpret_similarity(tanimoto),
        }

    def _interpret_similarity(self, tanimoto: float) -> str:
        """Interpret Tanimoto similarity."""
        if tanimoto > 0.85:
            return "Very similar (likely same scaffold)"
        elif tanimoto > 0.7:
            return "Similar (common substructure)"
        elif tanimoto > 0.5:
            return "Moderately similar"
        elif tanimoto > 0.3:
            return "Weakly similar"
        else:
            return "Dissimilar"
