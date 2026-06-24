"""
Virtual Chemistry Lab API - pKa Predictor
Machine learning model for pKa prediction using molecular descriptors.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs

from app.config import settings
from app.core.utils.exceptions import ValidationError


class PKaPredictor:
    """pKa prediction using XGBoost and molecular descriptors.

    Predicts acid dissociation constants for organic molecules.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained pKa model."""
        if os.path.exists(settings.PKA_MODEL_PATH):
            try:
                with open(settings.PKA_MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
            except Exception:
                pass

    def _get_features(self, smiles: str) -> np.ndarray:
        """Extract features for pKa prediction.

        Args:
            smiles: SMILES string.

        Returns:
            Feature vector.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValidationError(f"Invalid SMILES: {smiles}")

        # Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_arr = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        # Molecular descriptors
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.ExactMolWt(mol),
        ])

        # Check for acidic groups
        acidic_groups = self._detect_acidic_groups(mol)

        return np.concatenate([fp_arr, descriptors, acidic_groups])

    def _detect_acidic_groups(self, mol) -> np.ndarray:
        """Detect acidic functional groups.

        Returns:
            Binary vector indicating presence of acidic groups.
        """
        groups = [
            ("carboxylic_acid", "C(=O)O"),
            ("phenol", "Oc1ccccc1"),
            ("sulfonic_acid", "S(=O)(=O)O"),
            ("phosphoric_acid", "P(=O)(O)O"),
            ("ammonium", "[NH4+]"),
            ("aniline", "Nc1ccccc1"),
            ("amide", "NC=O"),
            ("alcohol", "[OH]"),
            ("thiol", "[SH]"),
            ("imidazole", "c1c[nH]cn1"),
        ]

        features = []
        for name, smarts in groups:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                features.append(1.0)
            else:
                features.append(0.0)

        return np.array(features)

    def predict(self, smiles: str, site: Optional[int] = None) -> Dict[str, Any]:
        """Predict pKa for a molecule.

        Args:
            smiles: SMILES string.
            site: Optional atom index for specific ionizable site.

        Returns:
            Dictionary with predicted pKa and confidence.
        """
        features = self._get_features(smiles)

        if self.model is not None:
            prediction = self.model.predict(features.reshape(1, -1))[0]
            prediction = float(prediction)
            model_status = "loaded"
        else:
            # Heuristic fallback
            prediction = self._heuristic_pka(smiles)
            model_status = "heuristic"

        # Detect ionizable sites
        sites = self._get_ionizable_sites(smiles)

        return {
            "predicted_pka": round(prediction, 2),
            "confidence_interval": [
                round(prediction - 1.5, 2),
                round(prediction + 1.5, 2),
            ],
            "smiles": smiles,
            "ionizable_sites": sites,
            "model_status": model_status,
            "interpretation": self._interpret_pka(prediction),
        }

    def _heuristic_pka(self, smiles: str) -> float:
        """Heuristic pKa prediction based on functional groups."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 7.0

        # Check for common acidic groups
        patterns = [
            ("S(=O)(=O)O", -2.0),    # Sulfonic acid
            ("C(=O)O", 4.5),          # Carboxylic acid
            ("Oc1ccccc1", 10.0),      # Phenol
            ("Nc1ccccc1", 4.6),       # Anilinium
            ("[NH4+]", 9.5),          # Ammonium
            ("c1c[nH]cn1", 6.0),      # Imidazole
            ("[OH]", 15.0),            # Alcohol
            ("[SH]", 10.5),           # Thiol
        ]

        for smarts, pka in patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return pka

        # Default based on molecular properties
        logp = Descriptors.MolLogP(mol)
        if logp > 3:
            return 8.0
        return 7.0

    def _get_ionizable_sites(self, smiles: str) -> List[Dict[str, Any]]:
        """Identify ionizable sites in the molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        sites = []

        # Check each atom
        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()

            # Potential acidic sites
            if atom_type == "O":
                # Check if it's part of OH
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if "H" in neighbors:
                    sites.append({
                        "atom_idx": atom.GetIdx(),
                        "type": "hydroxyl",
                        "estimated_pka_range": [15, 18],
                    })

            elif atom_type == "N":
                # Check if it's an amine
                num_h = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "H")
                if num_h > 0:
                    sites.append({
                        "atom_idx": atom.GetIdx(),
                        "type": "amine",
                        "estimated_pka_range": [9, 11],
                    })

            elif atom_type == "S":
                # Check for thiol
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                if "H" in neighbors:
                    sites.append({
                        "atom_idx": atom.GetIdx(),
                        "type": "thiol",
                        "estimated_pka_range": [10, 11],
                    })

        return sites

    def _interpret_pka(self, pka: float) -> str:
        """Interpret pKa value."""
        if pka < 0:
            return "Strong acid (fully dissociated at physiological pH)"
        elif pka < 4:
            return "Weak acid (mostly dissociated at physiological pH)"
        elif pka < 7:
            return "Moderate acid"
        elif pka < 10:
            return "Weak base / very weak acid"
        elif pka < 14:
            return "Weak base"
        else:
            return "Very weak acid / strong base"

    def train(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """Train the pKa prediction model.

        Args:
            X: Feature matrix.
            y: Target pKa values.
            save_path: Path to save model.
        """
        try:
            import xgboost as xgb

            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            self.model.fit(X, y)

            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump({"model": self.model}, f)

            return True
        except ImportError:
            raise ValidationError("XGBoost not installed")

    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        if self.model is None:
            raise ValidationError("No model to save")

        path = path or settings.PKA_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)
