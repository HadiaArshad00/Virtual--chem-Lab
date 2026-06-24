"""
Virtual Chemistry Lab API - LogP Predictor
Machine learning model for octanol-water partition coefficient prediction.
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


class LogPPredictor:
    """LogP prediction using XGBoost and molecular descriptors.

    Predicts octanol-water partition coefficient for drug-likeness assessment.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained LogP model."""
        if os.path.exists(settings.LOGP_MODEL_PATH):
            try:
                with open(settings.LOGP_MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
            except Exception:
                pass

    def _get_features(self, smiles: str) -> np.ndarray:
        """Extract features for LogP prediction.

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

        # Extended molecular descriptors
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),  # RDKit's own LogP as feature
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.MolMR(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
        ])

        return np.concatenate([fp_arr, descriptors])

    def predict(self, smiles: str) -> Dict[str, Any]:
        """Predict LogP for a molecule.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary with predicted LogP and confidence.
        """
        features = self._get_features(smiles)

        if self.model is not None:
            prediction = self.model.predict(features.reshape(1, -1))[0]
            prediction = float(prediction)
            model_status = "loaded"
        else:
            # Fallback to RDKit's built-in LogP
            mol = Chem.MolFromSmiles(smiles)
            prediction = Descriptors.MolLogP(mol)
            model_status = "rdkit_fallback"

        return {
            "predicted_logp": round(prediction, 4),
            "confidence_interval": [
                round(prediction - 0.5, 4),
                round(prediction + 0.5, 4),
            ],
            "smiles": smiles,
            "model_status": model_status,
            "interpretation": self._interpret_logp(prediction),
            "drug_likeness": self._assess_drug_likeness(prediction),
        }

    def _interpret_logp(self, logp: float) -> str:
        """Interpret LogP value."""
        if logp < 0:
            return "Very hydrophilic - poor membrane permeability"
        elif logp < 1:
            return "Hydrophilic"
        elif logp < 3:
            return "Moderately lipophilic - good oral absorption range"
        elif logp < 5:
            return "Lipophilic - acceptable for oral drugs"
        else:
            return "Very lipophilic - poor solubility, potential toxicity"

    def _assess_drug_likeness(self, logp: float) -> Dict[str, Any]:
        """Assess drug-likeness based on LogP."""
        lipinski_ok = logp <= 5
        veber_ok = True  # LogP doesn't directly affect Veber

        return {
            "lipinski_rule_of_five": lipinski_ok,
            "recommended_range": "0.5 - 3.0 (optimal for oral drugs)",
            "absorption_risk": "low" if 0 < logp < 5 else "high",
            "solubility_risk": "high" if logp > 4 else "low",
        }

    def batch_predict(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """Predict LogP for multiple molecules.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of prediction dictionaries.
        """
        results = []
        for smiles in smiles_list:
            try:
                result = self.predict(smiles)
                results.append(result)
            except Exception as e:
                results.append({
                    "smiles": smiles,
                    "error": str(e),
                })
        return results

    def train(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """Train the LogP prediction model.

        Args:
            X: Feature matrix.
            y: Target LogP values.
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

        path = path or settings.LOGP_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)
