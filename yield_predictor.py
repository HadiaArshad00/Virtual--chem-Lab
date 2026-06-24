"""
Virtual Chemistry Lab API - Yield Predictor
XGBoost-based reaction yield prediction using Morgan fingerprints and reaction conditions.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from app.config import settings
from app.core.utils.exceptions import ValidationError


class YieldPredictor:
    """Reaction yield predictor using XGBoost and Morgan fingerprints.

    Features:
    - Morgan fingerprints (2048 bits) for reactants
    - Reaction conditions (temperature, time, solvent)
    - Confidence intervals and feature importance
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load pre-trained XGBoost model."""
        if os.path.exists(settings.YIELD_MODEL_PATH):
            try:
                with open(settings.YIELD_MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
                    self.feature_names = data.get("feature_names")
            except Exception:
                pass

        """Generate Morgan fingerprint for a molecule.

        Args:
            smiles: SMILES string.
            n_bits: Number of bits in fingerprint.

        Returns:
            Numpy array of fingerprint bits.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValidationError(f"Invalid SMILES: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def predict(
        self,
        reactants_smiles: List[str],
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Predict reaction yield.

        Args:
            reactants_smiles: List of reactant SMILES strings.
            conditions: Reaction conditions dict with keys:
                - temperature: Temperature in Celsius (default: 25)
                - time_hours: Reaction time in hours (default: 1)
                - solvent: Solvent name (default: "DMF")
                - catalyst: Catalyst name (optional)

        Returns:
            Dictionary with predicted yield, confidence interval, and feature importance.
        """
        if conditions is None:
            conditions = {}

        # Generate fingerprints for all reactants
        fps = []
        for smiles in reactants_smiles:
            pass
        # Combine fingerprints (sum for multiple reactants, then clip)
        combined_fp = np.sum(fps, axis=0)
        combined_fp = np.clip(combined_fp, 0, 1)

        # Encode conditions
        temp = conditions.get("temperature", 25.0) / 100.0
        time_h = conditions.get("time_hours", 1.0) / 10.0

        # One-hot encode solvent
        common_solvents = ["DMF", "DMSO", "THF", "MeCN", "EtOH", "H2O", "toluene", "dioxane", "acetone", "DCM"]
        solvent = conditions.get("solvent", "DMF")
        solvent_features = [1.0 if s == solvent else 0.0 for s in common_solvents]

        # Combine all features
        features = np.concatenate([
            combined_fp,
            [temp, time_h],
            solvent_features,
        ])

        # Predict
        if self.model is not None:
            prediction = self.model.predict(features.reshape(1, -1))[0]
            prediction = float(np.clip(prediction, 0, 100))

            # Get feature importance if available
            importance = self._get_feature_importance()
        else:
            # Fallback heuristic prediction
            prediction = self._heuristic_predict(reactants_smiles, conditions)
            importance = None

        # Calculate confidence interval
        confidence = self._calculate_confidence(reactants_smiles, conditions)

        return {
            "predicted_yield": round(prediction, 2),
            "unit": "%",
            "confidence_interval": [
                round(max(0, prediction - confidence), 2),
                round(min(100, prediction + confidence), 2),
            ],
            "confidence": round(confidence, 2),
            "reactants": reactants_smiles,
            "conditions": conditions,
            "feature_importance": importance,
            "model_status": "loaded" if self.model else "heuristic",
        }

    def _heuristic_predict(self, reactants: List[str], conditions: Dict[str, Any]) -> float:
        """Heuristic yield prediction when model is not available."""
        base_yield = 65.0

        # Temperature effect
        temp = conditions.get("temperature", 25.0)
        if temp < 0:
            base_yield -= 20
        elif temp > 100:
            base_yield += 10

        # Time effect
        time_h = conditions.get("time_hours", 1.0)
        if time_h < 0.5:
            base_yield -= 15
        elif time_h > 12:
            base_yield += 5

        # Solvent effect
        solvent = conditions.get("solvent", "DMF")
        good_solvents = ["DMF", "DMSO", "NMP"]
        if solvent in good_solvents:
            base_yield += 10

        return float(np.clip(base_yield, 5, 95))

    def _calculate_confidence(self, reactants: List[str], conditions: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on data coverage."""
        # Higher confidence for common conditions
        confidence = 15.0  # Base uncertainty

        # Reduce uncertainty for common solvents
        common_solvents = ["DMF", "DMSO", "THF", "MeCN", "EtOH", "H2O", "toluene"]
        if conditions.get("solvent", "") in common_solvents:
            confidence -= 3

        # Reduce uncertainty for moderate temperatures
        temp = conditions.get("temperature", 25.0)
        if 20 <= temp <= 100:
            confidence -= 2

        return max(5, confidence)

    def _get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance from model."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None

        importances = self.model.feature_importances_

        # Get top features
        top_indices = np.argsort(importances)[-10:][::-1]

        return {
            "top_features": [
                {"feature_index": int(idx), "importance": round(float(importances[idx]), 6)}
                for idx in top_indices
            ],
            "num_features": len(importances),
        }

    def train(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """Train the yield prediction model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target yields (n_samples,).
            save_path: Path to save trained model.
        """
        try:
            import xgboost as xgb

            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            self.model.fit(X, y)

            if save_path:
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "model": self.model,
                        "feature_names": self.feature_names,
                    }, f)

            return True
        except ImportError:
            raise ValidationError("XGBoost not installed. Install with: pip install xgboost")

    def save_model(self, path: Optional[str] = None):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValidationError("No model to save")

        path = path or settings.YIELD_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
            }, f)

    def generate_training_data(
        self,
        reactions: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate training data from reaction database.

        Args:
            reactions: List of reaction dictionaries with keys:
                - reactants: List of SMILES
                - conditions: Dict of conditions
                - yield: Actual yield (0-100)

        Returns:
            Tuple of (X, y) arrays.
        """
        X_list = []
        y_list = []

        for reaction in reactions:
            try:
                fps = []
                for smiles in reaction["reactants"]:
                    fps.append(self._get_fingerprint(smiles))
                combined_fp = np.sum(fps, axis=0)
                combined_fp = np.clip(combined_fp, 0, 1)

                conditions = reaction.get("conditions", {})
                temp = conditions.get("temperature", 25.0) / 100.0
                time_h = conditions.get("time_hours", 1.0) / 10.0

                common_solvents = ["DMF", "DMSO", "THF", "MeCN", "EtOH", "H2O", "toluene", "dioxane", "acetone", "DCM"]
                solvent = conditions.get("solvent", "DMF")
                solvent_features = [1.0 if s == solvent else 0.0 for s in common_solvents]

                features = np.concatenate([
                    combined_fp,
                    [temp, time_h],
                    solvent_features,
                ])

                X_list.append(features)
                y_list.append(reaction["yield"])
            except Exception:
                continue

        return np.array(X_list), np.array(y_list)
