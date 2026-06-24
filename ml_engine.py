"""
Virtual Chemistry Lab API - ML Engine
Machine learning predictions for chemical properties.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.engines.base import AbstractEngine, CalculationResult
from app.core.utils.exceptions import ValidationError, CalculationError
from app.core.utils.citations import CitationManager


class MLEngine(AbstractEngine):
    """Machine learning prediction engine.

    Provides predictions for yield, pKa, LogP, and solvent recommendations
    using pre-trained XGBoost models and similarity-based methods.
    """

    def __init__(self):
        super().__init__(name="ml", version="1.0.0")
        self._models = {}
        self._solvent_db = None
        self._load_models()

    def _load_models(self):
        """Load pre-trained ML models."""
        try:
            import pickle
            import os
            from app.config import settings

            # Load yield predictor
            if os.path.exists(settings.YIELD_MODEL_PATH):
                with open(settings.YIELD_MODEL_PATH, "rb") as f:
                    self._models["yield"] = pickle.load(f)

            # Load pKa predictor
            if os.path.exists(settings.PKA_MODEL_PATH):
                with open(settings.PKA_MODEL_PATH, "rb") as f:
                    self._models["pka"] = pickle.load(f)

            # Load LogP predictor
            if os.path.exists(settings.LOGP_MODEL_PATH):
                with open(settings.LOGP_MODEL_PATH, "rb") as f:
                    self._models["logp"] = pickle.load(f)

            # Load solvent database
            if os.path.exists(settings.SOLVENT_DB_PATH):
                import json
                with open(settings.SOLVENT_DB_PATH, "r") as f:
                    self._solvent_db = json.load(f)

            self._available = True
        except Exception as e:
            self._available = False

    def _get_supported_methods(self) -> List[str]:
        return [
            "yield_prediction",
            "pka_prediction",
            "logp_prediction",
            "solvent_recommendation",
        ]

    async def validate_input(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters for ML predictions."""
        method = parameters.get("method", "")

        if method == "yield_prediction":
            if "reactants_smiles" not in parameters:
                return False, "reactants_smiles is required for yield prediction"
        elif method == "pka_prediction":
            if "smiles" not in parameters:
                return False, "smiles is required for pKa prediction"
        elif method == "logp_prediction":
            if "smiles" not in parameters:
                return False, "smiles is required for LogP prediction"
        elif method == "solvent_recommendation":
            if "reaction_type" not in parameters and "desired_properties" not in parameters:
                return False, "reaction_type or desired_properties is required for solvent recommendation"
        else:
            return False, f"Unknown method: {method}"

        return True, ""

    async def calculate(self, parameters: Dict[str, Any]) -> CalculationResult:
        """Execute ML prediction."""
        start_time = time.time()
        warnings = []

        try:
            valid, error = await self.validate_input(parameters)
            if not valid:
                raise ValidationError(error)

            method = parameters["method"]

            if method == "yield_prediction":
                result = self._predict_yield(parameters)
            elif method == "pka_prediction":
                result = self._predict_pka(parameters)
            elif method == "logp_prediction":
                result = self._predict_logp(parameters)
            elif method == "solvent_recommendation":
                result = self._recommend_solvent(parameters)
            else:
                raise ValidationError(f"Unknown method: {method}")

            calc_time = time.time() - start_time

            return CalculationResult(
                success=True,
                data=result,
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("yield_prediction"),
            )

        except Exception as e:
            calc_time = time.time() - start_time
            return CalculationResult(
                success=False,
                data={},
                engine_used=self.name,
                calculation_time=calc_time,
                warnings=warnings,
                citations=CitationManager.get_citations("yield_prediction"),
                error_message=str(e),
            )

        """Generate Morgan fingerprint for ML input."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise CalculationError(f"Invalid SMILES: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _predict_yield(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict reaction yield using XGBoost model."""
        reactants = parameters["reactants_smiles"]
        conditions = parameters.get("conditions", {})

        # Generate fingerprints for all reactants
        fps = []
        for smiles in reactants:
            pass
        # Combine fingerprints (sum for multiple reactants)
        combined_fp = np.sum(fps, axis=0)
        combined_fp = np.clip(combined_fp, 0, 1)

        # Add condition features
        temp = conditions.get("temperature", 25.0) / 100.0
        time_h = conditions.get("time_hours", 1.0) / 10.0

        # One-hot encode solvent (simplified)
        solvent = conditions.get("solvent", "DMF")
        common_solvents = ["DMF", "DMSO", "THF", "MeCN", "EtOH", "H2O", "toluene", "dioxane"]
        solvent_features = [1.0 if s == solvent else 0.0 for s in common_solvents]

        features = np.concatenate([
            combined_fp,
            [temp, time_h],
            solvent_features,
        ])

        # Predict
        if "yield" in self._models:
            model = self._models["yield"]
            prediction = model.predict(features.reshape(1, -1))[0]
            prediction = float(np.clip(prediction, 0, 100))
        else:
            # Fallback: heuristic prediction based on conditions
            prediction = self._heuristic_yield_prediction(reactants, conditions)

        return {
            "method": "yield_prediction",
            "predicted_yield": round(prediction, 2),
            "unit": "%",
            "confidence_interval": [round(max(0, prediction - 15), 2), round(min(100, prediction + 15), 2)],
            "reactants": reactants,
            "conditions": conditions,
            "model_used": "xgboost" if "yield" in self._models else "heuristic",
        }

    def _heuristic_yield_prediction(self, reactants: List[str], conditions: Dict[str, Any]) -> float:
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

    def _predict_pka(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict pKa using trained model."""
        smiles = parameters["smiles"]


        if "pka" in self._models:
            model = self._models["pka"]
            prediction = model.predict(fp.reshape(1, -1))[0]
        else:
            # Fallback: heuristic based on functional groups
            prediction = self._heuristic_pka(smiles)

        return {
            "method": "pka_prediction",
            "predicted_pka": round(float(prediction), 2),
            "confidence_interval": [round(float(prediction) - 1.5, 2), round(float(prediction) + 1.5, 2)],
            "smiles": smiles,
            "model_used": "xgboost" if "pka" in self._models else "heuristic",
        }

    def _heuristic_pka(self, smiles: str) -> float:
        """Heuristic pKa prediction based on functional groups."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 7.0

        # Check for common acidic groups
        carboxylic_acid = Chem.MolFromSmarts("C(=O)O")
        phenol = Chem.MolFromSmarts("cO")
        sulfonic = Chem.MolFromSmarts("S(=O)(=O)O")
        amine = Chem.MolFromSmarts("N")

        if mol.HasSubstructMatch(carboxylic_acid):
            return 4.5
        elif mol.HasSubstructMatch(phenol):
            return 10.0
        elif mol.HasSubstructMatch(sulfonic):
            return -2.0
        elif mol.HasSubstructMatch(amine):
            return 9.5

        return 7.0

    def _predict_logp(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict LogP using trained model or RDKit fallback."""
        smiles = parameters["smiles"]


        if "logp" in self._models:
            model = self._models["logp"]
            prediction = model.predict(fp.reshape(1, -1))[0]
        else:
            # Fallback to RDKit
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise CalculationError(f"Invalid SMILES: {smiles}")
            prediction = Descriptors.MolLogP(mol)

        return {
            "method": "logp_prediction",
            "predicted_logp": round(float(prediction), 4),
            "confidence_interval": [round(float(prediction) - 0.5, 4), round(float(prediction) + 0.5, 4)],
            "smiles": smiles,
            "model_used": "xgboost" if "logp" in self._models else "rdkit",
        }

    def _recommend_solvent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend solvents based on reaction type and desired properties."""
        reaction_type = parameters.get("reaction_type", "general")
        desired_props = parameters.get("desired_properties", {})

        # Default solvent database if not loaded
        if self._solvent_db is None:
            self._solvent_db = self._get_default_solvent_db()

        # Score solvents based on criteria
        scored_solvents = []

        for solvent in self._solvent_db:
            score = 0.0

            # Reaction type matching
            if reaction_type in solvent.get("suitable_for", []):
                score += 50

            # Property matching
            if "polarity" in desired_props:
                target_polarity = desired_props["polarity"]
                solvent_polarity = solvent.get("polarity", 0.5)
                score += 30 * (1 - abs(target_polarity - solvent_polarity))

            if "bp_min" in desired_props and "bp_max" in desired_props:
                bp = solvent.get("boiling_point", 100)
                if desired_props["bp_min"] <= bp <= desired_props["bp_max"]:
                    score += 20

            if "green" in desired_props and desired_props["green"]:
                if solvent.get("green", False):
                    score += 15

            scored_solvents.append({
                "name": solvent["name"],
                "smiles": solvent.get("smiles", ""),
                "score": round(score, 2),
                "properties": {
                    "polarity": solvent.get("polarity", None),
                    "boiling_point": solvent.get("boiling_point", None),
                    "dielectric_constant": solvent.get("dielectric_constant", None),
                },
            })

        # Sort by score
        scored_solvents.sort(key=lambda x: x["score"], reverse=True)

        return {
            "method": "solvent_recommendation",
            "reaction_type": reaction_type,
            "recommendations": scored_solvents[:5],
            "total_considered": len(scored_solvents),
        }

    def _get_default_solvent_db(self) -> List[Dict[str, Any]]:
        """Get default solvent database."""
        return [
            {"name": "Water", "smiles": "O", "polarity": 1.0, "boiling_point": 100, "dielectric_constant": 80.1, "green": True, "suitable_for": ["hydrolysis", "general"]},
            {"name": "Ethanol", "smiles": "CCO", "polarity": 0.65, "boiling_point": 78, "dielectric_constant": 24.3, "green": True, "suitable_for": ["general", "reduction"]},
            {"name": "Methanol", "smiles": "CO", "polarity": 0.76, "boiling_point": 65, "dielectric_constant": 32.7, "green": True, "suitable_for": ["general", "oxidation"]},
            {"name": "Acetone", "smiles": "CC(=O)C", "polarity": 0.47, "boiling_point": 56, "dielectric_constant": 20.7, "green": True, "suitable_for": ["general", "extraction"]},
            {"name": "DMF", "smiles": "CN(C)C=O", "polarity": 0.39, "boiling_point": 153, "dielectric_constant": 36.7, "green": False, "suitable_for": ["general", "coupling", "SNAr"]},
            {"name": "DMSO", "smiles": "CS(C)=O", "polarity": 0.44, "boiling_point": 189, "dielectric_constant": 46.7, "green": False, "suitable_for": ["general", "oxidation", "SNAr"]},
            {"name": "THF", "smiles": "C1CCOC1", "polarity": 0.21, "boiling_point": 66, "dielectric_constant": 7.5, "green": True, "suitable_for": ["general", "Grignard", "reduction"]},
            {"name": "Acetonitrile", "smiles": "CC#N", "polarity": 0.46, "boiling_point": 82, "dielectric_constant": 37.5, "green": False, "suitable_for": ["general", "electrochemistry"]},
            {"name": "Toluene", "smiles": "Cc1ccccc1", "polarity": 0.1, "boiling_point": 111, "dielectric_constant": 2.4, "green": False, "suitable_for": ["general", "dehydration"]},
            {"name": "Dichloromethane", "smiles": "ClCCl", "polarity": 0.3, "boiling_point": 40, "dielectric_constant": 8.9, "green": False, "suitable_for": ["general", "extraction"]},
            {"name": "Ethyl acetate", "smiles": "CCOC(=O)C", "polarity": 0.23, "boiling_point": 77, "dielectric_constant": 6.0, "green": True, "suitable_for": ["general", "extraction"]},
            {"name": "Hexane", "smiles": "CCCCCC", "polarity": 0.0, "boiling_point": 69, "dielectric_constant": 1.9, "green": True, "suitable_for": ["extraction", "chromatography"]},
            {"name": "Diethyl ether", "smiles": "CCOCC", "polarity": 0.12, "boiling_point": 35, "dielectric_constant": 4.3, "green": True, "suitable_for": ["extraction", "Grignard"]},
            {"name": "1,4-Dioxane", "smiles": "C1COCCO1", "polarity": 0.16, "boiling_point": 101, "dielectric_constant": 2.2, "green": False, "suitable_for": ["general"]},
            {"name": "NMP", "smiles": "CN1CCCC1=O", "polarity": 0.35, "boiling_point": 202, "dielectric_constant": 32.0, "green": False, "suitable_for": ["general", "coupling"]},
        ]
