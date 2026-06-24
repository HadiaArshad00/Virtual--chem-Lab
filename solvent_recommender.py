"""
Virtual Chemistry Lab API - Solvent Recommender
Cosine similarity-based solvent recommendation system.
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.config import settings
from app.core.utils.exceptions import ValidationError


@dataclass
class Solvent:
    """Represents a solvent with properties."""
    name: str
    smiles: str
    polarity: float  # 0-1 scale
    boiling_point: float  # °C
    dielectric_constant: float
    logp: float
    tpsa: float
    hbd: int  # Number of H-bond donors
    hba: int  # Number of H-bond acceptors
    viscosity: float  # cP at 25°C
    flash_point: float  # °C
    green: bool
    suitable_for: List[str]
    descriptors: Optional[Dict[str, float]] = None


class SolventRecommender:
    """Solvent recommendation system based on cosine similarity.

    Recommends solvents for reactions based on desired properties and reaction type.
    """

    def __init__(self):
        self.solvents = self._load_solvent_database()

    def _load_solvent_database(self) -> List[Solvent]:
        """Load solvent database from file or use defaults."""
        if os.path.exists(settings.SOLVENT_DB_PATH):
            try:
                with open(settings.SOLVENT_DB_PATH, "r") as f:
                    data = json.load(f)
                return [Solvent(**s) for s in data]
            except Exception:
                pass

        return self._get_default_solvents()

    def _get_default_solvents(self) -> List[Solvent]:
        """Get default solvent database."""
        return [
            Solvent("Water", "O", 1.0, 100.0, 80.1, -1.4, 17.1, 2, 1, 0.89, None, True, ["hydrolysis", "general", "extraction"]),
            Solvent("Methanol", "CO", 0.85, 64.7, 32.7, -0.77, 20.2, 1, 1, 0.54, 11.0, True, ["general", "reduction", "oxidation"]),
            Solvent("Ethanol", "CCO", 0.75, 78.4, 24.3, -0.31, 20.2, 1, 1, 1.07, 13.0, True, ["general", "reduction", "crystallization"]),
            Solvent("Isopropanol", "CC(C)O", 0.70, 82.6, 18.3, 0.05, 20.2, 1, 1, 2.04, 12.0, True, ["general", "reduction"]),
            Solvent("Acetone", "CC(=O)C", 0.47, 56.0, 20.7, -0.24, 17.1, 0, 1, 0.30, -20.0, True, ["general", "extraction", "crystallization"]),
            Solvent("Acetonitrile", "CC#N", 0.65, 81.6, 37.5, -0.34, 23.8, 0, 1, 0.34, 6.0, False, ["general", "electrochemistry", "SNAr"]),
            Solvent("DMF", "CN(C)C=O", 0.39, 153.0, 36.7, -1.0, 20.3, 0, 1, 0.80, 58.0, False, ["general", "coupling", "SNAr", "deprotonation"]),
            Solvent("DMSO", "CS(C)=O", 0.44, 189.0, 46.7, -1.35, 36.3, 0, 1, 1.99, 95.0, False, ["general", "oxidation", "SNAr"]),
            Solvent("THF", "C1CCOC1", 0.21, 66.0, 7.5, 0.46, 9.2, 0, 1, 0.46, -14.0, True, ["general", "Grignard", "reduction", "crystallization"]),
            Solvent("1,4-Dioxane", "C1COCCO1", 0.16, 101.1, 2.2, -0.27, 9.2, 0, 2, 1.20, 12.0, False, ["general"]),
            Solvent("Diethyl ether", "CCOCC", 0.12, 34.6, 4.3, 0.89, 9.2, 0, 1, 0.22, -45.0, True, ["extraction", "Grignard", "crystallization"]),
            Solvent("MTBE", "COC(C)(C)C", 0.10, 55.2, 4.5, 1.15, 9.2, 0, 1, 0.27, -28.0, False, ["extraction"]),
            Solvent("Toluene", "Cc1ccccc1", 0.10, 110.6, 2.4, 2.73, 0.0, 0, 0, 0.55, 4.0, False, ["general", "dehydration", "crystallization"]),
            Solvent("Benzene", "c1ccccc1", 0.08, 80.1, 2.3, 2.13, 0.0, 0, 0, 0.60, -11.0, False, ["general"]),
            Solvent("Hexane", "CCCCCC", 0.0, 68.7, 1.9, 3.9, 0.0, 0, 0, 0.29, -22.0, True, ["extraction", "chromatography"]),
            Solvent("Heptane", "CCCCCCC", 0.0, 98.4, 1.9, 4.66, 0.0, 0, 0, 0.39, -4.0, True, ["extraction", "chromatography"]),
            Solvent("Cyclohexane", "C1CCCCC1", 0.0, 80.7, 2.0, 3.44, 0.0, 0, 0, 0.89, -20.0, True, ["extraction", "crystallization"]),
            Solvent("Dichloromethane", "ClCCl", 0.30, 39.6, 8.9, 1.25, 0.0, 0, 0, 0.41, None, False, ["general", "extraction", "crystallization"]),
            Solvent("Chloroform", "ClC(Cl)Cl", 0.25, 61.2, 4.8, 1.97, 0.0, 0, 0, 0.54, None, False, ["general", "extraction"]),
            Solvent("Ethyl acetate", "CCOC(=O)C", 0.23, 77.1, 6.0, 0.73, 26.3, 0, 2, 0.43, -4.0, True, ["general", "extraction", "crystallization"]),
            Solvent("Isoamyl acetate", "CC(C)CCOC(=O)C", 0.15, 142.0, 4.6, 2.13, 26.3, 0, 2, 0.86, 25.0, True, ["extraction"]),
            Solvent("Pyridine", "c1ccncc1", 0.35, 115.2, 12.4, 0.65, 12.9, 0, 1, 0.88, 20.0, False, ["general", "deprotonation"]),
            Solvent("NMP", "CN1CCCC1=O", 0.35, 202.0, 32.0, -0.38, 20.3, 0, 1, 1.67, 91.0, False, ["general", "coupling", "SNAr"]),
            Solvent("DMAc", "CC(=O)N(C)C", 0.38, 165.0, 37.8, -0.77, 20.3, 0, 1, 1.96, 70.0, False, ["general", "coupling"]),
            Solvent("Diglyme", "COCCOCCOC", 0.25, 162.0, 7.4, -0.56, 27.7, 0, 3, 1.00, 67.0, False, ["general", "Grignard"]),
            Solvent("PEG-400", "OCCO", 0.40, None, 13.6, -1.5, 40.6, 2, 2, 90.0, None, True, ["general", "green"]),
            Solvent("Glycerol", "OCC(O)CO", 0.90, 290.0, 42.5, -1.76, 60.7, 3, 3, 1410.0, 160.0, True, ["general", "green"]),
            Solvent("Propylene carbonate", "CC1COC(=O)O1", 0.45, 242.0, 64.9, -0.41, 26.3, 0, 3, 2.53, 135.0, False, ["general", "electrochemistry"]),
            Solvent("Sulfolane", "O=S1(=O)CCCC1", 0.50, 285.0, 43.3, -0.77, 26.3, 0, 2, 10.3, 177.0, False, ["general", "electrochemistry"]),
        ]

    def _get_solvent_vector(self, solvent: Solvent) -> np.ndarray:
        """Convert solvent to feature vector."""
        return np.array([
            solvent.polarity,
            solvent.dielectric_constant / 100.0,
            (solvent.boiling_point or 100.0) / 300.0,
            solvent.logp / 5.0,
            solvent.tpsa / 100.0,
            solvent.hbd / 3.0,
            solvent.hba / 3.0,
            (solvent.viscosity or 1.0) / 10.0,
            1.0 if solvent.green else 0.0,
        ])

    def recommend(
        self,
        reaction_type: Optional[str] = None,
        desired_properties: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recommend solvents based on reaction type and desired properties.

        Args:
            reaction_type: Type of reaction (e.g., "coupling", "reduction").
            desired_properties: Dict of desired properties:
                - polarity: Target polarity (0-1).
                - bp_min/bp_max: Boiling point range.
                - dielectric_min/max: Dielectric constant range.
                - green: Prefer green solvents (bool).
                - hbd/hba: H-bond donor/acceptor counts.
            top_n: Number of recommendations to return.
            exclude: List of solvent names to exclude.

        Returns:
            Dictionary with ranked solvent recommendations.
        """
        if desired_properties is None:
            desired_properties = {}

        if exclude is None:
            exclude = []

        # Build query vector from desired properties
        query = np.array([
            desired_properties.get("polarity", 0.5),
            desired_properties.get("dielectric", 20.0) / 100.0,
            (desired_properties.get("bp", 100.0)) / 300.0,
            desired_properties.get("logp", 0.0) / 5.0,
            desired_properties.get("tpsa", 20.0) / 100.0,
            desired_properties.get("hbd", 0) / 3.0,
            desired_properties.get("hba", 1) / 3.0,
            (desired_properties.get("viscosity", 1.0)) / 10.0,
            1.0 if desired_properties.get("green", False) else 0.5,
        ])

        # Score each solvent
        scored = []
        for solvent in self.solvents:
            if solvent.name in exclude:
                continue

            score = 0.0
            details = {}

            # Reaction type matching
            if reaction_type and reaction_type in solvent.suitable_for:
                score += 30
                details["reaction_match"] = True

            # Cosine similarity for properties
            solvent_vec = self._get_solvent_vector(solvent)

            # Avoid division by zero
            norm_query = np.linalg.norm(query)
            norm_solvent = np.linalg.norm(solvent_vec)

            if norm_query > 0 and norm_solvent > 0:
                cosine_sim = np.dot(query, solvent_vec) / (norm_query * norm_solvent)
                score += cosine_sim * 40
                details["property_similarity"] = round(cosine_sim, 4)

            # Property-specific bonuses
            if "bp_min" in desired_properties and solvent.boiling_point:
                if solvent.boiling_point >= desired_properties["bp_min"]:
                    score += 10

            if "bp_max" in desired_properties and solvent.boiling_point:
                if solvent.boiling_point <= desired_properties["bp_max"]:
                    score += 10

            if desired_properties.get("green", False) and solvent.green:
                score += 15
                details["green_bonus"] = True

            scored.append({
                "name": solvent.name,
                "smiles": solvent.smiles,
                "score": round(score, 2),
                "properties": {
                    "polarity": solvent.polarity,
                    "boiling_point": solvent.boiling_point,
                    "dielectric_constant": solvent.dielectric_constant,
                    "logp": solvent.logp,
                    "tpsa": solvent.tpsa,
                    "hbd": solvent.hbd,
                    "hba": solvent.hba,
                    "viscosity": solvent.viscosity,
                    "green": solvent.green,
                },
                "suitable_for": solvent.suitable_for,
                "details": details,
            })

        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "reaction_type": reaction_type,
            "desired_properties": desired_properties,
            "recommendations": scored[:top_n],
            "total_considered": len(scored),
            "green_alternatives": [s for s in scored if s["properties"]["green"]][:3],
        }

    def get_solvent_info(self, solvent_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific solvent.

        Args:
            solvent_name: Name of the solvent.

        Returns:
            Solvent information dictionary or None.
        """
        for solvent in self.solvents:
            if solvent.name.lower() == solvent_name.lower():
                return {
                    "name": solvent.name,
                    "smiles": solvent.smiles,
                    "polarity": solvent.polarity,
                    "boiling_point": solvent.boiling_point,
                    "dielectric_constant": solvent.dielectric_constant,
                    "logp": solvent.logp,
                    "tpsa": solvent.tpsa,
                    "hbd": solvent.hbd,
                    "hba": solvent.hba,
                    "viscosity": solvent.viscosity,
                    "flash_point": solvent.flash_point,
                    "green": solvent.green,
                    "suitable_for": solvent.suitable_for,
                }
        return None

    def list_solvents(self, filter_green: bool = False) -> List[Dict[str, Any]]:
        """List all available solvents.

        Args:
            filter_green: Only return green solvents.

        Returns:
            List of solvent information dictionaries.
        """
        result = []
        for solvent in self.solvents:
            if filter_green and not solvent.green:
                continue
            result.append({
                "name": solvent.name,
                "smiles": solvent.smiles,
                "polarity": solvent.polarity,
                "boiling_point": solvent.boiling_point,
                "green": solvent.green,
            })
        return result

    def save_database(self, path: Optional[str] = None):
        """Save solvent database to file.

        Args:
            path: Path to save database.
        """
        path = path or settings.SOLVENT_DB_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = []
        for solvent in self.solvents:
            data.append({
                "name": solvent.name,
                "smiles": solvent.smiles,
                "polarity": solvent.polarity,
                "boiling_point": solvent.boiling_point,
                "dielectric_constant": solvent.dielectric_constant,
                "logp": solvent.logp,
                "tpsa": solvent.tpsa,
                "hbd": solvent.hbd,
                "hba": solvent.hba,
                "viscosity": solvent.viscosity,
                "flash_point": solvent.flash_point,
                "green": solvent.green,
                "suitable_for": solvent.suitable_for,
            })

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
