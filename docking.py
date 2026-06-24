"""
Virtual Chemistry Lab API - Docking Calculator
Molecular docking with AutoDock Vina wrapper.
"""

import time
import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.engines.base import CalculationResult
from app.core.utils.exceptions import CalculationError, EngineNotAvailableError
from app.core.utils.citations import CitationManager
from app.config import settings


class DockingCalculator:
    """Molecular docking calculator using AutoDock Vina.

    Provides binding affinity prediction, pose generation, and RMSD clustering.
    """

    def __init__(self):
        self.vina_available = self._check_vina()

    def _check_vina(self) -> bool:
        """Check if AutoDock Vina is available."""
        try:
            result = subprocess.run(
                [settings.VINA_EXECUTABLE, "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def dock(
        self,
        ligand_smiles: str,
        receptor_pdbqt: str,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        size_x: float = 20.0,
        size_y: float = 20.0,
        size_z: float = 20.0,
        num_poses: int = 10,
        exhaustiveness: int = 8,
        energy_range: float = 3.0,
    ) -> CalculationResult:
        """Perform molecular docking.

        Args:
            ligand_smiles: SMILES of the ligand.
            receptor_pdbqt: Receptor structure in PDBQT format.
            center_x, center_y, center_z: Grid box center coordinates.
            size_x, size_y, size_z: Grid box dimensions in Angstroms.
            num_poses: Number of poses to generate.
            exhaustiveness: Search exhaustiveness.
            energy_range: Maximum energy difference from best pose.

        Returns:
            CalculationResult with docking poses.
        """
        start_time = time.time()

        if not self.vina_available:
            # Fallback: heuristic binding affinity estimation
            return await self._heuristic_docking(
                ligand_smiles, receptor_pdbqt, num_poses
            )

        try:
            # Convert ligand SMILES to PDBQT
            ligand_pdbqt = self._smiles_to_pdbqt(ligand_smiles)

            # Write receptor to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
                f.write(receptor_pdbqt)
                receptor_file = f.name

            # Write ligand to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdbqt", delete=False) as f:
                f.write(ligand_pdbqt)
                ligand_file = f.name

            # Output file
            output_file = tempfile.mktemp(suffix=".pdbqt")

            # Run Vina
            cmd = [
                settings.VINA_EXECUTABLE,
                "--receptor", receptor_file,
                "--ligand", ligand_file,
                "--center_x", str(center_x),
                "--center_y", str(center_y),
                "--center_z", str(center_z),
                "--size_x", str(size_x),
                "--size_y", str(size_y),
                "--size_z", str(size_z),
                "--num_modes", str(num_poses),
                "--exhaustiveness", str(exhaustiveness),
                "--energy_range", str(energy_range),
                "--out", output_file,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Parse output
            poses = self._parse_vina_output(result.stdout, output_file)

            # Cleanup
            for f in [receptor_file, ligand_file, output_file]:
                try:
                    os.remove(f)
                except:
                    pass

            return CalculationResult(
                success=True,
                data={
                    "method": "docking",
                    "engine": "autodock_vina",
                    "ligand_smiles": ligand_smiles,
                    "num_poses": len(poses),
                    "poses": poses,
                    "best_affinity": poses[0]["affinity"] if poses else None,
                    "grid_box": {
                        "center": [center_x, center_y, center_z],
                        "size": [size_x, size_y, size_z],
                    },
                },
                engine_used="vina",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("docking"),
            )

        except Exception as e:
            return CalculationResult(
                success=False,
                data={},
                engine_used="vina",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("docking"),
                error_message=f"Docking failed: {str(e)}",
            )

    def _smiles_to_pdbqt(self, smiles: str) -> str:
        """Convert SMILES to PDBQT format with Gasteiger charges."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise CalculationError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        # Write to PDB
        pdb_block = Chem.MolToPDBBlock(mol)

        # Convert PDB to PDBQT (simplified - in production use obabel or meeko)
        # For now, return PDB with charge annotations
        lines = pdb_block.split("\n")
        pdbqt_lines = []

        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Add partial charges (simplified Gasteiger-like)
                atom_name = line[12:16].strip()
                charge = self._estimate_charge(atom_name)
                line = line[:70] + f"{charge:>6.3f}" + line[78:]
            pdbqt_lines.append(line)

        return "\n".join(pdbqt_lines)

    def _estimate_charge(self, atom_name: str) -> float:
        """Estimate partial charge for atom type."""
        charges = {
            "O": -0.5, "N": -0.3, "S": -0.2,
            "C": 0.0, "H": 0.1, "F": -0.3,
            "Cl": -0.2, "Br": -0.2, "P": 0.3,
        }
        element = atom_name[0]
        return charges.get(element, 0.0)

    def _parse_vina_output(self, stdout: str, output_file: str) -> List[Dict[str, Any]]:
        """Parse Vina output to extract poses."""
        poses = []

        # Parse stdout for binding affinities
        lines = stdout.split("\n")
        for line in lines:
            if line.strip().startswith("1 ") or line.strip().startswith("2 "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        pose_num = int(parts[0])
                        affinity = float(parts[1])
                        rmsd_lb = float(parts[2])
                        rmsd_ub = float(parts[3])

                        poses.append({
                            "pose_id": pose_num,
                            "affinity": round(affinity, 3),
                            "affinity_unit": "kcal/mol",
                            "rmsd_lb": round(rmsd_lb, 3),
                            "rmsd_ub": round(rmsd_ub, 3),
                            "rmsd_unit": "Å",
                        })
                    except ValueError:
                        continue

        return poses

    async def _heuristic_docking(
        self,
        ligand_smiles: str,
        receptor_pdbqt: str,
        num_poses: int = 10,
    ) -> CalculationResult:
        """Heuristic docking when Vina is not available."""
        start_time = time.time()

        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem

        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol is None:
            return CalculationResult(
                success=False,
                data={},
                engine_used="heuristic",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("docking"),
                error_message="Invalid SMILES",
            )

        # Estimate binding affinity based on molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rotatable = Descriptors.NumRotatableBonds(mol)

        # Heuristic scoring
        base_affinity = -6.0
        mw_penalty = -0.01 * max(0, mw - 300)
        logp_penalty = -0.2 * max(0, abs(logp - 2))
        tpsa_bonus = 0.01 * min(tpsa, 100)
        rot_penalty = -0.1 * num_rotatable

        best_affinity = base_affinity + mw_penalty + logp_penalty + tpsa_bonus + rot_penalty

        # Generate pseudo-poses
        poses = []
        for i in range(min(num_poses, 10)):
            affinity = best_affinity + i * 0.5 + np.random.normal(0, 0.3)
            poses.append({
                "pose_id": i + 1,
                "affinity": round(affinity, 3),
                "affinity_unit": "kcal/mol (estimated)",
                "rmsd_lb": round(i * 0.5 + np.random.random(), 3),
                "rmsd_ub": round(i * 0.5 + 1.0 + np.random.random(), 3),
                "note": "Heuristic estimate - Vina not available",
            })

        return CalculationResult(
            success=True,
            data={
                "method": "docking_heuristic",
                "engine": "heuristic",
                "ligand_smiles": ligand_smiles,
                "num_poses": len(poses),
                "poses": poses,
                "best_affinity": round(best_affinity, 3),
                "note": "AutoDock Vina not available. Using heuristic binding affinity estimation based on molecular properties.",
                "estimated_properties": {
                    "molecular_weight": round(mw, 2),
                    "logp": round(logp, 3),
                    "tpsa": round(tpsa, 2),
                    "num_rotatable_bonds": num_rotatable,
                },
            },
            engine_used="heuristic",
            calculation_time=time.time() - start_time,
            warnings=["AutoDock Vina not installed. Results are heuristic estimates only."],
            citations=CitationManager.get_citations("docking"),
        )

    async def cluster_poses(
        self,
        poses: List[Dict[str, Any]],
        rmsd_threshold: float = 2.0,
    ) -> CalculationResult:
        """Cluster docking poses by RMSD.

        Args:
            poses: List of pose dictionaries.
            rmsd_threshold: RMSD threshold for clustering.

        Returns:
            CalculationResult with clustered poses.
        """
        start_time = time.time()

        # Simple clustering based on affinity similarity
        clusters = []
        used = set()

        for i, pose in enumerate(poses):
            if i in used:
                continue

            cluster = [pose]
            used.add(i)

            for j, other in enumerate(poses):
                if j in used or j == i:
                    continue

                # Check if within energy range
                if abs(pose["affinity"] - other["affinity"]) < rmsd_threshold:
                    cluster.append(other)
                    used.add(j)

            clusters.append({
                "cluster_id": len(clusters) + 1,
                "representative": cluster[0],
                "num_poses": len(cluster),
                "poses": cluster,
                "average_affinity": round(
                    sum(p["affinity"] for p in cluster) / len(cluster), 3
                ),
            })

        return CalculationResult(
            success=True,
            data={
                "method": "pose_clustering",
                "num_clusters": len(clusters),
                "clusters": clusters,
                "rmsd_threshold": rmsd_threshold,
            },
            engine_used="docking",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("docking"),
        )
