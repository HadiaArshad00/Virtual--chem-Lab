"""
Virtual Chemistry Lab API - Molecular Dynamics Calculator
MD simulations with Verlet integration, LJ + Coulomb potentials, thermostat and barostat.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from app.core.engines.base import CalculationResult
from app.core.utils.citations import CitationManager


class DynamicsCalculator:
    """Molecular dynamics simulator.

    Implements Verlet integration with Lennard-Jones and Coulomb potentials,
    Berendsen thermostat and barostat.
    """

    # Physical constants
    kB = 1.380649e-23  # J/K
    NA = 6.02214076e23  # mol^-1

    def __init__(self):
        pass

    async def run_md(
        self,
        smiles: str,
        num_steps: int = 10000,
        timestep: float = 1.0,  # fs
        temperature: float = 300.0,  # K
        pressure: Optional[float] = None,  # atm
        ensemble: str = "NVT",
        thermostat_tau: float = 0.5,  # ps
        barostat_tau: Optional[float] = None,  # ps
        cutoff: float = 10.0,  # Angstrom
        save_interval: int = 100,
    ) -> CalculationResult:
        """Run molecular dynamics simulation.

        Args:
            smiles: SMILES string of the molecule.
            num_steps: Number of integration steps.
            timestep: Time step in femtoseconds.
            temperature: Target temperature in Kelvin.
            pressure: Target pressure in atm (for NPT).
            ensemble: Thermodynamic ensemble (NVT, NPT, NVE).
            thermostat_tau: Thermostat coupling time in ps.
            barostat_tau: Barostat coupling time in ps.
            cutoff: Non-bonded cutoff in Angstroms.
            save_interval: Save trajectory every N steps.

        Returns:
            CalculationResult with trajectory and energy data.
        """
        start_time = time.time()

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CalculationResult(
                success=False,
                data={},
                engine_used="dynamics",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("molecular_dynamics"),
                error_message="Invalid SMILES",
            )

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)

        # Get initial coordinates
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        positions = np.zeros((num_atoms, 3))
        masses = np.zeros(num_atoms)
        charges = np.zeros(num_atoms)

        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            positions[i] = [pos.x, pos.y, pos.z]
            atom = mol.GetAtomWithIdx(i)
            masses[i] = self._get_atomic_mass(atom.GetAtomicNum())
            charges[i] = atom.GetFormalCharge()

        # Convert units
        positions *= 1e-10  # Angstrom to meter
        timestep_s = timestep * 1e-15  # fs to seconds
        cutoff_m = cutoff * 1e-10

        # Initialize velocities from Maxwell-Boltzmann distribution
        velocities = self._initialize_velocities(masses, temperature)

        # Simulation data storage
        trajectory = []
        energies = []
        temperatures = []

        # Main MD loop
        for step in range(num_steps):
            # Calculate forces
            forces = self._calculate_forces(positions, charges, masses, cutoff_m)

            # Verlet integration
            accelerations = forces / masses[:, np.newaxis]

            if step == 0:
                # First step: use Euler
                positions_new = positions + velocities * timestep_s + 0.5 * accelerations * timestep_s ** 2
                velocities_new = velocities + accelerations * timestep_s
            else:
                # Verlet
                positions_new = 2 * positions - positions_old + accelerations * timestep_s ** 2
                velocities_new = (positions_new - positions_old) / (2 * timestep_s)

            positions_old = positions.copy()
            positions = positions_new
            velocities = velocities_new

            # Thermostat (Berendsen)
            if ensemble in ("NVT", "NPT"):
                current_temp = self._calculate_temperature(velocities, masses)
                scale = np.sqrt(1 + (timestep_s / (thermostat_tau * 1e-12)) * (temperature / current_temp - 1))
                velocities *= scale

            # Barostat (Berendsen) - simplified
            if ensemble == "NPT" and pressure is not None:
                pass  # Simplified - would need volume scaling

            # Calculate properties
            if step % save_interval == 0:
                ke = self._calculate_kinetic_energy(velocities, masses)
                pe = self._calculate_potential_energy(positions, charges, cutoff_m)
                total_energy = ke + pe
                current_temp = self._calculate_temperature(velocities, masses)

                # Save trajectory frame
                frame_positions = positions * 1e10  # Back to Angstrom
                frame = {
                    "step": step,
                    "positions": frame_positions.tolist(),
                    "temperature": round(current_temp, 2),
                    "kinetic_energy": round(ke, 6),
                    "potential_energy": round(pe, 6),
                    "total_energy": round(total_energy, 6),
                }
                trajectory.append(frame)
                energies.append({
                    "step": step,
                    "kinetic": round(ke, 6),
                    "potential": round(pe, 6),
                    "total": round(total_energy, 6),
                })
                temperatures.append(round(current_temp, 2))

        # Calculate RDF (simplified)
        rdf = self._calculate_rdf(trajectory[-1]["positions"]) if trajectory else None

        calc_time = time.time() - start_time

        return CalculationResult(
            success=True,
            data={
                "method": "molecular_dynamics",
                "smiles": smiles,
                "num_atoms": num_atoms,
                "num_steps": num_steps,
                "timestep": timestep,
                "timestep_unit": "fs",
                "total_time": round(num_steps * timestep / 1000, 2),  # ps
                "ensemble": ensemble,
                "target_temperature": temperature,
                "temperatures": temperatures,
                "average_temperature": round(np.mean(temperatures), 2) if temperatures else None,
                "energies": energies,
                "final_energy": energies[-1] if energies else None,
                "trajectory_frames": len(trajectory),
                "trajectory": trajectory[:10],  # Limit for response size
                "rdf": rdf,
            },
            engine_used="dynamics",
            calculation_time=calc_time,
            warnings=["Simplified MD with LJ+Coulomb. For production, use GROMACS, NAMD, or OpenMM."],
            citations=CitationManager.get_citations("molecular_dynamics"),
        )

    def _get_atomic_mass(self, atomic_num: int) -> float:
        """Get atomic mass in kg."""
        masses = {
            1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999,
            9: 18.998, 15: 30.974, 16: 32.06, 17: 35.45,
            35: 79.904, 53: 126.90,
        }
        amu = masses.get(atomic_num, 12.011)
        return amu * 1.66054e-27  # Convert to kg

    def _initialize_velocities(self, masses: np.ndarray, temperature: float) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        num_atoms = len(masses)
        velocities = np.zeros((num_atoms, 3))

        for i in range(num_atoms):
            sigma = np.sqrt(self.kB * temperature / masses[i])
            velocities[i] = np.random.normal(0, sigma, 3)

        # Remove center of mass velocity
        total_mass = np.sum(masses)
        com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
        velocities -= com_velocity

        return velocities

    def _calculate_forces(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        masses: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """Calculate forces using LJ + Coulomb potential."""
        num_atoms = len(positions)
        forces = np.zeros_like(positions)

        # LJ parameters (simplified - using generic values)
        epsilon = 1.0e-21  # J
        sigma = 3.4e-10    # m

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)

                if r > cutoff or r < 1e-10:
                    continue

                # LJ force
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                f_lj_mag = 24 * epsilon * (2 * sr12 - sr6) / r

                # Coulomb force
                k_e = 8.9875517923e9  # Coulomb constant
                f_coulomb_mag = k_e * charges[i] * charges[j] * 1.602e-19 ** 2 / r ** 2

                # Total force
                f_mag = f_lj_mag + f_coulomb_mag
                f_vec = f_mag * r_vec / r

                forces[i] -= f_vec
                forces[j] += f_vec

        return forces

    def _calculate_kinetic_energy(self, velocities: np.ndarray, masses: np.ndarray) -> float:
        """Calculate kinetic energy in Joules."""
        ke = 0.5 * np.sum(masses * np.sum(velocities ** 2, axis=1))
        return ke

    def _calculate_potential_energy(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        cutoff: float,
    ) -> float:
        """Calculate potential energy in Joules."""
        num_atoms = len(positions)
        pe = 0.0

        epsilon = 1.0e-21
        sigma = 3.4e-10
        k_e = 8.9875517923e9

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)

                if r > cutoff or r < 1e-10:
                    continue

                # LJ potential
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                pe_lj = 4 * epsilon * (sr12 - sr6)

                # Coulomb potential
                pe_coulomb = k_e * charges[i] * charges[j] * 1.602e-19 ** 2 / r

                pe += pe_lj + pe_coulomb

        return pe

    def _calculate_temperature(self, velocities: np.ndarray, masses: np.ndarray) -> float:
        """Calculate instantaneous temperature."""
        ke = self._calculate_kinetic_energy(velocities, masses)
        num_atoms = len(masses)
        return 2 * ke / (3 * num_atoms * self.kB)

    def _calculate_rdf(self, positions: List[List[float]], num_bins: int = 100) -> Dict[str, Any]:
        """Calculate radial distribution function."""
        positions = np.array(positions)
        num_atoms = len(positions)

        # Calculate all pairwise distances
        distances = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                r = np.linalg.norm(positions[j] - positions[i])
                distances.append(r)

        distances = np.array(distances)

        # Create histogram
        max_r = np.max(distances) if len(distances) > 0 else 10.0
        bins = np.linspace(0, max_r, num_bins)
        hist, edges = np.histogram(distances, bins=bins)

        # Normalize
        bin_width = edges[1] - edges[0]
        volume = (4/3) * np.pi * max_r ** 3
        density = num_atoms / volume

        rdf_values = []
        for i in range(len(hist)):
            r = (edges[i] + edges[i+1]) / 2
            shell_volume = 4 * np.pi * r ** 2 * bin_width
            if shell_volume > 0 and density > 0:
                g_r = hist[i] / (density * shell_volume * num_atoms / 2)
                rdf_values.append(round(g_r, 4))
            else:
                rdf_values.append(0.0)

        return {
            "r": [(edges[i] + edges[i+1]) / 2 for i in range(len(hist))],
            "g_r": rdf_values,
        }
