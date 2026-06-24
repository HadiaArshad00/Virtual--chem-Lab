"""
Virtual Chemistry Lab API - Spectra Calculator
IR, NMR, Mass Spec, and UV-Vis spectra simulation.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.engines.base import CalculationResult
from app.core.utils.citations import CitationManager


class SpectraCalculator:
    """Spectra simulation calculator.

    Provides IR, NMR, Mass Spec, and UV-Vis spectra predictions.
    """

    def __init__(self):
        pass

    async def calculate_ir(
        self,
        smiles: str,
        method: str = "harmonic",
        fwhm: float = 10.0,  # Full width at half maximum in cm^-1
    ) -> CalculationResult:
        """Calculate IR spectrum.

        Args:
            smiles: SMILES string.
            method: Calculation method (harmonic, anharmonic).
            fwhm: Peak broadening in cm^-1.

        Returns:
            CalculationResult with IR peak data.
        """
        start_time = time.time()

        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CalculationResult(
                success=False,
                data={},
                engine_used="spectra",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("dft"),
                error_message="Invalid SMILES",
            )

        # Generate approximate IR peaks from functional groups
        peaks = self._predict_ir_from_functional_groups(mol)

        # Generate spectrum data points
        wavenumbers = np.linspace(400, 4000, 3601)
        intensities = np.zeros_like(wavenumbers)

        for peak in peaks:
            freq = peak["frequency"]
            intensity = self._intensity_to_numeric(peak["intensity"])
            # Lorentzian broadening
            intensities += intensity * (fwhm / 2) ** 2 / ((wavenumbers - freq) ** 2 + (fwhm / 2) ** 2)

        spectrum_data = {
            "wavenumbers": wavenumbers.tolist(),
            "intensities": intensities.tolist(),
        }

        return CalculationResult(
            success=True,
            data={
                "method": "ir_simulation",
                "smiles": smiles,
                "peaks": peaks,
                "num_peaks": len(peaks),
                "spectrum": spectrum_data,
                "fwhm": fwhm,
                "note": "Approximate IR spectrum based on functional group analysis. For accurate spectra, use DFT frequency calculation.",
            },
            engine_used="spectra",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("dft"),
        )

    def _predict_ir_from_functional_groups(self, mol) -> List[Dict[str, Any]]:
        """Predict IR peaks from functional groups."""
        from rdkit import Chem

        peaks = []

        # Define SMARTS patterns and their characteristic frequencies
        patterns = [
            ("O-H (alcohol)", "[OH]", [(3200, 3600, "strong", "broad")]),
            ("O-H (carboxylic acid)", "C(=O)O", [(2500, 3300, "strong", "very broad")]),
            ("N-H (amine)", "[NH2]", [(3300, 3500, "medium", "sharp")]),
            ("N-H (amide)", "NC=O", [(3300, 3500, "medium", "sharp")]),
            ("C-H (alkane)", "[CH3]", [(2850, 3000, "strong", "sharp")]),
            ("C-H (alkene)", "C=C[CH]", [(3010, 3100, "medium", "sharp")]),
            ("C-H (aldehyde)", "[CH]=O", [(2720, 2820, "medium", "sharp")]),
            ("C≡C-H", "C#[CH]", [(3300, 3310, "strong", "sharp")]),
            ("C=O (aldehyde)", "[CH]=O", [(1720, 1740, "strong", "sharp")]),
            ("C=O (ketone)", "C(=O)C", [(1705, 1725, "strong", "sharp")]),
            ("C=O (ester)", "C(=O)O", [(1735, 1750, "strong", "sharp")]),
            ("C=O (amide)", "NC=O", [(1630, 1690, "strong", "sharp")]),
            ("C=O (acid)", "C(=O)O", [(1700, 1725, "strong", "sharp")]),
            ("C=C (alkene)", "C=C", [(1620, 1680, "medium", "variable")]),
            ("C≡C (alkyne)", "C#C", [(2100, 2260, "weak", "sharp")]),
            ("C≡N (nitrile)", "C#N", [(2220, 2260, "medium", "sharp")]),
            ("NO2", "[N+](=O)[O-]", [(1500, 1570, "strong", "sharp"), (1380, 1400, "strong", "sharp")]),
            ("C-O (ether)", "C-O-C", [(1000, 1300, "strong", "sharp")]),
            ("C-F", "C-F", [(1000, 1400, "strong", "sharp")]),
            ("C-Cl", "C-Cl", [(600, 800, "strong", "sharp")]),
            ("C-Br", "C-Br", [(500, 600, "strong", "sharp")]),
            ("S-H", "[SH]", [(2550, 2600, "weak", "sharp")]),
            ("S=O (sulfone)", "S(=O)(=O)", [(1300, 1350, "strong", "sharp"), (1150, 1180, "strong", "sharp")]),
        ]

        for name, smarts, frequencies in patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                for freq_range in frequencies:
                    peaks.append({
                        "frequency": round((freq_range[0] + freq_range[1]) / 2, 1),
                        "range": [freq_range[0], freq_range[1]],
                        "intensity": freq_range[2],
                        "shape": freq_range[3],
                        "assignment": name,
                    })

        # Remove duplicates (same frequency)
        seen_freqs = set()
        unique_peaks = []
        for peak in peaks:
            freq = peak["frequency"]
            if freq not in seen_freqs:
                seen_freqs.add(freq)
                unique_peaks.append(peak)

        unique_peaks.sort(key=lambda x: x["frequency"], reverse=True)
        return unique_peaks

    def _intensity_to_numeric(self, intensity: str) -> float:
        """Convert intensity string to numeric value."""
        mapping = {
            "very strong": 1.0,
            "strong": 0.8,
            "medium": 0.5,
            "weak": 0.3,
            "very weak": 0.1,
        }
        return mapping.get(intensity.lower(), 0.5)

    async def calculate_nmr(
        self,
        smiles: str,
        nuclei: str = "1H",
        solvent: str = "CDCl3",
    ) -> CalculationResult:
        """Calculate NMR spectrum.

        Args:
            smiles: SMILES string.
            nuclei: Nuclei type (1H, 13C).
            solvent: Solvent for NMR.

        Returns:
            CalculationResult with NMR chemical shifts.
        """
        start_time = time.time()

        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CalculationResult(
                success=False,
                data={},
                engine_used="spectra",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("nmr"),
                error_message="Invalid SMILES",
            )

        shifts = []

        if nuclei == "1H":
            shifts = self._predict_1h_nmr(mol, solvent)
        elif nuclei == "13C":
            shifts = self._predict_13c_nmr(mol, solvent)

        # Generate spectrum data
        if nuclei == "1H":
            ppm_range = np.linspace(-1, 12, 1301)
        else:
            ppm_range = np.linspace(0, 220, 2201)

        intensities = np.zeros_like(ppm_range)
        for shift in shifts:
            ppm = shift["chemical_shift"]
            # Lorentzian peak
            intensities += 1.0 * (0.5) ** 2 / ((ppm_range - ppm) ** 2 + (0.5) ** 2)

        return CalculationResult(
            success=True,
            data={
                "method": "nmr_simulation",
                "nuclei": nuclei,
                "smiles": smiles,
                "solvent": solvent,
                "shifts": shifts,
                "num_signals": len(shifts),
                "spectrum": {
                    "ppm": ppm_range.tolist(),
                    "intensities": intensities.tolist(),
                },
                "note": "Empirical NMR prediction based on substituent effects. For accurate GIAO calculations, use DFT.",
            },
            engine_used="spectra",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("nmr"),
        )

    def _predict_1h_nmr(self, mol, solvent: str = "CDCl3") -> List[Dict[str, Any]]:
        """Predict 1H NMR chemical shifts."""
        from rdkit import Chem

        shifts = []
        mol = Chem.AddHs(mol)

        # Solvent correction
        solvent_shifts = {
            "CDCl3": 0.0,
            "DMSO-d6": 0.3,
            "D2O": 0.0,
            "acetone-d6": 0.2,
            "methanol-d4": 0.1,
        }
        solvent_corr = solvent_shifts.get(solvent, 0.0)

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:  # Only hydrogens
                continue

            neighbors = atom.GetNeighbors()
            if not neighbors:
                continue

            attached = neighbors[0]
            attached_symbol = attached.GetSymbol()
            attached_idx = attached.GetIdx()

            # Base shift
            base_shift = 1.0

            # Adjust based on attached atom
            if attached_symbol == "C":
                # Check hybridization and environment
                if attached.GetIsAromatic():
                    base_shift = 7.2
                elif attached.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                    base_shift = 2.5
                elif attached.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                    base_shift = 5.5
                else:
                    base_shift = 1.5

                # Check for adjacent functional groups
                for neighbor in attached.GetNeighbors():
                    if neighbor.GetIdx() == atom.GetIdx():
                        continue
                    n_symbol = neighbor.GetSymbol()
                    if n_symbol == "O":
                        if neighbor.GetTotalDegree() == 1:  # OH
                            base_shift = 3.5
                        else:  # ether
                            base_shift = 3.8
                    elif n_symbol == "N":
                        base_shift = 2.8
                    elif n_symbol == "Cl":
                        base_shift = 3.5
                    elif n_symbol == "Br":
                        base_shift = 3.2
                    elif n_symbol == "F":
                        base_shift = 4.5
                    elif n_symbol == "S":
                        base_shift = 2.5

            elif attached_symbol == "O":
                base_shift = 4.5
                # Check if it's OH or ether
                if attached.GetTotalDegree() == 1:
                    base_shift = 2.0  # OH (broad, variable)

            elif attached_symbol == "N":
                base_shift = 2.5
                if attached.GetTotalDegree() == 1:  # NH2
                    base_shift = 1.5

            elif attached_symbol == "S":
                base_shift = 2.5

            # Apply solvent correction
            base_shift += solvent_corr

            # Count equivalent hydrogens
            num_h = 1

            shifts.append({
                "atom_idx": atom.GetIdx(),
                "attached_to": attached_symbol,
                "attached_atom_idx": attached_idx,
                "chemical_shift": round(base_shift, 2),
                "unit": "ppm",
                "multiplicity": "m",  # Simplified
                "num_protons": num_h,
            })

        return shifts

    def _predict_13c_nmr(self, mol, solvent: str = "CDCl3") -> List[Dict[str, Any]]:
        """Predict 13C NMR chemical shifts."""
        from rdkit import Chem

        shifts = []

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:  # Only carbons
                continue

            # Base shift
            base_shift = 30.0

            # Hybridization
            if atom.GetIsAromatic():
                base_shift = 128.0
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP:
                base_shift = 75.0
            elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                # Check if carbonyl
                is_carbonyl = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "O" and neighbor.GetTotalDegree() == 1:
                        is_carbonyl = True
                        break
                if is_carbonyl:
                    base_shift = 200.0
                else:
                    base_shift = 135.0

            # Adjust for attached electronegative atoms
            for neighbor in atom.GetNeighbors():
                n_symbol = neighbor.GetSymbol()
                if n_symbol == "O":
                    base_shift += 30.0
                elif n_symbol == "N":
                    base_shift += 20.0
                elif n_symbol == "F":
                    base_shift += 50.0
                elif n_symbol == "Cl":
                    base_shift += 25.0
                elif n_symbol == "Br":
                    base_shift += 10.0

            shifts.append({
                "atom_idx": atom.GetIdx(),
                "chemical_shift": round(base_shift, 2),
                "unit": "ppm",
                "hybridization": str(atom.GetHybridization()),
            })

        return shifts

    async def calculate_mass_spec(
        self,
        smiles: str,
        ionization: str = "EI",
    ) -> CalculationResult:
        """Predict mass spectrum.

        Args:
            smiles: SMILES string.
            ionization: Ionization method (EI, ESI, MALDI).

        Returns:
            CalculationResult with mass spectrum peaks.
        """
        start_time = time.time()

        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CalculationResult(
                success=False,
                data={},
                engine_used="spectra",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=[],
                error_message="Invalid SMILES",
            )

        mw = Descriptors.ExactMolWt(mol)

        # Predict fragments (simplified rule-based)
        fragments = self._predict_fragments(mol, ionization)

        # Add molecular ion
        fragments.insert(0, {
            "m_z": round(mw, 4),
            "intensity": 100.0,
            "assignment": "M+ (molecular ion)",
        })

        return CalculationResult(
            success=True,
            data={
                "method": "mass_spec_prediction",
                "ionization": ionization,
                "smiles": smiles,
                "molecular_weight": round(mw, 4),
                "fragments": fragments,
                "num_fragments": len(fragments),
                "note": "Rule-based fragmentation prediction. For accurate spectra, use experimental data.",
            },
            engine_used="spectra",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=[],
        )

    def _predict_fragments(self, mol, ionization: str) -> List[Dict[str, Any]]:
        """Predict mass spec fragments using rule-based approach."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        fragments = []

        # Common fragmentation patterns
        patterns = [
            ("alpha cleavage", "C-C", 15),  # CH3 loss
            ("beta cleavage", "C-C", 29),   # CHO or C2H5 loss
            ("McLafferty", "C-C", 28),     # CO or C2H4 loss
            ("water loss", "O", 18),        # H2O loss
            ("CO loss", "C=O", 28),         # CO loss from carbonyl
            ("NH3 loss", "N", 17),          # NH3 loss
        ]

        # Check for common functional groups
        mw = Descriptors.ExactMolWt(mol)

        # Alcohol - water loss
        oh_pattern = Chem.MolFromSmarts("[OH]")
        if oh_pattern and mol.HasSubstructMatch(oh_pattern):
            fragments.append({
                "m_z": round(mw - 18.0106, 4),
                "intensity": 60.0,
                "assignment": "[M-H2O]+",
            })

        # Carbonyl - CO loss
        co_pattern = Chem.MolFromSmarts("C=O")
        if co_pattern and mol.HasSubstructMatch(co_pattern):
            fragments.append({
                "m_z": round(mw - 28.0101, 4),
                "intensity": 40.0,
                "assignment": "[M-CO]+",
            })

        # Amine - NH3 loss
        nh2_pattern = Chem.MolFromSmarts("[NH2]")
        if nh2_pattern and mol.HasSubstructMatch(nh2_pattern):
            fragments.append({
                "m_z": round(mw - 17.0265, 4),
                "intensity": 30.0,
                "assignment": "[M-NH3]+",
            })

        # Methyl loss
        ch3_pattern = Chem.MolFromSmarts("[CH3]")
        if ch3_pattern and mol.HasSubstructMatch(ch3_pattern):
            fragments.append({
                "m_z": round(mw - 15.0235, 4),
                "intensity": 25.0,
                "assignment": "[M-CH3]+",
            })

        # Sort by intensity
        fragments.sort(key=lambda x: x["intensity"], reverse=True)
        return fragments

    async def calculate_uv_vis(
        self,
        smiles: str,
        method: str = "td-dft",
    ) -> CalculationResult:
        """Calculate UV-Vis absorption spectrum.

        Args:
            smiles: SMILES string.
            method: Calculation method (td-dft, empirical).

        Returns:
            CalculationResult with UV-Vis transitions.
        """
        start_time = time.time()

        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CalculationResult(
                success=False,
                data={},
                engine_used="spectra",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=[],
                error_message="Invalid SMILES",
            )

        # Empirical prediction based on chromophores
        transitions = self._predict_uv_transitions(mol)

        # Generate spectrum data
        wavelengths = np.linspace(200, 800, 601)
        absorbance = np.zeros_like(wavelengths)

        for trans in transitions:
            wl = trans["wavelength"]
            intensity = trans["oscillator_strength"]
            # Gaussian broadening
            fwhm = 30  # nm
            absorbance += intensity * np.exp(-((wavelengths - wl) / (fwhm / 2.355)) ** 2)

        return CalculationResult(
            success=True,
            data={
                "method": "uv_vis_prediction",
                "smiles": smiles,
                "transitions": transitions,
                "num_transitions": len(transitions),
                "spectrum": {
                    "wavelengths": wavelengths.tolist(),
                    "absorbance": absorbance.tolist(),
                },
                "note": "Empirical UV-Vis prediction based on chromophore analysis. For accurate TD-DFT calculations, use quantum chemistry software.",
            },
            engine_used="spectra",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=[],
        )

    def _predict_uv_transitions(self, mol) -> List[Dict[str, Any]]:
        """Predict UV-Vis transitions from chromophores."""
        from rdkit import Chem

        transitions = []

        # Check for chromophores
        chromophores = [
            ("C=C-C=C (diene)", "C=CC=C", 217, 0.3),
            ("benzene", "c1ccccc1", 255, 0.1),
            ("phenol", "Oc1ccccc1", 270, 0.2),
            ("aniline", "Nc1ccccc1", 280, 0.3),
            ("C=O (aldehyde)", "[CH]=O", 290, 0.1),
            ("C=O (ketone)", "C(=O)C", 280, 0.1),
            ("C=C-C=O (enone)", "C=CC=O", 320, 0.4),
            ("nitro", "[N+](=O)[O-]", 280, 0.2),
            ("azo", "N=N", 340, 0.5),
            ("nitroso", "N=O", 300, 0.2),
            ("C≡C", "C#C", 180, 0.3),
            ("C≡N", "C#N", 180, 0.1),
        ]

        for name, smarts, base_wl, base_f in chromophores:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                # Adjust based on conjugation
                num_matches = len(mol.GetSubstructMatches(pattern))
                wl = base_wl + (num_matches - 1) * 30  # Bathochromic shift
                f = min(base_f * num_matches, 1.0)

                transitions.append({
                    "wavelength": wl,
                    "wavelength_unit": "nm",
                    "oscillator_strength": round(f, 4),
                    "assignment": name,
                    "type": "π→π*" if "C=C" in name or "benzene" in name else "n→π*",
                })

        # Remove duplicates and sort by wavelength
        seen = set()
        unique = []
        for t in transitions:
            if t["wavelength"] not in seen:
                seen.add(t["wavelength"])
                unique.append(t)

        unique.sort(key=lambda x: x["wavelength"])
        return unique
