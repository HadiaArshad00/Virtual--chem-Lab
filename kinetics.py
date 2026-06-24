"""
Virtual Chemistry Lab API - Kinetics Calculator
Reaction kinetics calculations: Arrhenius, Eyring, rate constants, activation energy.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np

from app.core.engines.base import CalculationResult
from app.core.utils.citations import CitationManager


class KineticsCalculator:
    """Chemical kinetics calculator.

    Implements Arrhenius equation, Eyring equation, and transition state theory.
    """

    # Physical constants
    R = 8.314462618  # J/(mol·K)
    kB = 1.380649e-23  # J/K
    h = 6.62607015e-34  # J·s
    NA = 6.02214076e23  # mol^-1

    def __init__(self):
        pass

    async def arrhenius_rate(
        self,
        A: float,
        Ea: float,
        temperatures: List[float],
        Ea_unit: str = "J/mol",
    ) -> CalculationResult:
        """Calculate rate constant using Arrhenius equation: k = A * exp(-Ea/RT).

        Args:
            A: Pre-exponential factor (same units as desired k).
            Ea: Activation energy.
            temperatures: List of temperatures in Kelvin.
            Ea_unit: Unit of activation energy (J/mol, kJ/mol, kcal/mol).

        Returns:
            CalculationResult with rate constants at each temperature.
        """
        start_time = time.time()

        # Convert Ea to J/mol
        Ea_joules = self._convert_energy(Ea, Ea_unit, "J/mol")

        results = []
        for T in temperatures:
            if T <= 0:
                continue
            k = A * np.exp(-Ea_joules / (self.R * T))
            results.append({
                "temperature": T,
                "temperature_unit": "K",
                "rate_constant": float(k),
                "rate_constant_scientific": f"{k:.4e}",
            })

        return CalculationResult(
            success=True,
            data={
                "method": "arrhenius",
                "equation": "k = A * exp(-Ea/RT)",
                "A": A,
                "Ea": Ea,
                "Ea_unit": Ea_unit,
                "Ea_J_mol": Ea_joules,
                "temperatures": temperatures,
                "rate_constants": results,
                "num_temperatures": len(results),
            },
            engine_used="kinetics",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("kinetics"),
        )

    async def eyring_rate(
        self,
        delta_G: float,
        temperatures: List[float],
        delta_H: Optional[float] = None,
        delta_S: Optional[float] = None,
        delta_G_unit: str = "J/mol",
    ) -> CalculationResult:
        """Calculate rate constant using Eyring equation: k = (kB*T/h) * exp(-ΔG‡/RT).

        Args:
            delta_G: Gibbs free energy of activation.
            temperatures: List of temperatures in Kelvin.
            delta_H: Enthalpy of activation (optional).
            delta_S: Entropy of activation (optional).
            delta_G_unit: Unit of delta_G (J/mol, kJ/mol, kcal/mol).

        Returns:
            CalculationResult with rate constants.
        """
        start_time = time.time()

        # Convert delta_G to J/mol
        delta_G_joules = self._convert_energy(delta_G, delta_G_unit, "J/mol")

        results = []
        for T in temperatures:
            if T <= 0:
                continue

            # Eyring equation
            k = (self.kB * T / self.h) * np.exp(-delta_G_joules / (self.R * T))

            # Convert to M^-1 s^-1 for bimolecular or s^-1 for unimolecular
            result = {
                "temperature": T,
                "temperature_unit": "K",
                "rate_constant": float(k),
                "rate_constant_scientific": f"{k:.4e}",
                "rate_constant_unit": "s^-1 (unimolecular) or M^-1 s^-1 (bimolecular)",
            }

            # Calculate delta_G from delta_H and delta_S if provided
            if delta_H is not None and delta_S is not None:
                delta_H_j = self._convert_energy(delta_H, delta_G_unit, "J/mol")
                delta_S_j = delta_S  # Assume J/(mol·K)
                calc_delta_G = delta_H_j - T * delta_S_j
                result["calculated_delta_G"] = round(calc_delta_G, 2)
                result["delta_H"] = delta_H
                result["delta_S"] = delta_S

            results.append(result)

        return CalculationResult(
            success=True,
            data={
                "method": "eyring",
                "equation": "k = (kB*T/h) * exp(-ΔG‡/RT)",
                "delta_G": delta_G,
                "delta_G_unit": delta_G_unit,
                "delta_G_J_mol": delta_G_joules,
                "temperatures": temperatures,
                "rate_constants": results,
                "num_temperatures": len(results),
            },
            engine_used="kinetics",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("kinetics"),
        )

    async def calculate_activation_parameters(
        self,
        temperatures: List[float],
        rate_constants: List[float],
        method: str = "arrhenius",
    ) -> CalculationResult:
        """Calculate activation energy and pre-exponential factor from experimental data.

        Args:
            temperatures: List of temperatures in Kelvin.
            rate_constants: List of rate constants at corresponding temperatures.
            method: Fitting method (arrhenius or eyring).

        Returns:
            CalculationResult with fitted parameters.
        """
        start_time = time.time()

        if len(temperatures) != len(rate_constants):
            return CalculationResult(
                success=False,
                data={},
                engine_used="kinetics",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("kinetics"),
                error_message="Temperatures and rate constants must have same length",
            )

        temps = np.array(temperatures)
        k_vals = np.array(rate_constants)

        if method == "arrhenius":
            # Linear fit: ln(k) = ln(A) - Ea/(R*T)
            inv_T = 1.0 / temps
            ln_k = np.log(k_vals)

            # Linear regression
            coeffs = np.polyfit(inv_T, ln_k, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            Ea = -slope * self.R  # J/mol
            A = np.exp(intercept)

            # Calculate R-squared
            y_pred = intercept + slope * inv_T
            ss_res = np.sum((ln_k - y_pred) ** 2)
            ss_tot = np.sum((ln_k - np.mean(ln_k)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return CalculationResult(
                success=True,
                data={
                    "method": "arrhenius_fit",
                    "Ea": round(Ea, 2),
                    "Ea_kJ_mol": round(Ea / 1000, 4),
                    "Ea_kcal_mol": round(Ea / 4184, 4),
                    "A": float(A),
                    "A_scientific": f"{A:.4e}",
                    "R_squared": round(r_squared, 6),
                    "fit_equation": f"ln(k) = {intercept:.4f} + ({slope:.2f})/T",
                },
                engine_used="kinetics",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("kinetics"),
            )

        elif method == "eyring":
            # Linear fit: ln(k/T) = ln(kB/h) + ΔS/R - ΔH/(R*T)
            inv_T = 1.0 / temps
            ln_k_T = np.log(k_vals / temps)

            coeffs = np.polyfit(inv_T, ln_k_T, 1)
            slope = coeffs[0]
            intercept = coeffs[1]

            delta_H = -slope * self.R  # J/mol
            delta_S = (intercept - np.log(self.kB / self.h)) * self.R  # J/(mol·K)

            # Calculate delta_G at 298 K
            delta_G_298 = delta_H - 298.15 * delta_S

            # R-squared
            y_pred = intercept + slope * inv_T
            ss_res = np.sum((ln_k_T - y_pred) ** 2)
            ss_tot = np.sum((ln_k_T - np.mean(ln_k_T)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return CalculationResult(
                success=True,
                data={
                    "method": "eyring_fit",
                    "delta_H": round(delta_H, 2),
                    "delta_H_kJ_mol": round(delta_H / 1000, 4),
                    "delta_S": round(delta_S, 2),
                    "delta_S_J_mol_K": round(delta_S, 2),
                    "delta_G_298K": round(delta_G_298, 2),
                    "delta_G_298K_kJ_mol": round(delta_G_298 / 1000, 4),
                    "R_squared": round(r_squared, 6),
                },
                engine_used="kinetics",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("kinetics"),
            )

        else:
            return CalculationResult(
                success=False,
                data={},
                engine_used="kinetics",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("kinetics"),
                error_message=f"Unknown fitting method: {method}",
            )

    async def half_life(
        self,
        rate_constant: float,
        reaction_order: int = 1,
        initial_concentration: Optional[float] = None,
    ) -> CalculationResult:
        """Calculate half-life of a reaction.

        Args:
            rate_constant: Rate constant.
            reaction_order: Reaction order (0, 1, 2).
            initial_concentration: Initial concentration (required for 0 and 2 order).

        Returns:
            CalculationResult with half-life.
        """
        start_time = time.time()

        if reaction_order == 0:
            if initial_concentration is None:
                return CalculationResult(
                    success=False,
                    data={},
                    engine_used="kinetics",
                    calculation_time=time.time() - start_time,
                    warnings=[],
                    citations=CitationManager.get_citations("kinetics"),
                    error_message="Initial concentration required for 0-order reaction",
                )
            t_half = initial_concentration / (2 * rate_constant)
        elif reaction_order == 1:
            t_half = np.log(2) / rate_constant
        elif reaction_order == 2:
            if initial_concentration is None:
                return CalculationResult(
                    success=False,
                    data={},
                    engine_used="kinetics",
                    calculation_time=time.time() - start_time,
                    warnings=[],
                    citations=CitationManager.get_citations("kinetics"),
                    error_message="Initial concentration required for 2-order reaction",
                )
            t_half = 1 / (rate_constant * initial_concentration)
        else:
            return CalculationResult(
                success=False,
                data={},
                engine_used="kinetics",
                calculation_time=time.time() - start_time,
                warnings=[],
                citations=CitationManager.get_citations("kinetics"),
                error_message=f"Reaction order {reaction_order} not supported",
            )

        return CalculationResult(
            success=True,
            data={
                "method": "half_life",
                "reaction_order": reaction_order,
                "rate_constant": rate_constant,
                "half_life": round(float(t_half), 6),
                "half_life_unit": "seconds (or same unit as rate constant)",
            },
            engine_used="kinetics",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("kinetics"),
        )

    async def equilibrium_constant(
        self,
        delta_G: float,
        temperature: float = 298.15,
        delta_G_unit: str = "J/mol",
    ) -> CalculationResult:
        """Calculate equilibrium constant from Gibbs free energy.

        Args:
            delta_G: Gibbs free energy change.
            temperature: Temperature in Kelvin.
            delta_G_unit: Unit of delta_G.

        Returns:
            CalculationResult with equilibrium constant.
        """
        start_time = time.time()

        delta_G_joules = self._convert_energy(delta_G, delta_G_unit, "J/mol")
        K = np.exp(-delta_G_joules / (self.R * temperature))

        return CalculationResult(
            success=True,
            data={
                "method": "equilibrium_constant",
                "delta_G": delta_G,
                "delta_G_unit": delta_G_unit,
                "temperature": temperature,
                "equilibrium_constant": float(K),
                "equilibrium_constant_scientific": f"{K:.4e}",
                "ln_K": round(float(np.log(K)), 4),
            },
            engine_used="kinetics",
            calculation_time=time.time() - start_time,
            warnings=[],
            citations=CitationManager.get_citations("kinetics"),
        )

    def _convert_energy(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert energy between units."""
        conversions = {
            "J/mol": 1.0,
            "kJ/mol": 1000.0,
            "cal/mol": 4.184,
            "kcal/mol": 4184.0,
            "eV": 96485.0,
            "Hartree": 2625500.0,
        }

        from_factor = conversions.get(from_unit, 1.0)
        to_factor = conversions.get(to_unit, 1.0)

        return value * from_factor / to_factor

    def _generate_plot_data(
        self,
        temperatures: List[float],
        rate_constants: List[float],
        plot_type: str = "arrhenius",
    ) -> Dict[str, Any]:
        """Generate data for plotting.

        Args:
            temperatures: Temperatures in K.
            rate_constants: Rate constants.
            plot_type: Type of plot (arrhenius, eyring, vanthoff).

        Returns:
            Dictionary with plot data.
        """
        temps = np.array(temperatures)
        k_vals = np.array(rate_constants)

        if plot_type == "arrhenius":
            x = 1000.0 / temps  # 1000/T for better scale
            y = np.log(k_vals)
            x_label = "1000/T (K^-1)"
            y_label = "ln(k)"
        elif plot_type == "eyring":
            x = 1000.0 / temps
            y = np.log(k_vals / temps)
            x_label = "1000/T (K^-1)"
            y_label = "ln(k/T)"
        else:
            x = temps
            y = k_vals
            x_label = "T (K)"
            y_label = "k"

        return {
            "plot_type": plot_type,
            "x": x.tolist(),
            "y": y.tolist(),
            "x_label": x_label,
            "y_label": y_label,
        }
