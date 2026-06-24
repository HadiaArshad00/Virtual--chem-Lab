"""
Virtual Chemistry Lab API - Abstract Base Engine
Defines the interface for all calculation engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class CalculationResult:
    """Standard result container for all calculations."""
    success: bool
    data: Dict[str, Any]
    engine_used: str
    calculation_time: float
    warnings: List[str]
    citations: List[Dict[str, Any]]
    error_message: Optional[str] = None


class AbstractEngine(ABC):
    """Abstract base class for all calculation engines.

    All engines must implement these methods to ensure consistent
    behavior across the API.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._available = True

    @property
    def is_available(self) -> bool:
        """Check if the engine is available for use."""
        return self._available

    @abstractmethod
    async def calculate(self, parameters: Dict[str, Any]) -> CalculationResult:
        """Execute a calculation with the given parameters.

        Args:
            parameters: Dictionary containing calculation parameters.

        Returns:
            CalculationResult with results or error information.
        """
        pass

    @abstractmethod
    async def validate_input(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters before calculation.

        Args:
            parameters: Dictionary containing calculation parameters.

        Returns:
            Tuple of (is_valid, error_message).
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of this engine.

        Returns:
            Dictionary describing what this engine can do.
        """
        return {
            "name": self.name,
            "version": self.version,
            "available": self.is_available,
            "methods": self._get_supported_methods(),
        }

    @abstractmethod
    def _get_supported_methods(self) -> List[str]:
        """Get list of supported calculation methods.

        Returns:
            List of method names supported by this engine.
        """
        pass

    def get_citations(self) -> List[Dict[str, Any]]:
        """Get academic citations for this engine.

        Returns:
            List of citation dictionaries.
        """
        from app.core.utils.citations import CitationManager
        return CitationManager.get_citations(self.name)

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the engine.

        Returns:
            Dictionary with health status information.
        """
        return {
            "engine": self.name,
            "available": self.is_available,
            "version": self.version,
        }
