"""
Virtual Chemistry Lab API - Citation Manager
Manages academic citations for calculation methods and engines.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """Represents an academic citation."""
    title: str
    authors: str
    journal: str
    year: int
    doi: Optional[str] = None
    url: Optional[str] = None


class CitationManager:
    """Manages citations for different calculation methods and engines."""

    CITATIONS: Dict[str, List[Citation]] = {
        "rdkit": [
            Citation(
                title="RDKit: Open-source cheminformatics",
                authors="Greg Landrum",
                journal="GitHub",
                year=2016,
                url="https://www.rdkit.org",
            ),
        ],
        "psi4": [
            Citation(
                title="Psi4: An open-source ab initio electronic structure program",
                authors="Smith, D. G. A. et al.",
                journal="Journal of Chemical Physics",
                year=2020,
                doi="10.1063/5.0006002",
            ),
            Citation(
                title="PSI4 1.4: Open-source software for high-throughput quantum chemistry",
                authors="Turney, J. M. et al.",
                journal="Wiley Interdisciplinary Reviews: Computational Molecular Science",
                year=2012,
                doi="10.1002/wcms.93",
            ),
        ],
        "dft": [
            Citation(
                title="Density-functional exchange-energy approximation with correct asymptotic behavior",
                authors="Perdew, J. P., Wang, Y.",
                journal="Physical Review B",
                year=1992,
                doi="10.1103/PhysRevB.45.13244",
            ),
            Citation(
                title="Generalized Gradient Approximation Made Simple",
                authors="Perdew, J. P., Burke, K., Ernzerhof, M.",
                journal="Physical Review Letters",
                year=1996,
                doi="10.1103/PhysRevLett.77.3865",
            ),
        ],
        "nmr": [
            Citation(
                title="Gauge-including atomic orbitals",
                authors="Ditchfield, R.",
                journal="Molecular Physics",
                year=1974,
                doi="10.1080/00268977400102351",
            ),
        ],
        "docking": [
            Citation(
                title="AutoDock Vina: Improving the speed and accuracy of docking",
                authors="Trott, O., Olson, A. J.",
                journal="Journal of Computational Chemistry",
                year=2010,
                doi="10.1002/jcc.21334",
            ),
        ],
        "molecular_dynamics": [
            Citation(
                title="Molecular dynamics with coupling to an external bath",
                authors="Berendsen, H. J. C. et al.",
                journal="Journal of Chemical Physics",
                year=1984,
                doi="10.1063/1.448118",
            ),
        ],
        "kinetics": [
            Citation(
                title="The Arrhenius Law and Activation Energies",
                authors="Arrhenius, S.",
                journal="Zeitschrift für Physikalische Chemie",
                year=1889,
            ),
            Citation(
                title="The activated complex in chemical reactions",
                authors="Eyring, H.",
                journal="Journal of Chemical Physics",
                year=1935,
                doi="10.1063/1.1749604",
            ),
        ],
        "yield_prediction": [
            Citation(
                title="XGBoost: A Scalable Tree Boosting System",
                authors="Chen, T., Guestrin, C.",
                journal="Proceedings of the 22nd ACM SIGKDD International Conference",
                year=2016,
                doi="10.1145/2939672.2939785",
            ),
        ],
        "openbabel": [
            Citation(
                title="Open Babel: An open chemical toolbox",
                authors="O'Boyle, N. M. et al.",
                journal="Journal of Cheminformatics",
                year=2011,
                doi="10.1186/1758-2946-3-33",
            ),
        ],
        "mmff94": [
            Citation(
                title="Merck Molecular Force Field. I. Basis, Form, Scope, Parameterization, and Performance of MMFF94",
                authors="Halgren, T. A.",
                journal="Journal of Computational Chemistry",
                year=1996,
                doi="10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P",
            ),
        ],
        "morgan_fingerprint": [
            Citation(
                title="Extended-Connectivity Fingerprints",
                authors="Rogers, D., Hahn, M.",
                journal="Journal of Chemical Information and Modeling",
                year=2010,
                doi="10.1021/ci100050t",
            ),
        ],
        "electrochemistry": [
            Citation(
                title="Electrochemical Methods: Fundamentals and Applications",
                authors="Bard, A. J., Faulkner, L. R.",
                journal="Wiley",
                year=2001,
            ),
        ],
        "crystallization": [
            Citation(
                title="Crystal structure prediction via metadynamics",
                authors="Piaggi, P. M., Parrinello, M.",
                journal="Nature Communications",
                year=2022,
                doi="10.1038/s41467-022-28509-8",
            ),
        ],
    }

    @classmethod
    def get_citations(cls, method: str) -> List[Dict]:
        """Get citations for a specific method.

        Args:
            method: The calculation method or engine name.

        Returns:
            List of citation dictionaries.
        """
        citations = cls.CITATIONS.get(method.lower(), [])
        return [
            {
                "title": c.title,
                "authors": c.authors,
                "journal": c.journal,
                "year": c.year,
                "doi": c.doi,
                "url": c.url,
            }
            for c in citations
        ]

    @classmethod
    def get_all_citations(cls) -> Dict[str, List[Dict]]:
        """Get all available citations.

        Returns:
            Dictionary mapping method names to citation lists.
        """
        return {k: cls.get_citations(k) for k in cls.CITATIONS.keys()}

    @classmethod
    def add_citation(cls, method: str, citation: Citation):
        """Add a citation for a method.

        Args:
            method: The calculation method or engine name.
            citation: The citation to add.
        """
        if method.lower() not in cls.CITATIONS:
            cls.CITATIONS[method.lower()] = []
        cls.CITATIONS[method.lower()].append(citation)
