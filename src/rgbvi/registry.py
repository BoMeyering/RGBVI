"""
rgbvi.registry.py
Index specs and registrations
BoMeyering 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict

@dataclass
class IndexSpec:
    name: str
    formula: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    domain: Optional[Tuple[float, float]] # Theoretical raw range
    map01: bool # Whether to map raw domain to [0, 1]

EPS = 1e-6
INDEX_SPECS: Dict[str, IndexSpec] = {}

def register_index(spec: IndexSpec):
    """Register an index spec in INDEX_SPECS

    Parameters:
    -----------
        spec : IndexSpec
            An index specification that you want to add
    """

    INDEX_SPECS[spec.name.lower()] = spec