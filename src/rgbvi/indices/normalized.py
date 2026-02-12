"""
rgbvi.normalized.py
Normalized RGB vegetative indices
BoMeyering 2025
"""

__all__ = []

import numpy as np
from ..registry import register_index, IndexSpec
from ..core import compute_index

EPS = 1e-3

# Woebbecke Index: (G - B) / (|R - G| + EPS), with R,G,B in [0,1] => [-1000, 1000]
register_index(
    IndexSpec(
        name="wi",
        full_name="Woebbecke Index",
        formula=lambda R,G,B: (G - B) / (np.abs(R - G) + 1), # Add EPS=1 to denominator to avoid division by zero
        range=(-254.0, 254.0),
        map01=True,
        citation="Woebbecke D.M., Meyer G.E., Von Bargen K., Mortensen D.A. 'Color indices for weed identification under various soil, residue, and lighting conditions' (1995) Transactions of the American Society of Agricultural Engineers, 38 (1), pp. 259 - 269, https://www.scopus.com/inward/record.uri?eid=2-s2.0-0029110322&partnerID=40&md5=d3430f82764dc64892eb6dc77186596e"
    )
)

# VARI: (G - R) / (G + R - B + EPS), with R,G,B in [0,1] => [-1, 1]
register_index(
    IndexSpec(
        name="vari",
        full_name="Visible Atmospherically Resistant Index",
        formula=lambda R,G,B: (G - R) / (G + R - B),
        # range=(-254, 255),
        range=(-4.008, 4.008),
        map01=True,
        citation=""
    )
)

# MGRVI: (G - R) / (G + R + B + EPS), with R,G,B in [0,1] => [-1, 1]
register_index(
    IndexSpec(
        name="mgrvi",
        full_name="Modified Green Red Vegetation Index",
        formula=lambda R,G,B: (G**2 - R**2) / (G**2 + R**2),
        range=(-0.99997, 0.99997),
        map01=True,
        citation=""
    )
)

# RGRI: R / G, with R,G,B in [1, 255]
register_index(
    IndexSpec(
        name="rgri",
        full_name="Red Green Ratio Index",
        formula=lambda R,G,B: np.tanh(R / G),
        range=(0.003922, 255.0),
        map01=True,
        citation=""
    )
)
# Kawashima Index: (R - B) / (R + B), with R,G,B in [1, 255]
register_index(
    IndexSpec(
        name="ki",
        full_name="Kawashima Index",
        formula=lambda R,G,B: (R - B) / (R + B),
        range=(-0.9961, 0.9961),
        map01=True,
        citation="",
    )
)