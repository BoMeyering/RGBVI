"""
rgbvi.linear.py
Linear RGB vegetative indices
BoMeyering 2025
"""

__all__ = []

import numpy as np
from ..registry import register_index, IndexSpec
from ..core import compute_index

# Red minus Green: R - G, with R,G,B in [0,1] => [-1,1]
register_index(
    IndexSpec(
        name="rmg",
        full_name="Red Minus Green",
        formula=lambda R,G,B: R - G,
        domain=(-1, 1),
        map01=True,
        citation="Woebbecke D.M., Meyer G.E., Von Bargen K., Mortensen D.A. 'Color indices for weed identification under various soil, residue, and lighting conditions' (1995) Transactions of the American Society of Agricultural Engineers, 38 (1), pp. 259 - 269, https://www.scopus.com/inward/record.uri?eid=2-s2.0-0029110322&partnerID=40&md5=d3430f82764dc64892eb6dc77186596e"
    )
)

# Green Minus Blue: G - B, with R,G,B in [0,1] => [-1,1]
register_index(
    IndexSpec(
        name="gmb",
        full_name="Green Minus Blue",
        formula=lambda R,G,B: G - B,
        domain=(-1, 1),
        map01=True,
        citation="Woebbecke D.M., Meyer G.E., Von Bargen K., Mortensen D.A. 'Color indices for weed identification under various soil, residue, and lighting conditions' (1995) Transactions of the American Society of Agricultural Engineers, 38 (1), pp. 259 - 269, https://www.scopus.com/inward/record.uri?eid=2-s2.0-0029110322&partnerID=40&md5=d3430f82764dc64892eb6dc77186596e"
    )
)

# Excess Green: 2G - R - B, with R,G,B in [0,1] => [-2,2]
register_index(
    IndexSpec(
        name="exg",
        full_name="Excess Green",
        # formula=lambda R,G,B: 2*G/(G+R+B+1e-3) - R/(G+R+B+1e-3) - B/(G+R+B+1e-3),
        formula=lambda R,G,B: 2*G - R - B,
        domain=(-2.0, 2.0),
        map01=True
    )
)


register_index(
    IndexSpec(
        name="exr",
        full_name="Excess Red",
        formula=lambda R,G,B: 1.4*R - G,
        domain=(-1.0, 1.4),
        map01=True
    )
)
# CIVE: 0.441R - 0.811G + 0.385B + 18.78745, with R,G,B in [0,1] => [-0.7373, 0.8997]
register_index(
    IndexSpec(
        name="cive",
        full_name="Color Index of Vegetation",
        formula=lambda R,G,B: 0.441*R - 0.811*G + 0.385*B + 0.07367627450980392, # Constant 18.78745 normalized by division by 255
        domain=(-0.7373237254901961, 0.899676274509804),
        map01=True,
        citation="Kataoka, T.; Kaneko, T.; Okamoto, H.; Hata, S. 'Crop growth estimation system using machine vision', Proceedings 2003 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM 2003), Kobe, Japan, 2003, pp. b1079-b1083 vol.2, doi: 10.1109/AIM.2003.1225492."
    )
)

# Inverse CIVE: -0.441R + 0.811G - 0.385B - 18.78745, with R,G,B in [0,1] => [-0.8997, 0.7373]
register_index(
    IndexSpec(
        name="inv_cive",
        full_name="Color Index of Vegetation",
        formula=lambda R,G,B: -0.441*R + 0.811*G - 0.385*B + 0.07367627450980392, # Constant 18.78745 normalized by division by 255
        domain=(-0.899676274509804, 0.7373237254901961),
        map01=True,
        citation="Kataoka, T.; Kaneko, T.; Okamoto, H.; Hata, S. 'Crop growth estimation system using machine vision', Proceedings 2003 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM 2003), Kobe, Japan, 2003, pp. b1079-b1083 vol.2, doi: 10.1109/AIM.2003.1225492."
    )
)

# Excess Green minus Excess Red: (3G - 2.4R - B), with R,G,B in [0,1] => [-3.4, 3.0]
register_index(
    IndexSpec(
        name="exg_exr",
        full_name="Excess Green minus Excess Red",
        formula=lambda R,G,B: (3*G - 2.4*R - B),
        domain=(-3.4, 3.0),
        map01=True
    )
)




def exg(img: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the Excess Green (ExG) index for an RGB image.

    Parameters:
    -----------
        img : np.ndarray
            Input RGB image as a NumPy array of shape (H, W, 3) with values in [0, 1].

    Returns:
    --------
        np.ndarray
            ExG index array of shape (H, W) with values mapped to [0, 1].
    """
    return compute_index("exg", img)

__all__.append("exg")