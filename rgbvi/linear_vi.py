"""

"""

import numpy as np
from typing import Tuple


def green_blue(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Green Minus Blue RGBVI from an RGB image.
    First published in 
    ```
    Woebbecke, D.M., Meyer, G.E., Bargen, K.V., & Mortensen, D.A. (1994). 
    "Color indices for weed identification under various soil, residue, and lighting conditions."
    Transactions of the ASABE, 38, 259-269.
    ```

    Args:
        img (np.ndarray): An RGB image in BGR (OpenCV) format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the index in np.uint8 format and a [0,1] normalized index.
    """
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # Create the index and normalize to [0, 1]
    index = G - B
    n_index = (index + 255) / (510)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index


def red_green(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Green Minus Red RGBVI from an RGB image.
    First published in 
    ```
    Woebbecke, D.M., Meyer, G.E., Bargen, K.V., & Mortensen, D.A. (1994). 
    "Color indices for weed identification under various soil, residue, and lighting conditions."
    Transactions of the ASABE, 38, 259-269.
    ```

    Args:
        img (np.ndarray): An RGB image in BGR (OpenCV) format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the index in np.uint8 format and a [0,1] normalized index.
    """
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # Create the index and normalize to [0, 1]
    index = R - G
    n_index = (index + 255) / (510)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index


def exgr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Excess Green RGBVI from an RGB image.
    First published in 
    ```
    Woebbecke, D.M., Meyer, G.E., Bargen, K.V., & Mortensen, D.A. (1994). 
    "Color indices for weed identification under various soil, residue, and lighting conditions."
    Transactions of the ASABE, 38, 259-269.
    ```
    Args:
        img (np.ndarray): An RGB image in BGR (OpenCV) format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the index in np.uint8 format and a [0,1] normalized index.
    """

    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # Create the index and normalize to [0, 1]
    index = 2 * G - R - B
    n_index = (index + 510) / (510 + 510)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index


def exr(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Excess Red RGBVI from an RGB image.
    First published in 
    ```
    Meyer, G.E., Hindman, T.W., & Laksmi, K. (1999). 
    "Machine vision detection parameters for plant species identification."
    Proceedings of the SPIE, Volume 3543, p. 327-335.
    https://ui.adsabs.harvard.edu/link_gateway/1999SPIE.3543..327M/doi:10.1117/12.336896
    ```
    Args:
        img (np.ndarray): An RGB image in BGR (OpenCV) format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the index in np.uint8 format and a [0,1] normalized index.
    """

    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # Create the index and normalize to [0, 1]
    index = 1.4 * R - G
    n_index = (index + 255) / (357 + 255)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index

###### START HERE #######
def exgr_exr(img):
    """Summary"""
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = 2 * G - R - B - (1.4 * R - G)
    n_index = (index + 867) / (765 + 867)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index


def cive(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Color Index of Vegetation RGBVI from an RGB image
    First published in
    ```
    T. Kataoka, T. Kaneko, H. Okamoto and S. Hata, 
    "Crop growth estimation system using machine vision," 
    Proceedings 2003 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM 2003), 
    Kobe, Japan, 2003, pp. b1079-b1083 vol.2, doi: 10.1109/AIM.2003.1225492.
    ```

    CIVE is calculated as 0.441 * R - 0.811 * G + 0.385 * B + 18.78745 which corresponds to the "green" projection
    of field images after a principal component analysis.

    Args:
        img (np.ndarray): An RGB image in BGR (OpenCV) format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the index in np.uint8 format and a [0,1] normalized index.
    """
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # Create the index and normalize to [0, 1]
    index = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    n_index = (index + 188.01755) / (229.41745 + 188.01755)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index


def cive_inv(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the inverse of the Color Index of Vegetation RGBVI from an RGB image

    CIVE_INV is calculated as -0.441 * R + 0.811 * G - 0.385 * B - 18.78745 which corresponds to inverse the "green" projection
    of field images after a principal component analysis.

    Args:
        img (np.ndarray): An RGB image in BGR (OpenCV) format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the index in np.uint8 format and a [0,1] normalized index.
    """
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # Create the index and normalize to [0, 1]
    index = -0.441 * R + 0.811 * G - 0.385 * B - 18.78745
    n_index = (index + 229.41745) / (229.41745 + 188.01755)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index


def mexg(img):
    """MEXG index"""
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = 1.26 * G - 0.884 * R - 0.311 * B
    n_index = (index + 304.725) / (321.3 + 304.725)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index

def g4(img):
    """G**4 index"""
    # Grab individual channels
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = 4*G - R
    n_index = (index + 255) / (1020 + 255)
    index = (n_index * 255).astype(np.uint8)

    return index, n_index
