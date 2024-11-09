import numpy as np
from typing import Tuple

def ndi(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    First published in
    ```
    A.J. Pérez, F. López, J.V. Benlloch, S. Christensen,
    Colour and shape analysis techniques for weed detection in cereal fields,
    Computers and Electronics in Agriculture, Volume 25, Issue 3, 2000, Pages 197-212, ISSN 0168-1699,
    https://doi.org/10.1016/S0168-1699(99)00068-X.
    ```

    Args:
        img (np.ndarray): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """

        
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (G - R)/np.clip((G + R), a_min=1, a_max=510)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def woebbecke_index(img):
    """Wobbecke index this is very noisy"""
    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (G - B)/np.clip(np.abs(R - G), a_min=1, a_max=255)
    n_index = (index + 255)/(510)
    index = (index*255).astype(np.uint8)

    return index, n_index

def cive(img):
    """Color Index of Vegetation Extraction"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = 0.441*R - 0.811*G + 0.385*B + 18.78745
    n_index = (index + 188.01755)/(229.41745 + 188.01755)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def cive_inv(img):
    """Color Index of Vegetation Extraction Inverse"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = -0.441*R + 0.811*G - 0.385*B - 18.78745
    n_index = (index + 229.41745)/(229.41745 + 188.01755)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def veg(img):
    """Vegetation Index"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)
    
    index = G/np.clip(R**0.667 * B**0.333, a_min=0.1, a_max=255)


    index[np.isnan(index)] = 0
    print(np.any(np.isnan(index)))

    p1, p99 = np.nanpercentile(index, 1), np.nanpercentile(index, 90)
  
    index = np.clip(index, p1, p99)
    n_index = (index - index.min())/(index.max() - index.min())

    return index.astype(np.uint8)

def mgrvi(img):
    """MGVRI"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (G**2 - R**2)/np.clip((G**2 + R**2), a_min=1, a_max=130050)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def gli(img):
    """Color Index of Vegetation Extraction Inverse"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (2*G - R - B)/np.clip((2*G + R + B), a_min=1, a_max=1020)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)


    return index, n_index

def rgbvi(img):
    """Color Index of Vegetation Extraction Inverse"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (G**2 - (B*R))/np.clip(G**2 + (B*R), a_min=1, a_max=130050)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def rgri(img):
    """Red Green Ratio Index"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = R/np.clip(G, a_min=1, a_max=255)
    n_index = (index)/(255)
    index = (n_index*255).astype(np.uint8)

    p99 = np.percentile(index, 99)
    index = np.clip(index, 0, p99)

    return index, n_index

def ngrdi(img):
    """Normalized green red difference index"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (G - R)/np.clip(G + R, a_min=1, a_max=510)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def ngbdi(img):
    """Color Index of Vegetation Extraction Inverse"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    # index = (G - B)/np.clip(G + B, a_min=1, a_max=510)
    index = (G - B)/(G + B + 0.000001)
    index = np.clip(index, -1, 1)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def vari(img):
    """Color Index of Vegetation Extraction Inverse"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(np.float32)

    index = np.divide((G - R), (G + R - B + 0.00001))
    index = np.clip(index, -1, 1)
    n_index = (index + 1)/2

    # index = (G - R)/(G + R - B)
    # idx = np.isnan(index)
    # index[idx] = 0

    # idx = np.isinf(index)
    # index[idx] = 0


    # p3, p97 = np.percentile(index, 3), np.percentile(index, 97)
    # index = np.clip(index, p3, p97)

    # n_index = (index + 255)/(1 + 255)
    # print(n_index.min())
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def kawashima(img):
    """Kawashima index"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = (R - B)/np.clip(R + B, a_min=1, a_max=510)
    n_index = (index + 1)/(2)
    index = (n_index*255).astype(np.uint8)

    return index, n_index

def mexg(img):
    """MEXG index"""

    B, G, R = np.moveaxis(img, source=2, destination=0).astype(float)

    index = 1.26*G - 0.884*R - 0.311*B
    n_index = (index + 304.725)/(321.3 + 304.725)
    index = (n_index*255).astype(np.uint8)

    return index, n_index