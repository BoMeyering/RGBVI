"""
rgbvi.core.py
Core compute functions and shared utilities
BoMeyering 2025
"""

import torch
import cv2
import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
import scipy.ndimage as ndi

from .registry import INDEX_SPECS

def standardize_mask(mask: ArrayLike) -> np.ndarray:
    """Convert input mask to a boolean numpy array.

    Parameters:
    -----------
        mask : ArrayLike
            Input mask as an array-like object.
            Any non-zero value is considered True.

    Returns:
    -----------
        np.ndarray
            Boolean numpy array representing the mask.
    """
    # Validate input type
    if not isinstance(mask, (list, tuple, ArrayLike, np.ndarray, torch.Tensor)):
        raise ValueError("mask must be a NumPy array, torch tensor, or an array-like object.")
    # Convert to numpy array
    try:
        if isinstance(mask, torch.Tensor):
            mask_arr = mask.cpu().numpy()
        else:
            mask_arr = np.asarray(mask)
    except Exception as e:
        raise ValueError("Failed to convert mask to NumPy array.") from e
    
    if mask_arr.dtype != np.bool_:
        mask_arr = mask_arr.astype(bool)
    return mask_arr

def compute_index(
        index_name: str, 
        img: ArrayLike, 
        mask: Optional[ArrayLike]=None, 
        erode: Optional[int]=None,
        den_min: float=1e-3,
        robust_mean: bool=True,
        **kwargs
    ) -> np.ndarray:
    """
    Compute a vegetative index on the provided image data.

    Parameters:
    -----------
        index_name : str
            Name of the index to compute.
        img : ArrayLike
            Input image data as an array-like object. Should be of shape (H, W, 3) for RGB images with the last dimension in RGB order.
        mask : Optional[ArrayLike]
            Optional mask to apply to the image.
        erode : Optional[int]
            Optional erosion size for the mask.
        den_min : float
            Minimum denominator value to avoid division by zero.
        robust_mean : bool
            Whether to use robust mean calculation.
        **kwargs: Additional parameters for specific indices.

    Returns:
        np.ndarray
            Computed index as a NumPy array.
    """

    spec = INDEX_SPECS.get(index_name.lower())
    if spec is None:
        raise ValueError(f"Index '{index_name}' is not a registered index.")
    
    if not isinstance(img, (list, tuple, ArrayLike, np.ndarray, torch.Tensor)):
        raise ValueError("Input image must be a NumPy array, torch tensor, or an array-like object.")
    
    # Convert to numpy array
    try:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().astype(np.float32)
        else:
            img = np.asarray(img).astype(np.float32)
    except Exception as e:
        raise ValueError("Failed to convert input image to NumPy array.") from e

    img = np.where(img == 1, 1, img) # Fill in pixels with 1 to avoid division by zero
    img /= 255.0 # Normalize to [0, 1]
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    if mask is not None:
        print(True)
        mask = standardize_mask(mask)
        # if erode is not None and erode > 0:
        #     structure = np.ones((2*erode+1, 2*erode+1), dtype=bool)
        #     mask = ndi.binary_erosion(mask, structure=structure)
    else:
        mask = np.ones(R.shape, dtype=bool)

    idx_raw = spec.formula(R, G, B)
    valid = np.isfinite(idx_raw)

    # Mask the raw_idx to only valid pixels
    values = idx_raw * valid


    print("Values shape:", values.shape)
    if values.size == 0:
        return np.nan * np.ones_like(idx_raw)

    if spec.map01 and spec.domain is not None:
        dmin, dmax = spec.domain
        values = (idx_raw - dmin) / (dmax - dmin)
        values = np.clip(values, 0.0, 1.0)
    
    if robust_mean:
        p5, p95 = np.percentile(values, [5, 95])
        print("P5, P95:", p5, p95)
        # values = values[(values >= p5) & (values <= p95)]
        values = np.clip(values, p5, p95)

    # return values.mean().astype(np.float32)
    return values * mask

