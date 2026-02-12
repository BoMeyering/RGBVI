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
import matplotlib.pyplot as plt

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

    img = np.where(img == 0, 1, img) # Fill in pixels with 1 to avoid division by zero
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
    valid_mean = np.mean(mask & valid)
    invalid = ~(mask & valid)
    idx_raw[invalid] = valid_mean  # Set invalid pixels to mean of valid pixels
    print(np.min(idx_raw[valid]), np.max(idx_raw[valid]))

    plt.hist(idx_raw[valid].flatten(), bins=100)
    plt.title(f"Histogram of raw index values for {index_name}")
    plt.xlabel("Index Value")
    plt.ylabel("Frequency")
    plt.show()

    # Mask the raw_idx to only valid pixels
    values = idx_raw * valid


    if values.size == 0:
        return np.nan * np.ones_like(idx_raw)

    if robust_mean:
        p5, p95 = np.percentile(values, [.01, 99.99])
        print("P5, P95:", p5, p95)
        # values = values[(values >= p5) & (values <= p95)]
        values = np.clip(values, p5, p95)

    if spec.map01 and spec.range is not None:
        rmin, rmax = spec.range
        rmin, rmax = values.min(), idx_raw.max()
        # values = (idx_raw - rmin) / (rmax - rmin)
        values = np.clip(values, 0.0, 1.0)
        print(values)
    
    # if robust_mean:
    #     p5, p95 = np.percentile(values, [1, 99])
    #     print("P5, P95:", p5, p95)
    #     # values = values[(values >= p5) & (values <= p95)]
    #     values = np.clip(values, p5, p95)

    # return values.mean().astype(np.float32)
    return values * mask

