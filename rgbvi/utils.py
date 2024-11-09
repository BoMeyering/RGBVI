"""
rgbvi.utils
Utility functions to help out with imaging
"""

from glob import glob
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def show_img(img: np.ndarray, window_name: str='default'):
    """
    Show a Numpy array image in a GUI window

    Args:
        img (np.ndarray): Either a 3 channel or 1 channel image
        window_name (str, optional): The name of the GUI window to show the image in. Defaults to 'default'.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def clahe_channel(img: np.ndarray, clip_limit: int=10) -> np.ndarray:
    """
    Perform CLAHE on the lightness channel of an image in l*a*b basis.

    Args:
        img (np.ndarray): A 3 channel image in BGR (OpenCV) format.
        clip_limit (int, optional): integer clip limit for CLAHE operation. Defaults to 10.

    Returns:
        np.ndarray: A Numpy image array in BGR format processed by CLAHE
    """

    # Convert image to lab
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Apply CLAHE
    l_channel = lab_img[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=100)
    new_l = clahe.apply(l_channel)

    # Reset the l channel and convert to BGR
    lab_img[:, :, 0] = new_l
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)

    return bgr_img