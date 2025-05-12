"""
WienerFiltering.py

This module contains functions implementing different version of Wiener filtering.
"""

import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)


import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
import stattools
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np

import stattools
import wrappers

def filter_constant(image_ft, otf, w):
    """
    Applies a Wiener filtering (deconvolution) procedure with a regularization filter 
    set up to be a constant (standard procedure). The constant can be given or estimated from the image. 
    In case the ssnr calculator is provided, the constant is set to be equal to the 1/6. 

    Args:
        image (np.ndarray): The input image to be filtered.
        otf: Optical transfer function (OTF) of the system.
        w (float, optional): The constant noise value.
    Returns:
        np.ndarray: The filtered image.
    """
    if not w is None:
        filtered = image_ft * otf.conjugate() / (otf * otf.conjugate() + w)
    
    return filtered, w

def filter_flat_noise(image_ft, ssnr_calculator): 
    """
    Applies a Winer filtering (deconvolution) procedure with a regularization filter
    set up to ensure the flat noise. 

    Args:
        image (np.ndarray): The input image to be filtered.
        otf: Optical transfer function (OTF) of the system.
        ssnr_calculator: SSNR calculator object.

    Returns:
        np.ndarray: The filtered image.
    """
    if image_ft.ndim != ssnr_calculator.dj.ndim:
        raise ValueError("The image and SSNR calculator dimensions do not match.")
    
    w = ssnr_calculator.vj ** 0.5 - ssnr_calculator.dj
    # filtered = image_ft * otf.conjugate() / (otf * otf.conjugate() + w)
    filtered = image_ft * ssnr_calculator.dj.conjugate() / (ssnr_calculator.dj * ssnr_calculator.dj.conjugate() + w)
    return filtered, w

def filter_true_wiener(image_ft,
                    ssnr_calculator,
                    vja = None, 
                    dja = None,
                    average='rings',
                    numeric_noise=10**-4):    
    """
    Applies a Winer filtering (deconvolution) procedure with a regularization filter
    set up to ensure the best contrast (True Wiener). In the (realistic) case of the unknown
    ground object, the SSNR is estimated from the image for this purpose. 

    Args:
        image (np.ndarray): The input image to be filtered.
        otf: Optical transfer function (OTF) of the system.
        ssnr_calculator: SSNR calculator object.
        average (str): The method of averaging the SSNR for the self consistent estimation
        of the (full) ssnr (default option is ring averaging).

    Returns:
        np.ndarray: The filtered image.
    """
    if image_ft.ndim != ssnr_calculator.dj.ndim:
        raise ValueError("The image and SSNR calculator dimensions do not match.")

    center = np.array(image_ft.shape)//2
    f0 = image_ft[*center] / ssnr_calculator.dj[*center]
    bj2 = np.abs(image_ft) ** 2

    find_levels = stattools.find_decreasing_surface_levels3d if image_ft.ndim == 3 else stattools.find_decreasing_surface_levels2d
    average_rings = stattools.average_rings3d if image_ft.ndim == 3 else stattools.average_rings2d
    expand_ring_averages = stattools.expand_ring_averages3d if image_ft.ndim == 3 else stattools.expand_ring_averages2d
    
    if average == "surface_levels":
        mask = find_levels(np.copy(ssnr_calculator.dj), direction=0)
        obj2a = stattools.average_mask(bj2, mask)
    elif average == "rings":
        obj2ra = average_rings(bj2, ssnr_calculator.optical_system.otf_frequencies)
        obj2a = expand_ring_averages(obj2ra, ssnr_calculator.optical_system.otf_frequencies)
    else:
        raise AttributeError(f"Unknown method of averaging {average}")
    
    if vja is None or dja is None:
        if average == "surface_levels":
            dja = stattools.average_mask(np.copy(ssnr_calculator.dj), mask)
            vja = stattools.average_mask(np.copy(ssnr_calculator.vj), mask)

        elif average == "rings":
            djra = average_rings(np.copy(ssnr_calculator.dj), ssnr_calculator.optical_system.otf_frequencies)
            vjra = average_rings(np.copy(ssnr_calculator.vj), ssnr_calculator.optical_system.otf_frequencies)
            dja = expand_ring_averages(djra, ssnr_calculator.optical_system.otf_frequencies)
            vja = expand_ring_averages(vjra, ssnr_calculator.optical_system.otf_frequencies)


    ssnr = ((obj2a - vja * f0 - image_ft.size * ssnr_calculator.readout_noise_variance**2 * dja) /
                    (vja * f0 + image_ft.size * ssnr_calculator.readout_noise_variance**2 * dja)).real
    ssnr = np.nan_to_num(ssnr)
    # plt.imshow(np.log1p(10 ** 8 * np.abs(ssnr)))
    plt.imshow(np.log1p(10**8 * np.abs(np.where(ssnr_calculator.dj > numeric_noise, ssnr_calculator.dj, 0))))
    plt.show()
    w = (dja + 10**3 * numeric_noise)/(ssnr + numeric_noise)
    # filtered = image_ft * otf.conjugate() / (otf * otf.conjugate() + w)
    filtered = image_ft * ssnr_calculator.dj.conjugate() / (ssnr_calculator.dj * ssnr_calculator.dj.conjugate() + w)
    filtered = np.where(ssnr_calculator.dj > numeric_noise, filtered, 0)
    return filtered, w, ssnr

