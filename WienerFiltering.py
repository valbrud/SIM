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
import utils
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np

import utils
import hpc_utils

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
                    numeric_noise=10**-10, 
                    mask = None):    
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
    # print(image_ft[*center])
    # print(ssnr_calculator.dj[*center])
    f0 = image_ft[*center] / ssnr_calculator.dj[*center]
    bj2 = np.abs(image_ft) ** 2

    find_levels = utils.find_decreasing_surface_levels3d if image_ft.ndim == 3 else utils.find_decreasing_surface_levels2d
    average_rings = utils.average_rings3d if image_ft.ndim == 3 else utils.average_rings2d
    expand_ring_averages = utils.expand_ring_averages3d if image_ft.ndim == 3 else utils.expand_ring_averages2d
    
    if average == "surface_levels":
        if mask is None:
            mask = find_levels(np.copy(ssnr_calculator.dj), direction=0)
        obj2a = utils.average_mask(bj2, mask)
    elif average == "rings":
        obj2ra = average_rings(bj2, ssnr_calculator.optical_system.otf_frequencies)
        obj2a = expand_ring_averages(obj2ra, ssnr_calculator.optical_system.otf_frequencies)
    else:
        raise AttributeError(f"Unknown method of averaging {average}")
    
    if vja is None or dja is None:
        if average == "surface_levels":
            dja = utils.average_mask(np.copy(ssnr_calculator.dj), mask)
            vja = utils.average_mask(np.copy(ssnr_calculator.vj), mask)

        elif average == "rings":
            djra = average_rings(np.copy(ssnr_calculator.dj), ssnr_calculator.optical_system.otf_frequencies)
            vjra = average_rings(np.copy(ssnr_calculator.vj), ssnr_calculator.optical_system.otf_frequencies)
            dja = expand_ring_averages(djra, ssnr_calculator.optical_system.otf_frequencies)
            vja = expand_ring_averages(vjra, ssnr_calculator.optical_system.otf_frequencies)
    
    # print('total_counts', f0)
    # noise_power_ra = vjra * f0 + image_ft.size * ssnr_calculator.readout_noise_variance**2 * djra
    noise_power = vja * f0 + image_ft.size * ssnr_calculator.readout_noise_variance**2 * dja
    ssnr = ((obj2a - noise_power ) /
                 noise_power).real
    ssnr = np.nan_to_num(ssnr)
    ssnr = np.where(ssnr_calculator.dj > numeric_noise, ssnr, 0)
    ssnr = np.where(ssnr_calculator.dj > ssnr_calculator.dj[*center] * 10**(-5), ssnr, 0)
    # plt.plot(np.log(1 + (obj2ra).real), label='total')
    # plt.plot(np.log(1 + (noise_power_ra).real), label='noise')
    # plt.plot(np.log(1 + (np.abs(ssnr))[center[0], center[1]:]), label = 'ssnr')
    
    # plt.legend()
    # plt.show()

    w = ssnr_calculator.dj / ssnr
    w = np.where(ssnr < 1, 10**9, w)
    # w = noise_power 
    filtered = image_ft  / (ssnr_calculator.dj + w)
    filtered = np.where(ssnr_calculator.dj > numeric_noise, filtered, 0)
    filtered = np.nan_to_num(filtered)
    filtered = np.where(ssnr < 1, 0, filtered)
    return filtered, w, ssnr

