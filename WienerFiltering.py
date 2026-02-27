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
from SSNRCalculator import SSNRBase

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

def filter_flat_noise_sim(image_ft, ssnr_calculator): 
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

def filter_true_wiener_sim(image_ft,
                    ssnr_calculator,
                    vja = None, 
                    dja = None,
                    average='rings',
                    numeric_noise=10**-12, 
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
    print("f0:", f0)
    obj2 = np.abs(image_ft) ** 2

    find_levels = utils.find_decreasing_surface_levels3d if image_ft.ndim == 3 else utils.find_decreasing_surface_levels2d
    average_rings = utils.average_rings3d if image_ft.ndim == 3 else utils.average_rings2d
    expand_ring_averages = utils.expand_ring_averages3d if image_ft.ndim == 3 else utils.expand_ring_averages2d
    
    if average == "surface_levels":
        if mask is None:
            mask = find_levels(np.copy(ssnr_calculator.dj), direction=0)
        obj2a = utils.average_mask(obj2, mask)
    elif average == "rings":
        obj2ra = average_rings(obj2, ssnr_calculator.optical_system.otf_frequencies)
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
    noise_power = vja * f0 + image_ft.size * ssnr_calculator.readout_noise_variance**2 * dja
    noise_power_ra = vjra * f0 + image_ft.size * ssnr_calculator.readout_noise_variance**2 * djra

    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].plot(np.log1p(1 + 10**8 * (np.abs(obj2ra)))[:, 50], label='total power')
    # ax[0].plot(np.log1p(1 + 10**8 * (noise_power_ra).real)[:, 50], label='noise power') 
    # ax[1].plot(np.log1p(1 + 10**8 * ((np.abs(obj2ra) - noise_power_ra).real))[:, 50], label='signal power')
    # plt.show()

    ssnr = np.abs((obj2a - noise_power ))/ noise_power
    ssnr = np.nan_to_num(ssnr)




    # ssnr = np.where(ssnr_calculator.dj > numeric_noise, ssnr, numeric_noise)
    # ssnr = np.where(ssnr_calculator.dj > ssnr_calculator.dj[*center] * 10**(-9), ssnr, numeric_noise)

    # plt.plot(np.log(1 + (obj2ra).real), label='total')
    # plt.plot(np.log(1 + (noise_power_ra).real), label='noise')
    # plt.plot(np.log(1 + (np.abs(ssnr))[center[0], center[1]:]), label = 'ssnr')
    
    # plt.legend()
    # plt.show()

    # unfiltered_image = hpc_utils.wrapped_ifftn(image_ft)

    
    w = dja / ssnr 
    # fig, ax = plt.subplots(1, 3, figsize=(10,5))
    # ax[0].plot(np.log1p(10**8 * (ssnr))[50:, 50, 50])
    # ax[1].plot(np.log1p(10**8 * (dja))[50:, 50, 50])
    # ax[2].plot(np.log1p(10**8 * (w))[50:, 50, 50])
    # ax[0].set_title('SSNR')
    # ax[1].set_title('Dj average')
    # ax[2].set_title('Regularization w')
    # plt.show()

    # w = np.where(ssnr < 0.01, 10**9, w)
    # w = noise_power 
    filtered = image_ft  / (ssnr_calculator.dj + w)

    filtered = np.nan_to_num(filtered)
    # filtered = np.where(ssnr < 1, 0, filtered)
    # filtered_image = hpc_utils.wrapped_ifftn(filtered)
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(np.log1p(1 + (np.abs(image_ft)))[:, :, 50], cmap='viridis')
    # ax[1].imshow(np.log1p(1 + (np.abs(filtered)))[:, :, 50], cmap='viridis')
    # plt.show()

    # plt.imshow(np.abs(filtered_image), cmap='gray')
    # plt.show()


    return filtered, w, ssnr

def filter_simulated_object_wiener(image_ft,
                              ssnr_calculator: SSNRBase,
                              ground_object_ft,
                              numeric_noise=10**-12):
    """
    Applies a Winer filtering (deconvolution) procedure with a regularization filter
    set up to ensure the best contrast (True Wiener). In this function, the ground truth
    object is known (simulated case), so the SSNR is computed directly from it.

    Args:
        image (np.ndarray): The input image to be filtered.
        otf: Optical transfer function (OTF) of the system.
        ssnr_calculator: SSNR calculator object.
        ground_object_ft: The Fourier transform of the ground truth object.

    Returns:
        np.ndarray: The filtered image.
    """
    if image_ft.ndim != ssnr_calculator.dj.ndim:
        raise ValueError("The image and SSNR calculator dimensions do not match.")

    # plt.imshow(np.log1p(10**8 * np.abs(ground_object_ft[:, :, ground_object_ft.shape[2]//2])), cmap='gray')
    # plt.title('Ground truth object FT')
    # plt.show()
    ssnr = ssnr_calculator.compute_full_ssnr(ground_object_ft)

    w = ssnr_calculator.dj / ssnr
    # plt.imshow(np.log1p(np.abs(w[20:-20, 20:-20, 50])), cmap='gray')
    # plt.show()

    filtered = image_ft / (ssnr_calculator.dj + w)

    filtered = np.where(ssnr_calculator.dj > 10**-12, filtered, 0)
    filtered = np.nan_to_num(filtered)

    return filtered, w, ssnr
