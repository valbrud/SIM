"""
WienerFiltering.py

This module contains functions implementing different versions of Wiener filtering.

Functions:
    filter_constant - Wiener deconvolution with a constant regularization parameter.
    filter_flat_noise_sim - Wiener deconvolution ensuring flat noise power after filtering.
    filter_true_wiener_sim - True Wiener filter with SSNR estimated from the image itself.
    filter_simulated_object_wiener - True Wiener filter using the known ground truth object.
    filter_self_consistent_wiener - Self-consistent Wiener filter using measured signal power.
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
    
    
    return filtered, w, None

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
    filtered = image_ft / (ssnr_calculator.dj + w)
    return filtered, w, None

def filter_true_wiener_sim(image_ft,
                    ssnr_calculator,
                    vja = None, 
                    dja = None,
                    average='rings',
                    numeric_noise=10**-9, 
                    mask = None, 
                    watershed_confidence_interval=20):    
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
 

    # if noise_power_ra.ndim == 1:
    #     signal_power_ra = utils.comparative_watershed_filter(np.abs((obj2ra - noise_power_ra )), noise_power_ra)
    # else:
    #     signal_power_ra = np.zeros(noise_power_ra.shape)
    #     for i in range(noise_power_ra.shape[1]):
    #         signal_power_ra[:, i] = utils.comparative_watershed_filter(np.abs((obj2ra[:, i] - noise_power_ra[:, i])), noise_power_ra[:, i])

    signal_power_ra = np.abs((obj2ra - noise_power_ra ))
    ssnr_ra = signal_power_ra / noise_power_ra
    ssnr_ra = np.nan_to_num(ssnr_ra)
    
    ssnr = expand_ring_averages(ssnr_ra, ssnr_calculator.optical_system.otf_frequencies)

    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].plot(np.log1p(1+ (np.abs(obj2ra)))[:, obj2ra.shape[1]//2]    , label='total power')
    # ax[0].plot(np.log1p(1 + (noise_power_ra).real)[:, obj2ra.shape[1]//2], label='noise power') 
    # ax[1].plot(np.log1p(1  + ssnr_ra[:, obj2ra.shape[1]//2].real), label='SSNR')
    # plt.show()

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

    w = np.where(ssnr < 1, 10**9, w)
    # w = noise_power 
    filtered = image_ft  / (ssnr_calculator.dj + w + numeric_noise)

    filtered = np.nan_to_num(filtered)
    filtered = np.where(ssnr_calculator.dj < numeric_noise, 0, filtered)
    ssnr = np.where(ssnr_calculator.dj < numeric_noise, 0, ssnr)
    # filtered = np.where(ssnr < 1, 0, filtered)
    filtered_image = hpc_utils.wrapped_ifftn(filtered)
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

    filtered = image_ft / (ssnr_calculator.dj + w + numeric_noise)

    filtered = np.where(ssnr_calculator.dj > numeric_noise, filtered, 0)
    filtered = np.nan_to_num(filtered)

    return filtered, w, ssnr


def filter_self_consistent_wiener(image_ft,
                              ssnr_calculator: SSNRBase,
                              numeric_noise=10**-12):
    """
    Self-consistent Wiener filter that estimates the SSNR directly from
    the measured signal power divided by the noise variance.

    Args:
        image_ft (np.ndarray): Fourier transform of the reconstructed image.
        ssnr_calculator (SSNRBase): SSNR calculator object.
        numeric_noise (float): Small regularization constant to avoid division by zero.

    Returns:
        tuple: (filtered image FT, regularization array w, estimated SSNR).
    """
    f0 = np.amax(image_ft).real
    nomenator = np.abs(image_ft) ** 2 
    denominator = ssnr_calculator.vj * f0 
    ssnr_measured =  nomenator / (denominator + numeric_noise) 

    w = ssnr_calculator.dj / ssnr_measured
    
    filtered = image_ft / (ssnr_calculator.dj + w + numeric_noise)

    filtered = np.where(ssnr_calculator.dj > numeric_noise, filtered, 0)
    filtered = np.nan_to_num(filtered)

    return filtered, w, ssnr_measured

