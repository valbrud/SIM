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
from matplotlib.colors import LogNorm
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
                    numeric_noise=10**-7, 
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
    noise_power = vja * np.abs(f0) + image_ft.size * ssnr_calculator.readout_noise_variance**2 * dja
    noise_power_ra = vjra * np.abs(f0) + image_ft.size * ssnr_calculator.readout_noise_variance**2 * djra
 

    # if noise_power_ra.ndim == 1:
    #     signal_power_ra = utils.comparative_watershed_filter(np.abs((obj2ra - noise_power_ra )), noise_power_ra)
    # else:
    #     signal_power_ra = np.zeros(noise_power_ra.shape)
    #     for i in range(noise_power_ra.shape[1]):
    #         signal_power_ra[:, i] = utils.comparative_watershed_filter(np.abs((obj2ra[:, i] - noise_power_ra[:, i])), noise_power_ra[:, i])

    signal_power_ra = np.abs((obj2ra - noise_power_ra))
    # plt.plot(np.log1p(np.abs(obj2ra.real)))
    # plt.show()
    ssnr_ra = signal_power_ra / (noise_power_ra + 10 * numeric_noise)
    ssnr_ra = np.nan_to_num(ssnr_ra)
    # ssnr_ra = np.where(signal_power_ra, numeric_noise, ssnr_ra)
    # plt.plot(ssnr_ra)
    # plt.show()
    ssnr = expand_ring_averages(ssnr_ra, ssnr_calculator.optical_system.otf_frequencies)

    if image_ft.ndim == 3:
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].plot(np.log1p((np.abs(obj2ra)))[:, obj2ra.shape[1]//2]    , label='total power')
        ax[0].plot(np.log1p((noise_power_ra).real)[:, obj2ra.shape[1]//2], label='noise power') 
        ax[0].legend()
        ax[1].plot(np.log1p(ssnr_ra[:, obj2ra.shape[1]//2].real), label='SSNR')
        ax[1].legend()
        # plt.show()
    else: 
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].plot(np.log1p((np.abs(obj2ra))), label='total power')
        ax[0].plot(np.log1p((noise_power_ra).real), label='noise power') 
        ax[0].legend()
        ax[1].plot(np.log1p(ssnr_ra.real), label='SSNR')
        ax[1].legend()
        # plt.show()


    ssnr = np.where(ssnr_calculator.dj > numeric_noise, ssnr, numeric_noise)
    ssnr = np.where(ssnr_calculator.dj > ssnr_calculator.dj[*center] * 10**(-9), ssnr, numeric_noise)
    ssnr = np.where(ssnr_calculator.dj < numeric_noise, 0, ssnr)

    # plt.plot(np.log(1 + (obj2ra).real), label='total')
    # plt.plot(np.log(1 + (noise_power_ra).real), label='noise')
    # plt.plot(np.log(1 + (np.abs(ssnr))[center[0], center[1]:]), label = 'ssnr')
    
    # plt.legend()
    # plt.show()

    # # unfiltered_image = hpc_utils.wrapped_ifftn(image_ft)

    
    w = dja / ssnr 
    shape = image_ft.shape
    if len(shape) == 3: 
        fig, ax = plt.subplots(1, 3, figsize=(10,5))
        ax[0].plot(np.log1p(10**8 * (ssnr))[shape[0]//2:, shape[1]//2, shape[2]//2])
        ax[1].plot(np.log1p(10**8 * (dja))[shape[0]//2:, shape[1]//2, shape[2]//2])
        ax[2].plot(np.log1p(10**8 * (w))[shape[0]//2:, shape[1]//2, shape[2]//2])
        ax[0].set_title('SSNR')
        ax[1].set_title('Dj average')
        ax[2].set_title('Regularization w')
        # plt.show()
    else: 
        fig, ax = plt.subplots(1, 3, figsize=(10,5))
        ax[0].plot(np.log1p(ssnr)[shape[0]//2:, shape[1]//2])
        ax[1].plot(np.log1p((dja)[shape[0]//2:, shape[1]//2]))
        ax[2].plot(np.log1p((w)[shape[0]//2:, shape[1]//2]))    
        ax[0].set_title('SSNR')
        ax[1].set_title('Dj average')
        ax[2].set_title('Regularization w')
        # plt.show()

    w = np.where(ssnr < 1, 10**9, w)
    # w = noise_power 
    filtered = image_ft  / (ssnr_calculator.dj + w + numeric_noise)

    filtered = np.nan_to_num(filtered)
    filtered = np.where(ssnr_calculator.dj < numeric_noise, 0, filtered)
    # ssnr = np.where(ssnr_calculator.dj < numeric_noise, 0, ssnr)
    # filtered = np.where(ssnr < 1, 0, filtered)
    filtered_image = hpc_utils.wrapped_ifftn(filtered)
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # ax[0].imshow(np.log1p(1 + (np.abs(image_ft)))[:, :, 3], cmap='viridis')
    # ax[1].imshow(np.log1p(1 + (np.abs(filtered)))[:, :, 3], cmap='viridis')
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
                              numeric_noise=10**-12, 
                              kernel = None):
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


def _compute_measured_ssnr(image_ft, ssnr_calculator: SSNRBase, numeric_noise=10**-12):
    f0 = np.amax(image_ft).real / np.amax(ssnr_calculator.dj).real
    numerator = np.abs(image_ft) ** 2
    denominator = ssnr_calculator.vj * f0
    ssnr_measured = np.real(numerator / (denominator + numeric_noise))
    return np.maximum(ssnr_measured, 0)


def _iter_tile_sizes(shape):
    max_size = max(shape)
    tile_size = 1
    while tile_size < max_size:
        yield tile_size
        tile_size += 2
    yield max_size


def _iter_disjoint_tiles(shape, tile_size):
    ranges = [range(0, axis_size, tile_size) for axis_size in shape]
    for starts in np.ndindex(*(len(axis_range) for axis_range in ranges)):
        slices = []
        for axis, axis_index in enumerate(starts):
            start = ranges[axis][axis_index]
            stop = min(start + tile_size, shape[axis])
            slices.append(slice(start, stop))
        yield tuple(slices)


def _tile_debug_slice(array: np.ndarray):
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[..., array.shape[2] // 2]
    raise ValueError(f"Unsupported dimensionality for debug plotting: {array.ndim}")


def _edge_band_mean(array: np.ndarray, edge_width: int = 25):
    if array.ndim != 2:
        raise ValueError(f"Edge-band debug expects a 2D slice, got {array.ndim}D")

    clamped_width = min(edge_width, array.shape[0], array.shape[1])
    if clamped_width <= 0:
        return float('nan')

    edge_mask = np.zeros(array.shape, dtype=bool)
    edge_mask[:clamped_width, :] = True
    edge_mask[-clamped_width:, :] = True
    edge_mask[:, :clamped_width] = True
    edge_mask[:, -clamped_width:] = True
    return float(np.mean(array[edge_mask] ** 2))


def _imshow_ssnr(axis, ssnr_slice: np.ndarray, title: str):
    positive_values = ssnr_slice[ssnr_slice > 0]
    if positive_values.size:
        vmin = float(np.min(positive_values))
        vmax = float(np.max(positive_values))
        display = np.where(ssnr_slice > 0, ssnr_slice, np.nan)
        image = axis.imshow(
            display,
            origin='lower',
            norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * (1 + 10**-12)))
        )
    else:
        image = axis.imshow(ssnr_slice, origin='lower')
    axis.set_title(title)
    return image


def plot_tiling_debug(ssnr_measured: np.ndarray,
                      ssnr_tiled: np.ndarray,
                      removal_iteration: np.ndarray,
                      max_iterations_to_plot: int = 5,
                      edge_width: int = 25):
    measured_slice = _tile_debug_slice(ssnr_measured)
    tiled_slice = _tile_debug_slice(ssnr_tiled)
    removal_slice = _tile_debug_slice(removal_iteration)
    edge_mean = _edge_band_mean(measured_slice, edge_width=edge_width)
    print(f"Average measured SSNR in {edge_width}-pixel edge band: {edge_mean:.6g}")
    max_panels = max_iterations_to_plot + 1
    fig, axes = plt.subplots(2, max_panels, figsize=(5 * max_panels, 8))

    measured_image = _imshow_ssnr(axes[0, 0], measured_slice, 'SSNR measured')
    plt.colorbar(measured_image, ax=axes[0, 0])

    tiled_image = _imshow_ssnr(axes[1, 0], tiled_slice, 'SSNR estimated by tiling')
    plt.colorbar(tiled_image, ax=axes[1, 0])

    for iteration in range(max_iterations_to_plot):
        removed_mask = (removal_slice == iteration).astype(float)
        removal_image = axes[0, iteration + 1].imshow(removed_mask, origin='lower', cmap='magma', vmin=0.0, vmax=1.0)
        axes[0, iteration + 1].set_title(f'removed {2 * iteration + 1}x{2 * iteration + 1}')
        plt.colorbar(removal_image, ax=axes[0, iteration + 1])

        assigned_iteration = np.where(removal_slice == iteration, tiled_slice, 0.0)
        assigned_image = _imshow_ssnr(
            axes[1, iteration + 1],
            assigned_iteration,
            f'assigned SSNR {2 * iteration + 1}x{2 * iteration + 1}'
        )
        plt.colorbar(assigned_image, ax=axes[1, iteration + 1])

    plt.tight_layout()
    plt.show()


def _finalize_tiled_filter(image_ft,
                           ssnr_calculator: SSNRBase,
                           ssnr_tiled: np.ndarray,
                           numeric_noise=10**-6):
    w = np.where(ssnr_tiled > 10**-1, ssnr_calculator.dj  / (ssnr_tiled + numeric_noise), 10**9)
    # print(ssnr_tiled)
    # plt.imshow(ssnr_calculator.dj.real / ssnr_tiled.real, vmin=0, vmax=10)
    # plt.show()
    filtered = image_ft / (ssnr_calculator.dj + w + numeric_noise)
    filtered = np.where(ssnr_calculator.dj > numeric_noise, filtered, 0)
    filtered = np.nan_to_num(filtered)
    return filtered, w, ssnr_tiled


def _tile_acceptance_score(tile_ssnr_total: np.ndarray | float,
                           tile_pixel_count: np.ndarray | float):
    denominator = np.maximum(2 * tile_pixel_count + 4 * tile_ssnr_total, 10**-12)
    return tile_ssnr_total / np.sqrt(denominator)


def _tile_unbiased_total_ssnr(tile_ssnr_squared_sum: np.ndarray | float,
                                tile_pixel_count: np.ndarray | float):
    corrected_total = np.maximum(tile_ssnr_squared_sum - 2 * tile_pixel_count + 4, 0.0)
    return np.sqrt(corrected_total) - 2


def _tile_ssnr_value(tile_ssnr_squared_sum: np.ndarray | float,
                     tile_values: np.ndarray,
                     tile_pixel_count: np.ndarray | float):
    if np.size(tile_values) == 0:
        return 0.0
    unbiased_total = _tile_unbiased_total_ssnr(tile_ssnr_squared_sum, tile_pixel_count)
    return float(unbiased_total / tile_pixel_count)


def _tile_debiased_power(
                         tile_ssnr_squared_sum: np.ndarray | float,
                         tile_pixel_count: np.ndarray | float):
    return np.maximum(_tile_unbiased_total_ssnr(tile_ssnr_squared_sum, tile_pixel_count), 0.0)


def _print_tile_iteration_stats(tile_size: int,
                                tile_ssnr_total: np.ndarray,
                                tile_ssnr_squared_sum: np.ndarray,
                                tile_score: np.ndarray,
                                tile_debiased_power: np.ndarray,
                                accepted_tiles: np.ndarray):
    if tile_ssnr_total.size == 0:
        print(f"tile {tile_size}x{tile_size}: no eligible tiles")
        return

    print(
        f"tile {tile_size}x{tile_size}: eligible={tile_ssnr_total.size}, accepted={int(np.count_nonzero(accepted_tiles))}, "
        f"sumSSNR mean={np.mean(tile_ssnr_total):.6g}, max={np.max(tile_ssnr_total):.6g}, "
        f"sumSSNR2 mean={np.mean(tile_ssnr_squared_sum):.6g}, max={np.max(tile_ssnr_squared_sum):.6g}, "
        f"score mean={np.mean(tile_score):.6g}, max={np.max(tile_score):.6g}, "
        f"debiased mean={np.mean(tile_debiased_power):.6g}, max={np.max(tile_debiased_power):.6g}"
    )


def _tile_pixel_capacity(tile_slices):
    return int(np.prod([tile_slice.stop - tile_slice.start for tile_slice in tile_slices]))


def _remaining_tile_pixel_count(tile_remaining: np.ndarray):
    return int(np.count_nonzero(tile_remaining))


def _compute_tiled_ssnr_simple(ssnr_measured: np.ndarray,
                               harmonic_count: int,
                               SSNRc: float,
                               debug: bool = False,
                               debug_iterations: int = 5,
                               edge_width: int = 25):
    del harmonic_count
    remaining_mask = np.ones(ssnr_measured.shape, dtype=bool)
    ssnr_tiled = np.zeros_like(ssnr_measured, dtype=np.float64)
    removal_iteration = -np.ones(ssnr_measured.shape, dtype=np.int32)

    for iteration_index, tile_size in enumerate(_iter_tile_sizes(ssnr_measured.shape)):
        iteration_ssnr_total = []
        iteration_ssnr_squared_sum = []
        iteration_score = []
        iteration_debiased = []
        iteration_accepted = []
        for tile_slices in _iter_disjoint_tiles(ssnr_measured.shape, tile_size):
            tile_remaining = remaining_mask[tile_slices]
            if not np.any(tile_remaining):
                continue

            tile_values = ssnr_measured[tile_slices][tile_remaining]
            tile_ssnr_total = float(np.sum(tile_values))
            tile_ssnr_squared_sum = float(np.sum(tile_values ** 2))
            tile_pixel_count = _remaining_tile_pixel_count(tile_remaining)
            criterion = float(_tile_acceptance_score(tile_ssnr_squared_sum, tile_pixel_count))
            debiased_power = float(_tile_debiased_power(tile_ssnr_squared_sum, tile_ssnr_squared_sum, tile_pixel_count))
            is_accepted = criterion > SSNRc
            iteration_ssnr_total.append(tile_ssnr_total)
            iteration_ssnr_squared_sum.append(tile_ssnr_squared_sum)
            iteration_score.append(criterion)
            iteration_debiased.append(debiased_power)
            iteration_accepted.append(is_accepted)
            if is_accepted:
                tile_value = _tile_ssnr_value(tile_ssnr_squared_sum, tile_values, tile_pixel_count)
                tile_ssnr = ssnr_tiled[tile_slices]
                tile_ssnr[tile_remaining] = tile_value
                ssnr_tiled[tile_slices] = tile_ssnr

                tile_removals = removal_iteration[tile_slices]
                tile_removals[tile_remaining] = iteration_index
                removal_iteration[tile_slices] = tile_removals

                tile_remaining_updated = remaining_mask[tile_slices]
                tile_remaining_updated[tile_remaining] = False
                remaining_mask[tile_slices] = tile_remaining_updated

        if debug:
            _print_tile_iteration_stats(
                tile_size,
                np.asarray(iteration_ssnr_total, dtype=np.float64),
                np.asarray(iteration_ssnr_squared_sum, dtype=np.float64),
                np.asarray(iteration_score, dtype=np.float64),
                np.asarray(iteration_debiased, dtype=np.float64),
                np.asarray(iteration_accepted, dtype=bool),
            )

    if debug:
        plot_tiling_debug(
            ssnr_measured,
            ssnr_tiled,
            removal_iteration,
            max_iterations_to_plot=debug_iterations,
            edge_width=edge_width,
        )
    return ssnr_tiled, removal_iteration


def _block_reduce_sum(array: np.ndarray, block_size: int):
    pad_width = []
    padded_shape = []
    for axis_size in array.shape:
        remainder = axis_size % block_size
        pad_after = 0 if remainder == 0 else block_size - remainder
        pad_width.append((0, pad_after))
        padded_shape.append(axis_size + pad_after)
    padded = np.pad(array, pad_width, mode='constant', constant_values=0)

    reshape_dims = []
    reduction_axes = []
    axis_counter = 0
    for axis_size in padded_shape:
        reshape_dims.extend([axis_size // block_size, block_size])
        reduction_axes.append(axis_counter + 1)
        axis_counter += 2
    reduced = padded.reshape(reshape_dims).sum(axis=tuple(reduction_axes))
    return reduced, padded_shape


def _compute_tiled_ssnr_fast(ssnr_measured: np.ndarray,
                             harmonic_count: int,
                             SSNRc: float,
                             debug: bool = False,
                             debug_iterations: int = 5,
                             edge_width: int = 25):
    del harmonic_count
    remaining_mask = np.ones(ssnr_measured.shape, dtype=bool)
    ssnr_tiled = np.zeros_like(ssnr_measured, dtype=np.float64)
    removal_iteration = -np.ones(ssnr_measured.shape, dtype=np.int32)

    for iteration_index, tile_size in enumerate(_iter_tile_sizes(ssnr_measured.shape)):
        remaining_ssnr = np.where(remaining_mask, ssnr_measured, 0.0)
        remaining_ssnr_squared = remaining_ssnr ** 2
        remaining_counts = remaining_mask.astype(np.float64)
        reduced_ssnr, _ = _block_reduce_sum(remaining_ssnr, tile_size)
        reduced_ssnr_squared, _ = _block_reduce_sum(remaining_ssnr_squared, tile_size)
        reduced_counts, _ = _block_reduce_sum(remaining_counts, tile_size)

        criterion = _tile_acceptance_score(reduced_ssnr_squared, reduced_counts)
        debiased_power = _tile_debiased_power(reduced_ssnr_squared, reduced_counts)
        eligible_tiles = reduced_counts > 0
        accepted_tiles = criterion > SSNRc
        if debug:
            _print_tile_iteration_stats(
                tile_size,
                reduced_ssnr[eligible_tiles],
                reduced_ssnr_squared[eligible_tiles],
                criterion[eligible_tiles],
                debiased_power[eligible_tiles],
                accepted_tiles[eligible_tiles],
            )
        if not np.any(accepted_tiles):
            continue

        for tile_index in np.argwhere(accepted_tiles):
            tile_slices = []
            for axis, block_index in enumerate(tile_index):
                start = int(block_index) * tile_size
                stop = min(start + tile_size, ssnr_measured.shape[axis])
                tile_slices.append(slice(start, stop))
            tile_slices = tuple(tile_slices)
            tile_remaining = remaining_mask[tile_slices]
            if not np.any(tile_remaining):
                continue

            tile_values = ssnr_measured[tile_slices][tile_remaining]
            tile_ssnr_squared_sum = float(np.sum(tile_values**2))
            tile_pixel_count = _remaining_tile_pixel_count(tile_remaining)
            tile_value = _tile_ssnr_value(tile_ssnr_squared_sum, tile_values, tile_pixel_count)
            tile_ssnr = ssnr_tiled[tile_slices]
            tile_ssnr[tile_remaining] = tile_value
            ssnr_tiled[tile_slices] = tile_ssnr

            tile_removals = removal_iteration[tile_slices]
            tile_removals[tile_remaining] = iteration_index
            removal_iteration[tile_slices] = tile_removals

            tile_remaining_updated = remaining_mask[tile_slices]
            tile_remaining_updated[tile_remaining] = False
            remaining_mask[tile_slices] = tile_remaining_updated

    if debug:
        plot_tiling_debug(
            ssnr_measured,
            ssnr_tiled,
            removal_iteration,
            max_iterations_to_plot=debug_iterations,
            edge_width=edge_width,
        )
    return ssnr_tiled, removal_iteration


def filter_self_consistent_wiener_tiling(image_ft,
                                         ssnr_calculator: SSNRBase,
                                         SSNRc = 10,
                                         numeric_noise=10**-6,
                                         method: str = 'fast',
                                         debug: bool = False,
                                         debug_iterations: int = 5,
                                         edge_width: int = 25):
    """
    Self-consistent Wiener filter selecting regions to average over in a tiling fashion 
    from the limitations on statistics.

    Args:
        image_ft (np.ndarray): Fourier transform of the reconstructed image.
        ssnr_calculator (SSNRBase): SSNR calculator object.
        tile_size (int): Size of the tiles to process the image in.
        numeric_noise (float): Small regularization constant to avoid division by zero.

    Returns:
        tuple: (filtered image FT, regularization array w, estimated SSNR).
    """
    ssnr_measured = _compute_measured_ssnr(image_ft, ssnr_calculator, numeric_noise=numeric_noise)
    harmonic_count = len(ssnr_calculator.illumination.harmonics.keys())
    if debug:
        print(f"Harmonic count: {harmonic_count}")
    if method == 'simple':
        ssnr_tiled, _ = _compute_tiled_ssnr_simple(
            ssnr_measured,
            harmonic_count=harmonic_count,
            SSNRc=SSNRc,
            debug=debug,
            debug_iterations=debug_iterations,
            edge_width=edge_width,
        )
    elif method == 'fast':
        ssnr_tiled, _ = _compute_tiled_ssnr_fast(
            ssnr_measured,
            harmonic_count=harmonic_count,
            SSNRc=SSNRc,
            debug=debug,
            debug_iterations=debug_iterations,
            edge_width=edge_width,
        )
    else:
        raise ValueError(f"Unknown tiling method: {method}")
    return _finalize_tiled_filter(image_ft, ssnr_calculator, ssnr_tiled, numeric_noise=numeric_noise)
