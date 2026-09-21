"""
aberration_estimation_ssnr_based.py

This module implements aberration estimation for SIM without relying on
user-supplied convolution kernels. Instead, it decomposes the SIM reconstruction
into spatial-frequency orders and compares an ideal (kernel-independent) SSNR
to a guess-dependent SSNR to drive a gradient-descent optimization of Zernike
coefficients and NA.

Functions:
    compute_spatial_frequency_orders - Decompose reconstruction into per-harmonic frequency-order images.
    compute_SSNR_ideal - Compute the kernel-independent ideal SSNR from frequency orders.
    compute_SSNR_guess - Compute the SSNR under the current OTF guess.
    compute_loss_function - Loss function comparing ideal and guess-dependent SSNRs.
    estimate_true_otf - Full gradient-descent pipeline for aberration and NA estimation.
"""

from random import seed
import numpy as np
import matplotlib.pyplot as plt
import scipy
import hpc_utils
import Reconstructor
import OpticalSystems
import Illumination
import Reconstructor
import kernels
import WienerFiltering
import SSNRCalculator
import time
from PatternEstimation.ModulationEstimator import ModulationPeakHeightRatioEstimator, ModulationRatiosInversionEstimator
from itertools import combinations


def compute_spatial_frequency_orders(stack,
                                      illumination: Illumination.PlaneWavesSIM,
                                      effective_kernels_ft: dict,
                                      phase_modulation_patterns: dict,
                                      apodization_mask: np.ndarray = None,) -> dict[np.ndarray, tuple[int, ...]]:
    
    reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if illumination.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D
    reconstructor = reconstructor_class(illumination, 
                                        effective_kernels=effective_kernels_ft, 
                                        phase_modulation_patterns=phase_modulation_patterns,
                                        return_ft = True)
    
    spatial_frequency_orders = {}
    for key in effective_kernels_ft.keys():
        kernel_ft = effective_kernels_ft[key] + 1e-5
        # print(key, "eff kernel max", np.amax(np.abs(kernel_ft)))
        # print(key, 'max', round(np.max(ReI**2 / np.abs(kernel_ft)**2)/10**12))
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(np.log1p(ReI**2 / np.abs(kernel_ft)**2), origin='lower')
        # im = ax[1].imshow(np.log1p(ImI**2 / np.abs(kernel_ft)**2), origin='lower')
        # plt.colorbar(im, ax=ax[1])
        # plt.show()
        if not (np.abs(kernel_ft) > 0).all():
            raise ValueError("All values in the effective kernel FT must non-zero for this algorithm.")
        basis_vector = {k : kernel_ft if k == key else np.zeros_like(kernel_ft) for k in effective_kernels_ft.keys()}
        reconstructor.effective_kernels = basis_vector
        Ikey = reconstructor.reconstruct(stack)
        if apodization_mask is not None:
            Ikey = Ikey * apodization_mask
        ReI = Ikey.real / np.abs(kernel_ft)
        ImI = Ikey.imag / np.abs(kernel_ft)
        spatial_frequency_orders[key] = (ReI, ImI)

    return spatial_frequency_orders

def compute_SSNR_ideal(spatial_frequency_orders, 
                       illumination: Illumination.PlaneWavesSIM) -> np.ndarray:
    
    ReSSNR_ideal = np.zeros_like(next(iter(spatial_frequency_orders.values()))[0], dtype=np.float64)
    ImSSNR_ideal = np.zeros_like(next(iter(spatial_frequency_orders.values()))[1], dtype=np.float64)

    for key in spatial_frequency_orders.keys():
        ReI = spatial_frequency_orders[key][0]
        ImI = spatial_frequency_orders[key][1]
        ReSSNR_ideal += ReI**2 
        ImSSNR_ideal += ImI**2 
    ReSSNR_ideal /= illumination.Mr
    ImSSNR_ideal /= illumination.Mr

    # print(np.amax(ReSSNR_ideal)/10**9, np.amax(ImSSNR_ideal)/10**9)

    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(np.log1p(ReSSNR_ideal), origin='lower')
    # im = axes[1].imshow(np.log1p(ImSSNR_ideal), origin='lower')
    # plt.colorbar(im, ax=axes[1])    
    # plt.show() 

    return (ReSSNR_ideal, ImSSNR_ideal)

def compute_modulations(spatial_frequency_orders, 
                        illumination: Illumination.PlaneWavesSIM, 
                        optical_system: OpticalSystems.OpticalSystem2D, 
                        effective_kernels: dict[tuple[int, tuple[int, ...]], np.ndarray] = None) -> dict[int, np.ndarray]:
    modulation_estimator = ModulationRatiosInversionEstimator(
        illumination,
        optical_system,
        effective_otfs=effective_kernels,
    )
    return modulation_estimator.estimate_modulations(spatial_frequency_orders=spatial_frequency_orders)

def compute_SSNR_ideal_per_rotation(spatial_frequency_orders,
                                    illumination: Illumination.PlaneWavesSIM) -> dict[int, tuple[np.ndarray, np.ndarray]]:

    ssnr_per_rotation = {
        rotation: (
            np.zeros_like(next(iter(spatial_frequency_orders.values()))[0], dtype=np.float64),
            np.zeros_like(next(iter(spatial_frequency_orders.values()))[1], dtype=np.float64),
        )
        for rotation in range(illumination.Mr)
    }

    for key in spatial_frequency_orders.keys():
        rotation = key[0]
        ReI = spatial_frequency_orders[key][0]
        ImI = spatial_frequency_orders[key][1]
        ssnr_per_rotation[rotation][0][...] += ReI**2
        ssnr_per_rotation[rotation][1][...] += ImI**2

    for rotation in ssnr_per_rotation.keys():
        ssnr_per_rotation[rotation][0][...] /= illumination.Mr
        ssnr_per_rotation[rotation][1][...] /= illumination.Mr

    return ssnr_per_rotation

def compute_SSNR_guess(stack,
                        illumination: Illumination.PlaneWavesSIM,
                        optical_system: OpticalSystems.OpticalSystem2D,
                        apodization_mask: np.ndarray = None
                        ):

    ReSSNRg = np.zeros_like(stack[0, 0], dtype=np.float64)
    ImSSNRg = np.zeros_like(stack[0, 0], dtype=np.float64)

    reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if optical_system.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D
    ssnr_calc_class = SSNRCalculator.SSNRSIM2D if optical_system.dimensionality == 2 else SSNRCalculator.SSNRSIM3D
    # print("OTF max", np.amax(optical_system.otf.real))
    reconstructor = reconstructor_class(illumination, optical_system=optical_system, return_ft = True)
    ssnr_calc = ssnr_calc_class(illumination, optical_system)
    # for key in ssnr_calc.effective_kernels_ft.keys():
        # kernel_ft = ssnr_calc.effective_kernels_ft[key]
        # print(key, "eff kernel max", np.amax(np.abs(kernel_ft)))
    dj = ssnr_calc.dj.real
    # print('dj max', np.amax(dj))
    # plt.imshow(np.log1p(dj), origin='lower')
    # plt.show()

    Ibest = reconstructor.reconstruct(stack)
    if apodization_mask is not None:
        Ibest = Ibest * apodization_mask
    ReI = Ibest.real
    ImI = Ibest.imag

    f0 = np.sum(stack) / (stack.shape[0] * stack.shape[1])
    noise_term = np.random.chisquare(df=1, size=stack[0,0].shape) *  f0/np.sqrt(2)
    # im = plt.imshow(noise_term, origin='lower')
    # plt.colorbar(im)
    # plt.show()
    ReSSNRg = np.where(~np.isclose(dj, np.zeros_like(dj), atol=10**-10), ReI**2 / dj, noise_term)
    ImSSNRg = np.where(~np.isclose(dj, np.zeros_like(dj), atol=10**-10), ImI**2 / dj, noise_term)
    # print('ReSSNRg', 'max', round(np.amax(ReSSNRg)/10**12))

    # fig, axes = plt.subplots(1,2)
    # plt.suptitle("Guess-dependent dG part")
    # axes[0].imshow(np.log1p(ReSSNRg), origin='lower')
    # im = axes[1].imshow(np.log1p(ImSSNRg), origin='lower')
    # plt.colorbar(im, ax=axes[1])
    # plt.show() 
    # print(np.amax(ReSSNRg)/10**9, np.amax(ImSSNRg)/10**9)
    return (ReSSNRg, ImSSNRg)
                                   
    
def compute_loss_function(stack,
                          optical_system: OpticalSystems.OpticalSystem2D,
                          illumination: Illumination.PlaneWavesSIM,
                          zernike: dict,
                          SSNR_ideal: tuple[np.ndarray, np.ndarray],
                          apodization_mask: np.ndarray = None, 
                          estimate_amplitudes_dinamically: bool = False, 
                          spatial_frequency_orders = None,
                          interpolation=False
                          ):
    
    optical_system.compute_psf_and_otf(zernike=zernike)

    # plt.imshow(np.log1p(10**4 * optical_system.otf.real), origin='lower')
    # plt.gca().add_patch(plt.Circle((100, 100), 200/6, color='red', fill=False))
    # plt.title("OTF with aberrations")
    # plt.show()
    # end = time.time()
    # # print(f"Computed PSF in {end - start} seconds.")
    # start = time.time()
    if estimate_amplitudes_dinamically:
        if interpolation: 
            modulation_estimator = ModulationPeakHeightRatioEstimator(illumination, optical_system)
            am = modulation_estimator.estimate_modulations(stack=stack)
            illumination.update_amplitudes(am)
            # print('PEAK_HEIGHT', am)
        else: 
            illumination.equalize_amplitudes()
            _, effective_otfs = illumination.compute_effective_kernels(optical_system.psf, optical_system.psf_coordinates)
            for r in range(illumination.Mr):
                key_zero = (r, tuple([0] * illumination.dimensionality))
                
                I0 = spatial_frequency_orders[key_zero][0] + 1j * spatial_frequency_orders[key_zero][1]
                g0 = effective_otfs[key_zero]
                # a0 = float(np.real(illumination.harmonics[key_zero].amplitude))
                I0 = np.where(np.abs(g0) > 10**-3, I0, 0)

                for key in spatial_frequency_orders.keys():
                    if key[0] != r or key == key_zero:
                        continue
                    I = spatial_frequency_orders[key][0] + 1j * spatial_frequency_orders[key][1]
                    g = effective_otfs[key]

                    I0f = np.where(np.abs(g) > 10**-3, I0, 0)
                    I = np.where(np.abs(g) > 10**-3, I , 0)
                    I = np.where(np.abs(g0) > 10**-3, I, 0)

                    a = 1  

                    # Minimize L2: || a0*g0*I - a*g*I0f ||_2^2, with complex scalar a = a_re + 1j*a_im
                    def obj(x):
                        a = float(x[0])
                        diff =  g0 * I - a * g * I0f
                        return float(np.sum(np.abs(diff) ** 2))

                    x0 = [float(np.real(a))]
                    bounds = [(-10, 10)]
                    optimizer = scipy.optimize.minimize(
                        obj,
                        x0=x0,
                        method="L-BFGS-B",
                        bounds=bounds,
                    )

                    a = float(optimizer.x[0])
                    for z_index in illumination.rearranged_indices[key]:
                        index = illumination.glue_indices(key, z_index)
                        illumination.harmonics[index].amplitude = a
                    # overlap1 = hpc_utils.wrapped_fftn(g * I0f)
                    # overlap2 = hpc_utils.wrapped_fftn(g0 * I)

                    # print(np.abs(np.sum(overlap2)/np.sum(overlap1)))

                    # fig, ax = plt.subplots(1,2)
                    # ax[0].imshow(np.log1p(np.abs(overlap1)), origin='lower')
                    # ax[1].imshow(np.log1p(np.abs(overlap2)), origin='lower')
                    # plt.show()
                    # plt.imshow(np.log1p(np.abs(g0 * I - a * g* I0f)))
                    # plt.show()

            illumination.normalize_spatial_waves()
            # print('LS', illumination.get_all_amplitudes()[0])

    ReSSNRi, ImSSNRi = SSNR_ideal
    ReSSNRg, ImSSNRg =  compute_SSNR_guess(stack,
                                    illumination,
                                    optical_system, 
                                    apodization_mask=apodization_mask, 
                                    )
    
    # ReSSNRg /= np.amax(ReSSNRg) / np.amax(ReSSNRi)
    # ImSSNRg /= np.amax(ImSSNRg) / np.amax(ImSSNRi)

    ReD = ReSSNRi - ReSSNRg
    ImD = ImSSNRi - ImSSNRg

    # fig, axes = plt.subplots(1,2)
    # im = axes[0].imshow(np.log1p(ReD), origin='lower')
    # plt.colorbar(im, ax=axes[0])
    # im = axes[1].imshow(np.log1p(ImD), origin='lower')
    # plt.colorbar(im, ax=axes[1])
    # plt.show()
    
    f0 = np.sum(stack) / (stack.shape[0] * stack.shape[1])
    K = illumination.Mr * (illumination.Mt//2)

    # log_likelihood = -1/2 * (2 * np.log(2 * np.pi * K * f0**2) + (ReD**2 + ImD**2) / (K * f0**2))

    # plt.title("Log-likelihood map")
    # plt.imshow(np.log1p(-log_likelihood.real), origin='lower')
    # plt.show()

    # loss_function = -np.sum(log_likelihood)
    loss_function = np.mean(ReD + ImD)
    return loss_function.real


def estimate_true_otf(stack,
                    optical_system: OpticalSystems.OpticalSystem2D,
                    illumination: Illumination.PlaneWavesSIM,
                    initial_aberrations: dict,
                    NA_step: float = 0.005,
                    aberration_step: float = 0.0036, 
                    max_iterations: int = 50,
                    dynamic_gradient_step=False, 
                    gradient_range = 10.0, 
                    gradient_step_decay = 0.5,
                    fix_NA = False, 
                    apodization_mask: np.ndarray = None,
                    estimate_amplitudes_dinamically: bool = False, 

):
    optical_system.compute_psf_and_otf(zernike=initial_aberrations)
    # plt.imshow(kernel, origin='lower')
    # plt.show()
    kernel = kernels.sinc_kernel2d(1) if illumination.dimensionality == 2 else kernels.sinc_kernel3d(1, 1)
    reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if optical_system.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D 
    reconstructor1 = reconstructor_class(illumination, optical_system, kernel)

    effective_otfs = reconstructor1.effective_kernels
    phase_modulation_patterns= reconstructor1.phase_modulation_patterns

    spatial_frequency_orders = compute_spatial_frequency_orders(stack, illumination, effective_otfs, phase_modulation_patterns, apodization_mask=apodization_mask)
    SSNR_ideal = compute_SSNR_ideal(spatial_frequency_orders, illumination)

    if not estimate_amplitudes_dinamically:
        am = compute_modulations(spatial_frequency_orders, illumination, optical_system)
        for key, value in am.items():
            illumination.harmonics[key].amplitude = np.abs(value) / illumination.Mr / illumination.Mt

    zernike_old = initial_aberrations.copy()
    NA_old = optical_system.NA
    print("Starting NA:", NA_old)
    print("Starting aberrations:", zernike_old )
    loss_function_old = compute_loss_function(stack, optical_system, illumination, zernike_old, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
    if dynamic_gradient_step:
        aberration_step_stop_size = aberration_step / gradient_range**0.5
        print("Aberration step stop size:", aberration_step_stop_size)
        NA_step_stop_size = NA_step / gradient_range**0.5
        print("NA step stop size:", NA_step_stop_size)

        aberration_step *= gradient_range**0.5
        print("Initial aberration step:", aberration_step)

        NA_step *= gradient_range**0.5
        print("Initial NA step:", NA_step)

    for iteration in range(max_iterations):
        print(f"  Iteration {iteration}, loss function: {loss_function_old}")
        directional_derivative_dict = {}
        for aberration in initial_aberrations.keys():
            print("   Computing directional derivative for aberration ", aberration)
            zernike_adjacent = zernike_old.copy()
            zernike_adjacent[aberration] += aberration_step
            loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernike_adjacent, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
            directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
            print("Directional derivative: ", directional_derivative_dict[aberration])

        if not fix_NA:
            optical_system.NA = NA_old + NA_step
            loss_function_adjacent_NA = compute_loss_function(stack, optical_system, illumination, zernike_old, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
            print("Computing NA derivative", NA_old)
            directional_derivative_NA = loss_function_adjacent_NA - loss_function_old
            print("Directional derivative NA: ", directional_derivative_NA)
        else:
            directional_derivative_NA = 0.0
            
        total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in initial_aberrations.keys()] + [directional_derivative_NA**2]))
        zernike_new = {aberration: zernike_old[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in initial_aberrations.keys()}
        NA_new = NA_old - NA_step * directional_derivative_NA / total_change
        optical_system.NA = NA_new
        loss_function_new = compute_loss_function(stack, optical_system, illumination, zernike_new, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
        print(f" New loss function: {loss_function_new}")

        if loss_function_new > loss_function_old:
            if not dynamic_gradient_step:
                optical_system.NA = NA_old
                break
            else: 
                aberration_step *= gradient_step_decay
                NA_step *= gradient_step_decay
                if aberration_step < aberration_step_stop_size or NA_step < NA_step_stop_size:
                    break
                print("REDUCING GRADIENT STEPS TO ", aberration_step, NA_step)
                optical_system.NA = NA_old
                continue

        zernike_old = zernike_new
        NA_old = NA_new
        print("Updated aberrations: ", zernike_new, "Updated NA: ", NA_new)
        loss_function_old = loss_function_new

    return optical_system.NA, zernike_old, loss_function_old

# def estimate_sim_otf_per_frequency(stack, 
#                                    optical_system: OpticalSystems.OpticalSystem2D,
#                                    illumination: Illumination.PlaneWavesSIM,
#                                    apodization_mask = None):
#     kernel = kernels.sinc_kernel2d(1) if illumination.dimensionality == 2 else kernels.sinc_kernel3d(1, 1)
#     reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if optical_system.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D 
#     reconstructor1 = reconstructor_class(illumination, optical_system, kernel)

#     effective_otfs = reconstructor1.effective_kernels
#     phase_modulation_patterns= reconstructor1.phase_modulation_patterns
#     # ssnr_calc = SSNRSIM2D(illumination, optical_system)

#     spatial_frequency_orders = compute_spatial_frequency_orders(stack, illumination, effective_otfs, phase_modulation_patterns, apodization_mask=apodization_mask)
#     SSNR_ideal_per_rotation = compute_SSNR_ideal_per_rotation(spatial_frequency_orders, illumination)

#     order_keys = list(spatial_frequency_orders.keys())
#     zero_frequency = tuple([0] * illumination.dimensionality)
#     rotation_to_plot = 0
#     _, initial_effective_otfs = illumination.compute_effective_kernels(
#         optical_system.psf,
#         optical_system.psf_coordinates,
#     )

#     def _display_slice(array: np.ndarray) -> np.ndarray:
#         if array.ndim == 2:
#             return array
#         if array.ndim == 3:
#             return array[..., array.shape[2] // 2]
#         raise ValueError(f"Unsupported array dimensionality for plotting: {array.ndim}")

#     def _center_line(array: np.ndarray) -> np.ndarray:
#         plane = _display_slice(array)
#         return plane[plane.shape[0] // 2]

#     def _log_weight(array: np.ndarray) -> np.ndarray:
#         plane = _display_slice(array)
#         scaled = 10**4 * plane
#         return np.where(1 + scaled > 0, np.log1p(scaled), np.nan)

#     def _plot_maps(title: str, maps: dict):
#         map_items = list(maps.items())
#         ncols = min(3, max(1, len(map_items)))
#         nrows = int(np.ceil(len(map_items) / ncols))
#         fig, axes = plt.subplots(nrows, ncols, squeeze=False)
#         fig.suptitle(title)
#         for axis, (key, value) in zip(axes.flat, map_items):
#             image = axis.imshow(_log_weight(value), origin='lower')
#             axis.set_title(str(key))
#             plt.colorbar(image, ax=axis)
#         for axis in axes.flat[len(map_items):]:
#             axis.axis('off')
#         plt.tight_layout()
#         plt.show()

#     def _plot_rotation_comparison(title: str, weight_maps: dict, order_maps: dict, kernel_maps: dict):
#         pair_keys = [key for key in weight_maps.keys() if key in order_maps and key in kernel_maps]
#         if not pair_keys:
#             return
#         fig, axes = plt.subplots(3, len(pair_keys), squeeze=False, figsize=(5 * len(pair_keys), 12))
#         fig.suptitle(title)
#         for column, key in enumerate(pair_keys):
#             weight_image = axes[0, column].imshow(_log_weight(np.abs(weight_maps[key])), origin='lower')
#             axes[0, column].set_title(f"|weight| {key}")
#             plt.colorbar(weight_image, ax=axes[0, column])

#             order_image = axes[1, column].imshow(_log_weight(np.abs(order_maps[key])), origin='lower')
#             axes[1, column].set_title(f"|order| {key}")
#             plt.colorbar(order_image, ax=axes[1, column])

#             kernel_image = axes[2, column].imshow(_log_weight(np.abs(kernel_maps[key])), origin='lower')
#             axes[2, column].set_title(f"|kernel| {key}")
#             plt.colorbar(kernel_image, ax=axes[2, column])
#         plt.tight_layout()
#         plt.show()

#     def _plot_ssnr_comparison(title: str, target_map: np.ndarray, initial_map: np.ndarray, optimal_map: np.ndarray):
#         fig, axes = plt.subplots(1, 4, squeeze=False, figsize=(20, 4))
#         fig.suptitle(title)
#         comparison_maps = [
#             ("target", target_map),
#             ("initial", initial_map),
#             ("optimized", optimal_map),
#             ("abs diff", np.abs(optimal_map - target_map)),
#         ]
#         for axis, (label, value) in zip(axes.flat, comparison_maps):
#             image = axis.imshow(_log_weight(np.abs(value)), origin='lower')
#             axis.set_title(label)
#             plt.colorbar(image, ax=axis)
#         plt.tight_layout()
#         plt.show()

#     def _solve_weights(target: np.ndarray, component_index: int, component_name: str, rotation: int):
#         rotation_keys = [key for key in order_keys if key[0] == rotation]
#         zero_key = (rotation, zero_frequency)
#         if zero_key not in rotation_keys:
#             raise KeyError(f"Zero-order spatial frequency key {zero_key} is missing.")
#         zero_index = rotation_keys.index(zero_key)

#         order_stack = np.stack(
#             [spatial_frequency_orders[key][component_index] for key in rotation_keys],
#             axis=0,
#         ).astype(np.float64)
#         support_stack = np.stack(
#             [np.abs(initial_effective_otfs[key]) for key in rotation_keys],
#             axis=0,
#         ).astype(np.float64)
#         initial_weight_stack = np.stack(
#             [
#                 initial_effective_otfs[key].real if component_index == 0 else initial_effective_otfs[key].imag
#                 for key in rotation_keys
#             ],
#             axis=0,
#         ).astype(np.float64)
#         flat_shape = (order_stack.shape[0], -1)
#         order_stack_flat = order_stack.reshape(flat_shape)
#         support_flat = support_stack.reshape(flat_shape)
#         initial_weight_flat = initial_weight_stack.reshape(flat_shape)
#         target_flat = target.reshape(-1)

#         kernel_support_mask = np.max(support_flat, axis=0) > 10**-6
#         valid_target_mask = ~np.isclose(target_flat, 0.0)
#         order_norm_sq = np.sum(order_stack_flat ** 2, axis=0)
#         order_information_mask = ~np.isclose(order_norm_sq, 0.0)
#         solvable_mask = valid_target_mask & kernel_support_mask & order_information_mask
#         weights_flat = np.zeros_like(order_stack_flat, dtype=np.float64)
#         initial_weights_expanded = initial_weight_flat.copy()

#         if np.any(solvable_mask):
#             solvable_indices = np.where(solvable_mask)[0]
#             order_values = order_stack_flat[:, solvable_indices]
#             initial_weights = initial_weight_flat[:, solvable_indices].copy()
#             target_values = target_flat[solvable_indices]
#             max_target = order_norm_sq[solvable_indices]
#             target_values = np.clip(target_values, 0.0, max_target)

#             weights = initial_weights.copy()
#             regularization_strength = 1e-3
#             for _ in range(80):
#                 numerator = np.sum(weights * order_values, axis=0)
#                 denominator = np.sum(weights ** 2, axis=0) + 1e-12
#                 quotient = numerator ** 2 / denominator
#                 quotient_error = quotient - target_values
#                 gradient_quotient = (
#                     2.0 * numerator[None, :] * order_values / denominator[None, :]
#                     - 2.0 * (numerator ** 2)[None, :] * weights / (denominator ** 2)[None, :]
#                 )
#                 gradient = quotient_error[None, :] * gradient_quotient + regularization_strength * (weights - initial_weights)
#                 step = 0.2 / (order_norm_sq[solvable_indices][None, :] + regularization_strength + 1e-12)
#                 weights -= step * gradient

#             weights_flat[:, solvable_indices] = weights

#         optimal_guess_flat = np.zeros_like(target_flat)
#         initial_guess_flat = np.zeros_like(target_flat)
#         if np.any(solvable_mask):
#             numerator = np.sum(weights_flat[:, solvable_mask] * order_stack_flat[:, solvable_mask], axis=0)
#             denominator = np.sum(np.abs(weights_flat[:, solvable_mask]) ** 2, axis=0)
#             valid_denominator = ~np.isclose(denominator, 0.0)
#             optimal_guess_flat_indices = np.where(solvable_mask)[0]
#             optimal_guess_flat[optimal_guess_flat_indices[valid_denominator]] = (
#                 numerator[valid_denominator] ** 2 / denominator[valid_denominator]
#             )

#             initial_numerator = np.sum(
#                 initial_weights_expanded[:, solvable_mask] * order_stack_flat[:, solvable_mask],
#                 axis=0,
#             )
#             initial_denominator = np.sum(np.abs(initial_weights_expanded[:, solvable_mask]) ** 2, axis=0)
#             valid_initial_denominator = ~np.isclose(initial_denominator, 0.0)
#             initial_guess_flat[optimal_guess_flat_indices[valid_initial_denominator]] = (
#                 initial_numerator[valid_initial_denominator] ** 2 / initial_denominator[valid_initial_denominator]
#             )

#         weights = weights_flat.reshape(order_stack.shape)
#         optimal_guess = optimal_guess_flat.reshape(target.shape)
#         initial_guess = initial_guess_flat.reshape(target.shape)
#         solvable_mask_array = solvable_mask.reshape(target.shape)

#         ratio_mask = solvable_mask_array
#         ratios = {}
#         for key_index, key in enumerate(rotation_keys):
#             if key == zero_key:
#                 ratios[key] = 1.0
#                 continue
#             zero_weight = weights[zero_index]
#             zero_weight_sum = np.sum(zero_weight[ratio_mask]) if np.any(ratio_mask) else 0.0
#             if np.isclose(zero_weight_sum, 0.0):
#                 ratios[key] = np.nan
#             else:
#                 ratios[key] = float(np.sum(weights[key_index][ratio_mask]) / zero_weight_sum)

#         print(f"{component_name} rotation {rotation} weight ratios relative to {zero_key}:")
#         for key in rotation_keys:
#             print(f"  {key}: {ratios[key]}")
#         target_total = float(np.sum(target[ratio_mask])) if np.any(ratio_mask) else np.nan
#         initial_total = float(np.sum(initial_guess[ratio_mask])) if np.any(ratio_mask) else np.nan
#         optimized_total = float(np.sum(optimal_guess[ratio_mask])) if np.any(ratio_mask) else np.nan
#         print(
#             f"{component_name} rotation {rotation} weighted-square residual: "
#             f"mean={np.nanmean(np.abs(optimal_guess[ratio_mask] - target[ratio_mask])) if np.any(ratio_mask) else np.nan}, "
#             f"max={np.nanmax(np.abs(optimal_guess[ratio_mask] - target[ratio_mask])) if np.any(ratio_mask) else np.nan}"
#         )
#         print(
#             f"{component_name} rotation {rotation} SSNR totals: "
#             f"target={target_total}, initial={initial_total}, optimized={optimized_total}, "
#             f"optimized-initial={optimized_total - initial_total if np.isfinite(optimized_total) and np.isfinite(initial_total) else np.nan}"
#         )

#         if rotation == rotation_to_plot:
#             rotation_weight_maps = {key: weights[rotation_keys.index(key)] for key in rotation_keys}
#             rotation_order_maps = {key: spatial_frequency_orders[key][component_index] for key in rotation_keys}
#             rotation_kernel_maps = {key: initial_effective_otfs[key] for key in rotation_keys}
#             _plot_rotation_comparison(
#                 f"{component_name} absolute values rotation {rotation}",
#                 rotation_weight_maps,
#                 rotation_order_maps,
#                 rotation_kernel_maps,
#             )
#             _plot_ssnr_comparison(
#                 f"{component_name} SSNR comparison rotation {rotation}",
#                 target,
#                 initial_guess,
#                 optimal_guess,
#             )

#     for rotation in range(illumination.Mr):
#         _solve_weights(SSNR_ideal_per_rotation[rotation][0], component_index=0, component_name="Real", rotation=rotation)
#         _solve_weights(SSNR_ideal_per_rotation[rotation][1], component_index=1, component_name="Imag", rotation=rotation)
#     plt.show()

# def check_local_minima(stack,
#                        optical_system: OpticalSystems.OpticalSystem2D,
#                        illumination: Illumination.IlluminationPlaneWaves2D,
#                        estimated_zernike: dict,
#                        kernel1: np.ndarray,
#                        kernel2: np.ndarray,
#                        check_aberrations = True, 
#                        check_NA = True,
#                        aberration_range: tuple[float, float] = (-0.072, 0.072), 
#                        aberration_step: float = 0.0072, 
#                        NA_range: tuple[float, float] = (1.0, 1.5),
#                        NA_step: float =  0.01, 
#                        sectors=2, 
#                        theta0 = 0, 
#                        vectorial=False, 
#                        apodization_mask: np.ndarray = None): 
    
#     reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
#     reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

#     calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
#     calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)
#     print('DEBUG INFO', optical_system.NA, estimated_zernike)
#     refined_zernike = {aberration_type: 0.0 for aberration_type in estimated_zernike.keys()}

#     NA_estimated = optical_system.NA

#     loss_function_estimated = compute_loss_function(stack, optical_system, illumination, estimated_zernike, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
#     print("Loss function local", loss_function_estimated)
#     loss_functions_dict = {}
#     optimal_NA = NA_estimated

#     if check_NA:
#         NAs = np.arange(NA_range[0], NA_range[1] + NA_step/2, NA_step)
#         for NA in NAs: 
#             optical_system.NA = NA
#             loss_function = compute_loss_function(stack, optical_system, illumination, estimated_zernike, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
#             loss_functions_dict[NA] = loss_function
#             print(f" NA {NA}, loss function {loss_function}")
#         best_value = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
#         if loss_functions_dict[best_value[0]] < loss_function_estimated:
#             print(f"Globally best NA is estimated to be {best_value[0]}, instead of {NA_estimated} obtained previously.")
#             optimal_NA = best_value[0]

#     optical_system.NA = NA_estimated
#     loss_functions_dict = {}
#     if check_aberrations: 
#         for aberration_type in estimated_zernike.keys():
#             zernike = estimated_zernike.copy()
#             for aberration_value in np.arange(aberration_range[0], aberration_range[1] + aberration_step/2, aberration_step):
#                 zernike[aberration_type] = aberration_value
#                 loss_function = compute_loss_function(stack, optical_system, illumination, zernike, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
#                 loss_functions_dict[aberration_value] = loss_function
#                 print(f"Aberration{aberration_type} value {aberration_value}, loss function {loss_function}")
#             best_value = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
#             if loss_functions_dict[best_value[0]] < loss_function_estimated:
#                 print(f"Globally best aberration{aberration_type} is estimated to be {best_value[0]}, instead of {estimated_zernike[aberration_type]} obtained previously.")
#                 refined_zernike[aberration_type] = best_value[0]
#             else:
#                 refined_zernike[aberration_type] = estimated_zernike[aberration_type]
#             loss_functions_dict = {}
    
#     return optimal_NA, refined_zernike

