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
        print(key, "eff kernel max", np.amax(np.abs(kernel_ft)))
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
                          zernieke: dict,
                          SSNR_ideal: tuple[np.ndarray, np.ndarray],
                          apodization_mask: np.ndarray = None, 
                          estimate_amplitudes_dinamically: bool = False, 
                          spatial_frequency_orders = None,
                          interpolation=True
                          ):
    
    optical_system.compute_psf_and_otf(zernieke=zernieke)
    # plt.imshow(np.log1p(10**4 * optical_system.otf.real), origin='lower')
    # plt.title("OTF with aberrations")
    # plt.show()
    # end = time.time()
    # # print(f"Computed PSF in {end - start} seconds.")
    # start = time.time()
    if estimate_amplitudes_dinamically:
        if interpolation: 
            am = illumination.estimate_modulation_coefficients(stack, optical_system.psf, grid=optical_system.x_grid, update=True, method='peak_height_ratio')
            print('PEAK_HEIGHT', am)
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
                    illumination.harmonics[key].amplitude = a
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
            print('LS', illumination.get_all_amplitudes()[0])

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
    
    # f0 = np.sum(stack) / (stack.shape[0] * stack.shape[1])
    # K = illumination.Mr * (illumination.Mt//2)

    # log_likelihood = -1/2 * (2 * np.log(2 * np.pi * K * f0**2) + (ReD**2 + ImD**2) / (K * f0**2))

    # plt.title("Log-likelihood map")
    # plt.imshow(np.log1p(-log_likelihood.real), origin='lower')
    # plt.show()

    # loss_function = -np.sum(log_likelihood)
    loss_function = np.mean(ReD + ImD)
    return loss_function


def estimate_true_otf(  stack,
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
    optical_system.compute_psf_and_otf(zernieke=initial_aberrations)
    # plt.imshow(kernel, origin='lower')
    # plt.show()
    kernel = kernels.sinc_kernel2d(1) if illumination.dimensionality == 2 else kernels.sinc_kernel3d(1)
    reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if optical_system.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D 
    reconstructor1 = reconstructor_class(illumination, optical_system, kernel)

    effective_otfs = reconstructor1.effective_kernels
    phase_modulation_patterns= reconstructor1.phase_modulation_patterns

    spatial_frequency_orders = compute_spatial_frequency_orders(stack, illumination, effective_otfs, phase_modulation_patterns, apodization_mask=apodization_mask)
    SSNR_ideal = compute_SSNR_ideal(spatial_frequency_orders, illumination)
    
    zernieke_old = initial_aberrations.copy()
    NA_old = optical_system.NA
    print("Starting NA:", NA_old)
    print("Starting aberrations:", zernieke_old )
    loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke_old, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
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
            zernieke_adjacent = zernieke_old.copy()
            zernieke_adjacent[aberration] += aberration_step
            loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
            directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
            print("Directional derivative: ", directional_derivative_dict[aberration])

        if not fix_NA:
            optical_system.NA = NA_old + NA_step
            loss_function_adjacent_NA = compute_loss_function(stack, optical_system, illumination, zernieke_old, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
            print("Computing NA derivative", NA_old)
            directional_derivative_NA = loss_function_adjacent_NA - loss_function_old
            print("Directional derivative NA: ", directional_derivative_NA)
        else:
            directional_derivative_NA = 0.0
            
        total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in initial_aberrations.keys()] + [directional_derivative_NA**2]))
        zernieke_new = {aberration: zernieke_old[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in initial_aberrations.keys()}
        NA_new = NA_old - NA_step * directional_derivative_NA / total_change
        optical_system.NA = NA_new
        loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke_new, SSNR_ideal, apodization_mask=apodization_mask, estimate_amplitudes_dinamically=estimate_amplitudes_dinamically, spatial_frequency_orders=spatial_frequency_orders)
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

        zernieke_old = zernieke_new
        NA_old = NA_new
        print("Updated aberrations: ", zernieke_new, "Updated NA: ", NA_new)
        loss_function_old = loss_function_new

    return optical_system.NA, zernieke_old, loss_function_old


# def check_local_minima(stack,
#                        optical_system: OpticalSystems.OpticalSystem2D,
#                        illumination: Illumination.IlluminationPlaneWaves2D,
#                        estimated_zernieke: dict,
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
#     print('DEBUG INFO', optical_system.NA, estimated_zernieke)
#     refined_zernieke = {aberration_type: 0.0 for aberration_type in estimated_zernieke.keys()}

#     NA_estimated = optical_system.NA

#     loss_function_estimated = compute_loss_function(stack, optical_system, illumination, estimated_zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
#     print("Loss function local", loss_function_estimated)
#     loss_functions_dict = {}
#     optimal_NA = NA_estimated

#     if check_NA:
#         NAs = np.arange(NA_range[0], NA_range[1] + NA_step/2, NA_step)
#         for NA in NAs: 
#             optical_system.NA = NA
#             loss_function = compute_loss_function(stack, optical_system, illumination, estimated_zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
#             loss_functions_dict[NA] = loss_function
#             print(f" NA {NA}, loss function {loss_function}")
#         best_value = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
#         if loss_functions_dict[best_value[0]] < loss_function_estimated:
#             print(f"Globally best NA is estimated to be {best_value[0]}, instead of {NA_estimated} obtained previously.")
#             optimal_NA = best_value[0]

#     optical_system.NA = NA_estimated
#     loss_functions_dict = {}
#     if check_aberrations: 
#         for aberration_type in estimated_zernieke.keys():
#             zernieke = estimated_zernieke.copy()
#             for aberration_value in np.arange(aberration_range[0], aberration_range[1] + aberration_step/2, aberration_step):
#                 zernieke[aberration_type] = aberration_value
#                 loss_function = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
#                 loss_functions_dict[aberration_value] = loss_function
#                 print(f"Aberration{aberration_type} value {aberration_value}, loss function {loss_function}")
#             best_value = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
#             if loss_functions_dict[best_value[0]] < loss_function_estimated:
#                 print(f"Globally best aberration{aberration_type} is estimated to be {best_value[0]}, instead of {estimated_zernieke[aberration_type]} obtained previously.")
#                 refined_zernieke[aberration_type] = best_value[0]
#             else:
#                 refined_zernieke[aberration_type] = estimated_zernieke[aberration_type]
#             loss_functions_dict = {}
    
#     return optimal_NA, refined_zernieke

