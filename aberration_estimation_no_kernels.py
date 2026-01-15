from random import seed
import numpy as np
import matplotlib.pyplot as plt
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

def compute_guess_independent_dG_part(stack,
                                      illumination: Illumination.IlluminationPlaneWaves2D,
                                      effective_kernels_ft: dict,
                                      phase_modulation_patterns: dict,
                                      apodization_mask: np.ndarray = None,
                                    ) -> np.ndarray:
    
    RedG1 = np.zeros_like(stack[0, 0], dtype=np.float64)
    ImdG1 = np.zeros_like(stack[0, 0], dtype=np.float64)

    reconstructor = Reconstructor.ReconstructorFourierDomain2D(illumination, 
                                                               effective_kernels=effective_kernels_ft, 
                                                               phase_modulation_patterns=phase_modulation_patterns,
                                                               return_ft = True)
    for key in effective_kernels_ft.keys():
        kernel_ft = effective_kernels_ft[key]
        # im = plt.imshow(np.abs(kernel_ft), origin='lower')
        # plt.colorbar(im)
        # plt.title(f"Effective kernel FT for key {key}")
        # plt.show()
        if not (np.abs(kernel_ft) > 0).all():
            raise ValueError("All values in the effective kernel FT must non-zero for this algorithm.")
        basis_vector = {k : kernel_ft if k == key else np.zeros_like(kernel_ft) for k in effective_kernels_ft.keys()}
        reconstructor.effective_kernels = basis_vector
        Ikey = reconstructor.reconstruct(stack)
        if apodization_mask is not None:
            Ikey = Ikey * apodization_mask
        ReI = Ikey.real
        ImI = Ikey.imag
        RedG1 += ReI**2 / np.abs(kernel_ft)**2
        ImdG1 += ImI**2 / np.abs(kernel_ft)**2
        # print(key, "eff kernel max", np.amax(np.abs(kernel_ft)))
        # print(key, 'max', round(np.max(ReI**2 / np.abs(kernel_ft)**2)/10**12))
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(np.log1p(ReI**2 / np.abs(kernel_ft)**2), origin='lower')
        # im = ax[1].imshow(np.log1p(ImI**2 / np.abs(kernel_ft)**2), origin='lower')
        # plt.colorbar(im, ax=ax[1])
        # plt.show()
    RedG1 /= illumination.Mr
    ImdG1 /= illumination.Mr
    # print(np.amax(RedG1)/10**9, np.amax(ImdG1)/10**9)

    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(np.log1p(RedG1), origin='lower')
    # im = axes[1].imshow(np.log1p(ImdG1), origin='lower')
    # plt.colorbar(im, ax=axes[1])    
    # plt.show() 

    return (RedG1, ImdG1)

def compute_guess_dependent_dG_part(stack,
                                    illumination: Illumination.IlluminationPlaneWaves2D,
                                    optical_system: OpticalSystems.OpticalSystem2D,
                                    apodization_mask: np.ndarray = None
                                    ):
    
    RedG2 = np.zeros_like(stack[0, 0], dtype=np.float64)
    ImdG2 = np.zeros_like(stack[0, 0], dtype=np.float64)

    # print("OTF max", np.amax(optical_system.otf.real))
    reconstructor = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system=optical_system, return_ft = True)
    ssnr_calc = SSNRCalculator.SSNRSIM2D(illumination, optical_system)
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
    noise_term = np.random.chisquare(df=1, size=stack[0,0].shape) * 0 # f0/np.sqrt(2)
    # im = plt.imshow(noise_term, origin='lower')
    # plt.colorbar(im)
    # plt.show()
    RedG2 = np.where(~np.isclose(dj, np.zeros_like(dj), atol=10**-10), ReI**2 / dj, noise_term)
    ImdG2 = np.where(~np.isclose(dj, np.zeros_like(dj), atol=10**-10), ImI**2 / dj, noise_term)
    # print('RedG2', 'max', round(np.amax(RedG2)/10**12))

    # fig, axes = plt.subplots(1,2)
    # plt.suptitle("Guess-dependent dG part")
    # axes[0].imshow(np.log1p(RedG2), origin='lower')
    # im = axes[1].imshow(np.log1p(ImdG2), origin='lower')
    # plt.colorbar(im, ax=axes[1])
    # plt.show() 
    # print(np.amax(RedG2)/10**9, np.amax(ImdG2)/10**9)
    return (RedG2, ImdG2)
                                   
    
def compute_loss_function(stack,
                          optical_system: OpticalSystems.OpticalSystem2D,
                          illumination: Illumination.IlluminationPlaneWaves2D,
                          zernieke: dict,
                          dG1: tuple[np.ndarray, np.ndarray],
                          vectorial=False, 
                          high_NA=True,
                          apodization_mask: np.ndarray = None, 
                          ):
    
    optical_system.compute_psf_and_otf(high_NA=high_NA, vectorial=vectorial, zernieke=zernieke)
    # plt.imshow(np.log1p(10**4 * optical_system.otf.real), origin='lower')
    # plt.title("OTF with aberrations")
    # plt.show()
    end = time.time()
    # print(f"Computed PSF in {end - start} seconds.")
    start = time.time()
    # illumination.estimate_modulation_coefficients(stack, optical_system.psf, grid=optical_system.x_grid, update=True, method='least_squares')
    # print(f"Estimated illumination in {end - start} seconds.")
    # print(illumination.get_all_amplitudes()[0])

    end = time.time()
    ReG1, ImG1 = dG1
    RedG2, ImdG2 =  compute_guess_dependent_dG_part(stack,
                                    illumination,
                                    optical_system, 
                                    apodization_mask=apodization_mask
                                    )
    
    # RedG2 /= np.amax(RedG2) / np.amax(ReG1)
    # ImdG2 /= np.amax(ImdG2) / np.amax(ImG1)

    ReD = ReG1 - RedG2
    ImD = ImG1 - ImdG2
    # fig, axes = plt.subplots(1,2)
    # im = axes[0].imshow(np.log1p(ReD), origin='lower')
    # plt.colorbar(im, ax=axes[0])
    # im = axes[1].imshow(np.log1p(ImD), origin='lower')
    # plt.colorbar(im, ax=axes[1])
    # plt.show()
    
    f0 = np.sum(stack) / (stack.shape[0] * stack.shape[1])
    K = illumination.Mr * (illumination.Mt//2)

    log_likelihood = -1/2 * (2 * np.log(2 * np.pi * K * f0**2) + (ReD**2 + ImD**2) / (K * f0**2))

    # plt.title("Log-likelihood map")
    # plt.imshow(np.log1p(-log_likelihood.real), origin='lower')
    # plt.show()

    # loss_function = -np.sum(log_likelihood)
    loss_function = np.mean(ReD + ImD)
    return loss_function


def estimate_true_otf(  stack,
                        optical_system: OpticalSystems.OpticalSystem2D,
                        illumination: Illumination.IlluminationPlaneWaves2D,
                        initial_aberrations: dict,
                        NA_step: float = 0.005,
                        aberration_step: float = 0.0036, 
                        max_iterations: int = 50,
                        vectorial=False, 
                        high_NA=True,
                        dynamic_gradient_step=False, 
                        gradient_range = 10.0, 
                        gradient_step_decay = 0.5,
                        fix_NA = False, 
                        apodization_mask: np.ndarray = None 
):
    optical_system.compute_psf_and_otf(high_NA=high_NA, vectorial=vectorial, zernieke=initial_aberrations)
    kernel = kernels.psf_kernel2d(1)
    # plt.imshow(kernel, origin='lower')
    # plt.show()
    reconstructor1 = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, kernel)
    effective_kernels_ft = reconstructor1.effective_kernels
    phase_modulation_patterns= reconstructor1.phase_modulation_patterns

    dG1 = compute_guess_independent_dG_part(stack, illumination, effective_kernels_ft, phase_modulation_patterns, apodization_mask=apodization_mask)

    zernieke_old = initial_aberrations.copy()
    NA_old = optical_system.NA
    print("Starting NA:", NA_old)
    print("Starting aberrations:", zernieke_old )
    loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke_old, dG1, vectorial=vectorial, high_NA=high_NA, apodization_mask=apodization_mask)
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
            loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, dG1, vectorial=vectorial, high_NA=high_NA, apodization_mask=apodization_mask)
            directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
            print("Directional derivative: ", directional_derivative_dict[aberration])

        if not fix_NA:
            optical_system.NA = NA_old + NA_step
            loss_function_adjacent_NA = compute_loss_function(stack, optical_system, illumination, zernieke_old, dG1, vectorial=vectorial, high_NA=high_NA, apodization_mask=apodization_mask)
            print("Computing NA derivative", NA_old)
            directional_derivative_NA = loss_function_adjacent_NA - loss_function_old
            print("Directional derivative NA: ", directional_derivative_NA)
        else:
            directional_derivative_NA = 0.0
            
        total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in initial_aberrations.keys()] + [directional_derivative_NA**2]))
        zernieke_new = {aberration: zernieke_old[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in initial_aberrations.keys()}
        NA_new = NA_old - NA_step * directional_derivative_NA / total_change
        optical_system.NA = NA_new
        loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke_new, dG1, vectorial=vectorial, high_NA=high_NA, apodization_mask=apodization_mask)
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


def check_local_minima(stack,
                       optical_system: OpticalSystems.OpticalSystem2D,
                       illumination: Illumination.IlluminationPlaneWaves2D,
                       estimated_zernieke: dict,
                       kernel1: np.ndarray,
                       kernel2: np.ndarray,
                       check_aberrations = True, 
                       check_NA = True,
                       aberration_range: tuple[float, float] = (-0.072, 0.072), 
                       aberration_step: float = 0.0072, 
                       NA_range: tuple[float, float] = (1.0, 1.5),
                       NA_step: float =  0.01, 
                       sectors=2, 
                       theta0 = 0, 
                       vectorial=False, 
                       apodization_mask: np.ndarray = None): 
    
    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)
    print('DEBUG INFO', optical_system.NA, estimated_zernieke)
    refined_zernieke = {aberration_type: 0.0 for aberration_type in estimated_zernieke.keys()}

    NA_estimated = optical_system.NA

    loss_function_estimated = compute_loss_function(stack, optical_system, illumination, estimated_zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
    print("Loss function local", loss_function_estimated)
    loss_functions_dict = {}
    optimal_NA = NA_estimated

    if check_NA:
        NAs = np.arange(NA_range[0], NA_range[1] + NA_step/2, NA_step)
        for NA in NAs: 
            optical_system.NA = NA
            loss_function = compute_loss_function(stack, optical_system, illumination, estimated_zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
            loss_functions_dict[NA] = loss_function
            print(f" NA {NA}, loss function {loss_function}")
        best_value = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
        if loss_functions_dict[best_value[0]] < loss_function_estimated:
            print(f"Globally best NA is estimated to be {best_value[0]}, instead of {NA_estimated} obtained previously.")
            optimal_NA = best_value[0]

    optical_system.NA = NA_estimated
    loss_functions_dict = {}
    if check_aberrations: 
        for aberration_type in estimated_zernieke.keys():
            zernieke = estimated_zernieke.copy()
            for aberration_value in np.arange(aberration_range[0], aberration_range[1] + aberration_step/2, aberration_step):
                zernieke[aberration_type] = aberration_value
                loss_function = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
                loss_functions_dict[aberration_value] = loss_function
                print(f"Aberration{aberration_type} value {aberration_value}, loss function {loss_function}")
            best_value = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
            if loss_functions_dict[best_value[0]] < loss_function_estimated:
                print(f"Globally best aberration{aberration_type} is estimated to be {best_value[0]}, instead of {estimated_zernieke[aberration_type]} obtained previously.")
                refined_zernieke[aberration_type] = best_value[0]
            else:
                refined_zernieke[aberration_type] = estimated_zernieke[aberration_type]
            loss_functions_dict = {}
    
    return optimal_NA, refined_zernieke

