from random import seed
import numpy as np
import matplotlib.pyplot as plt
import hpc_utils
import Reconstructor
import OpticalSystems
import Illumination
import utils
import WienerFiltering
import SSNRCalculator
import time
from itertools import combinations


def compute_loss_function_multikernel(stack,
                          optical_system: OpticalSystems.OpticalSystem2D,
                          illumination: Illumination.IlluminationPlaneWaves2D,
                          zernieke: dict, 
                          reconstructors: list[Reconstructor.ReconstructorSpatialDomain2D],
                          calculators: list[SSNRCalculator.SSNRSIM2D]):
    
    start = time.time()
    # print(optical_system.NA)
    optical_system.compute_psf_and_otf( vectorial=False, zernieke=zernieke)
    end = time.time()
    # print(f"Computed PSF in {end - start} seconds.")
    start = time.time()
    illumination.estimate_modulation_coefficients(stack, optical_system.psf, grid=optical_system.x_grid, update=True, method='peak_height_ratio')
    end = time.time()
    # print(f"Estimated illumination in {end - start} seconds.")
    # print(illumination.get_all_amplitudes()[0])

    loss_function = 0
    for i in range (len(reconstructors)):
        for j in range (i+1, len(reconstructors)):
            reconstructor1 = reconstructors[i]
            reconstructor2 = reconstructors[j]
            calc1 = calculators[i]
            calc2 = calculators[j]

            reconstructed1 = reconstructor1.reconstruct(stack)
            reconstructed2 = reconstructor2.reconstruct(stack)

            calc1.optical_system = optical_system
            calc2.optical_system = optical_system

            _, _, ssnr1 = WienerFiltering.filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed1), calc1)
            _, _, ssnr2 = WienerFiltering.filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed2), calc2)
            
            N = optical_system.psf.shape[0]
            ratio_experimental = (ssnr1[N//2, N//2:])/(ssnr2[N//2, N//2:])
            ratio_theoretical = calc1.ssnri_like_sectorial_average()/calc2.ssnri_like_sectorial_average()

            R = np.where(ssnr1[N//2, N//2:] >=9,  ratio_experimental/ratio_theoretical, 0)
            R = np.where(ssnr2[N//2, N//2:] >=9,  R, 0)
            # R = np.where(r <= 1.5, R, 0)
            R = np.nan_to_num(R, nan=0.0, posinf=100, neginf=-100)
            loss_function += np.sum(np.where(R!=0, (R - 1)**2, 0))

    return loss_function 

def estimate_sim_aberrations_gradient_descent_2d(stack,
                                                 optical_system: OpticalSystems.OpticalSystem2D,
                                                 illumination: Illumination.IlluminationPlaneWaves2D,
                                                 aberrations_list: list,
                                                 kernel1: np.ndarray,
                                                 kernel2: np.ndarray,
                                                 gradient_step: float = 0.0072, 
                                                 maximal_aberration: float = 0.144,
                                                 max_iterations: int = 50,
                                                 num_initial_seeds: int = 10, 
                                                 sectors=2, 
                                                 theta0 = 0, 
                                                 vectorial=False):
    
    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)

    if num_initial_seeds > 1:   
        initial_seeds = np.linspace(0, maximal_aberration, num_initial_seeds)
    else:
        initial_seeds = np.array([maximal_aberration])
    
    loss_functions_dict = {}
    estimated_keys_dict = {}
    for seed in initial_seeds:
        print(f"Starting estimation with seed {seed}")
        zernieke = {aberration: seed for aberration in aberrations_list}
        loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
        for iteration in range(max_iterations):
            print(f"  Iteration {iteration}, loss function: {loss_function_old}")
            directional_derivative_dict = {}
            for aberration in aberrations_list:
                print("   Computing directional derivative for aberration ", aberration)
                zernieke_adjacent = zernieke.copy()
                zernieke_adjacent[aberration] += gradient_step
                loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
                directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
                print("   Directional derivative: ", directional_derivative_dict[aberration])
            total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in aberrations_list]))
            zernieke = {aberration: zernieke[aberration] - gradient_step * directional_derivative_dict[aberration] / total_change for aberration in aberrations_list}
            print("   Updated aberrations: ", zernieke)
            loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
            print(f"   New loss function: {loss_function_new}")
            if loss_function_new > loss_function_old:
                break
            loss_function_old = loss_function_new
        estimated_keys_dict[seed] = zernieke
        loss_functions_dict[seed] = loss_function_old
    return estimated_keys_dict, loss_functions_dict


def estimate_sim_aberrations_gradient_descent_2d_multikernel(stack,
                                                 optical_system: OpticalSystems.OpticalSystem2D,
                                                 illumination: Illumination.IlluminationPlaneWaves2D,
                                                 aberrations_list: list,
                                                 kernels_list: list,
                                                 gradient_step: float = 0.0072, 
                                                 maximal_aberration: float = 0.144,
                                                 max_iterations: int = 50,
                                                 num_initial_seeds: int = 10):
    
    reconstructors = []
    calculators = []
    for kernel in kernels_list:
        reconstructor = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel)
        reconstructors.append(reconstructor)
        calculator = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel)
        calculators.append(calculator)

    initial_seeds = np.linspace(0, maximal_aberration, num_initial_seeds)

    loss_functions_dict = {}
    estimated_keys_dict = {}
    for seed in initial_seeds:
        print(f"Starting estimation with seed {seed}")
        zernieke = {aberration: seed for aberration in aberrations_list}
        loss_function_old = compute_loss_function_multikernel(stack, optical_system, illumination, zernieke, reconstructors, calculators)
        for iteration in range(max_iterations):
            print(f"  Iteration {iteration}, loss function: {loss_function_old}")
            directional_derivative_dict = {}
            for aberration in aberrations_list:
                print("   Computing directional derivative for aberration ", aberration)
                zernieke_adjacent = zernieke.copy()
                zernieke_adjacent[aberration] += gradient_step
                loss_function_adjacent = compute_loss_function_multikernel(stack, optical_system, illumination, zernieke_adjacent, reconstructors, calculators)
                directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
                print("   Directional derivative: ", directional_derivative_dict[aberration])
            total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in aberrations_list]))
            zernieke = {aberration: zernieke[aberration] - gradient_step * directional_derivative_dict[aberration] / total_change for aberration in aberrations_list}
            print("   Updated aberrations: ", zernieke)
            loss_function_new = compute_loss_function_multikernel(stack, optical_system, illumination, zernieke, reconstructors, calculators)
            print(f"   New loss function: {loss_function_new}")
            if loss_function_new > loss_function_old:
                break
            loss_function_old = loss_function_new
        estimated_keys_dict[seed] = zernieke
        loss_functions_dict[seed] = loss_function_old
    return estimated_keys_dict, loss_functions_dict

# def estimate_sim_NA_2d(stack,
#                        optical_system: OpticalSystems.OpticalSystem2D,

def estimate_sim_aberrations_and_NA_2d(stack,
                                        optical_system: OpticalSystems.OpticalSystem2D,
                                        illumination: Illumination.IlluminationPlaneWaves2D,
                                        aberrations_list: list,
                                        kernel1: np.ndarray,
                                        kernel2: np.ndarray,
                                        NA_step: float = 0.05,
                                        aberration_step: float = 0.0072, 
                                        maximal_aberration: float = 0.144,
                                        NA_range: tuple[float, float] = (0.7, 1.3),
                                        max_iterations: int = 50,
                                        num_initial_seeds: int = 5,
                                        sectors=2, 
                                        theta0 = 0, 
                                        vectorial=False):

        reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
        reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

        calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
        calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)

        initial_seeds = np.linspace(0, maximal_aberration, num_initial_seeds)

        loss_functions_dict = {}
        estimated_keys_dict = {}
        estimated_NA_dict = {}
        for seed in initial_seeds:
            print(f"\n\nSTARTING ESTIMATION WITH A SEED {seed}")
            zernieke = {aberration: seed for aberration in aberrations_list}
            NA = NA_range[0] + seed / maximal_aberration * (NA_range[1] - NA_range[0])
            optical_system.NA = NA
            loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
            for iteration in range(max_iterations):
                print(f"  Iteration {iteration}, loss function: {loss_function_old}")
                directional_derivative_dict = {}
                for aberration in aberrations_list:
                    print("   Computing directional derivative for aberration ", aberration)
                    zernieke_adjacent = zernieke.copy()
                    zernieke_adjacent[aberration] += aberration_step
                    loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
                    directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
                    print("   Directional derivative: ", directional_derivative_dict[aberration])

                optical_system.NA += NA_step
                loss_function_adjacent_NA = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
                print("Computing NA derivative", NA)
                directional_derivative_NA = loss_function_adjacent_NA - loss_function_old
                print("   Directional derivative NA: ", directional_derivative_NA)
                total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in aberrations_list] + directional_derivative_NA**2))
                zernieke = {aberration: zernieke[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in aberrations_list}
                NA -= NA_step * directional_derivative_NA / total_change 
                optical_system.NA = NA
                print("   Updated aberrations: ", zernieke, "Updated NA: ", NA)
                loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
                print(f"   New loss function: {loss_function_new}")
                if loss_function_new > loss_function_old:
                    break
                loss_function_old = loss_function_new
            estimated_keys_dict[seed] = zernieke
            estimated_NA_dict[seed] = NA
            loss_functions_dict[seed] = loss_function_old
        return estimated_keys_dict, estimated_NA_dict, loss_functions_dict


# def compute_loss_function(stack,
#                           optical_system: OpticalSystems.OpticalSystem2D,
#                           illumination: Illumination.IlluminationPlaneWaves2D,
#                           zernieke: dict, 
#                           reconstructor1: Reconstructor.ReconstructorSpatialDomain2D,
#                           reconstructor2: Reconstructor.ReconstructorSpatialDomain2D,
#                           calc1: SSNRCalculator.SSNRSIM2D,
#                           calc2: SSNRCalculator.SSNRSIM2D, 
#                           sectors: 2,
#                           theta0: 0, 
#                           vectorial=False): 
    
#     start = time.time()
#     # print(optical_system.NA)
#     optical_system.compute_psf_and_otf( zernieke=zernieke)
#     # plt.imshow(np.log1p(10**4 * optical_system.otf.real), origin='lower')
#     # plt.title("OTF with aberrations")
#     # plt.show()
#     end = time.time()
#     # print(f"Computed PSF in {end - start} seconds.")
#     start = time.time()
#     illumination.estimate_modulation_coefficients(stack, optical_system.psf, grid=optical_system.x_grid, update=True, method='peak_height_ratio')
#     end = time.time()

#     # print(f"Estimated illumination in {end - start} seconds.")
#     # print(illumination.get_all_amplitudes()[0])
#     reconstructed1 = reconstructor1.reconstruct(stack)
#     reconstructed2 = reconstructor2.reconstruct(stack)

#     calc1.optical_system = optical_system
#     calc2.optical_system = optical_system

#     # print("Analyzing the 360 spectrum")
#     ssnr1 = calc1.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed1), 2, 0)
#     ssnr2 = calc2.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed2), 2, 0)
#     # fig, ax = plt.subplots(1, 2)
#     # im = ax[0].imshow(np.log1p(np.abs((calc1.ssnri))), origin='lower')
#     # fig.colorbar(im, ax=ax[0])
#     # ax[0].set_title("Calc 1 SSNR FT")
#     # im = ax[1].imshow(np.log1p(np.abs((calc2.ssnri))), origin='lower')
#     # fig.colorbar(im, ax=ax[1])
#     # ax[1].set_title("Calc 2 SSNR FT")
#     # plt.show()
    

#     ratio_experimental = ssnr1/ssnr2
#     ratio_theoretical = calc1.ssnri_like_sectorial_average(None, 2, 0)/calc2.ssnri_like_sectorial_average(None, 2, 0)
#     # plt.plot(ratio_experimental, label='Experimental ratio')
#     # plt.plot(ratio_theoretical, label='Theoretical ratio')
#     # plt.gca().set_ylim(0, 2)
#     # plt.legend()
#     # plt.show()
#     R = np.where(ssnr1 >=9,  ratio_experimental/ratio_theoretical, 1.2)
#     R = np.where(ssnr2 >=9,  R, 1.2)
#     # R = np.where(r <= 1.5, R, 0)
#     R = np.nan_to_num(R, nan=0.0, posinf=100, neginf=-100)
#     # print("  Loss function ra: ", loss_function)
#     # exp = np.zeros((sectors, len(ssnr1)))
#     # theor = np.zeros((sectors, len(ssnr1)))
#     loss_function = 0
#     loss_function += np.sum(np.where(R!=0, (R - 1)**2, 0))

#     if sectors >= 2:
#         for sector in range(sectors):
#             # print("Analyzing sector ", sector)
#             degree_of_symmetry = sectors * 2
#             theta_start = theta0 + sector * np.pi / sectors
#             # print(theta0, theta_start)
#             ssnr1 = calc1.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed1), degree_of_symmetry, theta_start)
#             # exp[sector, :] = ssnr1
#             ssnr2 = calc2.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed2), degree_of_symmetry, theta_start)
#             ssnr1t = calc1.ssnri_like_sectorial_average(None, degree_of_symmetry, theta_start)
#             ssnr2t = calc2.ssnri_like_sectorial_average(None, degree_of_symmetry, theta_start)
#             # theor[sector, :] = ssnr1t

#             ratio_experimental = ssnr1/ssnr2
#             ratio_theoretical = ssnr1t/ssnr2t
            
#             R = np.where(ssnr1 >=9,  ratio_experimental/ratio_theoretical, 1)
#             R = np.where(ssnr2 >=9,  R, 1)
#             R = np.nan_to_num(R, nan=0.0, posinf=100, neginf=-100)
#             # plt.plot(np.where(R!=1.2, R, 0), label=f'Sector {sector}')
#             # plt.plot(ratio_experimental, label=f'Experimental ratio sector {sector}')
#             # plt.plot(ratio_theoretical, label=f'Theoretical ratio sector {sector}')
#             # plt.gca().set_ylim(0, 2)
#             # plt.legend()
#             # plt.show()
#             loss_function += np.sum(np.where(R!=0, (R - 1)**2, 0))
#             # print("  Loss function sector ", sector, ": ", np.sum(np.where(R!=0, (R - 1)**2, 0)))

#     # fig, ax = plt.subplots(1, 2)
#     # for sector in range(len(exp)):
#     #     ax[0].plot(np.log1p(exp[sector, :]), label=f'Experimental sector {sector}')
#     #     ax[1].plot(theor[sector, :], label=f'Theoretical sector {sector}')
#     # ax[0].plot(np.log1p(calc1.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed1))), label='Full')
#     # ax[1].plot(calc1.ssnri_like_sectorial_average(), label='Full')
#     # ax[0].set_title("Experimental SSNR sectors")
#     # ax[1].set_title("Theoretical SSNR sectors")
#     # ax[0].legend()
#     # ax[1].legend()
#     # plt.show()
#     return loss_function 

def compute_loss_function(stack,
                          optical_system: OpticalSystems.OpticalSystem2D,
                          illumination: Illumination.IlluminationPlaneWaves2D,
                          zernieke: dict, 
                          reconstructor1: Reconstructor.ReconstructorSpatialDomain2D,
                          reconstructor2: Reconstructor.ReconstructorSpatialDomain2D,
                          calc1: SSNRCalculator.SSNRSIM2D,
                          calc2: SSNRCalculator.SSNRSIM2D, 
                          sectors: 2,
                          theta0: 0, 
                          vectorial=False, 
                          apodization_mask: np.ndarray = None, 
                          weighted: bool = False):
    
    optical_system.compute_psf_and_otf( zernieke=zernieke)
    # plt.imshow(np.log1p(10**4 * optical_system.otf.real), origin='lower')
    # plt.title("OTF with aberrations")
    # plt.show()
    end = time.time()
    # print(f"Computed PSF in {end - start} seconds.")
    start = time.time()
    illumination.estimate_modulation_coefficients(stack, optical_system.psf, grid=optical_system.x_grid, update=True, method='peak_height_ratio')
    end = time.time()

    # print(f"Estimated illumination in {end - start} seconds.")
    # print(illumination.get_all_amplitudes()[0])
    reconstructed1 = reconstructor1.reconstruct(stack)
    reconstructed2 = reconstructor2.reconstruct(stack)

    shapex, shapey = reconstructed1.shape[-2], reconstructed1.shape[-1]
    f0 = np.sum(reconstructed2).real

    calc1.optical_system = optical_system
    calc2.optical_system = optical_system

    reconstructed1_ft = hpc_utils.wrapped_fftn(reconstructed1)
    reconstructed2_ft = hpc_utils.wrapped_fftn(reconstructed2)
    
    # plt.imshow(np.abs(reconstructed1_ft/reconstructed2_ft), origin='lower', vmin=0, vmax=2)
    # plt.colorbar()
    # plt.show()

    if apodization_mask is not None:
        reconstructed1_ft = reconstructed1_ft * apodization_mask
        reconstructed2_ft = reconstructed2_ft * apodization_mask

    K1norm2 = np.sum(np.array([calc1.effective_kernels_ft[key] for key in calc1.effective_kernels_ft])**2, axis=0).real
    K2norm2 = np.sum(np.array([calc2.effective_kernels_ft[key] for key in calc2.effective_kernels_ft])**2, axis=0).real
    K1K2scalarprod = np.sum(np.array([calc1.effective_kernels_ft[key] * np.conj(calc2.effective_kernels_ft[key]) for key in calc1.effective_kernels_ft]), axis=0)   

    high_ssnr_mask = np.where(np.abs(reconstructed2_ft)**2 > 10 * f0 * K2norm2, 1, 0.0)
    high_ssnr_mask = high_ssnr_mask * np.where(np.abs(reconstructed1_ft)**2 > 10 * f0 * K1norm2, 1, 0.0)
    # plt.imshow(high_ssnr_mask, origin='lower')
    # plt.show()
    weighting_term = (calc2.dj.real / calc1.dj.real * np.abs(reconstructed1_ft) / np.abs(reconstructed2_ft))**2 if weighted else 1.0
    sigma2 = 2 * f0 * (calc1.dj.real **2 * K2norm2 + calc2.dj.real **2 * K1norm2 - 2 * calc1.dj.real * calc2.dj.real * K1K2scalarprod.real)
    log_likelihood = (calc1.dj.real * np.abs(reconstructed2_ft) - calc2.dj.real * np.abs(reconstructed1_ft))**2 * weighting_term / (2 * sigma2) + sigma2**0.5
    log_likelihood = np.where(np.isnan(log_likelihood), 0.0, log_likelihood)
    log_likelihood[high_ssnr_mask == 0] = 0.0
    log_likelihood[:, shapey//2] = 0.0 #To avoid annoying cross artifacts skewing the results
    log_likelihood[shapex//2, :] = 0.0
    # plt.imshow(np.log1p(log_likelihood.real), origin='lower')
    # plt.show()

    loss_function = np.sum(log_likelihood)

    return loss_function

    
    
def estimate_initial_aberrations(stack,
                       optical_system: OpticalSystems.OpticalSystem2D,
                       illumination: Illumination.IlluminationPlaneWaves2D,
                       aberration_list: list,
                       kernel1: np.ndarray,
                       kernel2: np.ndarray,
                       estimate_aberrations = True, 
                       estimate_NA = True,
                       aberration_range: tuple[float, float] = (0.0, 0.1), 
                       aberration_step: float = 0.005, 
                       NA_range: tuple[float, float] = (0.9, 1.3),
                       NA_step: float =  0.01, 
                       sectors=2, 
                       theta0 = 0, 
                       vectorial=False, 
                       apodization_mask: np.ndarray = None): 
    
    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)

    loss_functions_dict = {}
    estimated_zernieke = {aberration_type: 0.0 for aberration_type in aberration_list}

    if estimate_NA:
        NAs = np.arange(NA_range[0], NA_range[1] + NA_step/2, aberration_step)
        for NA in NAs: 
            optical_system.NA = NA
            loss_function = compute_loss_function(stack, optical_system, illumination, None, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
            loss_functions_dict[NA] = loss_function
            print(f" NA {NA}, loss function {loss_function}")
        initial_guess = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
        print(f"Initial NA is {initial_guess[0]}, NA loss function is {loss_functions_dict[initial_guess[0]]}")
        optical_system.NA = initial_guess[0]
    
    loss_functions_dict = {}
    if estimate_aberrations: 
        for aberration_value in np.arange(aberration_range[0], aberration_range[1] + aberration_step/2, aberration_step):
            zernieke = {aberration_type: aberration_value for aberration_type in aberration_list}
            loss_function = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
            loss_functions_dict[aberration_value] = loss_function
            print(f" Aberration value {aberration_value}, loss function {loss_function}")

        initial_guess = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
        print(f"Initial aberration is {initial_guess[0]}, aberration loss function is {loss_functions_dict[initial_guess[0]]}")
        initial_zernieke = {aberration_type: initial_guess[0] for aberration_type in aberration_list}
    
    return optical_system.NA, initial_zernieke


def refine_sim_aberration_estimation(stack,
                                    optical_system: OpticalSystems.OpticalSystem2D,
                                    illumination: Illumination.IlluminationPlaneWaves2D,
                                    initial_aberrations: dict,
                                    kernel1: np.ndarray,
                                    kernel2: np.ndarray,
                                    NA_step: float = 0.005,
                                    aberration_step: float = 0.0036, 
                                    max_iterations: int = 50,
                                    sectors=2,
                                    theta0=0,
                                    vectorial=False, 
                                    dynamic_gradient_step=False, 
                                    gradient_range = 10.0, 
                                    gradient_step_decay = 0.5,
                                    fix_NA = False, 
                                    apodization_mask: np.ndarray = None 
):
    optical_system.compute_psf_and_otf( zernieke=initial_aberrations)


    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)
    zernieke_old = initial_aberrations.copy()
    NA_old = optical_system.NA
    print("Starting NA:", NA_old)
    print("Starting aberrations:", zernieke_old )
    loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke_old, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
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
            loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
            directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
            print("   Directional derivative: ", directional_derivative_dict[aberration])

        if not fix_NA:
            optical_system.NA = NA_old + NA_step
            loss_function_adjacent_NA = compute_loss_function(stack, optical_system, illumination, zernieke_old, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
            print("Computing NA derivative", NA_old)
            directional_derivative_NA = loss_function_adjacent_NA - loss_function_old
            print("   Directional derivative NA: ", directional_derivative_NA)
        else:
            directional_derivative_NA = 0.0
            
        total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in initial_aberrations.keys()] + [directional_derivative_NA**2]))
        zernieke_new = {aberration: zernieke_old[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in initial_aberrations.keys()}
        NA_new = NA_old - NA_step * directional_derivative_NA / total_change
        optical_system.NA = NA_new
        loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke_new, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial, apodization_mask=apodization_mask)
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
                print(" REDUCING GRADIENT STEPS TO ", aberration_step, NA_step)
                optical_system.NA = NA_old
                continue

        zernieke_old = zernieke_new
        NA_old = NA_new
        print("   Updated aberrations: ", zernieke_new, "Updated NA: ", NA_new)
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

