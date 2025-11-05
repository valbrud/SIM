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
                          vectorial=False): 
    
    start = time.time()
    # print(optical_system.NA)
    optical_system.compute_psf_and_otf(high_NA=True, vectorial=vectorial, zernieke=zernieke)
    end = time.time()
    # print(f"Computed PSF in {end - start} seconds.")
    start = time.time()
    illumination.estimate_modulation_coefficients(stack, optical_system.psf, grid=optical_system.x_grid, update=True, method='peak_height_ratio')
    end = time.time()
    # print(f"Estimated illumination in {end - start} seconds.")
    # print(illumination.get_all_amplitudes()[0])
    reconstructed1 = reconstructor1.reconstruct(stack)
    reconstructed2 = reconstructor2.reconstruct(stack)

    calc1.optical_system = optical_system
    calc2.optical_system = optical_system

    ssnr1 = calc1.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed1))
    ssnr2 = calc2.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed2))
    
    ratio_experimental = ssnr1/ssnr2
    ratio_theoretical = calc1.ssnri_like_sectorial_average()/calc2.ssnri_like_sectorial_average()
    R = np.where(ssnr1 >=9,  ratio_experimental/ratio_theoretical, 0)
    R = np.where(ssnr2 >=9,  R, 0)
    # R = np.where(r <= 1.5, R, 0)
    R = np.nan_to_num(R, nan=0.0, posinf=100, neginf=-100)
    loss_function = np.sum(np.where(R!=0, (R - 1)**2, 0))
    # print("  Loss function ra: ", loss_function)
    if sectors >= 2:
        for sector in range(sectors):
            degree_of_symmetry = sectors * 2
            theta_start = theta0 + sector * np.pi / degree_of_symmetry
            ssnr1 = calc1.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed1), degree_of_symmetry, theta_start)
            ssnr2 = calc2.ssnr_like_sectorial_average_from_image(hpc_utils.wrapped_fftn(reconstructed2), degree_of_symmetry, theta_start)
            # plt.plot(ssnr1, label='SSNR 1')
            # plt.plot(ssnr2, label='SSNR 2')
            # plt.legend()
            # plt.show()

            ratio_experimental = ssnr1/ssnr2
            ratio_theoretical = calc1.ssnri_like_sectorial_average(None, degree_of_symmetry, theta_start)/calc2.ssnri_like_sectorial_average(None, degree_of_symmetry, theta_start)
            
            R = np.where(ssnr1 >=9,  ratio_experimental/ratio_theoretical, 1)
            R = np.where(ssnr2 >=9,  R, 1)
            R = np.nan_to_num(R, nan=0.0, posinf=100, neginf=-100)
            # plt.plot(ratio_experimental, label='Experimental ratio')
            # plt.plot(ratio_theoretical, label='Theoretical ratio')
            # plt.gca().set_ylim(0, 2)
            # plt.legend()
            # plt.show()
            loss_function += np.sum(np.where(R!=0, (R - 1)**2, 0))
            # print("  Loss function sector ", sector, ": ", np.sum(np.where(R!=0, (R - 1)**2, 0)))

    return loss_function 

def compute_loss_function_multikernel(stack,
                          optical_system: OpticalSystems.OpticalSystem2D,
                          illumination: Illumination.IlluminationPlaneWaves2D,
                          zernieke: dict, 
                          reconstructors: list[Reconstructor.ReconstructorSpatialDomain2D],
                          calculators: list[SSNRCalculator.SSNRSIM2D]):
    
    start = time.time()
    # print(optical_system.NA)
    optical_system.compute_psf_and_otf(high_NA=True, vectorial=False, zernieke=zernieke)
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

            _, _, ssnr1 = WienerFiltering.filter_true_wiener(hpc_utils.wrapped_fftn(reconstructed1), calc1)
            _, _, ssnr2 = WienerFiltering.filter_true_wiener(hpc_utils.wrapped_fftn(reconstructed2), calc2)
            
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

def estimate_initial_aberrations(stack,
                       optical_system: OpticalSystems.OpticalSystem2D,
                       illumination: Illumination.IlluminationPlaneWaves2D,
                       aberrations_list: list,
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
                       vectorial=False): 
    
    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)

    loss_functions_dict = {}
    estimated_zernieke = {aberration_type: 0.0 for aberration_type in aberrations_list}

    if estimate_NA:
        NAs = np.arange(NA_range[0], NA_range[1] + NA_step/2, aberration_step)
        for NA in NAs: 
            optical_system.NA = NA
            loss_function = compute_loss_function(stack, optical_system, illumination, None, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
            loss_functions_dict[NA] = loss_function
            print(f" NA {NA}, loss function {loss_function}")
        initial_guess = sorted(loss_functions_dict, key = lambda k: loss_functions_dict[k])
        print(f"Initial NA is {initial_guess[0]}, NA loss function is {loss_functions_dict[initial_guess[0]]}")
        optical_system.NA = initial_guess[0]
        loss_functions_dict = {}

    if estimate_aberrations: 
        for aberration_type in aberrations_list:
            if aberration_type == (2, -2):
                pass
            loss_function_old = np.inf
            for aberration_value in np.arange(aberration_range[0], aberration_range[1] + aberration_step/2, aberration_step):
                zernieke = {aberration_type: aberration_value}
                loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
                if loss_function_new > loss_function_old:
                    break
                loss_function_old = loss_function_new
                print(f"  Aberration{aberration_type} value {aberration_value}, loss function {loss_function_new}")

            print(f"Initial aberration{aberration_type} is {aberration_value - aberration_step}, NA loss function {loss_function_old}")
            estimated_zernieke[aberration_type] = aberration_value - aberration_step
            loss_functions_dict = {}
    
    return optical_system.NA, estimated_zernieke


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
                                    vectorial=False):

    optical_system.compute_psf_and_otf(high_NA=True, vectorial=vectorial, zernieke=initial_aberrations)
    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)
    zernieke = initial_aberrations.copy()
    NA = optical_system.NA
    loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
    for iteration in range(max_iterations):
        print(f"  Iteration {iteration}, loss function: {loss_function_old}")
        directional_derivative_dict = {}
        for aberration in initial_aberrations.keys():
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
        total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in initial_aberrations.keys()] + directional_derivative_NA**2))
        zernieke = {aberration: zernieke[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in initial_aberrations.keys()}
        NA -= NA_step * directional_derivative_NA / total_change
        optical_system.NA = NA
        print("   Updated aberrations: ", zernieke, "Updated NA: ", NA)
        loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2, sectors, theta0, vectorial)
        print(f"   New loss function: {loss_function_new}")
        if loss_function_new > loss_function_old:
            break
        loss_function_old = loss_function_new

    return optical_system.NA, zernieke, loss_function_old
