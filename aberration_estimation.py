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
                          calc2: SSNRCalculator.SSNRSIM2D): 
    
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
    reconstructed1 = reconstructor1.reconstruct(stack)
    reconstructed2 = reconstructor2.reconstruct(stack)

    calc1.optical_system = optical_system
    calc2.optical_system = optical_system

    _, _, ssnr1 = WienerFiltering.filter_true_wiener(hpc_utils.wrapped_fftn(reconstructed1), calc1)
    _, _, ssnr2 = WienerFiltering.filter_true_wiener(hpc_utils.wrapped_fftn(reconstructed2), calc2)
    
    N = optical_system.psf.shape[0]
    ratio_experimental = (ssnr1[N//2, N//2:])/(ssnr2[N//2, N//2:])
    ratio_theoretical = calc1.ring_average_ssnri_approximated()/calc2.ring_average_ssnri_approximated()

    # loss_function = (ratio_experimental - ratio_theoretical)**2
    # loss_function = np.where(ssnr1[N//2, N//2:] >=9,  loss_function, 1)
    # loss_function = np.where(ssnr2[N//2, N//2:] >=9,  loss_function, 1)
    # loss_function = np.nan_to_num(loss_function, nan=0.0, posinf=0, neginf=0)
    # plt.plot(loss_function)
    # plt.show()
    R = np.where(ssnr1[N//2, N//2:] >=9,  ratio_experimental/ratio_theoretical, 0)
    R = np.where(ssnr2[N//2, N//2:] >=9,  R, 0)
    # R = np.where(r <= 1.5, R, 0)
    R = np.nan_to_num(R, nan=0.0, posinf=100, neginf=-100)
    loss_function = np.sum(np.where(R!=0, (R - 1)**2, 0))
    loss_function = np.sum(loss_function)
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
            ratio_theoretical = calc1.ring_average_ssnri_approximated()/calc2.ring_average_ssnri_approximated()

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
                                                 num_initial_seeds: int = 10):
    
    reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
    reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

    calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
    calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)

    initial_seeds = np.linspace(0, maximal_aberration, num_initial_seeds)

    loss_functions_dict = {}
    estimated_keys_dict = {}
    for seed in initial_seeds:
        print(f"Starting estimation with seed {seed}")
        zernieke = {aberration: seed for aberration in aberrations_list}
        loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2)
        for iteration in range(max_iterations):
            print(f"  Iteration {iteration}, loss function: {loss_function_old}")
            directional_derivative_dict = {}
            for aberration in aberrations_list:
                print("   Computing directional derivative for aberration ", aberration)
                zernieke_adjacent = zernieke.copy()
                zernieke_adjacent[aberration] += gradient_step
                loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, reconstructor1, reconstructor2, calc1, calc2)
                directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
                print("   Directional derivative: ", directional_derivative_dict[aberration])
            total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in aberrations_list]))
            zernieke = {aberration: zernieke[aberration] - gradient_step * directional_derivative_dict[aberration] / total_change for aberration in aberrations_list}
            print("   Updated aberrations: ", zernieke)
            loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2)
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
                                        num_initial_seeds: int = 5):
        
        reconstructor1 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel1)
        reconstructor2 = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel2)

        calc1  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel1)
        calc2  = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel2)

        initial_seeds = np.linspace(0, maximal_aberration, num_initial_seeds)

        loss_functions_dict = {}
        estimated_keys_dict = {}
        estimated_NA_dict = {}
        for seed in initial_seeds:
            print(f"Starting estimation with seed {seed}")
            zernieke = {aberration: seed for aberration in aberrations_list}
            NA = NA_range[0] + seed / maximal_aberration * (NA_range[1] - NA_range[0])
            optical_system.NA = NA
            loss_function_old = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2)
            for iteration in range(max_iterations):
                print(f"  Iteration {iteration}, loss function: {loss_function_old}")
                directional_derivative_dict = {}
                for aberration in aberrations_list:
                    print("   Computing directional derivative for aberration ", aberration)
                    zernieke_adjacent = zernieke.copy()
                    zernieke_adjacent[aberration] += aberration_step
                    loss_function_adjacent = compute_loss_function(stack, optical_system, illumination, zernieke_adjacent, reconstructor1, reconstructor2, calc1, calc2)
                    directional_derivative_dict[aberration] = loss_function_adjacent - loss_function_old
                optical_system.NA += NA_step
                loss_function_adjacent_NA = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2)
                print("Computing NA derivative", NA)
                directional_derivative_NA = loss_function_adjacent_NA - loss_function_old
                total_change = np.sqrt(np.sum([directional_derivative_dict[aberration]**2 for aberration in aberrations_list] + directional_derivative_NA**2))
                zernieke = {aberration: zernieke[aberration] - aberration_step * directional_derivative_dict[aberration] / total_change for aberration in aberrations_list}
                NA -= NA_step * directional_derivative_NA / total_change 
                optical_system.NA = NA
                print("   Updated aberrations: ", zernieke, "Updated NA: ", NA)
                loss_function_new = compute_loss_function(stack, optical_system, illumination, zernieke, reconstructor1, reconstructor2, calc1, calc2)
                print(f"   New loss function: {loss_function_new}")
                if loss_function_new > loss_function_old:
                    break
                loss_function_old = loss_function_new
            estimated_keys_dict[seed] = zernieke
            estimated_NA_dict[seed] = NA
            loss_functions_dict[seed] = loss_function_old
        return estimated_keys_dict, estimated_NA_dict, loss_functions_dict