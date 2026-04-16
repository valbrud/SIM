import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from config.BFPConfigurations import *
from matplotlib.widgets import Slider
import SSNRCalculator
from OpticalSystems import System4f3D, System4f2D
import kernels
import utils

configurations = BFPConfiguration()


if __name__ == "__main__":
    alpha = 2 * np.pi / 5
    theta = np.arcsin(0.9 * np.sin(alpha))
    
    dx = 1 / (8 * np.sin(alpha))
    dy = dx
    dz = 1 / (4 * (1 - np.cos(alpha)))
    N = 201
    max_r = N//2 * dx
    # max_z = N//2 * dz

    # kernel[0, 0, 0] = 1

    NA = np.sin(alpha)
    psf_size = 2 * np.array((max_r, max_r))
    dV = dx * dy * dz
    x = np.linspace(-max_r, max_r, N)
    y = np.copy(x)
    # z = np.linspace(-max_z, max_z, N)

    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)
    # fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , N)

    arg = N // 2
    # print(fz[arg])

    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    # two_NA_fz = fz / (1 - np.cos(alpha))

    optical_system = System4f2D(alpha=alpha)
    optical_system.compute_psf_and_otf((psf_size, N))

    illumination_square = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
    # illumination_hexagonal = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
    illumination_hexagonal = configurations.get_3_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
    plt.imshow(illumination_hexagonal.get_illumination_density(optical_system.x_grid))
    plt.show()
    illumination_conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1, dimensionality=2)
    illumination_widefield = configurations.get_widefield(dimensionality=2)

    configurations = {"Conventional" : illumination_conventional,
                     "Square" : illumination_square,
                     "Hexagonal" : illumination_hexagonal}

    noise_estimator = SSNRCalculator.SSNRSIM2D(illumination_widefield, optical_system)
    ssnr_widefield = noise_estimator.ssnri
    ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
    volume_widefield = noise_estimator.compute_ssnri_volume()
    entropy_widefield = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_square
    ssnr_square = np.abs(noise_estimator.ssnri)
    ssnr_square_ra = noise_estimator.ring_average_ssnri()
    volume_squareSP = noise_estimator.compute_ssnri_volume()
    entropy_square = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_hexagonal
    ssnr_hexagonal = np.abs(noise_estimator.ssnri)
    ssnr_hexagonal_ra = noise_estimator.ring_average_ssnri()
    volume_hexagonal = noise_estimator.compute_ssnri_volume()
    entropy_hexagonal = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_conventional
    ssnr_conventional = np.abs(noise_estimator.ssnri)
    ssnr_conventional_ra = noise_estimator.ring_average_ssnri()
    volume_conventional = noise_estimator.compute_ssnri_volume()
    entropy_conventional = noise_estimator.compute_ssnri_entropy()

    for kernel_type, file_name in zip([kernels.psf_kernel2d, kernels.sinc_kernel2d], ['RadiallySymmetricKernels.csv', 'SincKernels.csv']):
        with open(current_dir + '/' + file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ["Configuration", 'Factor', "Volume", "Entropy"]
            writer.writerow(headers)
            writer.writerow(["Widefield", 0, volume_widefield, entropy_widefield])
            writer.writerow(["Conventional", 0, volume_conventional, entropy_conventional])
            writer.writerow(["Square", 0, volume_squareSP, entropy_square])
            writer.writerow(["Hexagonal", 0, volume_hexagonal, entropy_hexagonal])

            print("Volume ideal widefield = ", volume_widefield)
            print("Entropy ideal widefield = ", entropy_widefield)


            print("Volume ideal square = ", volume_squareSP)
            print("Entropy ideal s_polarized = ", entropy_square)


            print("Volume ideal conventional = ", volume_conventional)
            print("Entropy ideal conventional = ", entropy_conventional)


            print("Volume ideal hexagonal = ", volume_hexagonal)
            print("Entropy ideal hexagonal = ", entropy_hexagonal)
            kernel = np.zeros((1,1))
            kernel[0,0] = 1
            noise_estimator_finite = SSNRCalculator.SSNRSIM2D(illumination_widefield, optical_system, kernel)
            correction_factor_l = 2 / (1 + np.sin(theta)/np.sin(alpha))
            
            for configuration in configurations:
                illumination = configurations[configuration]
                noise_estimator_finite.illumination = illumination
                print(configuration)

                for factor in np.arange(0.1, 3.01, 0.05):
                    cut_off_frequency = 1 / 4 / (factor * dx)
                    print(f"Factor {configuration} = ", factor)
                    kernel = utils.expand_kernel(kernel_type(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency), 101)
                    plt.plot(kernel[kernel.shape[0]//2, :], label=f"Factor {factor}")

                    
                    noise_estimator_finite.kernel = kernel

                    ssnr_finite= noise_estimator_finite.ssnri
                    ssnr_finite_ra = noise_estimator_finite.ring_average_ssnri()
                    # if configuration == "Conventional":
                    #     noise_estimator_finite.plot_effective_kernel_and_otf()
                    volume_finite = noise_estimator_finite.compute_ssnri_volume()
                    entropy_finite = noise_estimator_finite.compute_ssnri_entropy()

                    print(f"Volume finite {configuration} = ", volume_finite)
                    print(f"Entropy finite {configuration} = ", entropy_finite)

                    writer.writerow([configuration, factor, volume_finite, entropy_finite])

                # plt.legend()
                # plt.show()