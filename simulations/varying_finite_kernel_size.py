
import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
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
from OpticalSystems import System4f3D
import kernels
sys.path.append('../')
configurations = BFPConfiguration()


if __name__ == "__main__":
    theta = np.pi / 4
    alpha = np.pi / 4
    dx = 1 / (8 * np.sin(alpha))
    dy = dx
    dz = 1 / (4 * (1 - np.cos(alpha)))
    N = 51
    max_r = N//2 * dx
    max_z = N//2 * dz

    # kernel[0, 0, 0] = 1

    NA = np.sin(alpha)
    psf_size = 2 * np.array((max_r, max_r, max_z))
    dV = dx * dy * dz
    x = np.linspace(-max_r, max_r, N)
    y = np.copy(x)
    z = np.linspace(-max_z, max_z, N)

    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)
    fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , N)

    arg = N // 2
    print(fz[arg])

    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / (1 - np.cos(alpha))

    optical_system = System4f3D(alpha=alpha)
    optical_system.compute_psf_and_otf((psf_size, N),
                                       apodization_function="Sine")

    illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
    illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
    illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
    illumination_widefield = configurations.get_widefield()

    configurations = {"Conventional" : illumination_3waves,
                     "Square" : illumination_s_polarized,
                     "Hexagonal" : illumination_seven_waves}

    noise_estimator = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system)
    ssnr_widefield = noise_estimator.ssnri
    ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
    volume_widefield = noise_estimator.compute_ssnri_volume()
    entropy_widefield = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_s_polarized
    ssnr_s_polarized = np.abs(noise_estimator.ssnri)
    ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri()
    volume_squareSP = noise_estimator.compute_ssnri_volume()
    entropy_s_polarized = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_seven_waves
    ssnr_seven_waves = np.abs(noise_estimator.ssnri)
    ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri()
    volume_hexagonal = noise_estimator.compute_ssnri_volume()
    entropy_seven_waves = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_3waves
    ssnr_3waves = np.abs(noise_estimator.ssnri)
    ssnr_3waves_ra = noise_estimator.ring_average_ssnri()
    volume_conventional = noise_estimator.compute_ssnri_volume()
    entropy_3waves = noise_estimator.compute_ssnri_entropy()

    with open('RadiallySymmetricKernel.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Configuration", 'Size', "Volume", "Entropy"]
        writer.writerow(headers)
        writer.writerow(["Widefield", 0, volume_widefield, entropy_widefield])
        writer.writerow(["Conventional", 0, volume_conventional, entropy_3waves])
        writer.writerow(["Square", 0, volume_squareSP, entropy_s_polarized])
        writer.writerow(["Hexagonal", 0, volume_hexagonal, entropy_seven_waves])

        print("Volume ideal widefield = ", volume_widefield)
        print("Entropy ideal widefield = ", entropy_widefield)


        print("Volume ideal square = ", volume_squareSP)
        print("Entropy ideal s_polarized = ", entropy_s_polarized)


        print("Volume ideal conventional = ", volume_conventional)
        print("Entropy ideal conventional = ", entropy_3waves)


        print("Volume ideal hexagonal = ", volume_hexagonal)
        print("Entropy ideal hexagonal = ", entropy_seven_waves)
        kernel = np.zeros((1,1,1))
        kernel[0,0,0] = 1
        noise_estimator_finite = SSNRCalculator.SSNRSIM3DFiniteKernel(illumination_widefield, optical_system, kernel)

        for configuration in configurations:
            illumination = configurations[configuration]
            noise_estimator_finite.illumination = illumination
            print(configuration)

            for kernel_size in range(1, 31, 2):
                print(f"Kernel size {configuration} = ", kernel_size)
                # kernel = kernels.sinc_kernel(kernel_size, kernel_z_size=1)
                kernel = kernels.psf_kernel2d(kernel_size, (dx, dy))

                noise_estimator_finite.kernel = kernel

                ssnr_finite= noise_estimator_finite.compute_ssnr()
                ssnr_finite_ra = noise_estimator_finite.ring_average_ssnri()
                # if configuration == "Conventional":
                #     plt.plot(1 + 10**4 * ssnr_finite_ra[N//4  , :], label=f"{kernel_size}")
                #     plt.gca().set_yscale("log")
                volume_finite = noise_estimator_finite.compute_ssnri_volume()
                entropy_finite = noise_estimator_finite.compute_ssnri_entropy()

                print(f"Volume finite {configuration} = ", volume_finite)
                print(f"Entropy finite {configuration} = ", entropy_finite)

                writer.writerow([configuration, kernel_size, volume_finite, entropy_finite])
