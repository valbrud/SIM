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
    alpha_paraxial= 2 * np.pi / 5
    alpha_vectorial = 2 * np.pi / 5

    theta = np.arcsin(0.9 * np.sin(alpha_vectorial))
    dx = 1 / (8 * np.sin(alpha_vectorial))
    dy = dx
    dz = 1 / (4 * (1 - np.cos(alpha_vectorial)))
    N = 201
    max_r = N//2 * dx
    # max_z = N//2 * dz

    # kernel[0, 0, 0] = 1

    NA_paraxial = np.sin(alpha_paraxial)
    NA_vectorial = np.sin(alpha_vectorial)

    NA_ratio = NA_vectorial / NA_paraxial

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

    two_NA_fx = fx / (2 * NA_vectorial)
    two_NA_fy = fy / (2 * NA_vectorial)
    # two_NA_fz = fz / (1 - np.cos(alpha))

    optical_system_paraxial = System4f2D(alpha=alpha_paraxial)
    optical_system_paraxial.compute_psf_and_otf((psf_size, N))

    optical_system_vectorial = System4f2D(alpha=alpha_vectorial)
    optical_system_vectorial.compute_psf_and_otf((psf_size, N), high_NA=True, vectorial=True)

    illumination= configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1, dimensionality=2)
    illumination_widefield = configurations.get_widefield(dimensionality=2)

    noise_estimator_paraxial = SSNRCalculator.SSNRSIM2D(illumination_widefield, optical_system_paraxial)
    ssnr_paraxial_widefield = noise_estimator_paraxial.ssnri
    ssnr_paraxial_ra_widefield = noise_estimator_paraxial.ring_average_ssnri()
    volume_paraxial_widefield = noise_estimator_paraxial.compute_ssnri_volume()
    entropy_paraxial_widefield = noise_estimator_paraxial.compute_ssnri_entropy()

    noise_estimator_vectorial = SSNRCalculator.SSNRSIM2D(illumination_widefield, optical_system_vectorial)
    ssnr_vectorial_widefield = noise_estimator_vectorial.ssnri
    ssnr_vectorial_ra_widefield = noise_estimator_vectorial.ring_average_ssnri()
    volume_vectorial_widefield = noise_estimator_vectorial.compute_ssnri_volume()
    entropy_vectorial_widefield = noise_estimator_vectorial.compute_ssnri_entropy()


    with open(current_dir + '/DifferentModels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Model", 'Factor', "Volume", "Entropy"]
        writer.writerow(headers)
        writer.writerow(["Paraxial", 0, volume_paraxial_widefield, entropy_paraxial_widefield])
        writer.writerow(["Vectorial", 0, volume_vectorial_widefield, entropy_vectorial_widefield])
        print("Volume ideal paraxial = ", volume_paraxial_widefield)
        print("Entropy ideal paraxial = ", entropy_paraxial_widefield)
        
        noise_estimator_paraxial.illumination = illumination
        noise_estimator_vectorial.illumination = illumination

        for factor in np.arange(0.2, 10, 0.2):
            cut_off_frequency_paraxial = 1 / 2 / (factor * NA_ratio * dx)
            kernel_paraxial = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency_paraxial), 201)

            cut_off_frequency_vectorial = 1 / 2 / (factor * dx)
            kernel_vectorial = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency_vectorial), 201)
            print(f"Factor finite = ", factor)
            noise_estimator_paraxial.kernel = kernel_paraxial
            noise_estimator_vectorial.kernel = kernel_vectorial

            ssnr_finite_paraxial= noise_estimator_paraxial.ssnri
            ssnr_finite_ra_paraxial = noise_estimator_paraxial.ring_average_ssnri()
            volume_finite = noise_estimator_paraxial.compute_ssnri_volume()
            entropy_finite = noise_estimator_paraxial.compute_ssnri_entropy()

            writer.writerow(["Paraxial", factor, volume_finite, entropy_finite])

            ssnr_finite_vectorial= noise_estimator_vectorial.ssnri
            ssnr_finite_ra_vectorial = noise_estimator_vectorial.ring_average_ssnri()
            volume_finite = noise_estimator_vectorial.compute_ssnri_volume()
            entropy_finite = noise_estimator_vectorial.compute_ssnri_entropy()

            writer.writerow(["Vectorial", factor, volume_finite, entropy_finite])

        # plt.legend()
        # plt.show()