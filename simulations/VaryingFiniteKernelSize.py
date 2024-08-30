import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from config.IlluminationConfigurations import *
from matplotlib.widgets import Slider
import SSNRCalculator
from OpticalSystems import Lens
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
    dx = 2 * max_r / N
    dy = 2 * max_r / N
    dz = 2 * max_z / N
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

    optical_system = Lens(alpha=alpha)
    optical_system.compute_psf_and_otf((psf_size, N),
                                       apodization_filter=None)

    illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
    illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
    illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
    illumination_widefield = configurations.get_widefield()

    illuminations = {"Conventional" : illumination_3waves,
                     "Square" : illumination_s_polarized,
                     "Hexagonal" : illumination_seven_waves}

    noise_estimator = SSNRCalculator.SSNR3dSIM2dShifts(illumination_widefield, optical_system)
    ssnr_widefield = noise_estimator.compute_ssnr()
    ssnr_widefield_ra = noise_estimator.ring_average_ssnr()
    volume_widefield = noise_estimator.compute_ssnr_volume()
    entropy_widefield = noise_estimator.compute_true_ssnr_entropy()

    noise_estimator.illumination = illumination_s_polarized
    ssnr_s_polarized = np.abs(noise_estimator.compute_ssnr())
    ssnr_s_polarized_ra = noise_estimator.ring_average_ssnr()
    volume_s_polarized = noise_estimator.compute_ssnr_volume()
    entropy_s_polarized = noise_estimator.compute_true_ssnr_entropy()

    noise_estimator.illumination = illumination_seven_waves
    ssnr_seven_waves = np.abs(noise_estimator.compute_ssnr())
    ssnr_seven_waves_ra = noise_estimator.ring_average_ssnr()
    volume_seven_waves = noise_estimator.compute_ssnr_volume()
    entropy_seven_waves = noise_estimator.compute_true_ssnr_entropy()

    noise_estimator.illumination = illumination_3waves
    ssnr_3waves = np.abs(noise_estimator.compute_ssnr())
    ssnr_3waves_ra = noise_estimator.ring_average_ssnr()
    volume_3waves = noise_estimator.compute_ssnr_volume()
    entropy_3waves = noise_estimator.compute_true_ssnr_entropy()

    with open('SincKernels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Configuration", 'Size', "Volume", "Entropy"]
        writer.writerow(headers)
        writer.writerow(["Widefield", 1, volume_widefield, entropy_widefield])
        writer.writerow(["Conventional", 1, volume_3waves, entropy_3waves])
        writer.writerow(["Square", 1, volume_s_polarized, entropy_s_polarized])
        writer.writerow(["Hexagonal", 1, volume_seven_waves, entropy_seven_waves])

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)


        print("Volume ssnr s_polarized = ", volume_s_polarized)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)


        print("Volume ssnr 3waves = ", volume_3waves)
        print("Entropy ssnr 3waves = ", entropy_3waves)


        print("Volume ssnr seven_waves = ", volume_seven_waves)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)

        for kernel_size in range(3, 31, 2):

            kernel = np.zeros((kernel_size, kernel_size, 1))
            func = np.zeros(kernel_size)
            func[0:kernel_size//2 + 1] = np.linspace(0, 1, kernel_size//2+1)
            func[kernel_size//2: kernel_size] = np.linspace(1, 0, kernel_size//2 + 1)
            func2d = func[:, None] * func[None, :]
            kernel[:, :,  0] = func

            noise_estimator_finite = SSNRCalculator.SSNR3dSIM2dShiftsFiniteRealKernel(illumination_widefield, optical_system, kernel)
            ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
            ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnr()
            volume_finite_widefield = noise_estimator_finite.compute_ssnr_volume()
            entropy_finite_widefield = noise_estimator_finite.compute_true_ssnr_entropy()

            noise_estimator_finite.illumination = illumination_s_polarized
            ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
            ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnr()
            volume_finite_s_polarized = noise_estimator_finite.compute_ssnr_volume()
            entropy_finite_s_polarized = noise_estimator_finite.compute_true_ssnr_entropy()

            noise_estimator_finite.illumination = illumination_seven_waves
            ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
            ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnr()
            volume_finite_seven_waves = noise_estimator_finite.compute_ssnr_volume()
            entropy_finite_seven_waves = noise_estimator_finite.compute_true_ssnr_entropy()

            noise_estimator_finite.illumination = illumination_3waves
            ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
            ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnr()
            volume_finite_3waves = noise_estimator_finite.compute_ssnr_volume()
            entropy_finite_3waves = noise_estimator_finite.compute_true_ssnr_entropy()

            print("Volume finite s_polarized = ", volume_finite_s_polarized)
            print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

            print("Volume finite 3waves = ", volume_finite_3waves)
            print("Entropy finite 3waves = ", entropy_finite_3waves)

            print("Volume finite seven_waves = ", volume_finite_seven_waves)
            print("Entropy finite seven_waves = ", entropy_finite_seven_waves)
