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
import pickle

configurations = BFPConfiguration()


if __name__ == "__main__":
    exit()
    alpha = 2 * np.pi / 5
    theta = np.arcsin(0.9 * np.sin(alpha))
    nmedium = 1.5
    nsample = 1.5

    dx = 1 / (8 * nmedium * np.sin(alpha))
    dy = dx
    dz = 1 / (4 * nmedium * (1 - np.cos(alpha)))
    Nl = 101
    Nz = 51
    max_r = Nl//2 * dx
    max_z = Nz//2 * dz


    NA = np.sin(alpha)
    psf_size = 2 * np.array((max_r, max_r, max_z))
    dV = dx * dy * dz
    x = np.linspace(-max_r, max_r, Nl)
    y = np.copy(x)
    z = np.linspace(-max_z, max_z, Nz)

    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , Nl)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , Nl)
    fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , Nz)

    arg = Nl // 2
    print(fz[arg])

    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / (1 - np.cos(alpha))
    
    if not os.path.exists(current_dir + '/optical_system3D_vectorial.pkl'):
        optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nsample)
        optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Nz)), high_NA=True, vectorial=True)
        with open(current_dir + '/optical_system3D_vectorial.pkl', 'wb') as f:
            pickle.dump(optical_system, f)
    else:
        with open(current_dir + '/optical_system3D_vectorial.pkl', 'rb') as f:
            optical_system = pickle.load(f)

    illumination_s_polarized = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 1, Mt=1, dimensionality=3)
    illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1, dimensionality=3)
    illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, Mr=3, Mt=1, dimensionality=3)
    illumination_widefield = configurations.get_widefield(dimensionality=3)

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

    correction_factor_l = 2 / (1 + np.sin(theta)/np.sin(alpha))
    correction_factor_z = 2 / (1 + np.cos(theta)/np.cos(alpha))

    with open(current_dir + '/MixedKernels3D.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ["Configuration", 'Lateral factor', "Axial factor", "Volume", "Entropy"]
        writer.writerow(headers)
        writer.writerow(["Widefield", 0, 0, volume_widefield, entropy_widefield])
        writer.writerow(["Conventional", 0, 0,  volume_conventional, entropy_3waves])
        writer.writerow(["Square", 0, 0, volume_squareSP, entropy_s_polarized])
        writer.writerow(["Hexagonal", 0, 0, volume_hexagonal, entropy_seven_waves])

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
        noise_estimator_finite = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system, kernel)

        for configuration in configurations:
            illumination = configurations[configuration]
            noise_estimator_finite.illumination = illumination
            print(configuration)

            for factor_l in np.arange(0.4, 5.2, 0.2):
                cut_off_frequency_l = 1 / 2 / (factor_l * dx)
                print(f"Factor {configuration} = ", factor_l)
                for factor_z in np.arange(0.4, 5.2, 0.2):
                    cut_off_frequency_z = 1 / 2 / (factor_z * dz)
                    print(f"    Factor z {configuration} = ", factor_z)
                    # kernel = utils.expand_kernel(kernels.sinc_kernel3d(pixel_size=(dx, dy, dz), first_zero_frequency_r=cut_off_frequency_l, first_zero_frequency_z=cut_off_frequency_z), (31, 31, 31))
                    kernel = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency_l)[..., None] * kernels.sinc_kernel1d(pixel_size=dz, first_zero_frequency=cut_off_frequency_z), (31, 31, 31))

                    noise_estimator_finite.kernel = kernel

                    ssnr_finite= noise_estimator_finite.ssnri
                    ssnr_finite_ra = noise_estimator_finite.ring_average_ssnri()

                    volume_finite = noise_estimator_finite.compute_ssnri_volume()
                    entropy_finite = noise_estimator_finite.compute_ssnri_entropy()

                    print(f"Volume finite {configuration} = ", volume_finite)
                    print(f"Entropy finite {configuration} = ", entropy_finite)

                    writer.writerow([configuration, factor_l, factor_z, volume_finite, entropy_finite])
