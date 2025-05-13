import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from config.BFPConfigurations import *
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from ShapesGenerator import generate_random_lines
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from SSNRCalculator import SSNRSIM2D, SSNRWidefield2D
from OpticalSystems import System4f2D
from kernels import psf_kernel2d
from SIMulator import SIMulator2D
import csv
from copy import deepcopy


if __name__=="__main__": 
    configurations = BFPConfiguration(refraction_index=1.5)
    alpha = 2 * np.pi / 5
    nmedium = 1.5
    nobject = 1.5
    NA = nmedium * np.sin(alpha)
    theta = np.asin(0.9 * np.sin(alpha))
    fz_max_diff = nmedium * (1 - np.cos(alpha))
    dx = 1 / (16 * NA)
    dy = dx
    N = 255
    max_r = N // 2 * dx
    psf_size = 2 * np.array((max_r, max_r))
    dV = dx * dy 
    x = np.linspace(-max_r, max_r, N)
    y = np.copy(x)
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)

    arg = N // 2
    # print(fz[arg])

    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)

    multiplier = 10 ** 5
    ylim = 10 ** 2


    from config.SIM_N100_NA15 import *

    optical_system = System4f2D(alpha=alpha, refractive_index=nobject)
    optical_system.compute_psf_and_otf(((psf_size[0], psf_size[1]), N))

    illumination = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
    illumination = IlluminationPlaneWaves2D.init_from_3D(illumination)
    illumination.set_spatial_shifts_diagonally()
    for r in range(illumination.Mr):
        illumination.harmonics[(r, (2, 0))].amplitude = 0.25
        illumination.harmonics[(r, (-2, 0))].amplitude = 0.25

    illumination_reconstruction = deepcopy(illumination)

    image = generate_random_lines((10, 10), N, 0.2, 100, 100)
    simulator = SIMulator2D(
        illumination=illumination,
        optical_system=optical_system,
        readout_noise_variance=1
    )
 
    
    # simulator = SIMulator2D(illumination, optical_system)
    # sim_images = simulator.generate_sim_images(image)
    # sim_images = simulator.generate_noisy_images(sim_images)

    ssnr_calculator_fourier = SSNRSIM2D(
        illumination=illumination,
        optical_system=optical_system,
        readout_noise_variance=1, 
        illumination_reconstruction=illumination_reconstruction
    )

    ssnr_calculator_spatial = SSNRSIM2D(
        illumination=illumination,
        optical_system=optical_system,
        kernel=psf_kernel2d(1, (dx, dx)),
        readout_noise_variance=1, 
        illumination_reconstruction=illumination_reconstruction 
    )

    ssnr_calculator_finite = SSNRSIM2D(
        illumination=illumination,
        optical_system=optical_system,
        kernel=psf_kernel2d(7, (dx, dx)),
        readout_noise_variance=1, 
        illumination_reconstruction=illumination_reconstruction
    )

    modulations = np.linspace(0, 1/2., 21)
    headers = ['Modulation', 'SSNR_Fourier', 'SSNR_Spatial', 'SSNR_Finite_7']
    with open("simulations/Tables/ModulationInfluence.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for modulation in modulations:
            for r in range(illumination.Mr):
                illumination_reconstruction.harmonics[(r, (2, 0))].amplitude = modulation
                illumination_reconstruction.harmonics[(r, (-2, 0))].amplitude = modulation

            ssnr_calculator_fourier._compute_effective_kernels_ft()
            ssnr_calculator_finite._compute_effective_kernels_ft()
            ssnr_calculator_spatial._compute_effective_kernels_ft()

            ssnr_calculator_fourier._compute_ssnri()
            ssnr_calculator_finite._compute_ssnri()
            ssnr_calculator_spatial._compute_ssnri()

            entropy_fourier = ssnr_calculator_fourier.compute_ssnri_volume()
            entropy_spatial = ssnr_calculator_spatial.compute_ssnri_volume()
            entropy_finite = ssnr_calculator_finite.compute_ssnri_volume()
            writer.writerow([np.round(modulation, 2), np.round(entropy_fourier, 1), np.round(entropy_spatial, 1), np.round(entropy_finite, 1)])

        # fourier_reconstructor = ReconstructorFourierDomain2D(
        #     illumination=conventional,
        #     optical_system=optical_system,
        # )

        # spatial_reconstructor1 = ReconstructorSpatialDomain2D(
        #     illumination=conventional,
        #     optical_system=optical_system,
        #     kernel=psf_kernel2d(1, (dx, dx))
        # )

        # spatial_reconstructor5 = ReconstructorSpatialDomain2D(
        #     illumination=conventional,
        #     optical_system=optical_system,
        #     kernel=psf_kernel2d(5, (dx, dx))
        # )

    # widefield = fourier_reconstructor.get_widefield(sim_images)
    # reconstructed_fdr = fourier_reconstructor.reconstruct(sim_images)
    # reconstructed_sdr = spatial_reconstructor1.reconstruct(sim_images)
    # reconstructed_fk = spatial_reconstructor5.reconstruct(sim_images)
