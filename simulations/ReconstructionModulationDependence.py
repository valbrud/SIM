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


if __name__=="__main__": 
    from config.SIM_N100_NA15 import *

    optical_system = System4f2D(alpha=alpha, refractive_index=nobject)
    optical_system.compute_psf_and_otf(((psf_size[0], psf_size[1]), N))
    conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
    conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)
    conventional.set_spatial_shifts_diagonally()

    image = generate_random_lines((10, 10), N, 0.2, 100, 100)
    simulator = SIMulator2D(
        illumination=conventional,
        optical_system=optical_system,
        readout_noise_variance=1
    )
 
    
    simulator = SIMulator2D(conventional, optical_system)
    sim_images = simulator.generate_sim_images(image)
    sim_images = simulator.generate_noisy_images(sim_images)

    ssnr_calculator_fourier = SSNRSIM2D(
        illumination=conventional,
        optical_system=optical_system,
        readout_noise_variance=1
    )

    ssnr_calculator_spatial = SSNRSIM2D(
        illumination=conventional,
        optical_system=optical_system,
        kernel=psf_kernel2d(1, (dx, dx)),
        readout_noise_variance=1
    )

    ssnr_calculator_finite = SSNRSIM2D(
        illumination=conventional,
        optical_system=optical_system,
        kernel=psf_kernel2d(7, (dx, dx)),
        readout_noise_variance=1
    )

    modulations = np.linspace(0, 1/2., 20)
    headers = ['Modulation', 'SSNR_Fourier', 'SSNR_Spatial', 'SSNR_Finite_7']
    with open("simulations/Modulation_influence.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for modulation in modulations:
            for r in range(conventional.Mr):
                conventional.harmonics[(r, (2, 0))].amplitude = modulation
                conventional.harmonics[(r, (-2, 0))].amplitude = modulation
            
            ssnr_calculator_fourier.illumination = conventional
            ssnr_calculator_spatial.illumination = conventional
            ssnr_calculator_finite.illumination = conventional

            entropy_fourier = ssnr_calculator_fourier.compute_ssnri_entropy()
            entropy_spatial = ssnr_calculator_spatial.compute_ssnri_entropy()
            entropy_finite = ssnr_calculator_finite.compute_ssnri_entropy()
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
