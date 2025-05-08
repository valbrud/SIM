"""
Unit‑tests for IlluminationPatternEstimator2D

The test generates a synthetic SIM 2‑D stack with your SIMulator2D and checks
that the estimator recovers:
    * zero rotational phase increment  (Mr = 1)
    * ~120° translational phase increment  (2π / 3)
    * a modulation depth a₁ close to the simulated value
"""

import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

# ---------------------------------------------------------------------
# project imports (exact names from your repo)
# ---------------------------------------------------------------------
from wrappers import wrapped_fftn, wrapped_ifftn
from PatternEstimator import *
from OpticalSystems import System4f2D
from SIMulator import SIMulator2D
from Illumination import IlluminationPlaneWaves2D
from Sources import IntensityHarmonic2D
import unittest
import numpy as np
import ShapesGenerator
import matplotlib.pyplot as plt

# one of your config files – adjust if you prefer another
from config.SIM_N100_NA15 import (
    alpha, theta, dx, configurations, psf_size, nmedium
)

theta = np.arcsin(0.8 * np.sin(alpha))  # 0.9 is a factor to avoid total internal reflection
# wavelength_ex = 540
# wavelength_em = 600
# theta_corrected = np.arcsin(theta * wavelength_ex / wavelength_em)

def build_experimental_illumination():
    print('ratio to lens semi-oepning', np.sin(theta) / np.sin(alpha))
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta, 1, 0, Mr=3, angles=(-0 / 180 * np.pi, 58 / 180 * np.pi, 115 / 180 * np.pi),
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum2d

def build_theoretical_illumination():
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        0.98 * theta, 1, 0, Mr=3,
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum2d
# ---------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------
class TestPatternEstimator2D(unittest.TestCase):

    def setUp(self):
        # ---------- optics --------------------------------------------------
        N = 101                                   # keep small for speed
        max_r = N // 2 * dx
        psf_size = 2 * np.array((max_r, max_r))
        self.optical_system = System4f2D(alpha=alpha-0.2, refractive_index=nmedium)
        self.optical_system.compute_psf_and_otf((psf_size, N))
        # plt.imshow(self.optical_system.otf.real, cmap='gray',)
        # plt.show()

        # ---------- illumination & simulator -------------------------------
        self.experimenatal_illumination = build_experimental_illumination()
        self.theoretical_illumination = build_theoretical_illumination()
        self.simulator = SIMulator2D(self.experimenatal_illumination, self.optical_system)

        # synthetic object: random dots
        self.sample = ShapesGenerator.generate_random_lines(psf_size, N, 0.3, 1000, 10)
        print('total_photon_counts = ', np.sum(self.sample))  # check that the sample is not empty
        print('averaged_photon_counts = ', np.sum(self.sample) / N**2)  # check that the sample is not empty
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()


    def test_cross_correlation_estimate(self):
        np.random.seed(1234)
        self.estimator = PatternEstimatorCrossCorrelation2D(
            self.theoretical_illumination, self.optical_system
        )
        """Estimator recovers phases and modulation depth on clean data."""
        raw_stack = self.simulator.generate_sim_images(self.sample)
        raw_stack = self.simulator.add_noise(raw_stack)  # (3, 3, N, N)
        plt.imshow(np.log1p(np.abs(wrapped_fftn(raw_stack[0, 0]))).T, cmap='gray', origin='lower')
        plt.show()
        plt.imshow(np.log1p(np.abs(wrapped_fftn(raw_stack[1, 0]))).T, cmap='gray', origin='lower')
        plt.show()
        plt.imshow(np.log1p(np.abs(wrapped_fftn(raw_stack[2, 0]))).T, cmap='gray', origin='lower')
        plt.show()

        base_vectors = np.array(self.theoretical_illumination.get_base_vectors(0))/(2 * np.pi)
        true_vectors = np.array(self.experimenatal_illumination.get_base_vectors(0)) / (2  * np.pi)
        dq = self.estimator.optical_system.otf_frequencies[0][1] - self.estimator.optical_system.otf_frequencies[0][0]
        print('dq size = ', dq )
        print("initial_guess", np.array(base_vectors) * 4 * np.pi)
        print('true_wavevectors =', np.array(true_vectors) * 4 * np.pi)

        # print("base_vectors =", base_vectors)

        illumination_estimated = self.estimator.estimate_illumination_parameters(
            raw_stack,
            zooming_factor=1.3,
            peak_neighborhood_size=7, 
            max_iterations=10,             
        )

        print(f"rotation_angles,  {np.round(illumination_estimated.angles * 180 / np.pi, 1)} degrees")
        print("refined_vectors", illumination_estimated.get_all_wavevectors()[0])
        print('true_wavevectors =', np.array(true_vectors) * 4 * np.pi)
        # print('phase_matrix = ', illumination_estimated.phase_matrix)
        # precision = (true_vectors -  illumination_estimated.get_all_wavevectors()[0]) / dq
        # print('achieved_precision = ', precision, 'pixels')
        am = illumination_estimated.estimate_modulation_coefficients(raw_stack, self.optical_system.psf, self.optical_system.x_grid)
        print("modulation_coefficients", am)

    def test_interpolation_estimate(self):
        self.estimator = PatternEstimatorInterpolation2D(
            self.theoretical_illumination, self.optical_system
        )
        """Estimator recovers phases and modulation depth on clean data."""
        # self.sample = 10000 * self.experimenatal_illumination.get_illumination_density(self.optical_system.x_grid)
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()
        raw_stack = self.simulator.generate_sim_images(self.sample)
        # for n in range(3):
        #     plt.imshow(raw_stack[1, n], cmap='gray')
        #     plt.show()
        np.random.seed(1234)
        raw_stack = self.simulator.add_noise(raw_stack)  # (3, 3, N, N)
        # raw_stack = np.stack([self.sample]*3, axis=0)  # (3, 3, N, N)
        # plt.imshow(raw_stack[0, 0], cmap='gray')
        # plt.show()

        base_vectors = np.array(self.theoretical_illumination.get_base_vectors(0))/(2 * np.pi)
        true_vectors = np.array(self.experimenatal_illumination.get_base_vectors(0))/(2 * np.pi)
        dq = self.estimator.optical_system.otf_frequencies[0][1] - self.estimator.optical_system.otf_frequencies[0][0]
        print('dq size = ', dq )
        print("initial_guess", np.array(base_vectors))
        print('true_wavevectors =', np.array(true_vectors))

        # print("base_vectors =", base_vectors)
    
        illumination_estimated = self.estimator.estimate_illumination_parameters(
            raw_stack,
            interpolation_factor= 2,
            peak_search_area_size=3,
            peak_interpolation_area_size=3, 
            iteration_number=10, 
            deconvolve_stacks=False,
            correct_peak_position=True, 
            ssnr_estimation_iters=100
        )
        rotation_angles = illumination_estimated.angles
        refined_wavevectors = illumination_estimated.get_all_wavevectors()[0]
        phase_matrix = illumination_estimated.phase_matrix
        modulation_coefficients = illumination_estimated.get_all_amplitudes()
        print(f"rotation_angles,  {np.round(np.array(rotation_angles) * 180 / np.pi, 1)} degrees")
        print("refined_vectors", refined_wavevectors / (2 * np.pi))
        print('true_wavevectors =', np.array(true_vectors))
        # print('phase_matrix = ', phase_matrix)
        print('modulation_coefficients = ', modulation_coefficients)
        # precision = (true_vectors -  illumination_estimated.get_all_wavevectors()[0]) / dq
        # print('achieved_precision = ', precision, 'pixels')

        # print('phase_matrix = ', phase_matrix)
        # am = illumination_estimated.estimate_modulation_coefficients(raw_stack, self.optical_system.psf, self.optical_system.x_grid)
        # print("modulation_coefficients", am)
if __name__ == '__main__':
    unittest.main(verbosity=2)
