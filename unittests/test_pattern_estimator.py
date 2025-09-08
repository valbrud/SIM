
import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from wrappers import wrapped_fftn, wrapped_ifftn
from PatternEstimator import *
from OpticalSystems import System4f2D, System4f3D
from SIMulator import SIMulator2D, SIMulator3D
from Illumination import IlluminationPlaneWaves2D
from Sources import IntensityHarmonic2D
import unittest
import numpy as np
import ShapesGenerator
import matplotlib.pyplot as plt

from config.SIM_N100_NA15 import (
    alpha, theta, dx, dz, configurations, nmedium
)

theta = np.arcsin(0.8 * np.sin(alpha))  # 0.9 is a factor to avoid total internal reflection
# wavelength_ex = 540
# wavelength_em = 600
# theta_corrected = np.arcsin(theta * wavelength_ex / wavelength_em)

def build_experimental_illumination_2d():
    print('ratio to lens semi-oepning', np.sin(theta) / np.sin(alpha))
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta, 1, 0, Mr=3, angles=(-5 / 180 * np.pi, 62 / 180 * np.pi, 115 / 180 * np.pi),
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum2d

def build_theoretical_illumination_2d():
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        0.98 * theta, 1, 0, Mr=3,
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum2d

class TestPatternEstimator2D(unittest.TestCase):

    def setUp(self):
        N = 101                                   # keep small for speed
        max_r = N // 2 * dx
        psf_size = 2 * np.array((max_r, max_r))
        self.optical_system = System4f2D(alpha=alpha-0.2, refractive_index=nmedium)
        self.optical_system.compute_psf_and_otf((psf_size, N))
        # plt.imshow(self.optical_system.otf.real, cmap='gray',)
        # plt.show()

        self.experimenatal_illumination = build_experimental_illumination_2d()
        self.theoretical_illumination = build_theoretical_illumination_2d()
        self.simulator = SIMulator2D(self.experimenatal_illumination, self.optical_system)

        # synthetic object: random dots
        self.sample = ShapesGenerator.generate_random_lines(psf_size, N, 0.3, 1000, 100)
        print('total_photon_counts = ', np.sum(self.sample))  # check that the sample is not empty
        print('averaged_photon_counts = ', np.sum(self.sample) / N**2)  # check that the sample is not empty
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()


    def test_cross_correlation_estimate(self):
        np.random.seed(1234)
        self.estimator = IlluminationPatternEstimator2D(
            self.theoretical_illumination, self.optical_system
        )
        """Estimator recovers phases and modulation depth on clean data."""
        raw_stack = self.simulator.generate_noiseless_sim_images(self.sample)
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
        self.estimator = IlluminationPatternEstimator2D(
            self.theoretical_illumination, self.optical_system
        )
        """Estimator recovers phases and modulation depth on clean data."""
        # self.sample = 10000 * self.experimenatal_illumination.get_illumination_density(self.optical_system.x_grid)
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()
        raw_stack = self.simulator.generate_noiseless_sim_images(self.sample)
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
            peaks_estimation_method='interpolation', 
            phase_estimation_method='autocorrelation',
            modulation_coefficients_method='default',
            peak_search_area_size = 5,
            zooming_factor = 3, 
            max_iterations = 10,
            ssnr_estimation_iters=100, 
            debug_info_level=2
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


def build_experimental_illumination_3d():
    print('ratio to lens semi-oepning', np.sin(theta) / np.sin(alpha))
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta, 1, 1, Mr=3, angles=(-5 / 180 * np.pi, 58 / 180 * np.pi, 115 / 180 * np.pi),
    )
    illum3d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum3d

def build_theoretical_illumination_3d():
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta-0.05, 1, 1, Mr=3,
    )
    illum3d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum3d

from windowing import make_mask_cosine_edge2d
class TestPatternEstimator3D(unittest.TestCase):
    
    def setUp(self):
        # ---------- optics --------------------------------------------------
        N = (255, 255, 5)                               
        max_r = N[0] // 2 * dx
        max_z = N[2] // 2 * dz
        psf_size = 2 * np.array((max_r, max_r, max_z))
        self.optical_system2d = System4f2D(alpha=alpha, refractive_index=nmedium)
        self.optical_system2d.compute_psf_and_otf((psf_size[:2], N[:2]))

        self.optical_system3d = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nmedium)
        self.optical_system3d.compute_psf_and_otf((psf_size, N))
        # plt.imshow(self.optical_system.otf.real, cmap='gray',)
        # plt.show()

        # ---------- illumination & simulator -------------------------------
        self.experimenatal_illumination = build_experimental_illumination_3d()
        self.theoretical_illumination = build_theoretical_illumination_3d()
        density = self.experimenatal_illumination.get_illumination_density(grid = self.optical_system3d.x_grid)
        # for r in range(3):
        # plt.imshow(density[:, :, 3], cmap='gray')
        # plt.show()
        self.auxiliary_illumination = build_theoretical_illumination_2d()
        self.auxiliary_illumination.set_spatial_shifts_diagonally(5)

        self.simulator = SIMulator3D(self.experimenatal_illumination, self.optical_system3d)

        # self.sample = ShapesGenerator.generate_random_spherical_particles(psf_size, N, 1, 10000, 1000)
        self.sample = np.ones(N) * 10000
        mask = make_mask_cosine_edge2d(self.sample.shape[:2], 20)
        self.sample *= mask[:, :, None]
        print('total_photon_counts = ', np.sum(self.sample))  # check that the sample is not empty
        print('averaged_photon_counts = ', np.sum(self.sample) / np.prod(np.array(N)))  # check that the sample is not empty
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()


    def test_estimate_by_averaging_of2d_slices(self):
        self.estimator = IlluminationPatternEstimator2D(
            self.auxiliary_illumination, self.optical_system2d
        )

        """Estimator recovers phases and modulation depth on clean data."""
        # self.sample = 10000 * self.experimenatal_illumination.get_illumination_density(self.optical_system.x_grid)
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()
        raw_stack = self.simulator.generate_noiseless_sim_images(self.sample)
        # for n in range(3):
        #     plt.imshow(raw_stack[1, n], cmap='gray')
        #     plt.show()
        np.random.seed(1234)
        raw_stack = self.simulator.add_noise(raw_stack) 
        # for r in range(3):
        # plt.imshow(raw_stack[1, 0][:,:,3], cmap='gray')
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
            peaks_estimation_method='interpolation', 
            phase_estimation_method='autocorrelation',
            modulation_coefficients_method='default',
            peak_search_area_size = 35,
            zooming_factor = 3, 
            max_iterations = 7,
            ssnr_estimation_iters=100, 
            debug_info_level=2
        )

        illumination3d = IlluminationPatternEstimator3D.illumination_3d_from_illuminaton_2d_and_total_wavelength(
            self.theoretical_illumination, illumination_estimated, 1
        )
        rotation_angles = illumination_estimated.angles
        refined_wavevectors = illumination_estimated.get_all_wavevectors()[0]
        phase_matrix = illumination_estimated.phase_matrix
        modulation_coefficients = illumination_estimated.get_all_amplitudes()
        
        illumination3d = IlluminationPatternEstimator3D.illumination_3d_from_illuminaton_2d_and_total_wavelength(
            self.theoretical_illumination, illumination_estimated, 1)
        
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
