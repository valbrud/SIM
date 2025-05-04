import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys

# --- Imports from your simulation modules ---
from OpticalSystems import System4f2D
from SIMulator import SIMulator2D
from config.BFPConfigurations import BFPConfiguration
from Illumination_experimental import IlluminationPlaneWaves2D, IlluminationNonLinearSIM2D
import ShapesGenerator
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from kernels import sinc_kernel, psf_kernel2d


class TestReconstruction(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        # Set simulation parameters similar to the provided example.
        self.N = 511
        self.alpha = 2 * np.pi / 5
        # self.theta = 2 * np.pi / 12
        self.nmedium = 1.5
        self.theta = np.arcsin(0.9 * np.sin(self.alpha))
        self.dimensions = (1, 1)
        NA = self.nmedium * np.sin(self.alpha)
        self.dx = 1 / (64 * NA)
        self.max_r = self.N // 2 * self.dx

        self.psf_size = np.array((2 * self.max_r, 2 * self.max_r))
        self.x = np.linspace(-self.max_r, self.max_r, self.N)
        y = np.copy(self.x)

        self.image = ShapesGenerator.generate_random_lines(
            image_size=self.psf_size,
            point_number=self.N,
            line_width=0.25,
            num_lines=150,
            intensity=100
        )
        # plt.title("Ground truth")
        # plt.imshow(self.image)
        # plt.show()
        self.optical_system = System4f2D(alpha=self.alpha, refractive_index=self.nmedium)
        self.optical_system.compute_psf_and_otf((self.psf_size, self.N))

        # self.image = 10**5 * np.ones(self.optical_system.psf.shape)

        self.widefield = scipy.signal.convolve(self.image, self.optical_system.psf, mode='same')
        # plt.title("Widefield image")
        # plt.imshow(self.widefield)
        # plt.show()
        
        configurations = BFPConfiguration(refraction_index=1.5)
        illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(
            self.theta, 1, 0, 3, Mt=1
        )
        self.illumination = IlluminationPlaneWaves2D.init_from_3D(
            illumination_3waves3d, self.dimensions
        )

        # spatial_shifts = np.array(((0., 0.), (1, 0), (2, 0)))
        # spatial_shifts /= (3 * 2 * self.nmedium * np.sin(self.theta))
        # self.illumination.spatial_shifts = spatial_shifts
        self.illumination.set_spatial_shifts_diagonally()
        # plt.imshow(self.illumination.get_illumination_density(coordinates=(self.x, y)))
        # plt.show()

        # Create the simulator and generate simulated images.
        self.simulator = SIMulator2D(self.illumination, self.optical_system)
        self.sim_images = self.simulator.generate_sim_images(self.image)
        # for r in range(self.illumination.Mr):
        #     for n in range(self.illumination.Mt):
        #         image = self.sim_images[r, n]
        #         plt.title(f"Simulated image{r, n}")
        #         plt.imshow(image)
        #         plt.show()


    def test_widefield_reconstruction(self):
        reconstructor = ReconstructorFourierDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system
        )
        reconstructed_image = reconstructor.get_widefield(self.sim_images)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield)
        axes[0].set_title("Widefield")
        axes[1].imshow(reconstructed_image)
        axes[1].set_title("Reconstructed Widefield")
        plt.show()

    def test_fourier_reconstruction(self):
        # self.sim_images += np.random.normal(0, 20, self.sim_images.shape)

        fourier_reconstructor = ReconstructorFourierDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            # regularization_filter=self.optical_system.otf**2 + 0.01
            # apodization_filter =
        )
        # Reconstruct the image.
        reconstructed_image = fourier_reconstructor.reconstruct(self.sim_images)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield)
        axes[1].imshow(reconstructed_image)
        plt.show()

    def test_spatial_reconstruction(self):
        spatial_reconstructor = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
        )
        # Reconstruct the image.
        reconstructed_image = spatial_reconstructor.reconstruct(self.sim_images)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield)
        axes[1].imshow(reconstructed_image)
        plt.show()

    def test_spatial_reconstruction_finite_kernel(self):
        spatial_reconstructor = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(5, (self.dx, self.dx))
        )
        # Reconstruct the image.
        plt.title("Spatial-domain reconstruction with finite kernel")
        reconstructed_image = spatial_reconstructor.reconstruct(self.sim_images)
        plt.imshow(reconstructed_image)
        plt.show()

    def test_compare_kernel_size_effect(self):
        spatial_reconstructor1 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(1, (self.dx, self.dx))
        )

        spatial_reconstructor3 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(3, (self.dx, self.dx))
        )
        spatial_reconstructor5 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(5, (self.dx, self.dx))
        )
        spatial_reconstructor7 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(5, (self.dx, self.dx))
        )

        self.sim_images += np.random.normal(0, 2, self.sim_images.shape)
        reconstructed_image1 = spatial_reconstructor1.reconstruct(self.sim_images)
        reconstructed_image3 = spatial_reconstructor3.reconstruct(self.sim_images)
        reconstructed_image5 = spatial_reconstructor5.reconstruct(self.sim_images)
        reconstructed_image7 = spatial_reconstructor7.reconstruct(self.sim_images)

        fig, axes = plt.subplots(1, 4)
        fig.suptitle("Spatial-domain reconstruction with finite kernels")
        axes[0].set_title("Kernel size 1")
        axes[1].set_title("Kernel size 3")  
        axes[2].set_title("Kernel size 5")
        axes[3].set_title("Kernel size 7")
        axes[0].imshow(reconstructed_image1)
        axes[1].imshow(reconstructed_image3)
        axes[2].imshow(reconstructed_image5)
        axes[3].imshow(reconstructed_image7)
        plt.show()


class TestNonlinearReconstruction(unittest.TestCase):
    def setUp(self):
        self.N = 255
        self.alpha = 2 * np.pi / 5
        self.nmedium = 1.5
        self.theta = np.arcsin(0.9 * np.sin(self.alpha))
        self.dimensions = (1, 1)
        NA = self.nmedium * np.sin(self.alpha)
        self.dx = 1 / (64 * NA)
        self.max_r = self.N // 2 * self.dx

        self.psf_size = np.array((2 * self.max_r, 2 * self.max_r))
        self.x = np.linspace(-self.max_r, self.max_r, self.N)
        y = np.copy(self.x)

        self.image = ShapesGenerator.generate_random_lines(
            image_size=self.psf_size,
            point_number=self.N,
            line_width=0.25,
            num_lines=150,
            intensity=100
        )
        # plt.title("Ground truth")
        # plt.imshow(self.image)
        # plt.show()
        self.optical_system = System4f2D(alpha=self.alpha, refractive_index=self.nmedium)
        self.optical_system.compute_psf_and_otf((self.psf_size, self.N))

        # self.image = 10**5 * np.ones(self.optical_system.psf.shape)

        self.widefield = scipy.signal.convolve(self.image, self.optical_system.psf, mode='same')
        # plt.title("Widefield image")
        # plt.imshow(self.widefield)
        # plt.show()

        self.configurations = BFPConfiguration(refraction_index=1.5)
        self.Mr = 3

        illumination_3waves3d = self.configurations.get_2_oblique_s_waves_and_s_normal(
            self.theta, 1, 0, self.Mr, Mt=1
        )
        self.illumination_linear = IlluminationPlaneWaves2D.init_from_3D(
            illumination_3waves3d, self.dimensions
        )

        self.illumination_linear.set_spatial_shifts_diagonally()

        self.simulator_linear = SIMulator2D(self.illumination_linear, self.optical_system)
        self.sim_images = self.simulator_linear.generate_sim_images(self.image)

        # self.sim_images += np.random.normal(0, 20, self.sim_images.shape)

        self.linear_reconstructor = ReconstructorSpatialDomain2D(
            illumination=self.illumination_linear,
            optical_system=self.optical_system,
        )
        # Reconstruct the image.
        self.reconstructed_linear = self.linear_reconstructor.reconstruct(self.sim_images)


    def test_exponential_intensity_dependence(self):
        p = 4
        nonlinear_expansion_coefficients = [0, ]
        n = 1
        from scipy.special import factorial
        while (p ** n / factorial(n)) > 10 ** -14:
            nonlinear_expansion_coefficients.append(p ** n / factorial(n) * (-1) ** (n + 1))
            n += 1

        illumination_non_linear = IlluminationNonLinearSIM2D.init_from_linear_illumination(self.illumination_linear, tuple(nonlinear_expansion_coefficients))
        illumination_non_linear.set_spatial_shifts_diagonally()

        plt.imshow(illumination_non_linear.get_illumination_density(coordinates=(self.x, self.x)))
        plt.show()

        simulator_non_linear = SIMulator2D(illumination_non_linear, self.optical_system)
        sim_images_non_linear = simulator_non_linear.generate_sim_images(self.image)
        nonlinear_reconstructor = ReconstructorSpatialDomain2D(
            illumination=illumination_non_linear,
            optical_system=self.optical_system
        )

        reconstructed_non_linear = nonlinear_reconstructor.reconstruct(sim_images_non_linear)
        fig, axes = plt.subplots(1, 4)
        axes[0].imshow(self.image)
        axes[1].imshow(self.widefield)
        axes[2].imshow(self.reconstructed_linear)
        axes[3].imshow(reconstructed_non_linear)
        plt.show()

