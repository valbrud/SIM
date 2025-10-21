"""
Unit tests for FRC (Fourier Ring Correlation) functionality.

This module contains tests for the FRC functions in ResolutionMeasures.py,
testing both two-image FRC and single-image FRC with noise simulation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import unittest

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import required modules
from ShapesGenerator import generate_random_lines
from OpticalSystems import System4f2D
from resolution_measures import frc, frc_one_image
import scipy.signal


class TestFRC(unittest.TestCase):
    """
    AI-generated tests.

    Test cases for Fourier Ring Correlation functionality.
    """

    def setUp(self):
        """Set up test parameters."""
        self.N = 128  # Smaller size for faster testing
        self.alpha = np.pi/4
        self.nmedium = 1.5
        self.readout_noise = 2.0

    def test_frc_two_images_with_noise(self):
        """
        Test FRC with two noisy images created from the same ground truth.

        This test:
        1. Generates a test image using ShapesGenerator
        2. Convolves it with PSF using System4f2DCoherent
        3. Creates two noisy versions with different shot noise realizations
        4. Computes FRC between the two noisy images
        5. Verifies FRC values are reasonable
        """
        # Generate test image using ShapesGenerator
        image_size = (self.N, self.N)
        point_number = self.N
        line_width = 2.0
        num_lines = 100
        intensity = 100.0

        ground_truth = generate_random_lines(
            image_size=image_size,
            point_number=point_number,
            line_width=line_width,
            num_lines=num_lines,
            intensity=intensity
        )

        # Create optical system and compute PSF
        optical_system = System4f2D(
            alpha=self.alpha,
            refractive_index=self.nmedium,
            interpolation_method="linear",
            normalize_otf=True
        )

        psf_size = (self.N, self.N)
        optical_system.compute_psf_and_otf_coordinates(psf_size, self.N)
        optical_system.compute_psf_and_otf()

        # Convolve ground truth with PSF
        convolved_image = scipy.signal.convolve2d(ground_truth, optical_system.psf, mode='same')

        # Create two noisy images with different shot noise realizations
        rng = np.random.default_rng(42)

        # First noisy image
        noisy_image1 = rng.poisson(convolved_image.astype(np.float64))
        noisy_image1 = noisy_image1 + rng.normal(0, self.readout_noise, noisy_image1.shape)

        # Second noisy image (different realization)
        noisy_image2 = rng.poisson(convolved_image.astype(np.float64))
        noisy_image2 = noisy_image2 + rng.normal(0, self.readout_noise, noisy_image2.shape)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(noisy_image1, cmap='gray')
        axes[0].set_title("Noisy Image 1")
        axes[1].imshow(noisy_image2, cmap='gray')
        axes[1].set_title("Noisy Image 2")
        plt.show()

        # Compute FRC
        frc_values, freq = frc(noisy_image1, noisy_image2, is_fourier=False, num_bins=30)

        # Basic assertions
        self.assertEqual(len(frc_values), 30)
        self.assertEqual(len(freq), 30)
        self.assertTrue(np.all(frc_values >= 0))
        self.assertTrue(np.all(frc_values <= 1))
        self.assertTrue(np.max(frc_values) > 0.1)  # Should have some correlation

        # FRC should generally decrease with frequency
        self.assertTrue(frc_values[0] > frc_values[-1])

        plt.plot(freq, frc_values)
        plt.title("FRC Values")
        plt.xlabel("Frequency (bins)")
        plt.ylabel("FRC")
        plt.show()


    def test_frc_one_image_with_noise(self):
        """
        Test FRC with one noisy image using the single-image FRC method.

        This test:
        1. Generates a test image using ShapesGenerator
        2. Convolves it with PSF using System4f2DCoherent
        3. Creates one noisy version
        4. Uses frc_one_image function to compute FRC
        5. Verifies FRC values are reasonable
        """
        # Generate test image using ShapesGenerator
        image_size = (self.N, self.N)
        point_number = self.N
        line_width = 2.0
        num_lines = 100
        intensity = 100.0

        ground_truth = generate_random_lines(
            image_size=image_size,
            point_number=point_number,
            line_width=line_width,
            num_lines=num_lines,
            intensity=intensity
        )

        # Create optical system and compute PSF
        optical_system = System4f2D(
            alpha=self.alpha,
            refractive_index=self.nmedium,
            interpolation_method="linear",
            normalize_otf=True
        )
        psf_size = (self.N, self.N)
        optical_system.compute_psf_and_otf_coordinates(psf_size, self.N)
        optical_system.compute_psf_and_otf()

        # Convolve ground truth with PSF
        convolved_image = scipy.signal.convolve2d(ground_truth, optical_system.psf, mode='same')

        # Create one noisy image
        rng = np.random.default_rng(42)
        noisy_image = rng.poisson(convolved_image.astype(np.float64))

        plt.imshow(noisy_image, cmap='gray')
        plt.title("Noisy Image")
        plt.axis("off")
        plt.show()

        # Compute FRC using single image method (without readout noise correction)
        frc_values, freq = frc_one_image(noisy_image, num_bins=30, readout_noise=0)

        # Basic assertions
        self.assertEqual(len(frc_values), 30)
        self.assertEqual(len(freq), 30)
        self.assertTrue(np.all(frc_values >= 0))
        self.assertTrue(np.all(frc_values <= 1))
        self.assertTrue(np.max(frc_values) > 0.1) 

        plt.plot(freq, frc_values)
        plt.title("FRC Values")
        plt.xlabel("Frequency (bins)")
        plt.ylabel("FRC")
        plt.show()

    def test_frc_identical_images(self):
        """Test FRC with identical images (should give FRC = 1)."""
        # Create a simple test image
        image = np.random.rand(64, 64)

        # Compute FRC between identical images
        frc_values, freq = frc(image, image, is_fourier=False, num_bins=20)

        # FRC should be 1 for identical images
        np.testing.assert_allclose(frc_values, 1.0, rtol=1e-10)

    def test_frc_uncorrelated_images(self):
        """Test FRC with uncorrelated images (should give FRC ≈ 0)."""
        rng = np.random.default_rng(42)
        image1 = rng.normal(0, 1, (64, 64))
        image2 = rng.normal(0, 1, (64, 64))

        # Compute FRC between uncorrelated images
        frc_values, freq = frc(image1, image2, is_fourier=False, num_bins=20)

        # FRC should be close to 0 for uncorrelated images
        # Note: Due to finite sampling, we expect values much less than 0.5
        self.assertTrue(np.max(frc_values) < 0.3)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFRC)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
