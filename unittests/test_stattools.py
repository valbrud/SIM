import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import numpy as np
import matplotlib.pyplot as plt
import utils
import wrappers
import unittest
import sys
from OpticalSystems import System4f3D
from config.BFPConfigurations import BFPConfiguration
from SSNRCalculator import SSNRWidefield2D

configurations = BFPConfiguration()
class TestRingAveraging(unittest.TestCase):
    def test_averaging_over_uniform(self):
        array = np.ones((100, 100))
        averages = utils.average_rings2d(array)
        assert np.allclose(averages, np.ones(50))

    def test_different_axes(self):
        x = np.arange(100)
        y = np.arange(0, 100, 2)
        array = np.ones((x.size, y.size))
        averages = utils.average_rings2d(array, (x, y))
        assert np.allclose(averages, np.ones(50))

    def test_averaging_over_sine(self):
        x = np.arange(1000)
        y = np.arange(0, 1000, 2)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X**2 + Y**2)**0.5/100)
        plt.imshow(sine_array)
        plt.show()
        averages = utils.average_rings2d(sine_array.T, (x, y))
        plt.plot(averages, label='computed')
        plt.legend()
        plt.plot(np.sin(y/100), label='theoretical')
        plt.show()

    def test_SSNR_averaging(self):
        alpha = np.pi / 4
        theta = np.pi / 4
        NA = np.sin(alpha)
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), N)

        arg = N // 2
        # print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        print(two_NA_fx)
        two_NA_fy = fy / (2 * NA)
        scaled_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        noise_estimator_widefield = SSNRWidefield(optical_system)
        ssnr = noise_estimator_widefield.ssnri
        plt.figure(1)
        plt.imshow(np.log(1 + 10**8*ssnr[N//2, :, :]))
        ssnr_widefield_ra = noise_estimator_widefield.ring_average_ssnri()
        plt.figure(2)
        plt.imshow(np.log(1 + 10**8*ssnr_widefield_ra))
        ssnr_diff = ssnr[N//2, N//2:, :] - ssnr_widefield_ra
        plt.figure(3)
        plt.imshow(np.log(1 + 10**8 * np.abs(ssnr_diff)))
        plt.show()





class TestRingExpansion(unittest.TestCase):
    def test_uniform_expansion(self):
        array = np.ones((100, 100))
        averages = utils.average_rings2d(array)
        assert np.allclose(averages, np.ones(50))
        expanded = utils.expand_ring_averages2d(averages)
        plt.imshow(expanded)
        plt.show()

    def test_sine_expansion(self):
        x = np.arange(1000)
        y = np.arange(0, 1000, 2)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X ** 2 + Y ** 2) ** 0.5 / 100)
        # plt.imshow(sine_array)
        # plt.show()
        averages = utils.average_rings2d(sine_array.T, (x, y))
        plt.plot(averages)
        plt.plot(np.sin(y / 100))
        # plt.show()
        expanded = utils.expand_ring_averages2d(averages, (x, y))
        plt.imshow(expanded)
        plt.show()

    def test_inhomogeneous_expansion(self):
        x = np.arange(-100, 100)
        y = np.arange(-100, 100)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X ** 2 + Y ** 2) ** 0.5 / 10)
        phi = np.arctan(Y/X)
        sine_array *= np.abs(np.sin(phi))
        plt.imshow(sine_array)
        plt.show()
        averages = utils.average_rings2d(sine_array, (x, y))
        plt.plot(averages)
        # plt.plot(np.sin(y / 100))
        plt.show()
        expanded = utils.expand_ring_averages2d(averages, (x, y))
        plt.plot(x, expanded[:, 100])
        # plt.imshow(expanded)
        plt.show()

class TestSurfaceLevels(unittest.TestCase):
    def test_split_spherically_symmetric(self):
        x = np.linspace(-100, 100, 201)
        y = np.copy(x)
        X, Y = np.meshgrid(x, y)
        R = (X**2 + Y**2)**0.5
        f = 1 / (1 + R)
        mask = utils.find_decreasing_surface_levels2d(f)
        plt.imshow(mask)
        plt.show()

    def test_split_flower(self):
        x = np.linspace(-100, 100, 201)
        y = np.copy(x)
        X, Y = np.meshgrid(x, y)
        R1 = (X ** 2 + Y ** 2 / 10) ** 0.5
        R2 = (X**2 / 10 + Y**2)**0.5
        f = 1 / (1 + R1/10 + R2/10)
        # plt.imshow(f)
        # plt.show()
        mask = utils.find_decreasing_surface_levels2d(f)
        plt.imshow(mask)
        plt.show()

    def test_split_3d(self):
        x = np.linspace(-100, 100, 201)
        y = np.copy(x)
        z = np.copy(y)
        X, Y, Z = np.meshgrid(x, y, z)
        R = (X ** 2 + Y ** 2 + Z**2) ** 0.5
        f = 1 / (1 + R)
        mask = utils.find_decreasing_surface_levels3d(f, direction=0)
        plt.imshow(mask[:, :, 100])
        plt.show()


class TestMiscellaneous(unittest.TestCase):
    def test_upsample(self):
        # Generate an image with a circle
        image = np.zeros((30, 30))
        radius = 6
        center = (15, 15)
        y, x = np.indices(image.shape)
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = 1

        # Display the original image
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("Original Image")
        axes[0].imshow(image, cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(np.log1p(np.abs(wrappers.wrapped_fftn(image))), cmap='gray')
        axes[1].axis('off')
        plt.show()
        # You can also generate other features like lines (uncomment below if needed)
        # image = np.zeros((100, 100))
        # image[50, :] = 1  # horizontal line
        # image[:, 50] = 1  # vertical line
        # plt.figure()
        # plt.title("Original Image with Lines")
        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.show()
        upsampled_image = utils.upsample(image, factor=2)
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("Original Image")
        axes[0].imshow(np.abs(upsampled_image), cmap='gray')
        axes[0].axis('off')
        axes[1].imshow(np.log1p(np.abs(wrappers.wrapped_fftn(upsampled_image))), cmap='gray')
        axes[1].axis('off')
        plt.show()