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
import hpc_utils
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
        axes[1].imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(image))), cmap='gray')
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
        axes[1].imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(upsampled_image))), cmap='gray')
        axes[1].axis('off')
        plt.show()


class TestVisualisation(unittest.TestCase):
    def test_axis3d_wrappers(self):
        """
        Demonstrate imshow3D (single panel) and wrap_axes3d (multi-panel).

        Two synthetic 3-D volumes are created:
          vol_a  – a sphere whose radius grows along z
          vol_b  – a Gaussian blob centred at the middle z-slice

        Part 1: imshow3D on a single axes.
        Part 2: wrap_axes3d on a 2×2 subplot grid, skipping one panel.
        """
        N = 64

        # --- build two synthetic volumes ---
        lin = np.linspace(-1, 1, N)
        X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
        R = np.sqrt(X**2 + Y**2)

        # vol_a: disk whose radius shrinks from 0.8 (z=0) to 0.2 (z=N-1)
        z_norm = np.linspace(0, 1, N)            # 0 … 1 along z-axis
        radius = 0.8 - 0.6 * z_norm              # shape (N,) broadcast over [x,y,z]
        vol_a = (R < radius[np.newaxis, np.newaxis, :]).astype(np.float64)

        # vol_b: Gaussian blob, centre drifts in x as z increases
        x_centre = np.linspace(-0.5, 0.5, N)
        vol_b = np.exp(-((X - x_centre[np.newaxis, np.newaxis, :]) ** 2
                         + Y ** 2) / 0.1)

        # ------------------------------------------------------------------ #
        # Part 1 – imshow3D: single interactive panel                         #
        # ------------------------------------------------------------------ #
        fig1, ax1, slider1 = utils.imshow3D(
            vol_a,
            mode='abs',
            axis='z',
            cmap='hot',
            vmin=0,
            vmax=1,
            origin='lower',
        )
        ax1.set_title("imshow3D demo – shrinking disk (z-scan)")
        ax1.set_xlabel("y")
        ax1.set_ylabel("x")

        # ------------------------------------------------------------------ #
        # Part 2 – wrap_axes3d: retrofit an existing 2×2 grid                 #
        # ------------------------------------------------------------------ #
        fig2, axes2 = plt.subplots(2, 2, figsize=(9, 8))

        # Populate each panel with a static 2-D slice (index 0) – the usual
        # workflow before handing off to wrap_axes3d.
        axes2[0, 0].imshow(vol_a[:, :, 0], cmap='gray', vmin=0, vmax=1,
                            origin='lower')
        axes2[0, 0].set_title("vol_a  (|array|) – z-scan")
        axes2[0, 0].set_xlabel("y")
        axes2[0, 0].set_ylabel("x")

        axes2[0, 1].imshow(vol_b[:, :, 0], cmap='hot', vmin=0, vmax=1,
                            origin='lower')
        axes2[0, 1].set_title("vol_b  (|array|) – z-scan")

        # axes2[1, 0] – intentionally left as a line plot; pass None to skip
        axes2[1, 0].plot(lin, vol_a[:, N // 2, N // 2], label="vol_a mid-y")
        axes2[1, 0].plot(lin, vol_b[:, N // 2, N // 2], label="vol_b mid-y")
        axes2[1, 0].set_title("x-cuts at z=0 (not wrapped)")
        axes2[1, 0].legend()

        axes2[1, 1].imshow(np.log1p(5 * vol_b[:, :, 0]), cmap='viridis',
                            origin='lower')
        axes2[1, 1].set_title("log1p(5 · vol_b) – z-scan")

        # Hand off to wrap_axes3d – None skips the line-plot panel
        slider2 = utils.wrap_axes3d(
            axes2,
            [vol_a, vol_b, None, vol_b],
            mode='abs',
            axis='z',
        )

        plt.show()