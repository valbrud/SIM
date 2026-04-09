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


class TestWatershedFilter(unittest.TestCase):
    """Tests for utils.low_pass_adaptive_watershed_filter."""

    def test_clean_decaying_signal_unchanged(self):
        """A purely decaying signal with no noise should stay untouched."""
        x = np.arange(100, 0, -1, dtype=np.float64)  # 100, 99, …, 1
        filtered = utils.low_pass_adaptive_watershed_filter(x, confidence_interval=10)

        plt.figure()
        plt.title("Clean decaying signal (should be unchanged)")
        plt.plot(x, label='original')
        plt.plot(filtered, '--', label='filtered')
        plt.legend()
        plt.show()

        np.testing.assert_array_equal(filtered, x)

    def test_decaying_with_noisy_tail(self):
        """Signal decays from 100 to ~20, then a noisy tail of small values.
        The noisy tail should be zeroed out."""
        n_signal = 60
        n_noise = 40
        signal = np.linspace(100, 20, n_signal)
        noise = np.random.RandomState(42).uniform(0, 2, n_noise)  # values in [0, 2)
        array = np.concatenate([signal, noise])

        filtered = utils.low_pass_adaptive_watershed_filter(array, confidence_interval=10)

        plt.figure()
        plt.title("Decaying signal + noisy tail")
        plt.plot(array, label='original')
        plt.plot(filtered, '--', label='filtered')
        plt.axvline(n_signal, color='gray', linestyle=':', label='signal/noise boundary')
        plt.legend()
        plt.show()

        # The first part of the signal should be preserved (values >> noise)
        self.assertTrue((filtered[:n_signal] > 0).all(),
                        "Signal region should not be zeroed out")
        # The noise tail should be zeroed
        self.assertTrue((filtered[n_signal:] == 0).all(),
                        "Noise tail should be zeroed out")

    def test_flat_noise_array(self):
        """An array of uniform random noise.  Most / all values should be zeroed
        because there is no clear 'signal' region."""
        noise = np.random.RandomState(7).uniform(0, 1, 100)
        filtered = utils.low_pass_adaptive_watershed_filter(noise, confidence_interval=10)

        plt.figure()
        plt.title("Flat noise (no signal)")
        plt.plot(noise, label='original')
        plt.plot(filtered, '--', label='filtered')
        plt.legend()
        plt.show()

        # At least half the array should be zeroed (the filter should cut something)
        self.assertGreater(np.sum(filtered == 0), len(noise) // 2)

    def test_short_array(self):
        """Arrays shorter than confidence_interval should be returned as-is."""
        short = np.array([5.0, 3.0, 1.0])
        filtered = utils.low_pass_adaptive_watershed_filter(short, confidence_interval=10)

        plt.figure()
        plt.title("Short array (shorter than confidence_interval)")
        plt.plot(short, 'o-', label='original')
        plt.plot(filtered, 'x--', label='filtered')
        plt.legend()
        plt.show()

        np.testing.assert_array_equal(filtered, short)

    # ---------- comparative_watershed_filter tests ----------

    def test_comparative_signal_crosses_noise(self):
        """Signal decays and eventually dips below a flat noise floor."""
        n = 100
        signal = np.linspace(50, 0, n)        # decays from 50 to 0
        noise_floor = np.full(n, 10.0)         # constant noise at 10
        # crossover happens where signal < 10 → around index 80

        filtered = utils.comparative_watershed_filter(signal, noise_floor)

        crossover_idx = np.argmax(noise_floor >= signal)

        plt.figure()
        plt.title("Comparative filter: signal vs constant noise floor")
        plt.plot(signal, label='signal (array1)')
        plt.plot(noise_floor, label='noise floor (array2)')
        plt.plot(filtered, '--', label='filtered signal')
        plt.axvline(crossover_idx, color='gray', linestyle=':', label='crossover')
        plt.legend()
        plt.show()

        self.assertTrue((filtered[:crossover_idx] > 0).all())
        self.assertTrue((filtered[crossover_idx:] == 0).all())

    def test_comparative_no_crossing(self):
        """Signal is always above the noise floor – nothing is filtered."""
        n = 100
        signal = np.linspace(100, 20, n)
        noise_floor = np.linspace(5, 1, n)

        filtered = utils.comparative_watershed_filter(signal, noise_floor)

        plt.figure()
        plt.title("Comparative filter: signal always above noise")
        plt.plot(signal, label='signal (array1)')
        plt.plot(noise_floor, label='noise floor (array2)')
        plt.plot(filtered, '--', label='filtered signal')
        plt.legend()
        plt.show()

        np.testing.assert_array_equal(filtered, signal)

    def test_comparative_immediate_crossing(self):
        """Noise exceeds signal from the very first element."""
        n = 50
        signal = np.linspace(5, 1, n)
        noise_floor = np.full(n, 10.0)

        filtered = utils.comparative_watershed_filter(signal, noise_floor)

        plt.figure()
        plt.title("Comparative filter: noise exceeds signal immediately")
        plt.plot(signal, label='signal (array1)')
        plt.plot(noise_floor, label='noise floor (array2)')
        plt.plot(filtered, '--', label='filtered signal')
        plt.legend()
        plt.show()

        self.assertTrue((filtered == 0).all())

class TestCutoffRatioInterpolation(unittest.TestCase):
    def test_circle(self):
        """Circle surface: ratio should equal r/R everywhere → smooth cone."""
        N, R = 201, 80
        c = (N - 1) / 2
        yy, xx = np.indices((N, N))
        rr = np.sqrt((xx - c)**2 + (yy - c)**2)
        S = ((rr >= R - 0.5) & (rr < R + 0.5)).astype(float)
        X = np.zeros((N, N))

        ratio = utils.radial_ratio(X, S)

        # Interior midpoint along x-axis: ratio ≈ 0.5
        self.assertAlmostEqual(ratio[int(c), int(c + R/2)], 0.5, delta=0.02)
        # On circle: ratio ≈ 1
        self.assertAlmostEqual(ratio[int(c), int(c + R)],   1.0, delta=0.02)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im = axes[0].imshow(ratio, cmap='gray', vmin=0, vmax=1,
                            origin='lower')
        axes[0].set_title('2D circle: ratio map')
        plt.colorbar(im, ax=axes[0])
        axes[1].imshow(S, cmap='gray', origin='lower')
        axes[1].set_title('Surface S (circle)')
        plt.tight_layout()
        # plt.savefig('test_2d_circle.png', dpi=120)
        plt.show()

    def test_rhombus(self):
        """Rhombus (L1 ball) surface: ratio should equal (|dx|+|dy|)/R."""
        N, R = 201, 70
        c = (N - 1) / 2
        yy, xx = np.indices((N, N))
        l1 = np.abs(xx - c) + np.abs(yy - c)
        S  = ((l1 >= R - 0.5) & (l1 < R + 0.5)).astype(float)
        X  = np.zeros((N, N))

        ratio = utils.radial_ratio(X, S)

        # On x-axis at half-range: dx=R/2, dy=0 → r_surf=R, ratio=0.5
        self.assertAlmostEqual(ratio[int(c), int(c + R/2)], 0.5, delta=0.03)
        # On the contour along x-axis: ratio ≈ 1
        self.assertAlmostEqual(ratio[int(c), int(c + R)],   1.0, delta=0.03)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im = axes[0].imshow(ratio, cmap='gray', vmin=0, vmax=1,
                            origin='lower')
        axes[0].set_title('2D rhombus: ratio map')
        plt.colorbar(im, ax=axes[0])
        axes[1].imshow(S, cmap='gray', origin='lower')
        axes[1].set_title('Surface S (rhombus / L1 ball)')
        plt.tight_layout()
        # plt.savefig('test_2d_rhombus.png', dpi=120)
        plt.show()

    def test_sphere(self):
        """Sphere surface: ratio = r/R; middle slice should look like a 2-D cone."""
        N, R = 61, 22
        c = (N - 1) / 2
        zz, yy, xx = np.indices((N, N, N))
        rr = np.sqrt((xx - c)**2 + (yy - c)**2 + (zz - c)**2)
        S = ((rr >= R - 0.5) & (rr < R + 0.5)).astype(float)
        X = np.zeros((N, N, N))

        ratio = utils.radial_ratio(X, S)

        mid = N // 2
        # On sphere along x-axis at z=mid: ratio ≈ 1
        self.assertAlmostEqual(ratio[mid, mid, mid + R], 1.0, delta=0.05)
        # Half-way: ratio ≈ 0.5
        self.assertAlmostEqual(ratio[mid, mid, mid + R//2], 0.5, delta=0.05)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im = axes[0].imshow(ratio[mid], cmap='gray',
                            vmin=0, vmax=1, origin='lower')
        axes[0].set_title(f'3D sphere: ratio, z={mid} slice')
        plt.colorbar(im, ax=axes[0])
        axes[1].imshow(S[mid], cmap='gray', origin='lower')
        axes[1].set_title('Surface S (sphere), middle slice')
        plt.tight_layout()
        # plt.savefig('test_3d_sphere.png', dpi=120)
        plt.show()

    def test_cylinder(self):
        """
        Cylinder (lateral surface, axis along z): ratio ≈ rho/R in xy-slices,
        and should grow with |z| in xz-slices since the 3-D distance grows.
        """
        N, R = 61, 20
        c = (N - 1) / 2
        zz, yy, xx = np.indices((N, N, N))
        rho = np.sqrt((xx - c)**2 + (yy - c)**2)
        S   = ((rho >= R - 0.5) & (rho < R + 0.5)).astype(float)
        X   = np.zeros((N, N, N))

        ratio = utils.radial_ratio(X, S)

        mid = N // 2
        # At (z=mid, y=mid, x=mid+R): on the cylinder wall → ratio ≈ 1
        self.assertAlmostEqual(ratio[mid, mid, mid + R], 1.0, delta=0.1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(ratio[mid], cmap='gray',
                             vmin=0, vmax=1, origin='lower')
        axes[0].set_title('Cylinder ratio — xy slice (z=mid)')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(ratio[:, mid, :], cmap='gray',
                             vmin=0, vmax=1, origin='lower')
        axes[1].set_title('Cylinder ratio — xz slice (y=mid)')
        plt.colorbar(im1, ax=axes[1])

        axes[2].imshow(S[mid], cmap='gray', origin='lower')
        axes[2].set_title('Surface S — xy slice')
        plt.tight_layout()
        # plt.savefig('test_3d_cylinder.png', dpi=120)
        plt.show()
