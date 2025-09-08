import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from config.BFPConfigurations import *
import unittest
import time
from matplotlib.widgets import Slider
from OpticalSystems import System4f2D, System4f3D
import ShapesGenerator
from SIMulator import SIMulator2D, SIMulator3D
from Camera import Camera
# from simulations.mutual_information_filtering import psf_size
from config.SIM_N100_NA15 import *
import wrappers
from windowing import make_mask_cosine_edge2d
class TestSIMImages(unittest.TestCase):
    def test_generate_images2d(self):
        
        N = 101
        max_r = N // 2 * dx

        psf_size = 2 * np.array((2 * max_r, 2 * max_r))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        dimensions = (1, 1)
        X, Y = np.meshgrid(x, y)
        image = ShapesGenerator.generate_random_lines(psf_size, N, line_width=0.25, num_lines=150, intensity=10000)

        optical_system = System4f2D(alpha=alpha, refractive_index=nmedium)
        optical_system.compute_psf_and_otf((psf_size, N), )
        # plt.imshow(optical_system.psf)
        # plt.show()

        illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_3waves = IlluminationPlaneWaves2D.init_from_3D(illumination_3waves3d, dimensions)

        spatial_shifts = np.array(((0., 0.), (1, 0), (2, 0)))
        spatial_shifts /= (3 * np.sin(theta))
        illumination_3waves.spatial_shifts = spatial_shifts

        simulator = SIMulator2D(illumination_3waves, optical_system)
        widefield = simulator.generate_widefield(image)
        noisy_widefield = simulator.add_noise(widefield)
        # plt.imshow(widefield)
        # plt.show()
        # image = np.ones((N, N)) * 1000
        images = simulator.generate_noiseless_sim_images(image)
        noisy_images = simulator.add_noise(images)

        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_title("SIM 2D", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Widefield", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(noisy_images[0, 0], vmin=0)
        im2 = ax2.imshow(noisy_widefield, vmin=0)
        plt.show()

        ax1.imshow(np.abs(np.log1p(wrappers.wrapped_fftn(noisy_images[0, 0]))), vmin=0)
        ax2.imshow(np.abs(np.log1p(wrappers.wrapped_fftn(noisy_widefield))), vmin=0)

        plt.show()

    def test_generate_images2d_with_camera(self):
        N = 500
        max_r = N // 2 * dx
        x = np.linspace(-max_r, max_r, N)
        psf_size= np.array((2*max_r, 2*max_r))
        y = np.copy(x)
        dimensions = (1, 1)
        X, Y = np.meshgrid(x, y)
        image = ShapesGenerator.generate_random_lines(psf_size, N, line_width=0.4, num_lines=100, intensity=10 ** 4)
        image = np.ones((N, N)) * 1000
        optical_system = System4f2D(alpha=alpha, refractive_index=nmedium)
        optical_system.compute_psf_and_otf((psf_size, N), )
        plt.imshow(optical_system.psf)

        illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_3waves = IlluminationPlaneWaves2D.init_from_3D(illumination_3waves3d, dimensions)

        spatial_shifts = np.array(((0., 0.), (1, 0), (2, 0)))
        spatial_shifts /= (3 * np.sin(theta))
        illumination_3waves.spatial_shifts = spatial_shifts

        cam = Camera(pixel_number=(500, 500), pixel_size=(dx, dx), mode='2D',
                     readout_noise_variance=10,  # variance=4 => sigma=2
                     hot_pixel_fraction=0.0,  # 1% hot pixels
                     exposure_time=1.0, dark_current_rate=10.0,
                     saturation_level=1e5
                     )

        simulator = SIMulator2D(illumination_3waves, optical_system, cam)
        widefield = simulator.generate_widefield(image)
        noisy_widefield = simulator.add_noise(widefield)
        # plt.imshow(widefield)
        # plt.show()

        images = simulator.generate_noiseless_sim_images(image)
        noisy_images = simulator.add_noise(images)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_title("SIM 2D with camera", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Widefield", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(noisy_images[0, 0], vmin=0)
        im2 = ax2.imshow(noisy_widefield, vmin=0)
        plt.show()

    def test_generate_images3d(self):
        theta = np.pi / 3
        alpha = np.pi / 2
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 4
        N = 51
        psf_size = 2 * np.array((max_r, max_r, max_z))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)
        dimensions = (1, 1, 0)
        X, Y, Z = np.meshgrid(x, y, z)
        # image = np.zeros((N, N, N))
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.25, N=1000, I=10 ** 4)
        # image[:, :, N//2] = 100000
        image = np.ones((N, N, N)) * 10**9
        image += 100
        arg = N // 2

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N))
        # plt.imshow(optical_system.psf[:, :, N // 2])
        # plt.show()
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_3waves.set_spatial_shifts_diagonally()

        simulator = SIMulator3D(illumination_3waves, optical_system)
        images = simulator.generate_noiseless_sim_images(image)
        mask = make_mask_cosine_edge2d(images[0, 0, :, :, arg].shape, 10)
        images *= mask[None, None, :, :, None]
        images_ft = np.array([np.array([wrappers.wrapped_fftn(images[r, n])for n in range(5)])  for r in range(3)])
        fig, axes = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
        for r in range(3):
            for n in range(5):
                axes[r, n].imshow(np.log1p(np.abs(images_ft[r, n, :, :, N//2])), vmin=0)
                axes[r, n].set_title(f"r={r}, n={n}", fontsize=15)
                axes[r, n].tick_params(labelsize=10)
        plt.show()
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Square SIM (s)", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Illumination", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(images[0, 0, :, :, int(N // 2)], vmin=0)

        # plt.colorbar(im1)
        illumination_density = illumination_3waves.get_illumination_density(grid=optical_system.x_grid, r=0, n=0)
        im2 = ax2.imshow(illumination_density[:, :, int(N//2)])
        # plt.colorbar(im2)


        def update1(val):
            ax1.clear()
            ax2.clear()

            ax1.set_title("Square (s)", fontsize=25, pad=15)
            ax2.set_title("Illumination", fontsize=25)

            im1 = ax1.imshow(images[0, 0, :, :, int(val)], vmin=0)
            im2 = ax2.imshow(illumination_density[:, :, int(val)], vmin=0)
            # im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            # im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, illumination_3waves.Mt)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # plt.show()

        plt.show()
