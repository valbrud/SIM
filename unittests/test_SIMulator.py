import sys

import scipy.signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import wrappers
from config.IlluminationConfigurations import *
import unittest
import time
from matplotlib.widgets import Slider
from OpticalSystems import Lens
import ShapesGenerator
from SIMulator import SIMulator
sys.path.append('../')
configurations = BFPConfiguration()

class TestSIMImages(unittest.TestCase):
    def test_generate_images(self):
        theta = np.pi/3
        alpha = np.pi/2
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 4
        N = 51
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)
        X, Y, Z = np.meshgrid(x, y, z)
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r=0.5,  N=1000, I=1000)
        # image = np.zeros((N, N, N))
        # image[:, :, N//2] = 100000
        image += 100
        arg = N // 2

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        plt.imshow(optical_system.psf[:, :, N//2])
        # plt.show()
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        spacial_shifts = np.array(((0., 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)))
        spacial_shifts /= (5 * np.sin(theta))
        # spacial_shifts = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0)))
        # spacial_shifts /= (11 * np.sin(np.pi / 4))
        illumination_3waves.spacial_shifts = spacial_shifts

        simulator = SIMulator(illumination_3waves, optical_system, psf_size, N)
        images = simulator.simulate_sim_images(image)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)


        ax1.set_title("Square SIM (s)", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Illumination", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(images[0, 0, :, :, int(N//2)], vmin=0)
        # plt.colorbar(im1)

        im2 = ax2.imshow(np.abs(simulator.illuminations_shifted[0, 0, :, :, int(N//2)]), vmin=0)
        # # plt.colorbar(im2)
        #
        # im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)
        #
        # im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)


        def update1(val):
            ax1.clear()
            ax2.clear()


            ax1.set_title("Square (s)", fontsize=25, pad=15)
            ax2.set_title("Illumination", fontsize=25)


            im1 = ax1.imshow(images[1, int(val), :, :, N//2], vmin=0)
            im2 = ax2.imshow(simulator.illuminations_shifted[1, int(val), :, :, N//2], vmin=0)
            # im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            # im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, illumination_3waves.Mt)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()


class TestReconstruction(unittest.TestCase):
    def test_SDR(self):
        theta = np.pi / 2
        alpha = np.pi / 2
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 10
        max_z = 20
        N = 99
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)
        X, Y, Z = np.meshgrid(x, y, z)
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r=0.25, N=1000, I=10**4)
        # image = np.zeros((N, N, N))
        # R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r/10] = 10**9
        # image[N//2+1, N//2+1, N//2+1] = 10**9
        image += 100
        arg = N // 2

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        # plt.imshow(optical_system.psf[:, :, N // 2])
        # plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 1, 1, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        # spacial_shifts = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0))) - np.array((1., 9, 0))
        # spacial_shifts /= (11 * np.sin(theta))
        spacial_shifts = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        spacial_shifts /= (10 * np.sin(theta))
        illumination_s_polarized.spacial_shifts = spacial_shifts

        simulator = SIMulator(illumination_s_polarized, optical_system, psf_size, N)
        images = simulator.simulate_sim_images(image)
        image_sr = simulator.reconstruct_real_space(images)
        image_widefield = simulator.generate_widefield(images)
        # plt.imshow(image_sr[:, :, N//2])
        # plt.show()
        # plt.imshow(image_widefield[:, :, N//2])
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_title("SR", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Widefield", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(image_sr[:, :, N//2], vmin=0)
        im2 = ax2.imshow(image_widefield[:, :, N//2], vmin=0)

        # # plt.colorbar(im2)
        #
        # im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)
        #
        # im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)

        def update1(val):
            ax1.clear()
            ax2.clear()

            ax1.set_title("SR", fontsize=25, pad=15)
            ax2.set_title("Widefield", fontsize=25)

            im1 = ax1.imshow(image_sr[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(image_widefield[:, :, int(val)], vmin=0)
            # im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            # im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()