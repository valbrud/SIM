import sys

import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import kernels
import wrappers
from config.IlluminationConfigurations import *
import unittest
import time
from matplotlib.widgets import Slider
from OpticalSystems import System4f3D
import ShapesGenerator
from SIMulator import SIMulator
from SSNRCalculator import SSNR3dSIM2dShifts, SSNR3dSIM2dShiftsFiniteKernel
from SSNRBasedFiltering import WienerFilter3dModel, FlatNoiseFilter3d, WienerFilter3dReconstruction
sys.path.append('../')
configurations = BFPConfiguration()

class TestReconstruction(unittest.TestCase):
    def test_SDR(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N//2 * dx
        max_z = N//2 * dz
        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        image = np.zeros((N, N, N))
        # image[N//2, N//2, N//2] = 10**9
        # image[(3*N+1)//4, N//2+4, N//2] = 10**9
        # image[(N+1)//4, N//2-3, N//2] = 10**9
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**9
        # image += 10**6
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.25, N=1000, I=10 ** 4)
        image += 100
        arg = N // 2

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        plt.imshow(image[:, :, N // 2])
        plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 0.5, 1, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        # spatial_shifts_s_polarized11 = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0))) - np.array((1., 9, 0))
        # spatial_shifts_s_polarized11 /= (11 * np.sin(theta))
        # spatial_shifts_s_polarized10 = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        # spatial_shifts_s_polarized10 /= (10 * np.sin(theta))
        # illumination_s_polarized.spatial_shifts = spatial_shifts_s_polarized10
        # spatial_shifts_conventional = np.array(((0., 0., 0), (1, 0, 0), (2, 0, 0), (3., 0, 0), (4, 0, 0)))
        # spatial_shifts_conventional /= (5 * np.sin(theta))
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination_2waves.spatial_shifts = spatial_shifts_conventional2d

        simulator = SIMulator(illumination_2waves, optical_system, psf_size, N)
        images = simulator.generate_sim_images(image)
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
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_FDR(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        image = np.zeros((N, N, N))
        # image[N//2, N//2, N//2] = 10**9
        # image[(3*N+1)//4, N//2+4, N//2] = 10**9
        # image[(N+1)//4, N//2-3, N//2] = 10**9
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**9
        # image += 10**6
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=10000, I=10 ** 4)
        image += 100
        arg = N // 2

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        plt.imshow(image[:, :, N // 2])
        plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 0.5, 1, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        # spatial_shifts_s_polarized11 = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0))) - np.array((1., 9, 0))
        # spatial_shifts_s_polarized11 /= (11 * np.sin(theta))
        spatial_shifts_s_polarized10 = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        spatial_shifts_s_polarized10 /= (10 * np.sin(theta))
        illumination_s_polarized.spatial_shifts = spatial_shifts_s_polarized10
        spatial_shifts_conventional3d = np.array(((0., 0., 0), (1, 0, 0), (2, 0, 0), (3., 0, 0), (4, 0, 0)))
        spatial_shifts_conventional3d /= (5 * np.sin(theta))
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination_2waves.spatial_shifts = spatial_shifts_conventional2d
        illumination_3waves.spatial_shifts = spatial_shifts_conventional3d
        simulator = SIMulator(illumination_s_polarized, optical_system, psf_size, N)
        images = simulator.generate_sim_images(image)
        image_sr_ft, image_sr = simulator.reconstruct_Fourier_space(images)
        image_widefield = simulator.generate_widefield(images)

        noise_estimator = SSNR3dSIM2dShifts(illumination_s_polarized, optical_system, 0)
        noise_estimator.compute_ssnr()

        wiener = WienerFilter3dModel(noise_estimator)
        expected_image, ssnr, wj, otf_sim, tj = wiener.filter_object(image, real_space=True)
        # filtered_image = wrappers.wrapped_ifftn(tj * image_sr_ft).real
        wiener_rec = WienerFilter3dReconstruction(noise_estimator)
        filtered_image, ssnr, wj, otf_sim, tj = wiener_rec.filter_object(image_sr)
        plt.plot(image_sr[:, N//2, N//2])
        plt.imshow(image_sr[:, :, N//2])
        plt.show()
        plt.imshow(image_widefield[:, :, N//2])
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_title("Widefield", fontsize=25, pad=15)
        # ax1.tick_params(labelsize=20)

        ax2.set_title("Filtered", fontsize=25)
        # ax2.tick_params(labelsize=20)

        ax3.set_title("Expected", fontsize=25)
        # ax3.tick_params(labelsize=20)
        im1 = ax1.imshow(image_widefield[:, :, N // 2], vmin=0, vmax=np.amax(image_widefield))
        im2 = ax2.imshow(filtered_image[:, :, N // 2], vmin=0)
        im3 = ax3.imshow(expected_image[:, :, N // 2], vmin=0, vmax=np.amax(expected_image))
        # # plt.colorbar(im2)
        #
        # im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)
        #
        # im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)

        def update1(val):
            ax1.clear()
            ax2.clear()

            ax1.set_title("Widefield", fontsize=25, pad=15)
            ax2.set_title("Filtered", fontsize=25)
            ax3.set_title("Expected", fontsize=25)

            im1 = ax1.imshow(image_widefield[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(filtered_image[:, :, int(val)], vmin=0)
            im3 = ax3.imshow(expected_image[:, :, int(val)], vmin=0)
            # im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()


    def test_SDR_finite_kernel(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz

        kernel_size = 1
        kernel = kernels.psf_kernel2d(kernel_size, (dx, dy))

        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , N)

        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        image = np.zeros((N, N, N))
        # image[N//2, N//2, N//2] = 10**9
        # image[(3*N+1)//4, N//2+4, N//2] = 10**9
        # image[(N+1)//4, N//2-3, N//2] = 10**9
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**9
        # image += 10**6
        image = ShapesGenerator.generate_sphere_slices(psf_size, N, r=0.5, N=1000, I=10 ** 3)
        image += 100
        arg = N // 2

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        # plt.imshow(image[:, :, N // 2])
        # plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 0.5, 1, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        # spatial_shifts_s_polarized11 = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0))) - np.array((1., 9, 0))
        # spatial_shifts_s_polarized11 /= (11 * np.sin(theta))
        # spatial_shifts_s_polarized10 = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        # spatial_shifts_s_polarized10 /= (10 * np.sin(theta))
        # illumination_s_polarized.spatial_shifts = spatial_shifts_s_polarized10
        # spatial_shifts_conventional = np.array(((0., 0., 0), (1, 0, 0), (2, 0, 0), (3., 0, 0), (4, 0, 0)))
        # spatial_shifts_conventional /= (5 * np.sin(theta))
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination_2waves.spatial_shifts = spatial_shifts_conventional2d

        simulator = SIMulator(illumination_2waves, optical_system, psf_size, N)
        images = simulator.generate_sim_images(image)
        image_sr = simulator.reconstruct_real2d_finite_kernel(images, kernel)
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
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("SR", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Widefield", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(image_sr[:, :, N // 2], vmin=0)
        im2 = ax2.imshow(image_widefield[:, :, N // 2], vmin=0)

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
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_FDR_finite_kernel(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        kernel_r_size = 5
        kernel_z_size = 1
        kernel = np.zeros((kernel_r_size, kernel_r_size, kernel_z_size))
        func_r = np.zeros(kernel_r_size)
        func_r[0:kernel_r_size // 2 + 1] = np.linspace(0, 1, (kernel_r_size + 1) // 2 + 1)[1:]
        func_r[kernel_r_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_r_size + 1) // 2 + 1)[:-1]
        # func_r = np.ones(kernel_r_size)
        func_z = np.zeros(kernel_z_size)
        func_z[0:kernel_z_size // 2 + 1] = np.linspace(0, 1, (kernel_z_size + 1) // 2 + 1)[1:]
        func_z[kernel_z_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_z_size + 1) // 2 + 1)[:-1]
        func2d = func_r[:, None] * func_r[None, :]
        # func3d = func_r[:, None, None] * func_r[None, :, None] * func_z[None, None, :]
        # kernel[kernel_r_size//2, kernel_r_size//2, kernel_z_size//2] = 1
        # kernel[0,:, 0] = func_r
        kernel[:, :, 0] = func2d
        # kernel = func3ds

        image = np.zeros((N, N, N))
        # image[N//2, N//2, N//2] = 10**9
        # image[(3*N+1)//4, N//2+4, N//2] = 10**9
        # image[(N+1)//4, N//2-3, N//2] = 10**9
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**9
        # image += 10**6
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.25, N=1000, I=10 ** 5)
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=10000, I=10 ** 4)
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=10000, I=10 ** 10)
        image += 100
        arg = N // 2

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        # plt.imshow(image[:, :, N // 2])
        # plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 0.5, 1, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        # spatial_shifts_s_polarized11 = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0))) - np.array((1., 9, 0))
        # spatial_shifts_s_polarized11 /= (11 * np.sin(theta))
        # spatial_shifts_s_polarized10 = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        # spatial_shifts_s_polarized10 /= (10 * np.sin(theta))
        # illumination_s_polarized.spatial_shifts = spatial_shifts_s_polarized10
        # spatial_shifts_conventional = np.array(((0., 0., 0), (1, 0, 0), (2, 0, 0), (3., 0, 0), (4, 0, 0)))
        # spatial_shifts_conventional /= (5 * np.sin(theta))
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination_2waves.spatial_shifts = spatial_shifts_conventional2d

        simulator = SIMulator(illumination_2waves, optical_system, psf_size, N)
        images = simulator.generate_sim_images(image)

        noise_estimator = SSNR3dSIM2dShiftsFiniteKernel(illumination_2waves, optical_system, kernel=kernel)
        noise_estimator.compute_ssnr()

        image_sr_ft, image_sr = simulator.reconstruct_Fourier2d_finite_kernel(images, shifted_kernels=noise_estimator.effective_kernels_ft)
        image_widefield = simulator.generate_widefield(images)

        wiener = WienerFilter3dModel(noise_estimator)
        expected_image, ssnr, wj, otf_sim, tj = wiener.filter_object(image, real_space=True)

        filtered_image = wrappers.wrapped_ifftn(tj * image_sr_ft).real

        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_title("Widefield", fontsize=25, pad=15)
        # ax1.tick_params(labelsize=20)

        ax2.set_title("Filtered", fontsize=25)
        # ax2.tick_params(labelsize=20)

        ax3.set_title("Expected", fontsize=25)
        # ax3.tick_params(labelsize=20)
        im1 = ax1.imshow(image_widefield[:, :, N // 2], vmin=0, vmax=np.amax(image_widefield))
        im2 = ax2.imshow(filtered_image[:, :, N // 2], vmin=0, vmax=np.amax(filtered_image))
        im3 = ax3.imshow(expected_image[:, :, N // 2], vmin=0, vmax=np.amax(expected_image))

        # # plt.colorbar(im2)
        #
        # im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)
        #
        # im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)

        def update1(val):
            ax1.clear()
            ax2.clear()

            ax1.set_title("Widefield", fontsize=25, pad=15)
            ax2.set_title("Filtered", fontsize=25)
            ax3.set_title("Expected", fontsize=25)

            im1 = ax1.imshow(image_widefield[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(filtered_image[:, :, int(val)], vmin=0)
            im3 = ax3.imshow(expected_image[:, :, int(val)], vmin=0)
            # im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
