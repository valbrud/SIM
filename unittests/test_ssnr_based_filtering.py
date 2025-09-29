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

import WienerFiltering
import hpc_utils
from config.BFPConfigurations import *
import unittest
import time
import skimage
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from SSNRCalculator import SSNRSIM3D, SSNRSIM3D3dShifts
from OpticalSystems import System4f3D
import ShapesGenerator
import SIMulator
from windowing import make_mask_cosine_edge3d
configurations = BFPConfiguration()


class TestWiener(unittest.TestCase):
    def test_model_object(self):
        theta = np.pi/4
        alpha = np.pi/4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 4
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_r, max_z, N)

        fx = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r = 0.5,  N=100)
        image_ft = hpc_utils.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRSIM3D(illumination_3waves, optical_system)
        noise_estimator_widefield = SSNRSIM3D(illumination_widefield, optical_system)
        noise_estimator.ssnri
        noise_estimator_widefield.ssnri

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnri[:, :, N//2])))
        # plt.show()
        wiener = WienerFiltering.WienerFilter3dModel(noise_estimator)
        wiener_widefield = WienerFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_filtered, ssnr, wj, otf_sim, tj = wiener.filter_object(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.filter_object(image_ft, real_space=False)
        wjr = hpc_utils.wrapped_ifftn(wj)
        wjr /= np.amax(wjr)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title("Object", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Blurred", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("Wiener widefield", fontsize=25, pad=15)
        ax3.tick_params(labelsize=20)

        ax4.set_title("Wiener SIM", fontsize=25, pad=15)
        ax4.tick_params(labelsize=20)

        im1 = ax1.imshow(image[:, :, int(N//2)], vmin=0)
        # plt.colorbar(im1)

        im2 = ax2.imshow(np.abs(image_blurred[:, :, int(N//2)]), vmin=0)
        # plt.colorbar(im2)

        im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)

        im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)


        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.set_title("Object", fontsize=25, pad=15)
            ax2.set_title("Blurred", fontsize=25)
            ax3.set_title("Wiener widefield", fontsize=25, pad=15)
            ax4.set_title("Wiener SIM", fontsize=25, pad=15)

            im1 = ax1.imshow(image[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(np.abs(image_blurred[:, :, int(val)]), vmin=0)
            im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_wj_real_space(self):
        theta = np.pi/4
        alpha = np.pi/4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 4
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_r, max_z, N)

        fx = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r = 0.5,  N=100, I=1000)
        image_ft = hpc_utils.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRSIM3D(illumination_3waves, optical_system, readout_noise_variance=1)
        noise_estimator_widefield = SSNRSIM3D(illumination_widefield, optical_system, readout_noise_variance=1)
        noise_estimator.ssnri
        noise_estimator_widefield.ssnri

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnri[:, :, N//2])))
        # plt.show()
        wiener = WienerFiltering.WienerFilter3dModel(noise_estimator)
        wiener_widefield = WienerFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_filtered, ssnr, wj, otf_sim, tj = wiener.filter_object(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.filter_object(image_ft, real_space=False)

        wjr = hpc_utils.wrapped_ifftn(wj)
        wjrw = hpc_utils.wrapped_ifftn(wjw)
        # plt.imshow(np.abs(wjr - wjrw)[:, :, N//2])
        # plt.show()
        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title("Widefield Fourier", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Widefield Real", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("SIM Fourier", fontsize=25, pad=15)
        ax3.tick_params(labelsize=20)

        ax4.set_title("SIM Real", fontsize=25, pad=15)
        ax4.tick_params(labelsize=20)

        mp = 10**4
        im1 = ax1.imshow(np.log10(1 + mp * wjw[:, :, int(N//2)]), vmin=0)
        im2 = ax2.imshow(np.real(wjrw[:, :, int(N//2)]))
        im3 = ax3.imshow(np.log10(1 + mp * wj[:, :, int(N//2)]), vmin=0)
        im4 = ax4.imshow(np.real(wjr[:, :, int(N//2)]))


        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.set_title("Widefield Fourier", fontsize=25, pad=15)
            ax2.set_title("Widefield Real", fontsize=25, pad=15)
            ax3.set_title("SIM Fourier", fontsize=25, pad=15)
            ax4.set_title("SIM Real", fontsize=25, pad=15)

            im1 = ax1.imshow(np.log10(1 + mp * wjw[:, :, int(val)]), vmin=0)
            im2 = ax2.imshow(np.real(wjrw[:, :, int(val)]))
            im3 = ax3.imshow(np.log10(1 + mp * wj[:, :, int(val)]), vmin=0)
            im4 = ax4.imshow(np.real(wjr[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()


    def test_tj_real_space(self):
        theta = np.pi/4
        alpha = np.pi/4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 4
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_r, max_z, N)

        fx = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1,  N=500, I=100)
        image_ft = hpc_utils.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRSIM3D(illumination_s_polarized, optical_system)
        noise_estimator_widefield = SSNRSIM3D(illumination_widefield, optical_system)
        noise_estimator.ssnri
        noise_estimator_widefield.ssnri

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnri[:, :, N//2])))
        # plt.show()
        wiener = WienerFiltering.WienerFilter3dModel(noise_estimator)
        wiener_widefield = WienerFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_filtered, ssnr, wj, otf_sim, tj = wiener.filter_object(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.filter_object(image_ft, real_space=False)

        tjr = hpc_utils.wrapped_ifftn(tj)
        tjrw = hpc_utils.wrapped_ifftn(tjw)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title("Widefield Fourier", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Widefield Real", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("SIM Fourier", fontsize=25, pad=15)
        ax3.tick_params(labelsize=20)

        ax4.set_title("SIM Real", fontsize=25, pad=15)
        ax4.tick_params(labelsize=20)

        mp = 10**4
        im1 = ax1.imshow(np.real(tjw[:, :, int(N//2)]), vmin=0)
        im2 = ax2.imshow(np.real(tjrw[:, :, int(N//2)]))
        im3 = ax3.imshow(np.real(tj[:, :, int(N//2)]), vmin=0)
        im4 = ax4.imshow(np.real(tjr[:, :, int(N//2)]))


        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.set_title("Widefield Fourier", fontsize=25, pad=15)
            ax2.set_title("Widefield Real", fontsize=25, pad=15)
            ax3.set_title("SIM Fourier", fontsize=25, pad=15)
            ax4.set_title("SIM Real", fontsize=25, pad=15)

            im1 = ax1.imshow(np.real(tjw[:, :, int(val)]), vmin=0)
            im2 = ax2.imshow(np.real(tjrw[:, :, int(val)]))
            im3 = ax3.imshow(np.real(tj[:, :, int(val)]), vmin=0)
            im4 = ax4.imshow(np.real(tjr[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_different_illuminations(self):
        theta = np.pi/4
        alpha = np.pi/4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 4
        N = 50
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_r, max_z, N)

        fx = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r= 0.1,  N=1000, I=200)
        image = ShapesGenerator.generate_sphere_slices(psf_size, N, r= 0.1,  N=100, I=200000)
        image_ft = hpc_utils.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same') + 10**-10
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_linear = SSNRSIM3D(illumination_3waves, optical_system)
        noise_estimator_s_polarized = SSNRSIM3D(illumination_s_polarized, optical_system)
        noise_estimator_circular = SSNRSIM3D(illumination_circular, optical_system)
        noise_estimator_hexagonal = SSNRSIM3D(illumination_seven_waves, optical_system)
        noise_estimator_widefield = SSNRSIM3D(illumination_widefield, optical_system)
        noise_estimator_linear.compute_ssnr()
        noise_estimator_s_polarized.compute_ssnr()
        noise_estimator_circular.compute_ssnr()
        noise_estimator_hexagonal.compute_ssnr()
        noise_estimator_widefield.ssnri

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnri[:, :, N//2])))
        # plt.show()
        wiener_linear = WienerFiltering.WienerFilter3dModel(noise_estimator_linear)
        wiener_s = WienerFiltering.WienerFilter3dModel(noise_estimator_s_polarized)
        wiener_circular = WienerFiltering.WienerFilter3dModel(noise_estimator_circular)
        wiener_hexagonal = WienerFiltering.WienerFilter3dModel(noise_estimator_hexagonal)
        wiener_widefield = WienerFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_linear = wiener_linear.filter_object(image_ft, real_space=False)[0]
        image_s = wiener_s.filter_object(image_ft, real_space=False)[0]
        image_circular = wiener_circular.filter_object(image_ft, real_space=False)[0]
        image_hexagonal = wiener_hexagonal.filter_object(image_ft, real_space=False)[0]
        image_widefield = wiener_widefield.filter_object(image_ft, real_space=False)[0]


        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

        ax1.set_title("Widefield", fontsize=25, pad=15)
        ax2.set_title("Linear", fontsize=25)
        ax3.set_title("Square (s)", fontsize=25, pad=15)
        ax4.set_title("Square (c)", fontsize=25, pad=15)
        ax5.set_title("Hexagonal", fontsize=25, pad = 15)
        ax6.set_title("Object", fontsize=25, pad=15)

        im1 = ax1.imshow(image_widefield[:, :, int(N//2)], vmin=0)
        im2 = ax2.imshow(image_linear[:, :, int(N//2)], vmin=0)
        im3 = ax3.imshow(image_s[:, :, int(N//2)], vmin=0)
        im4 = ax4.imshow(image_circular[:, :, int(N//2)], vmin=0)
        im5 = ax5.imshow(image_hexagonal[:, :, int(N//2)], vmin=0)
        im6 = ax6.imshow(image[:, :, int(N//2)], vmin=0)


        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax1.set_title("Widefield", fontsize=25, pad=15)
            ax2.set_title("Linear", fontsize=25)
            ax3.set_title("Square (s)", fontsize=25, pad=15)
            ax4.set_title("Square (c)", fontsize=25, pad=15)
            ax5.set_title("Hexagonal", fontsize=25, pad=15)

            im1 = ax1.imshow(image_widefield[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(image_linear[:, :, int(val)], vmin=0)
            im3 = ax3.imshow(image_s[:, :, int(val)], vmin=0)
            im4 = ax4.imshow(image_circular[:, :, int(val)], vmin=0)
            im5 = ax5.imshow(image_hexagonal[:, :, int(val)], vmin=0)
            im6 = ax6.imshow(image[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_SDR_Wiener(self):
        theta = np.pi/4
        alpha = np.pi/4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 4
        max_z = 10
        N = 71
        psf_size = 2 * np.array((max_r, max_r, max_z))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z , N)
        X, Y, Z = np.meshgrid(x, y, z)
        fx = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)

        R = (X**2 + Y**2 + Z**2)**0.5
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.25, N=1000, I=10 ** 6)
        image = np.zeros((N, N, N))
        # image[R < max_r/10] = 10**5
        image[N//2, N//2, N//2] = 10**5
        image[(3*N+1)//4, N//2+4, N//2] = 10**5
        image[(N+1)//4, N//2-3, N//2] = 10**5
        image[(N+1)//4, (3 * N+1)//4, N//2] = 10**5
        image += 100
        X, Y, Z = np.meshgrid(x, y, z)
        image_ft = hpc_utils.wrapped_fftn(image)
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        plt.imshow(image_blurred[:, :, N//2])
        plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 1, 1, Mt=10)
        illumination_widefield = configurations.get_widefield()
        spatial_shifts = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        spatial_shifts /= (10 * np.sin(theta))
        illumination_s_polarized.spatial_shifts = spatial_shifts
        noise_estimator = SSNRSIM3D(illumination_s_polarized, optical_system)
        noise_estimator_widefield = SSNRSIM3D(illumination_widefield, optical_system)
        noise_estimator.ssnri
        noise_estimator_widefield.ssnri


        wiener = WienerFiltering.WienerFilter3dModelSDR(noise_estimator)
        wiener_widefield = WienerFiltering.WienerFilter3dModelSDR(noise_estimator_widefield)
        simulator = SIMulator.SIMulator(illumination_s_polarized, optical_system, psf_size, N)
        images = simulator.generate_noiseless_sim_images(image)
        image_sr = simulator.reconstruct_real_space(images)
        image_sr_ft = hpc_utils.wrapped_fftn(image_sr)
        plt.plot(np.log(1 + 10**4 * np.abs(image_sr_ft[:, 25, 25])), label='ft rec')
        plt.plot(np.log(1 + 10**4 * np.abs(hpc_utils.wrapped_fftn(image)[:, 25, 25] * noise_estimator.dj[:, 25, 25])))
        plt.legend()
        plt.show()
        image_widefield = simulator.generate_widefield(images)
        image_filtered, ssnr,  wj, geff, uj = wiener.filter_SDR_reconstruction(image, image_sr)
        widefield_benchmark, ssnrw, wjw, geff, uj = wiener_widefield.filter_SDR_reconstruction(image, image_widefield)
        wjr = hpc_utils.wrapped_ifftn(wj)
        wjr /= np.amax(wjr)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title("Blurred", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("No filter SDR", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("Wiener widefield", fontsize=25, pad=15)
        ax3.tick_params(labelsize=20)

        ax4.set_title("Wiener SDR", fontsize=25, pad=15)
        ax4.tick_params(labelsize=20)

        im1 = ax1.imshow(np.abs(image_widefield[:, :, int(N//2)]), vmin=0)
        # plt.colorbar(im1)

        im2 = ax2.imshow(np.abs(image_sr[:, :, int(N//2)]), vmin=0)
        # plt.colorbar(im2)

        im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)

        im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)


        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.set_title("Object", fontsize=25, pad=15)
            ax2.set_title("Blurred", fontsize=25)
            ax3.set_title("SDR", fontsize=25, pad=15)
            ax4.set_title("Wiener SDR SIM", fontsize=25, pad=15)

            im1 = ax1.imshow(image[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(np.abs(image_blurred[:, :, int(val)]), vmin=0)
            im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_FDR_reconstruction(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 3
        max_z = 8
        N = 51
        psf_size = 2 * np.array((max_r, max_r, max_z))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)
        X, Y, Z = np.meshgrid(x, y, z)
        image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=1000, I=10 ** 6)
        # image = ShapesGenerator.generate_sphere_slices(psf_size, N, r = 0.1, N = 100, I=10**5)
        R = (X**2 + Y**2 + Z**2)**0.5

        # image = np.zeros((N, N, N))
        # image[R < max_r/10] = 10**5
        # image[N//2, N//2, N//2] = 10**5
        # image[(3*N+1)//4, N//2+4, N//2] = 10**5
        # image[(N+1)//4, N//2-3, N//2] = 10**5
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**5
        image += 100
        mask = make_mask_cosine_edge3d(image.shape, 10)
        image *= mask
        arg = N // 2

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        # plt.imshow(optical_system.psf[:, :, N // 2])
        # plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 1, 0, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        # spatial_shifts = np.array(((1., 9, 0), (2, 2, 0), (3, 6, 0), (4, 10, 0), (5, 3, 0), (6, 7, 0), (7, 11, 0), (8, 4, 0), (9, 8, 0), (10, 1, 0), (11, 5, 0))) - np.array((1., 9, 0))
        # spatial_shifts /= (11 * np.sin(theta))
        spatial_shifts = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        spatial_shifts /= (10 * np.sin(theta))
        illumination_s_polarized.spatial_shifts = spatial_shifts

        spatial_shifts = np.array(((0., 0., 0), (1, 0, 0), (2, 0, 0), (3., 0, 0), (4, 0, 0)))
        spatial_shifts /= (5 * np.sin(theta))
        illumination_3waves.spatial_shifts = spatial_shifts

        illumination = illumination_3waves
        noise_estimator = SSNRSIM3D(illumination, optical_system, readout_noise_variance=0)
        noise_estimator.ssnri

        simulator = SIMulator.SIMulator(illumination, optical_system, psf_size, N, readout_noise_variance=0)
        images = simulator.generate_noiseless_sim_images(image)
        image_sr_ft, image_sr = simulator.reconstruct_Fourier_space(images)
        image_widefield = simulator.generate_widefield(images)
        plt.plot(np.log(1 + 10**4 * np.abs(image_sr_ft[:, N//2, N//2])), label='ft rec')
        plt.plot(np.log(1 + 10**4 * np.abs(hpc_utils.wrapped_fftn(image)[:, N//2, N//2] * noise_estimator.dj[:, N//2, N//2])))
        # plt.plot(np.log(1 + 10**4 * np.abs(hpc_utils.wrapped_fftn(image_widefield)[:, N//2, N//2])))
        plt.legend()
        plt.show()
        wiener_model = WienerFiltering.WienerFilter3dModel(noise_estimator)
        expected_image, ssnr, wj, otf_sim, tj = wiener_model.filter_object(image, real_space=True)

        wiener_reconstruction = WienerFiltering.WienerFilter3dReconstruction(noise_estimator)
        filtered_image, ssnr_rec, wj_rec, otf_sim_rec, tj_rec = wiener_reconstruction.filter_object(image_sr, real_space=True, average="surface_levels_3d")
        # filtered_image = np.abs(hpc_utils.wrapped_ifftn(tj_rec * hpc_utils.wrapped_fftn(image_sr)))
        plt.plot(np.log(1 + 10 ** 4 * np.abs(ssnr_rec[:, N//2, N//2])), label='ssnr rec')
        plt.plot(np.log(1 + 10 ** 4 * np.abs(ssnr[:, N//2, N//2])))
        plt.legend()
        plt.show()
        print("COMPUTED AT THE END", np.abs(ssnr_rec[N//2, N//2, N//2]))
        print("WHAT IT SHOULD BE", np.abs(ssnr[N//2, N//2, N//2]))
        plt.plot(np.log(1 + 10 ** 4 * np.abs((hpc_utils.wrapped_fftn(filtered_image[:, N//2, N//2])))), label='freqs rec')
        plt.plot(np.log(1 + 10 ** 4 * np.abs((hpc_utils.wrapped_fftn(expected_image[:, N//2, N//2])))))
        plt.legend()
        plt.show()
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title("Widefield", fontsize=25, pad=15)
        ax2.set_title("Reconstuction", fontsize=25, pad=15)
        # ax3.set_title("Filtered reconstruction", fontsize=25)
        ax3.set_title("SSNR rec", fontsize=25)
        ax4.set_title("SSNR true", fontsize=25)
        # ax4.set_title("Filtered model", fontsize=25)
        rec_exp = hpc_utils.wrapped_ifftn(noise_estimator.dj * hpc_utils.wrapped_fftn(image)).real
        # ax3.tick_params(labelsize=20)
        im1 = ax1.imshow(rec_exp[:, :, N // 2], vmin=0, vmax=np.amax(rec_exp))
        im2 = ax2.imshow(image_sr[:, :, N // 2], vmin=0, vmax=np.amax(image_sr))
        im3 = ax3.imshow(filtered_image[:, :, N // 2], vmin=0, vmax=np.amax(filtered_image))
        # im3 = ax3.imshow(np.log(1 + 10 ** 4 * np.abs(ssnr_rec[:, :, N // 2])), vmin=0, vmax=np.amax(ssnr_rec))
        im3 = ax4.imshow(expected_image[:, :, N // 2], vmin=0, vmax=np.amax(expected_image))
        # im4 = ax4.imshow(np.log(1 + 10 ** 4 * np.abs(ssnr[:, :, N // 2])), vmin=0, vmax=np.amax(ssnr))
        # # plt.colorbar(im2)
        #
        # im3 = ax3.imshow(widefield_benchmark[:, :, int(N//2)], vmin=0)
        #
        # im4 = ax4.imshow(image_filtered[:, :, int(N//2)], vmin=0)

        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.set_title("Reconstuction_expected", fontsize=25, pad=15)
            ax2.set_title("Image SR", fontsize=25, pad=15)
            ax3.set_title("Filtered", fontsize=25)
            # ax3.set_title("SSNR diff", fontsize=25)
            ax4.set_title("Filtered xpected", fontsize=25)
            # ax4.set_title("SSNR true", fontsize=25)

            im1 = ax1.imshow(rec_exp[:, :, int(val)], vmin=0, vmax = np.amax(rec_exp))
            im2 = ax2.imshow(image_sr[:, :, int(val)], vmin=0, vmax=np.amax(image_sr))
            im3 = ax3.imshow(filtered_image[:, :, int(val)], vmin=0, vmax=np.amax(filtered_image))
            # im3 = ax3.imshow(np.log(1 + 10 ** 4 * np.abs((ssnr_rec)[:, :, int(val)])), vmin=0)
            im4 = ax4.imshow(expected_image[:, :, int(val)], vmin=0, vmax = np.amax(expected_image))
            # im4 = ax4.imshow(np.log(1 + 10 ** 4 * np.abs(ssnr[:, :, int(val)])), vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()
class TestFlat(unittest.TestCase):
    def test_model_object(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 40
        max_z = 160
        N = 101
        psf_size = 2 * np.array((max_r, max_r, max_z))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_r, max_z, N)

        fx = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-N / (4 * max_r), N / (4 * max_r), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
        image[N//2, N//2, N//2] = 10000000

        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=100)
        image_ft = hpc_utils.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        # image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRSIM3D(illumination_3waves, optical_system)
        noise_estimator_widefield = SSNRSIM3D(illumination_widefield, optical_system)
        noise_estimator.ssnri
        noise_estimator_widefield.ssnri

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnri[:, :, N//2])))
        # plt.show()
        flat = WienerFiltering.FlatNoiseFilter3dModel(noise_estimator)
        flat_widefield = WienerFiltering.FlatNoiseFilter3dModel(noise_estimator_widefield)

        image_filtered, wj, otf_sim, tj = flat.filter_object(image_ft, real_space=False)
        widefield_benchmark, wjw, otf_simw, tjw = flat_widefield.filter_object(image_ft, real_space=False)
        wjr = hpc_utils.wrapped_ifftn(wj)
        wjr /= np.amax(wjr)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.set_title("Object", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Blurred", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("Flat Noise widefield", fontsize=25, pad=15)
        ax3.tick_params(labelsize=20)

        ax4.set_title("Flat Noise SIM", fontsize=25, pad=15)
        ax4.tick_params(labelsize=20)

        im1 = ax1.imshow(image[:, :, int(N // 2)], vmin=0)
        # plt.colorbar(im1)

        im2 = ax2.imshow(np.abs(image_blurred[:, :, int(N // 2)]), vmin=0)
        # plt.colorbar(im2)

        im3 = ax3.imshow(widefield_benchmark[:, :, int(N // 2)], vmin=0)

        im4 = ax4.imshow(image_filtered[:, :, int(N // 2)], vmin=0)

        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            ax1.set_title("Object", fontsize=25, pad=15)
            ax2.set_title("Blurred", fontsize=25)
            ax3.set_title("Flat noise widefield", fontsize=25, pad=15)
            ax4.set_title("Flat noise SIM", fontsize=25, pad=15)

            im1 = ax1.imshow(image[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(np.abs(image_blurred[:, :, int(val)]), vmin=0)
            im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            im4 = ax4.imshow(image_filtered[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig(f'{path_to_figures}square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()




