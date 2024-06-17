import sys

import scipy.signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import SSNRBasedFiltering
import wrappers
from config.IlluminationConfigurations import *
import unittest
import time
import skimage
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from SNRCalculator import SSNRCalculatorProjective3dSIM, SSNRCalculatorTrue3dSIM
from OpticalSystems import Lens
import ShapesGenerator
import SIMulator
sys.path.append('../')
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
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r = 0.5,  N=100)
        image_ft = wrappers.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRCalculatorProjective3dSIM(illumination_3waves, optical_system)
        noise_estimator_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system)
        noise_estimator.compute_ssnr()
        noise_estimator_widefield.compute_ssnr()

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnr[:, :, N//2])))
        # plt.show()
        wiener = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator)
        wiener_widefield = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_filtered, ssnr, wj, otf_sim, tj = wiener.filter_model_object(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.filter_model_object(image_ft, real_space=False)
        wjr = wrappers.wrapped_ifftn(wj)
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
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
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
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r = 0.5,  N=100, I=1000)
        image_ft = wrappers.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRCalculatorProjective3dSIM(illumination_3waves, optical_system, readout_noise_variance=1)
        noise_estimator_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system, readout_noise_variance=1)
        noise_estimator.compute_ssnr()
        noise_estimator_widefield.compute_ssnr()

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnr[:, :, N//2])))
        # plt.show()
        wiener = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator)
        wiener_widefield = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_filtered, ssnr, wj, otf_sim, tj = wiener.filter_model_object(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.filter_model_object(image_ft, real_space=False)

        wjr = wrappers.wrapped_ifftn(wj)
        wjrw = wrappers.wrapped_ifftn(wjw)
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
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
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
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r=0.1,  N=500, I=100)
        image_ft = wrappers.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRCalculatorProjective3dSIM(illumination_s_polarized, optical_system)
        noise_estimator_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system)
        noise_estimator.compute_ssnr()
        noise_estimator_widefield.compute_ssnr()

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnr[:, :, N//2])))
        # plt.show()
        wiener = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator)
        wiener_widefield = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_filtered, ssnr, wj, otf_sim, tj = wiener.filter_model_object(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.filter_model_object(image_ft, real_space=False)

        tjr = wrappers.wrapped_ifftn(tj)
        tjrw = wrappers.wrapped_ifftn(tjw)

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
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
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
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        # image = ShapesGenerator.generate_random_spheres(psf_size, N, r= 0.1,  N=1000, I=200)
        image = ShapesGenerator.generate_sphere_slices(psf_size, N, r= 0.1,  N=100, I=200)
        image_ft = wrappers.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same') + 10**-10
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_linear = SSNRCalculatorProjective3dSIM(illumination_3waves, optical_system)
        noise_estimator_s_polarized = SSNRCalculatorProjective3dSIM(illumination_s_polarized, optical_system)
        noise_estimator_circular = SSNRCalculatorProjective3dSIM(illumination_circular, optical_system)
        noise_estimator_hexagonal = SSNRCalculatorProjective3dSIM(illumination_seven_waves, optical_system)
        noise_estimator_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system)
        noise_estimator_linear.compute_ssnr()
        noise_estimator_s_polarized.compute_ssnr()
        noise_estimator_circular.compute_ssnr()
        noise_estimator_hexagonal.compute_ssnr()
        noise_estimator_widefield.compute_ssnr()

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnr[:, :, N//2])))
        # plt.show()
        wiener_linear = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_linear)
        wiener_s = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_s_polarized)
        wiener_circular = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_circular)
        wiener_hexagonal = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_hexagonal)
        wiener_widefield = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_widefield)

        image_linear = wiener_linear.filter_model_object(image_ft, real_space=False)[0]
        image_s = wiener_s.filter_model_object(image_ft, real_space=False)[0]
        image_circular = wiener_circular.filter_model_object(image_ft, real_space=False)[0]
        image_hexagonal = wiener_hexagonal.filter_model_object(image_ft, real_space=False)[0]
        image_widefield = wiener_widefield.filter_model_object(image_ft, real_space=False)[0]


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
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()

    def test_SDR_Wiener(self):
        theta = np.pi/2
        alpha = np.pi/2
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 5
        max_z = 12
        N = 89
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X**2 + Y**2 + Z**2)**0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r = 0.3,  N=1000, I = 10**5)
        image+=10
        # image[N//2+1, N//2 + 1, N//2 + 1] = 10**9
        image_ft = wrappers.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()
        illumination_s_polarized = configurations.get_5_s_waves(theta, 1, 1, Mt=10)
        illumination_widefield = configurations.get_widefield()
        spacial_shifts = np.array(((0., 0, 0), (1, 3, 0), (2, 6, 0), (3, 9, 0), (4, 2, 0), (5, 5, 0), (6, 8, 0), (7, 1, 0), (8, 4, 0), (9, 7, 0)))
        spacial_shifts /= (10 * np.sin(theta))
        illumination_s_polarized.spacial_shifts = spacial_shifts
        noise_estimator = SSNRCalculatorProjective3dSIM(illumination_s_polarized, optical_system)
        noise_estimator_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system)
        noise_estimator.compute_ssnr()
        noise_estimator_widefield.compute_ssnr()

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnr[:, :, N//2])))
        # plt.show()
        wiener = SSNRBasedFiltering.WienerFilter3dModelSDR(noise_estimator)
        wiener_widefield = SSNRBasedFiltering.WienerFilter3dModelSDR(noise_estimator_widefield)
        simulator = SIMulator.SIMulator(illumination_s_polarized, optical_system, psf_size, N)
        images = simulator.simulate_sim_images(image)
        image_sr = simulator.reconstruct_real_space(images)
        image_widefield = simulator.generate_widefield(images)
        image_filtered, ssnr,  wj, geff, uj = wiener.filter_SDR_reconstruction(image, image_sr)
        widefield_benchmark, ssnrw, wjw, geff, uj = wiener_widefield.filter_SDR_reconstruction(image, image_widefield)
        wjr = wrappers.wrapped_ifftn(wj)
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
            ax3.set_title("Wiener widefield", fontsize=25, pad=15)
            ax4.set_title("Wiener SDR SIM", fontsize=25, pad=15)

            im1 = ax1.imshow(image[:, :, int(val)], vmin=0)
            im2 = ax2.imshow(np.abs(image_blurred[:, :, int(val)]), vmin=0)
            im3 = ax3.imshow(widefield_benchmark[:, :, int(val)], vmin=0)
            im4 = ax4.imshow(image_sr[:, :, int(val)], vmin=0)

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()
class TestFlat(unittest.TestCase):
    def test_model_object(self):
        theta = np.pi / 4
        alpha = np.pi / 4
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
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        image = np.zeros((N, N, N))

        X, Y, Z = np.meshgrid(x, y, z)
        R = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
        # image[R < max_r//2] = 1000
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r=0.5, N=100)
        image_ft = wrappers.wrapped_fftn(image)
        # plt.imshow(image[:, :, N//2])
        # plt.show()
        # plt.imshow(np.abs(image_ft[:, :, N//2]))
        # plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)
        image_blurred = scipy.signal.convolve(image, optical_system.psf, mode='same')
        image_blurred = np.random.poisson(image_blurred)
        # plt.imshow(image_blurred[:, :, N//2])
        # plt.show()

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNRCalculatorProjective3dSIM(illumination_3waves, optical_system)
        noise_estimator_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system)
        noise_estimator.compute_ssnr()
        noise_estimator_widefield.compute_ssnr()

        # plt.imshow(np.abs(np.log10(1 + 10 ** 4 * noise_estimator.ssnr[:, :, N//2])))
        # plt.show()
        flat = SSNRBasedFiltering.FlatNoiseFilter3dModel(noise_estimator)
        flat_widefield = SSNRBasedFiltering.FlatNoiseFilter3dModel(noise_estimator_widefield)

        image_filtered, wj, otf_sim, tj = flat.filter_model_object(image_ft, real_space=False)
        widefield_benchmark, wjw, otf_simw, tjw = flat_widefield.filter_model_object(image_ft, real_space=False)
        wjr = wrappers.wrapped_ifftn(wj)
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
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparison_of_3d_SIM_modalities_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/square_sim_anisotropies_fz={:.2f}_r_={:.2f}.png'.format(two_NA_fz[arg], r))
        plt.show()




