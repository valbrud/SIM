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

        image_filtered, ssnr, wj, otf_sim, tj = wiener.compute_filtered_image(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.compute_filtered_image(image_ft, real_space=False)
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
        image = ShapesGenerator.generate_random_spheres(psf_size, N, r = 0.1,  N=500, I=100)
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

        image_filtered, ssnr, wj, otf_sim, tj = wiener.compute_filtered_image(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.compute_filtered_image(image_ft, real_space=False)

        wjr = wrappers.wrapped_ifftn(wj)
        wjrw = wrappers.wrapped_ifftn(wjw)

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

        image_filtered, ssnr, wj, otf_sim, tj = wiener.compute_filtered_image(image_ft, real_space=False)
        widefield_benchmark, ssnrw, wjw, otf_simw, tjw = wiener_widefield.compute_filtered_image(image_ft, real_space=False)

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

        image_filtered, wj, otf_sim, tj = flat.compute_filtered_image(image_ft, real_space=False)
        widefield_benchmark, wjw, otf_simw, tjw = flat_widefield.compute_filtered_image(image_ft, real_space=False)
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
            ax3.set_title("Wiener widefield", fontsize=25, pad=15)
            ax4.set_title("Wiener SIM", fontsize=25, pad=15)

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
