import sys
import Box
sys.path.append('../')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from config.IlluminationConfigurations import *
import unittest
import time
import skimage
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from Illumination import Illumination
from SNRCalculator import SNRCalculator
from OpticalSystems import Lens
from Sources import IntensityPlaneWave
import tqdm

configurations = BFPConfiguration()
class Testssnr(unittest.TestCase):
    def test_ssnr_interpolations(self):
        max_r = 6
        max_z = 6
        N = 80
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        waves = configurations.get_4_oblique_s_waves_and_circular_normal(np.pi/4, 1, 1)

        illumination_polarized = Illumination(waves)
        optical_system_fourier = Lens(interpolation_method="Fourier")
        optical_system_fourier.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N),
                                                   apodization_filter=None)
        optical_system_linear = Lens()
        optical_system_linear.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N),
                                                  apodization_filter=None)
        wavevectors = illumination_polarized.get_wavevectors()

        noise_estimator = SNRCalculator(illumination_polarized, optical_system_fourier)

        begin_shifted = time.time()
        optical_system_fourier.prepare_Fourier_interpolation(wavevectors)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        ssnr_F = np.abs(noise_estimator.compute_ssnr(q_axes))
        end_shifted = time.time()
        print("fourier time is ", end_shifted - begin_shifted)

        noise_estimator.optical_system = optical_system_linear
        begin_interpolation = time.time()
        ssnr_L = np.abs(noise_estimator.compute_ssnr(q_axes))
        end_interpolation = time.time()
        print("linear time is ", end_interpolation - begin_interpolation)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure()
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        mp1 = ax1.imshow(np.log10(ssnr_F[:, :, int(N / 2)]))
        cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        mp2 = ax2.imshow(np.log10(ssnr_L[:, :, int(N / 2)]))
        cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)

        def update1(val):
            ax1.set_title("ssnr_F, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fx, $\lambda^{-1}$")
            ax1.set_ylabel("fy, $\lambda^{-1}$")
            Z = (np.log10(ssnr_F[:, :, int(val)]))
            mp1.set_data(Z)
            mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            plt.draw()
            print(np.amax(ssnr_F[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        def update2(val):
            ax2.set_title("ssnr_L, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax2.set_xlabel("fx, $\lambda^{-1}$")
            ax2.set_ylabel("fy, $\lambda^{-1}$")
            print(int(val))
            Z = (np.log10(ssnr_L[:, :, int(val)]))
            mp2.set_data(Z)
            min = Z.min()
            max = Z.max()
            mp2.set_clim(vmin=min, vmax=max)
            # plt.draw()

        slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnri = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnri.on_changed(update2)

        plt.show()

    def test_ssnr(self):
        NA = np.sin(np.pi/4)
        max_r = 10
        max_z = 40
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)
        print(x)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)
        print(fx)

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        # print(fw2z_illumination)
        theta = np.pi/4
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1/2**0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()


        illumination_list = {
            illumination_s_polarized : ("4 s-polarized oblique waves", "4s1c"),
            illumination_circular : ("4 circularly polarized oblique waves", "5c"),
            illumination_seven_waves : ("6 s-polarized oblique waves", "6s1c"),
            illumination_3waves : ("State of Art SIM", "state_of_art"),
            illumination_widefield : ("Widefield", "widefield"),
            illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        optical_system = Lens(alpha=theta)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        for illumination in illumination_list:
            noise_estimator = SNRCalculator(illumination, optical_system)

            q_axes = 2 * np.pi * np.array((fx, fy, fz))
            ssnr = np.abs(noise_estimator.compute_ssnr(q_axes,))
            scaling_factor = 10**8
            ssnr_scaled = 1 + scaling_factor * ssnr
            ssnr_ring_averaged = noise_estimator.ring_average_ssnr(q_axes, ssnr)
            # ssnr_ra_scaled = 1 + scaling_factor * ssnr_ring_averaged

            Fx, Fy = np.meshgrid(fx, fy)
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            fig.suptitle(illumination_list[illumination][0], fontsize = 30)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.1,
                                hspace=0)
            ax1 = fig.add_subplot(121)
            ax1.tick_params(labelsize = 20)
            # ax1.set_title(title)
            ax1.set_xlabel("$f_z$", fontsize = 25)
            ax1.set_ylabel("$f_y$", fontsize = 25)
            mp1 = ax1.imshow(ssnr_scaled[:, int(N / 2), :], extent=(fz[0]/(2 * NA), fz[-1]/(2 * NA), fy[0]/(2 * NA), fy[-1]/(2 * NA)), norm=colors.LogNorm())
            # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
            # cb1.set_label("$1 + 10^8$ ssnr")
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2 = fig.add_subplot(122, sharey=ax1)
            ax2.set_xlabel("$f_x$", fontsize = 25)
            ax2.tick_params(labelsize = 20)
            ax2.tick_params('y', labelleft=False)
            # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
            mp2 = ax2.imshow(ssnr_scaled[:, :, int(N//2)].T, extent=(fy[0]/(2 * NA), fy[-1]/(2 * NA), fx[0]/(2 * NA), fx[-1]/(2 * NA)), norm=colors.LogNorm())
            cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=20)
            cb2.set_label("$1 + 10^8$ SSNR", fontsize = 25)
            ax2.set_aspect(1. / ax2.get_data_ratio())

            fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/'
                     + illumination_list[illumination][1] + '_waves_SSNR.png')

            # def update1(val):
            #     ax1.set_title("SSNR, fy = {:.2f}, ".format(fy[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            #     ax1.set_xlabel("fz, $\lambda^{-1}$")
            #     ax1.set_ylabel("fx, $\lambda^{-1}$")
            #     Z = (SSNR_scaled[:, int(val), :])
            #     mp1.set_data(Z)
            #     mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            #     ax1.set_aspect(1. / ax1.get_data_ratio())
            #     plt.draw()
            #
            #     ax2.set_title("SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            #     ax2.set_xlabel("fz, $\lambda^{-1}$")
            #     ax2.set_ylabel("fx, $\lambda^{-1}$")
            #     Z = (SSNR_scaled[:, :, int(val)])
            #     mp2.set_data(Z)
            #     mp2.set_clim(vmin=Z.min(), vmax=Z.max())
            #     ax2.set_aspect(1. / ax2.get_data_ratio())
            #     plt.draw()
            #
            # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
            # slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
            # slider_ssnr.on_changed(update1)
            #
            # plt.show()

    def test_SSNR_ring_averaged_color_maps(self):
        NA = np.sin(np.pi/4)
        max_r = 10
        max_z = 40
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)
        print(x)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)
        print(fx)

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        # print(fw2z_illumination)
        theta = np.pi/4
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1/2**0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()


        illumination_list = {
            illumination_s_polarized : ("4 s-polarized oblique waves", "4s1c"),
            illumination_circular : ("4 circularly polarized oblique waves", "5c"),
            illumination_seven_waves : ("6 s-polarized oblique waves", "6s1c"),
            illumination_3waves : ("State of Art SIM", "state_of_art"),
            illumination_widefield : ("Widefield", "widefield"),
            illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        optical_system = Lens(alpha=theta)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        for illumination in illumination_list:
            noise_estimator = SNRCalculator(illumination, optical_system)

            q_axes = 2 * np.pi * np.array((fx, fy, fz))
            SSNR = np.abs(noise_estimator.compute_ssnr(q_axes,))
            scaling_factor = 10**8
            SSNR_scaled = 1 + scaling_factor * SSNR
            SSNR_ring_averaged = noise_estimator.ring_average_SSNR(q_axes, SSNR)
            SSNR_ra_scaled = 1 + scaling_factor * SSNR_ring_averaged

            Fx, Fy = np.meshgrid(fx, fy)
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            fig.suptitle(illumination_list[illumination][0], fontsize = 30)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.1,
                                hspace=0)
            ax1 = fig.add_subplot(121)
            ax1.tick_params(labelsize = 20)
            # ax1.set_title(title)
            ax1.set_xlabel("$f_x$", fontsize = 25)
            ax1.set_ylabel("$f_z$", fontsize = 25)
            mp1 = ax1.imshow(SSNR_scaled[int(N/2):, int(N / 2), :].T, extent=(0, fy[-1]/(2 * NA), fz[0]/(2 * NA), fz[-1]/(2 * NA)), norm=colors.LogNorm())
            # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
            # cb1.set_label("$1 + 10^8$ SSNR")
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2 = fig.add_subplot(122, sharey=ax1)
            ax2.set_xlabel("$f_r$", fontsize = 25)
            ax2.tick_params(labelsize = 20)
            ax2.tick_params('y', labelleft=False)
            # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
            mp2 = ax2.imshow(SSNR_ra_scaled[:, :].T, extent=(0, fy[-1]/(2 * NA), fz[0]/(2 * NA), fz[-1]/(2 * NA)), norm=colors.LogNorm())
            cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=20)
            cb2.set_label("$1 + 10^8$ SSNR", fontsize = 25)
            ax2.set_aspect(1. / ax2.get_data_ratio())

            fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/'
                     + illumination_list[illumination][1] + '_waves_SSNR_ra.png')

            # def update1(val):
            #     ax1.set_title("SSNR, fy = {:.2f}, ".format(fy[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            #     ax1.set_xlabel("fz, $\lambda^{-1}$")
            #     ax1.set_ylabel("fx, $\lambda^{-1}$")
            #     Z = (SSNR_scaled[:, int(val), :])
            #     mp1.set_data(Z)
            #     mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            #     ax1.set_aspect(1. / ax1.get_data_ratio())
            #     plt.draw()
            #
            #     ax2.set_title("SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            #     ax2.set_xlabel("fz, $\lambda^{-1}$")
            #     ax2.set_ylabel("fx, $\lambda^{-1}$")
            #     Z = (SSNR_scaled[:, :, int(val)])
            #     mp2.set_data(Z)
            #     mp2.set_clim(vmin=Z.min(), vmax=Z.max())
            #     ax2.set_aspect(1. / ax2.get_data_ratio())
            #     plt.draw()
            #
            # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
            # slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
            # slider_ssnr.on_changed(update1)
            #
            # plt.show()

    def test_isosurface_visualisation(self):
        NA = np.pi/4
        max_r = 10
        max_z = 40
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dfx = 1 / (2 * max_r)
        dfy = 1 / (2 * max_r)
        dfz = 1 / (2 * max_z)
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        theta = np.pi/4
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1/2**0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()


        illumination_list = {
            # illumination_s_polarized : ("4 s-polarized oblique waves", "4s1c"),
            # illumination_circular : ("4 circularly polarized oblique waves", "5c"),
            # illumination_seven_waves : ("6 s-polarized oblique waves", "6s1c"),
            # illumination_3waves : ("State of Art SIM", "state_of_art"),
            # illumination_widefield : ("Widefield", "widefield"),
            illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        optical_system = Lens(alpha=np.pi/4)
        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        for illumination in illumination_list:
            noise_estimator = SNRCalculator(illumination, optical_system)
            SSNR = np.abs(noise_estimator.compute_ssnr(q_axes,))
            SSNR_scaled = np.log10(1 + 10**8 * np.abs(SSNR))

            noise_estimator.illumination = configurations.get_widefield()
            ssnr_widefield = np.abs(noise_estimator.compute_ssnr(q_axes))
            ssnr_widefield_scaled = np.log10(1 + 10**8 * np.abs(ssnr_widefield))

            constant_value = 0.2
            verts, faces, _, _ = skimage.measure.marching_cubes(SSNR_scaled, level=constant_value)
            w_verts, w_faces, _, _ = skimage.measure.marching_cubes(ssnr_widefield_scaled, level=constant_value)
            Fx, Fy, Fz = np.meshgrid(fx, fy, fz)
            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.set_title(illumination_list[illumination][0], fontsize=20)
            ax1.set_xlabel(r"$f_x$", fontsize=20, labelpad=15)
            ax1.set_ylabel(r"$f_y$", fontsize=20, labelpad=15)
            ax1.set_zlabel(r"$f_z$", fontsize=20, labelpad=15)
            ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],  alpha=0.7)
            ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2],  alpha=1, color="red")

            ax1.set_xlim(N // 4, 3 * N // 4)
            ax1.set_ylim(N // 4, 3 * N // 4)
            ax1.set_zlim(0, N)
            xticks = np.round((ax1.get_xticks() - N / 2) * dfx / (2 * NA), 2)
            yticks = np.round((ax1.get_yticks() - N / 2) * dfy / (2 * NA), 2)
            zticks = np.round((ax1.get_zticks() - N / 2) * dfz / (2 * NA), 2)
            ax1.set_xticklabels(xticks)
            ax1.set_yticklabels(yticks)
            ax1.set_zticklabels(zticks)
            ax1.tick_params(labelsize=15)
            ax1.view_init(elev=20, azim=45)
            plt.draw()
            fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/'
                     + illumination_list[illumination][1] + '_waves_SSNR_isosurface.png')

            def update1(val):
                ax1.clear()
                ax1.set_title(illumination_list[illumination][0] + "\n1 + $10^8 SSNR$ = {:.2f}".format(val/15), fontsize=18)
                ax1.set_xlabel(r"$f_x$", fontsize=18, labelpad=15)
                ax1.set_ylabel(r"$f_y$", fontsize=18, labelpad=15)
                ax1.set_zlabel(r"$f_z$", fontsize=18, labelpad=15)
                verts, faces, _, _ = skimage.measure.marching_cubes(SSNR_scaled, level=val/15)
                w_verts, w_faces, _, _ = skimage.measure.marching_cubes(ssnr_widefield_scaled, level=val/15)
                ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, color="blue")
                ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2], alpha=1, color="red")

                ax1.set_xlim(N//4, 3 * N//4)
                ax1.set_ylim(N//4, 3 * N//4)
                ax1.set_zlim(0, N)

                xticks = np.round((ax1.get_xticks() - N / 2) * dfx / (2 * NA), 2)
                yticks = np.round((ax1.get_yticks() - N / 2) * dfy / (2 * NA), 2)
                zticks = np.round((ax1.get_zticks() - N / 2) * dfz / (2 * NA), 2)
                ax1.set_xticklabels(xticks)
                ax1.set_yticklabels(yticks)
                ax1.set_zticklabels(zticks)
                plt.draw()

            slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
            slider_ssnr = Slider(slider_loc, 'fz', 0, 50)  # slider properties
            slider_ssnr.on_changed(update1)

            # ani = FuncAnimation(fig, update1, frames=range(2, 40), repeat=False, interval=100)
            # ani.save('/home/valerii/Documents/projects/SIM/SSNR_article_1/Animations/'
            #          'Animation_' + illumination_list[illumination][1] + '_waves_SSNR_isosurface.mp4', writer="ffmpeg")
            plt.show()

    def test_compare_SSNR(self):
        theta = np.pi/4
        NA = np.sin(theta)
        max_r = 10
        max_z = 20
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

        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (2 * NA)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = Lens()
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1/2**0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()


        noise_estimator = SNRCalculator(illumination_s_polarized, optical_system)
        SSNR_s_polarized = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_s_polarized_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_s_polarized)
        volume_s_polarized = noise_estimator.compute_SSNR_volume(SSNR_s_polarized, dV)
        measure_s_polarized, threshold_s_polarized = noise_estimator.compute_ssnr_measure(SSNR_s_polarized)
        entropy_s_polarized = noise_estimator.compute_true_ssnr_entropy(SSNR_s_polarized)

        noise_estimator.illumination = illumination_circular
        SSNR_circular = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_circular_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_circular)
        volume_circular = noise_estimator.compute_SSNR_volume(SSNR_circular, dV)
        measure_circular, threshold_circular= noise_estimator.compute_ssnr_measure(SSNR_circular)
        entropy_circular = noise_estimator.compute_true_ssnr_entropy(SSNR_circular)

        noise_estimator.illumination = illumination_seven_waves
        SSNR_seven_waves = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_seven_waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_seven_waves)
        volume_seven_waves = noise_estimator.compute_SSNR_volume(SSNR_seven_waves, dV)
        measure_seven_waves, threshold7waves = noise_estimator.compute_ssnr_measure(SSNR_seven_waves)
        entropy_seven_waves = noise_estimator.compute_true_ssnr_entropy(SSNR_seven_waves)

        noise_estimator.illumination = illumination_3waves
        SSNR_3waves = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_3waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_3waves)
        volume_3waves = noise_estimator.compute_SSNR_volume(SSNR_3waves, dV)
        measure_3waves, threshold3waves  = noise_estimator.compute_ssnr_measure(SSNR_3waves)
        entropy_3waves = noise_estimator.compute_true_ssnr_entropy(SSNR_3waves)

        noise_estimator.illumination = illumination_widefield
        ssnr_widefield = np.abs(noise_estimator.compute_ssnr(q_axes))
        ssnr_widefield_ra = noise_estimator.ring_average_SSNR(q_axes, ssnr_widefield)
        volume_widefield = noise_estimator.compute_SSNR_volume(ssnr_widefield, dV)
        measure_widefield, _ = noise_estimator.compute_ssnr_measure(ssnr_widefield)

        print("Volume SSNR widefield = ", volume_widefield)
        print("Measure SSNR widefield = ", measure_widefield)

        print("Volume SSNR s_polarized = ", volume_s_polarized)
        print("Measure SSNR s_polarized = ", measure_s_polarized, threshold_s_polarized)
        print("Entropy SSNR s_polarized = ", entropy_s_polarized)

        print("Volume SSNR 3waves = ", volume_3waves)
        print("Measure SSNR 3waves = ", measure_3waves, threshold3waves)
        print("Entropy SSNR 3waves = ", entropy_3waves)

        print("Volume SSNR seven_waves = ", volume_seven_waves)
        print("Measure SSNR seven_waves = ", measure_seven_waves, threshold7waves)
        print("Entropy SSNR seven_waves = ", entropy_seven_waves)

        print("Volume SSNR circular = ", volume_circular)
        print("Measure SSNR circular = ", measure_circular, threshold_circular)
        print("Entropy SSNR circular = ", entropy_circular)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 6), constrained_layout = True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Ring averaged", fontsize=20, pad=15)
        ax1.set_xlabel(r"$f_r$", fontsize=20)
        ax1.set_ylabel(r"$SSNR_{ra}$", fontsize=20)
        ax1.set_yscale("log")
        ax1.set_ylim(1, 3 * 10**2)
        ax1.set_xlim(0, fx[-1]/(2 * NA))
        ax1.grid(which = 'major')
        ax1.grid(which='minor', linestyle = '--')
        ax1.tick_params(labelsize = 15)

        ax2.set_title("Slice $f_y$ = 0", fontsize = 20)
        ax2.set_xlabel(r"$f_x$", fontsize=20)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 3 * 10**2)
        ax2.set_xlim(0, fx[-1]/(2 * NA))
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize = 15)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, arg], label="lattice SIM, 5 waves, s-polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, arg],    label="lattice SIM, 5 waves, circularly polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_seven_waves_ra[:, arg],    label="7 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, arg],      label="Conventional SIM")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg],   label="Widefield")
        ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
        ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
        ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
        ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_s_polarized[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, s-polarized")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_circular[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, circularly polarized")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_seven_waves[:, int(N / 2), arg][fx >= 0], label="7 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[:, int(N / 2), arg][fx >= 0], label="Conventional SIM")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="Widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("Ring averaged", fontsize = 30)
            ax1.set_xlabel(r"$f_r$", fontsize = 25)
            ax1.set_ylabel(r"$SSNR$", fontsize = 25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 3 * 10**2)
            ax1.set_xlim(0, fx[-1]/(2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, int(val)], label="S-polarized")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, int(val)],    label="Circular")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_seven_waves_ra[:, int(val)],    label="7 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, int(val)],      label="3 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)],   label="Widefield")
            ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
            ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
            ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
            ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')
            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())
            
            ax2.clear()
            ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"SSNR")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10**2)
            ax2.set_xlim(0, fx[-1]/(2 * NA))
            ax2.grid()

            # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_s_polarized[:, :, int(val)])[q_axes[1] >= 0], label="S-polarized")
            # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_circular[:, :, int(val)])[q_axes[1] >= 0], label="Circular")

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_s_polarized[:, int(N / 2), int(val)][fx >= 0], label="S-polarized")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_circular[:, int(N / 2), int(val)][fx >= 0],    label="Circular"   )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_seven_waves[:, int(N / 2),  int(val)][fx >= 0],    label="7 waves"   )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[:, int(N / 2), int(val)][fx >= 0],      label="3 waves"    )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0],   label="Widefield"  )
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, 100)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR_conf_version.png'.format(two_NA_fz[arg]))
        plt.show()

    def test_alternative_visualisations(self):
        theta = np.pi / 4
        NA = np.sin(theta)
        max_r = 10
        max_z = 20
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

        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (2 * NA)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = Lens()
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SNRCalculator(illumination_s_polarized, optical_system)
        SSNR_s_polarized = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_s_polarized_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_s_polarized)
        # volume_s_polarized = noise_estimator.compute_SSNR_volume(SSNR_s_polarized, dV)
        # measure_s_polarized, threshold_s_polarized = noise_estimator.compute_ssnr_measure(SSNR_s_polarized)

        noise_estimator.illumination = illumination_circular
        SSNR_circular = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_circular_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_circular)
        # volume_circular = noise_estimator.compute_SSNR_volume(SSNR_circular, dV)
        # measure_circular, threshold_circular = noise_estimator.compute_ssnr_measure(SSNR_circular)

        noise_estimator.illumination = illumination_seven_waves
        SSNR_seven_waves = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_seven_waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_seven_waves)
        # volume_seven_waves = noise_estimator.compute_SSNR_volume(SSNR_seven_waves, dV)
        # measure_seven_waves, threshold7waves = noise_estimator.compute_ssnr_measure(SSNR_seven_waves)

        noise_estimator.illumination = illumination_3waves
        SSNR_3waves = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_3waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_3waves)
        # volume_3waves = noise_estimator.compute_SSNR_volume(SSNR_3waves, dV)
        # measure_3waves, threshold3waves = noise_estimator.compute_ssnr_measure(SSNR_3waves)

        noise_estimator.illumination = illumination_widefield
        ssnr_widefield = np.abs(noise_estimator.compute_ssnr(q_axes))
        ssnr_widefield_ra = noise_estimator.ring_average_SSNR(q_axes, ssnr_widefield)
        # volume_widefield = noise_estimator.compute_SSNR_volume(ssnr_widefield, dV)
        # measure_widefield, _ = noise_estimator.compute_ssnr_measure(ssnr_widefield)

        # print("Volume SSNR widefield = ", volume_widefield)
        # print("Measure SSNR widefield = ", measure_widefield)
        # print("Volume SSNR s_polarized = ", volume_s_polarized)
        # print("Measure SSNR s_polarized = ", measure_s_polarized, threshold_s_polarized)
        # print("Volume SSNR three_waves = ", volume_3waves)
        # print("Measure SSNR 3waves = ", measure_3waves, threshold3waves)
        # print("Volume SSNR seven_waves = ", volume_seven_waves)
        # print("Measure SSNR seven_waves = ", measure_seven_waves, threshold7waves)
        # print("Volume SSNR circular = ", volume_circular)
        # print("Measure SSNR circular = ", measure_circular, threshold_circular)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Ring averaged", fontsize=20, pad=15)
        ax1.set_xlabel(r"$f_r$", fontsize=20)
        ax1.set_ylabel(r"$SSNR_{ra}$", fontsize=20)
        ax1.set_yscale("log")
        ax1.set_ylim(1, 3 * 10 ** 3)
        ax1.set_xlim(0, fx[-1] / (2 * NA))
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=15)

        ax2.set_title("Slice $f_y$ = 0", fontsize=20)
        ax2.set_xlabel(r"$f_x$", fontsize=20)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 3 * 10 ** 3)
        ax2.set_xlim(0, fx[-1] / (2 * NA))
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=15)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(SSNR_s_polarized_ra, axis = 1), label="lattice SIM, 5 waves, s-polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(SSNR_circular_ra, axis = 1), label="lattice SIM, 5 waves, circularly polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(SSNR_seven_waves_ra, axis = 1), label="7 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(SSNR_3waves_ra, axis = 1), label="Conventional SIM")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(ssnr_widefield_ra, axis = 1), label="Widefield")
        # ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
        # ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
        # ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
        # ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')

        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_s_polarized[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, s-polarized")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_circular[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, circularly polarized")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_seven_waves[:, int(N / 2), arg][fx >= 0], label="7 waves")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[:, int(N / 2), arg][fx >= 0], label="Conventional SIM")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="Widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        # ax2.set_aspect(1. / ax2.get_data_ratio())

        # def update1(val):
        #     ax1.clear()
        #     ax1.set_title("Ring averaged", fontsize=30)
        #     ax1.set_xlabel(r"$f_r$", fontsize=25)
        #     ax1.set_ylabel(r"$SSNR$", fontsize=25)
        #     ax1.set_yscale("log")
        #     ax1.set_ylim(1, 3 * 10 ** 2)
        #     ax1.set_xlim(0, fx[-1] / (2 * NA))
        #     ax1.grid()
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, int(val)], label="S-polarized")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, int(val)], label="Circular")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_seven_waves_ra[:, int(val)], label="7 waves")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, int(val)], label="3 waves")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="Widefield")
        #     # ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
        #     # ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
        #     # ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
        #     # ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')
        #     ax1.legend()
        #     ax1.set_aspect(1. / ax1.get_data_ratio())
        #
        #     ax2.clear()
        #     ax2.set_title("Slice $f_y$ = 0")
        #     ax2.set_xlabel(r"$f_x$")
        #     # ax2.set_ylabel(r"SSNR")
        #
        #     ax2.set_yscale("log")
        #     ax2.set_ylim(1, 3 * 10 ** 2)
        #     ax2.set_xlim(0, fx[-1] / (2 * NA))
        #     ax2.grid()
        #
        #     # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_s_polarized[:, :, int(val)])[q_axes[1] >= 0], label="S-polarized")
        #     # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_circular[:, :, int(val)])[q_axes[1] >= 0], label="Circular")
        #
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_s_polarized[:, int(N / 2), int(val)][fx >= 0], label="S-polarized")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_circular[:, int(N / 2), int(val)][fx >= 0], label="Circular")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_seven_waves[:, int(N / 2), int(val)][fx >= 0], label="7 waves")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[:, int(N / 2), int(val)][fx >= 0], label="3 waves")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0], label="Widefield")
        #     ax2.legend()
        #     ax2.set_aspect(1. / ax2.get_data_ratio())
        #
        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, 100)  # slider properties
        # slider_ssnr.on_changed(update1)

        ax1.legend()
        # ax2.legend()
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR_conf_version.png'.format(two_NA_fz[arg]))
        plt.show()
    def test_compare_SSNR_weird_configurations(self):
        theta = np.pi / 4
        NA = np.sin(theta)
        max_r = 10
        max_z = 40
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

        arg = N // 2 - 24
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (2 * NA)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = Lens()
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SNRCalculator(illumination_two_triangles_not_rotated, optical_system)
        SSNR_ttnr = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_ttnr_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_ttnr)
        volume_ttnr = np.sum(SSNR_ttnr)
        volume_ttnr_a = noise_estimator.compute_analytic_total_ssnr()

        noise_estimator.illumination = illumination_two_triangles_rotated
        SSNR_ttr= np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_ttr_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_ttr)
        volume_ttr = np.sum(np.abs(SSNR_ttr))
        volume_ttr_a = noise_estimator.compute_analytic_total_ssnr()

        noise_estimator.illumination = illumination_two_squares_not_rotated
        SSNR_tsnr = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_tsnr_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_tsnr)
        volume_tsnr = np.sum(np.abs(SSNR_tsnr))
        volume_tsnr_a = noise_estimator.compute_analytic_total_ssnr()

        noise_estimator.illumination = illumination_two_squares_rotated
        SSNR_tsr = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_tsr_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_tsr)
        volume_tsr = np.sum(np.abs(SSNR_tsr))
        volume_tsr_a = noise_estimator.compute_analytic_total_ssnr()

        noise_estimator.illumination = illumination_five_waves_two_angles
        SSNR_5w = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_5w_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_5w)
        volume_5w = np.sum(np.abs(SSNR_5w))
        volume_5w_a = noise_estimator.compute_analytic_total_ssnr()

        noise_estimator.illumination = illumination_widefield
        ssnr_widefield = np.abs(noise_estimator.compute_ssnr(q_axes))
        ssnr_widefield_ra = noise_estimator.ring_average_SSNR(q_axes, ssnr_widefield)
        volume_widefield = np.sum(np.abs(ssnr_widefield))
        volume_widefield_a = noise_estimator.compute_analytic_total_ssnr()

        print("Volume SSNR widefield = ", volume_widefield)
        print("Volume SSNR widefield_a = ", volume_widefield_a)
        print("Volume SSNR ttnr = ", volume_ttnr)
        print("Volume SSNR ttnr_a = ", volume_ttnr_a)
        print("Volume SSNR ttr = ", volume_ttr)
        print("Volume SSNR ttr_a = ", volume_ttr_a)
        print("Volume SSNR tsnr = ", volume_tsnr)
        print("Volume SSNR tsnr_a = ", volume_tsnr_a)
        print("Volume SSNR tsr = ", volume_tsr)
        print("Volume SSNR tsr_a = ", volume_tsr_a)
        print("Volume SSNR 5w = ", volume_5w)
        print("Volume SSNR 5w_a = ", volume_5w_a)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Ring averaged", fontsize=20, pad=15)
        ax1.set_xlabel(r"$f_r$", fontsize=20)
        ax1.set_ylabel(r"$SSNR_{ra}$", fontsize=20)
        ax1.set_yscale("log")
        ax1.set_ylim(1, 3 * 10 ** 2)
        ax1.set_xlim(0, fx[-1] / (2 * NA))
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=15)

        ax2.set_title("Slice $f_y$ = 0", fontsize=20)
        ax2.set_xlabel(r"$f_x$", fontsize=20)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 3 * 10 ** 2)
        ax2.set_xlim(0, fx[-1] / (2 * NA))
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=15)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_ttnr_ra[:, arg], label="two triangles not rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_ttr_ra[:, arg], label="two triangles rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_tsnr_ra[:, arg], label="two squares not rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_tsr_ra[:, arg], label="two squares rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_5w_ra[:, arg], label="fiwe waves two angles")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_ttr[:, int(N / 2), arg][fx >= 0], label="two triangles not rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_ttr[:, int(N / 2), arg][fx >= 0], label="two triangles rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_tsnr[:, int(N / 2), arg][fx >= 0], label="two squares not rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_tsr[:, int(N / 2), arg][fx >= 0], label="two squares rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_5w[:, int(N / 2), arg][fx >= 0], label="five waves two angles")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("Ring averaged", fontsize=30)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$SSNR$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 3 * 10 ** 2)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_ttnr_ra[:, int(val)], label="two triangles not rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_ttr_ra[:, int(val)], label="two triangles rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_tsnr_ra[:, int(val)], label="two squares not rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_tsr_ra[:, int(val)], label="two squares rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_5w_ra[:, int(val)], label="five waves two angles")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="widefield")

            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"SSNR")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10 ** 2)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_ttr[:, int(N / 2), int(val)][fx >= 0],
                     label="two triangles not rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_ttr[:, int(N / 2), int(val)][fx >= 0],
                     label="two triangles rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_tsnr[:, int(N / 2), int(val)][fx >= 0],
                     label="two squares not rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_tsr[:, int(N / 2), int(val)][fx >= 0],
                     label="two squares rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_5w[:, int(N / 2), int(val)][fx >= 0],
                     label="five waves two angles")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0],
                     label="widefield")
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, 50)  # slider properties
        # slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR_conf_version.png'.format(two_NA_fz[arg]))
        # plt.show()

    def testSSNRFromNumericalSpacialWaves(self):
        max_r = 10
        max_z = 40
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

        arg = N // 2  # - 24

        NA = np.sin(np.pi/3)
        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (2 * NA)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = Lens(alpha = np.pi/3)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        k = 2 * np.pi
        theta = np.pi/3
        vec_x = np.array((k * np.sin(np.pi/3), 0, k * np.cos(np.pi/3)))
        vec_mx = np.array((-k * np.sin(np.arcsin(5/7 * np.sin(np.pi/3))), 0, k * np.cos(np.arcsin(5/7 * np.sin(np.pi/3)))))
        ax_z = np.array((0, 0, 1))

        sources = [
            Sources.PlaneWave(0, 1, 0, 0, vec_x),
            Sources.PlaneWave(0, 1, 0, 0, vec_mx),
            Sources.PlaneWave(0, 1, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 2 * np.pi/3)),
            Sources.PlaneWave(0, 1, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 2 * np.pi/3)),
            Sources.PlaneWave(0, 1, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 4 * np.pi/3)),
            Sources.PlaneWave(0, 1, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 4 * np.pi/3)),

            Sources.PlaneWave(1, 1j, 0, 0, np.array((0, 10**-10, 2 * np.pi))),
        ]
        size = (2 * max_r, 2 * max_r, 2 * max_z)
        box = Box.Box(sources, size, N)
        box.compute_intensity_and_spacial_waves_numerically()
        iwaves = box.get_approximated_intensity_sources()
        illumination_2z = Illumination.init_from_list(iwaves, (k * np.sin(theta) / 14, k * np.sin(theta) * 3**0.5/ 14, k / 14), Mr = 1)
        illumination_2z.normalize_spacial_waves()
        noise_estimator = SNRCalculator(illumination_2z, optical_system)
        SSNR_2z = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_2z_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_2z)
        volume_2z = noise_estimator.compute_SSNR_volume(SSNR_2z, dV)
        b = 1/2**0.5
        sources = [
            Sources.PlaneWave(0, b, 0, 0, np.array((0, k * NA,k * np.cos(np.pi/3)))),
            Sources.PlaneWave(0, -b, 0, 0, np.array((0, -k * NA, k * np.cos(np.pi/3)))),
            Sources.PlaneWave(0, 1, 0, 0, np.array((0, 10 ** -10, k))),
        ]
        box = Box.Box(sources, size, N)
        box.compute_intensity_and_spacial_waves_numerically()
        iwaves = box.get_approximated_intensity_sources()
        il = Illumination.index_frequencies(iwaves, (10**10 , k * np.sin(theta), k * (1 - np.cos(theta))))
        illumination_3w = Illumination(il, Mr=5)
        illumination_3w.Mt = 1
        illumination_3w.normalize_spacial_waves()
        noise_estimator = SNRCalculator(illumination_3w, optical_system)
        SSNR_3w = np.abs(noise_estimator.compute_ssnr(q_axes))
        SSNR_3w_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_3w)
        volume_3w = noise_estimator.compute_SSNR_volume(SSNR_3w, dV)

        widefield = Illumination({
        (0, 0, 0) : Sources.IntensityPlaneWave(1, 0, np.array((0, 0, 0)))}, Mr=1)
        widefield.Mt = 1
        noise_estimator.illumination = widefield
        ssnr_widefield = np.abs(noise_estimator.compute_ssnr(q_axes))
        ssnr_widefield_ra= noise_estimator.ring_average_SSNR(q_axes, ssnr_widefield)
        volume_w = noise_estimator.compute_SSNR_volume(ssnr_widefield, dV)
        print("volume 3 waves = ", volume_3w)
        print("volume two triangles = ", volume_2z)
        print("volume widefield = ", volume_w)
        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Ring averaged", fontsize=20, pad=15)
        ax1.set_xlabel(r"$f_r$", fontsize=20)
        ax1.set_ylabel(r"$SSNR_{ra}$", fontsize=20)
        ax1.set_yscale("log")
        ax1.set_ylim(1, 3 * 10 ** 2)
        ax1.set_xlim(0, fx[-1] / (2 * NA))
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=15)

        ax2.set_title("Slice $f_y$ = 0", fontsize=20)
        ax2.set_xlabel(r"$f_x$", fontsize=20)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 3 * 10 ** 2)
        ax2.set_xlim(0, fx[-1] / (2 * NA))
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=15)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_2z_ra[:, arg], label="2 triangles")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3w_ra[:, arg], label="3 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_2z[:, int(N / 2), arg][fx >= 0], label="5 waves, 2 z angles")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3w[:, int(N / 2), arg][fx >= 0], label="3 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("Ring averaged", fontsize=30)
            ax1.set_xlabel(r"$f_r$",  fontsize=25)
            ax1.set_ylabel(r"$SSNR$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 3 * 10 ** 2)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_2z_ra[:, int(val)], label=" 5 waves, 2 z angles")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3w_ra[:, int(val)], label=" 3 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="widefield")

            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"SSNR")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10 ** 2)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_2z[:, int(N / 2), int(val)][fx >= 0], label="5 waves, 2z angles")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3w[:, int(N / 2), int(val)][fx >= 0], label="3 waves")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0], label="widefield")

            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, 100)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        # fig.savefig(
        #     '/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR_conf_version.png'.format(
        #         fz[arg]))
        plt.show()

    def testFourWavesVSFiveWaves(self):
        max_r = 10
        max_z = 20
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

        arg = N // 2  # - 24

        NA = np.sin(2 * np.pi / 5)
        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (2 * NA)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = Lens(alpha= 2 * np.pi / 5)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        angle = 8 * np.pi / 20
        a = 0.9
        b, c = 1, 1

        k = 2 * np.pi
        il4 = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(angle, a, b, c, 0)
        noise4 = SNRCalculator(il4, optical_system)
        SSNR4 = noise4.SSNR(q_axes)

        il5 = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(angle, a, b, c, 1)
        noise5 = SNRCalculator(il5, optical_system)
        SSNR5 = noise5.SSNR(q_axes)
        ra4 = noise4.ring_average_SSNR(q_axes, SSNR4)
        ra5 = noise5.ring_average_SSNR(q_axes, SSNR5)
        volume4 = noise4.compute_SSNR_volume(SSNR4, 1)
        volume5 = noise5.compute_SSNR_volume(SSNR5, 1)

        print("volume 4 waves = ", volume4)
        print("volume 5 waves = ", volume5)
        # print("volume widefield = ", volume_w)
        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_title("Ring averaged", fontsize=20, pad=15)
        ax1.set_xlabel(r"$f_r$", fontsize=20)
        ax1.set_ylabel(r"$SSNR_{ra}$", fontsize=20)
        ax1.set_yscale("log")
        ax1.set_ylim(1, 3 * 10 ** 2)
        ax1.set_xlim(0, fx[-1] / (2 * NA))
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=15)

        ax2.set_title("Slice $f_y$ = 0", fontsize=20)
        ax2.set_xlabel(r"$f_x$", fontsize=20)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 3 * 10 ** 2)
        ax2.set_xlim(0, fx[-1] / (2 * NA))
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=15)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ra4[:, arg], label="4 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ra5[:, arg], label="5 waves")
        # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR4[:, int(N / 2), arg][fx >= 0], label="4 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR5[:, int(N / 2), arg][fx >= 0], label="5 waves")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("fz = {:.2f}".format(two_NA_fz[int(val)]), fontsize=30)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$SSNR$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 3 * 10 ** 2)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ra4[:, int(val)], label="4 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ra5[:, int(val)], label="5 waves")
            # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="widefield")

            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_title("Slice $f_y$ = {:.2f}".format(two_NA_fy[int(val)]))
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"SSNR")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10 ** 2)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR4[:, int(val), arg][fx >= 0], label="4 waves")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR5[:, int(val), arg][fx >= 0], label="5 waves")
            # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0],
            #          label="widefield")

            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, 100)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        # fig.savefig(
        #     '/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR_conf_version.png'.format(
        #         fz[arg]))
        plt.show()

