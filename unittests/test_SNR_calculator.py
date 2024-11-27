import sys
import Box
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
from SSNRCalculator import SSNR3dSIM2dShifts, SSNR2dSIM, SSNRWidefield, SSNRConfocal
from OpticalSystems import Lens3D, Lens2D
import stattools
from Sources import IntensityPlaneWave
import tqdm
sys.path.append('../')


configurations = BFPConfiguration(refraction_index=1)
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

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        waves = configurations.get_4_oblique_s_waves_and_circular_normal(np.pi/4, 1, 1)

        illumination_polarized = Illumination(waves)
        optical_system_fourier = Lens3D(interpolation_method="Fourier")
        optical_system_fourier.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N),
                                                   apodization_function="Sine")
        optical_system_linear = Lens3D()
        optical_system_linear.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N),
                                                  apodization_function="Sine")
        wavevectors = illumination_polarized.get_all_wavevectors()

        noise_estimator = SSNR3dSIM2dShifts(illumination_polarized, optical_system_fourier)

        begin_shifted = time.time()
        optical_system_fourier.prepare_Fourier_interpolation(wavevectors)
        ssnr_F = np.abs(noise_estimator.compute_ssnr())
        end_shifted = time.time()
        print("fourier time is ", end_shifted - begin_shifted)

        noise_estimator.optical_system = optical_system_linear
        begin_interpolation = time.time()
        ssnr_L = np.abs(noise_estimator.compute_ssnr())
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

    def test_ssnr2d(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        N = 51
        max_r = N // 2 * dx
        x = np.linspace(-max_r, max_r, N)
        fr = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        NA = np.sin(alpha)
        illumination = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3)
        spacial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spacial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination.spacial_shifts = spacial_shifts_conventional2d


        fx_normalized = fr / (2 * NA)
        fy_normalized = fr / (2 * NA)
        fx_normalized = fr / (1 - np.cos(theta))
        # print(fw2z_illumination)
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 0, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        # illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        # illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        # illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        # illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        # illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()

        illumination_list = {
            illumination_s_polarized: ("4 s-polarized oblique waves", "4s1c"),
            illumination_circular: ("4 circularly polarized oblique waves", "5c"),
            illumination_seven_waves: ("6 s-polarized oblique waves", "6s1c"),
            illumination_3waves: ("State of Art SIM", "state_of_art"),
            illumination_widefield: ("Widefield", "widefield"),
            # illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            # illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            # illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            # illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            # illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        optical_system = Lens2D(alpha=NA)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r), N))

        noise_estimator_wf = SSNR2dSIM(illumination_widefield, optical_system)
        ssnr_wf = np.abs(noise_estimator_wf.compute_ssnr())

        for illumination in illumination_list:
            noise_estimator = SSNR2dSIM(illumination, optical_system)

            ssnr = np.abs(noise_estimator.compute_ssnr())
            scaling_factor = 10 ** 4
            # ssnr_diff = ssnr - ssnr_wf
            # ssnr_scaled = 1 + scaling_factor * ssnr
            ssnr_ring_averaged = noise_estimator.ring_average_ssnr()
            # ssnr_ra_scaled = 1 + scaling_factor * ssnr_ring_averaged

            Fx, Fy = np.meshgrid(fr, fr)
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            fig.suptitle(illumination_list[illumination][0], fontsize=30)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.1,
                                hspace=0)
            ax1 = fig.add_subplot(111)
            ax1.tick_params(labelsize=20)
            # ax1.set_title(title)
            ax1.set_xlabel("$f_z$", fontsize=25)
            ax1.set_ylabel("$f_y$", fontsize=25)
            mp1 = ax1.imshow(ssnr, extent=(fr[0] / (2 * NA), fr[-1] / (2 * NA), fr[0] / (2 * NA), fr[-1] / (2 * NA)), norm=colors.LogNorm())
            # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
            # cb1.set_label("$1 + 10^8$ ssnr")
            ax1.set_aspect(1. / ax1.get_data_ratio())

            plt.show()
    def test_ssnr(self):
        theta = np.pi/4
        NA = np.sin(theta)
        max_r = 8
        max_z = 20
        N = 101
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.linspace(-max_r, max_r - dx, N)
        y = np.copy(x)
        z = np.arange(-max_z, max_z - dz, N)
        print(x)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)
        print(fx)

        fx_normalized = fx / (2 * NA)
        fy_normalized = fy / (2 * NA)
        fx_normalized = fz / (1 - np.cos(theta))
        # print(fw2z_illumination)
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1/2**0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 1, Mt=1)
        # illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        # illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        # illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        # illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        # illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()


        illumination_list = {
            illumination_s_polarized : ("4 s-polarized oblique waves", "4s1c"),
            illumination_circular : ("4 circularly polarized oblique waves", "5c"),
            illumination_seven_waves : ("6 s-polarized oblique waves", "6s1c"),
            illumination_3waves : ("State of Art SIM", "state_of_art"),
            illumination_widefield : ("Widefield", "widefield"),
            # illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            # illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            # illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            # illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            # illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        optical_system = Lens3D(alpha=NA)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        noise_estimator_wf = SSNR3dSIM2dShifts(illumination_widefield, optical_system)
        ssnr_wf = np.abs(noise_estimator_wf.compute_ssnr())

        for illumination in illumination_list:
            noise_estimator = SSNR3dSIM2dShifts(illumination, optical_system)

            ssnr = np.abs(noise_estimator.compute_ssnr())
            scaling_factor = 10**4
            # ssnr_diff = ssnr - ssnr_wf
            ssnr_scaled = 1 + scaling_factor * ssnr
            ssnr_ring_averaged = noise_estimator.ring_average_ssnr()
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
            mp2 = ax2.imshow(ssnr_scaled[:, :, int(N/2)], extent=(fy[0]/(2 * NA), fy[-1]/(2 * NA), fx[0]/(2 * NA), fx[-1]/(2 * NA)), norm=colors.LogNorm())
            cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=20)
            cb2.set_label("$1 + 10^4$ ssnr", fontsize = 25)
            ax2.set_aspect(1. / ax2.get_data_ratio())

            # fig.savefig(f'{path_to_figures}'
            #          + illumination_list[illumination][1] + '_waves_ssnr.png')

            def update1(val):
                ax1.set_title("ssnr, fy = {:.2f}, ".format(fy[int(val)]) + "$\\frac{2NA}{\\lambda}$")
                ax1.set_xlabel("fz, $\lambda^{-1}$")
                ax1.set_ylabel("fx, $\lambda^{-1}$")
                Z = (ssnr_scaled[:, int(val), :])
                mp1.set_data(Z)
                mp1.set_clim(vmin=Z.min(), vmax=Z.max())
                ax1.set_aspect(1. / ax1.get_data_ratio())
                plt.draw()

                ax2.set_title("ssnr, fz = {:.2f}, ".format(fz[int(val)]) + "$\\frac{2NA}{\\lambda}$")
                ax2.set_xlabel("fz, $\lambda^{-1}$")
                ax2.set_ylabel("fx, $\lambda^{-1}$")
                Z = (ssnr_scaled[:, :, int(val)])
                mp2.set_data(Z)
                mp2.set_clim(vmin=Z.min(), vmax=Z.max())
                ax2.set_aspect(1. / ax2.get_data_ratio())
                plt.draw()

            slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
            slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
            slider_ssnr.on_changed(update1)

            plt.show()


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
            illumination_s_polarized : ("Square SIM, s-polarized", "4s1c"),
            illumination_circular : ("Square SIM, circular", "5c"),
            illumination_seven_waves : ("Hexagonal SIM", "6s1c"),
            illumination_3waves : ("State of Art SIM", "state_of_art"),
            illumination_widefield : ("Widefield", "widefield"),
            # illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            # illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            # illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            # illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            # illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        optical_system = Lens3D(alpha=np.pi/4)
        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        for illumination in illumination_list:
            noise_estimator = SSNR3dSIM2dShifts(illumination, optical_system)
            ssnr = np.abs(noise_estimator.compute_ssnr())
            ssnr_scaled = np.log10(1 + 10**8 * np.abs(ssnr))

            noise_estimator.illumination = configurations.get_widefield()
            ssnr_widefield = np.abs(noise_estimator.compute_ssnr())
            ssnr_widefield_scaled = np.log10(1 + 10**8 * np.abs(ssnr_widefield))

            constant_value = 0.2
            verts, faces, _, _ = skimage.measure.marching_cubes(ssnr_scaled, level=constant_value)
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
            ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=18, labelpad=15)
            ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=18, labelpad=15)
            ax1.set_zlabel(r"$f_z, \frac{NA(1 - cos(\alpha)}{\lambda}$", fontsize=18, labelpad=15)
            ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],  alpha=0.7)
            ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2],  alpha=1, color="red")

            ax1.set_xlim(N // 8, 7 * N // 8)
            ax1.set_ylim(N // 8, 7 * N // 8)
            ax1.set_zlim(0, N)
            xticks = np.round((ax1.get_xticks() - N / 2) * dfx / (2 * NA), 2)
            yticks = np.round((ax1.get_yticks() - N / 2) * dfy / (2 * NA), 2)
            zticks = np.round((ax1.get_zticks() - N / 2) * dfz / (1 - np.cos(NA)), 2)
            ax1.set_xticklabels(xticks)
            ax1.set_yticklabels(yticks)
            ax1.set_zticklabels(zticks)
            ax1.tick_params(labelsize=15)
            ax1.view_init(elev=20, azim=45)
            plt.draw()
            # fig.savefig(f'{path_to_figures}'
            #          + illumination_list[illumination][1] + '_waves_ssnr_isosurface.png')

            def update1(val):
                ax1.clear()
                ax1.set_title(illumination_list[illumination][0] + "\n1 + $10^8 ssnr$ = {:.2f}".format(val/15), fontsize=18)
                ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=18, labelpad=15)
                ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=18, labelpad=15)
                ax1.set_zlabel(r"$f_z, \frac{NA(1 - cos(\alpha)}{\lambda}$", fontsize=18, labelpad=15)
                verts, faces, _, _ = skimage.measure.marching_cubes(ssnr_scaled, level=val/15)
                w_verts, w_faces, _, _ = skimage.measure.marching_cubes(ssnr_widefield_scaled, level=val/15)
                ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, color="blue")
                ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2], alpha=1, color="red")

                ax1.set_xlim(N//8, 7 * N//8)
                ax1.set_ylim(N//8, 7 * N//8)
                ax1.set_zlim(0, N)

                xticks = np.round((ax1.get_xticks() - N / 2) * dfx / (2 * NA), 2)
                yticks = np.round((ax1.get_yticks() - N / 2) * dfy / (2 * NA), 2)
                zticks = np.round((ax1.get_zticks() - N / 2) * dfz / (1 - np.cos(NA)), 2)
                ax1.set_xticklabels(xticks)
                ax1.set_yticklabels(yticks)
                ax1.set_zticklabels(zticks)
                plt.draw()
            def update2(val):
                ax1.clear()
                ax1.set_title(illumination_list[illumination][0] + "\n1 + $10^8 ssnr$ = {:.2f}".format(0.2), fontsize=18)
                ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=18, labelpad=15)
                ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=18, labelpad=15)
                ax1.set_zlabel(r"$f_z, \frac{NA(1 - cos(\alpha))}{\lambda}$", fontsize=18, labelpad=15)
                verts, faces, _, _ = skimage.measure.marching_cubes(ssnr_scaled, level=0.2)
                w_verts, w_faces, _, _ = skimage.measure.marching_cubes(ssnr_widefield_scaled, level=0.2)
                ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, color="blue")
                ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2], alpha=1, color="red")
                if val <= 20:
                    ax1.view_init(elev=20, azim=15 * val)
                elif val <=30:
                    ax1.view_init(elev=10 * (val - 20), azim=15)
                else:
                    ax1.view_init(elev=(40 - val) * (val - 20), azim = 15)
                ax1.set_xlim(N//8, 7 * N//8)
                ax1.set_ylim(N//8, 7 * N//8)
                ax1.set_zlim(0, N)

                xticks = np.round((ax1.get_xticks() - N / 2) * dfx / (2 * NA), 2)
                yticks = np.round((ax1.get_yticks() - N / 2) * dfy / (2 * NA), 2)
                zticks = np.round((ax1.get_zticks() - N / 2) * dfz / (1 - np.cos(NA)), 2)
                ax1.set_xticklabels(xticks)
                ax1.set_yticklabels(yticks)
                ax1.set_zticklabels(zticks)
                plt.draw()



            ani = FuncAnimation(fig, update2, frames=range(1, 40), repeat=False, interval=300)
            ani.save(path_to_animations+
                     'Animation_' + illumination_list[illumination][1] + '_waves_ssnr_different_angles.mp4', writer="ffmpeg")
            # plt.show()

    def test_ring_averaged_ssnr(self):
        alpha = 4 * np.pi / 5
        theta = 0.8 * alpha
        nmedium = 1.5
        nobject = 1.5
        NA = nmedium * np.sin(alpha)
        fz_max_diff = nmedium * (1 - np.cos(alpha))
        dx = 1 / (8 * NA)
        dy = dx
        dz = 1 / (4 * fz_max_diff)
        N = 101
        max_r = N // 2 * dx
        max_z = N // 2 * dz
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
        scaled_fz = fz / fz_max_diff

        multiplier = 10 ** 3
        ylim = 10**2

        optical_system = Lens3D(alpha=alpha, refractive_index_sample=nobject, refractive_index_medium=nmedium)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.55, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_widefield = SSNR3dSIM2dShifts(illumination_widefield, optical_system)
        noise_estimator_widefield.compute_ssnr()
        ssnr_widefield = noise_estimator_widefield.ssnr
        ssnr_widefield_ra = noise_estimator_widefield.ring_average_ssnr()

        noise_estimator_confocal = SSNRConfocal(optical_system)
        ssnr_confocal = noise_estimator_confocal.ssnr
        ssnr_confocal_ra = noise_estimator_confocal.ring_average_ssnr()

        noise_estimator = SSNR3dSIM2dShifts(illumination_s_polarized, optical_system)
        # noise_estimator_true = SSNR3dSIM3dShifts(illumination_s_polarized, optical_system)
        ssnr_s_polarized = np.abs(noise_estimator.compute_ssnr())
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnr()
        volume_squareSP = noise_estimator.compute_ssnr_volume()
        measure_s_polarized, threshold_s_polarized = noise_estimator.compute_ssnr_waterline_measure()
        entropy_s_polarized = noise_estimator.compute_true_ssnr_entropy()

        noise_estimator.illumination = illumination_circular
        ssnr_circular = np.abs(noise_estimator.compute_ssnr())
        ssnr_circular_ra = noise_estimator.ring_average_ssnr()
        volume_circular = noise_estimator.compute_ssnr_volume()
        measure_circular, threshold_circular = noise_estimator.compute_ssnr_waterline_measure()
        entropy_circular = noise_estimator.compute_true_ssnr_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnr()
        volume_hexagonal = noise_estimator.compute_ssnr_volume()
        measure_seven_waves, threshold7waves = noise_estimator.compute_ssnr_waterline_measure()
        entropy_seven_waves = noise_estimator.compute_true_ssnr_entropy()
        #
        noise_estimator.illumination = illumination_3waves
        # noise_estimator_true.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_3waves_ra = noise_estimator.ring_average_ssnr()
        # ssnr_3waves_true = np.abs(noise_estimator_true.compute_ssnr())
        # ssnr_3waves_true_ra = noise_estimator_true.ring_average_ssnr()
        volume_conventional = noise_estimator.compute_ssnr_volume()
        volume_a_3waves = noise_estimator.compute_analytic_ssnr_volume()
        measure_3waves, threshold3waves = noise_estimator.compute_ssnr_waterline_measure()
        entropy_3waves = noise_estimator.compute_true_ssnr_entropy()

        # noise_estimator.illumination = illumination_widefield
        # ssnr_widefield = np.abs(noise_estimator.compute_ssnr())
        # ssnr_widefield_ra = noise_estimator.ring_average_ssnr()
        # # volume_widefield = noise_estimator.compute_ssnr_volume()
        # measure_widefield, _ = noise_estimator.compute_ssnr_waterline_measure()


        # print("Volume ssnr widefield = ", volume_widefield)
        # print("Measure ssnr widefield = ", measure_widefield)
        #
        # print("Volume ssnr s_polarized = ", volume_squareSP)
        # print("Measure ssnr s_polarized = ", measure_s_polarized, threshold_s_polarized)
        # print("Entropy ssnr s_polarized = ", entropy_s_polarized)

        # print("Volume ssnr 3waves = ", volume_conventional)
        # print("Measure ssnr 3waves = ", measure_3waves, threshold3waves)
        # print("Entropy ssnr 3waves = ", entropy_3waves)
        #
        # print("Volume ssnr seven_waves = ", volume_hexagonal)
        # print("Measure ssnr seven_waves = ", measure_seven_waves, threshold7waves)
        # print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        #
        # print("Volume ssnr circular = ", volume_circular)
        # print("Measure ssnr circular = ", measure_circular, threshold_circular)
        # print("Entropy ssnr circular = ", entropy_circular)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        # fig.suptitle("Ring averaged SSNR for different configurations", fontsize=30)

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # ax1.set_title("Projective 3D SIM anisotropy \n $f_z = ${:.1f}".format(two_NA_fz[arg]) + "$(\\frac{n - \sqrt{n^2 - NA^2}}{\lambda})$", fontsize=25, pad=15)
        ax1.set_title("$f_z = ${:.1f}".format(scaled_fz[arg]) + "$\; [\\frac{n - \sqrt{n^2 - NA^2}}{\lambda}]$", fontsize=25, pad=15)
        ax1.set_xlabel(r"$f_r \; [\frac{2NA}{\lambda}]$", fontsize=25)
        ax1.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax1.set_yscale("log")
        ax1.set_ylim(1, ylim)
        ax1.set_xlim(0, two_NA_fx[-1])
        ax1.grid(which = 'major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=20)

        # ax2.set_title("Slice $f_y$ = 0", fontsize=25)
        ax2.set_title("$f_z = ${:.1f}".format(scaled_fz[arg//2]) + "$\; [\\frac{n - \sqrt{n^2 - NA^2}}{\lambda}]$", fontsize=25, pad=15)
        ax2.set_xlabel(r"$f_r \; [\frac{2NA}{\lambda}]$", fontsize=25)
        ax2.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax2.set_yscale("log")
        ax2.set_ylim(1, ylim)
        ax2.set_xlim(0, two_NA_fx[-1])
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=20)

        # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, arg], label="Square SIM, radial average")
        # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[:, int(N / 2), arg][fy >= 0], label="Square SIM, $f_y=0$")
        # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[int(N/2), :, arg][fx >= 0], label="Square SIM, $f_x=0$")
        # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="Widefield")
        # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves_true_ra[:, arg], label="State Of Art SIM, True 3D SIM")
        # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves_ra[:, arg], label="State Of Art SIM, Projective 3D SIM")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, arg],      label="Conventional")
        # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves[arg, :, arg][fy >= 0],      label="test")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_circular_ra[:, arg],    label="SquareC")
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, arg], label="SquareL")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, arg], label="Hexagonal")
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg],   label="Widefield")
        # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_confocal_ra[:, arg],    label="Confocal")

        # ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
        # ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
        # ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
        # ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves_ra[:, arg//2],      label="Conventional")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_circular_ra[:, arg//2],    label="SquareC")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, arg//2], label="SquareL")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, arg//2], label="Hexagonal")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg//2],   label="Widefield")
        # ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_confocal[:, arg,  arg//2][fx >= 0],      label="Confocal")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            # ax1.set_title("Comparison of 3D SIM modalities\n $f_z = ${:.1f}".format(two_NA_fz[arg]) + "$(\\frac{n - \sqrt{n^2 - NA^2}}{\lambda})$", fontsize=25, pad=15)
            # ax1.set_title("Directional anisotropies in Projective 3D SIM", fontsize = 30)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$ssnr$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1,  10**4)
            ax1.set_xlim(0, fx[-1]/(2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, int(val)], label="SquareL")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_circular_ra[:, int(val)],    label="SquareC")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, int(val)],      label="Conventional")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)],   label="Widefield")
            # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_confocal_ra[:, int(val)],         label="Confocal")

            # ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
            # ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
            # ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
            # ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')
            # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[:, int(N/2), int(val)][fy >= 0], label="Square SIM, $f_y=0$")
            # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[int(N/2), :, int(val)][fx >= 0], label="Square SIM, $f_x=0$")
            # ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N/2), int(val)][fx >= 0], label="Widefield")
            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            # ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"ssnr")

            ax2.set_yscale("log")
            # ax2.set_ylim(1, 3 * 10**2)
            ax2.set_xlim(0, fx[-1]/(2 * NA))
            ax2.grid()

            # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(ssnr_s_polarized[:, :, int(val)])[q_axes[1] >= 0], label="S-polarized")
            # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(ssnr_circular[:, :, int(val)])[q_axes[1] >= 0], label="Circular")

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[:, int(val), int(N / 2)][fx >= 0], label="S-polarized")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_circular[:, int(val), int(N / 2)][fx >= 0],    label="Circular"   )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_seven_waves[:, int(val), int(N / 2)][fx >= 0],    label="7 waves"   )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves[:, int(val),  int(N / 2)][fx >= 0],      label="3 waves"    )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(val), int(N / 2)][fx >= 0],   label="Widefield"  )
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_confocal[:, int(val), int(N / 2)][fx >= 0], label="Confocal")
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        # fig.savefig(f'{path_to_figures}comparison_of_3d_SIM_modalities_fz={:.2f}_optimal_b.png'.format(two_NA_fz[arg]))
        # fig.savefig(f'{path_to_figures}ring_averaged_ssnr')
        plt.show()

    def test_compare_ssnr_weird_configurations(self):
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

        optical_system = Lens3D()
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        # illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 0,  Mt=32, mutually_rotated=False)
        # illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        # illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        # illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNR3dSIM2dShifts(illumination_five_waves_two_angles, optical_system)
        # ssnr_ttnr = np.abs(noise_estimator.compute_ssnr())
        # ssnr_ttnr_ra = noise_estimator.ring_average_ssnr()
        # volume_ttnr = np.sum(ssnr_ttnr)
        # volume_ttnr_a = noise_estimator.compute_analytic_total_ssnr()

        # noise_estimator.illumination = illumination_two_triangles_rotated
        # ssnr_ttr= np.abs(noise_estimator.compute_ssnr())
        # ssnr_ttr_ra = noise_estimator.ring_average_ssnr()
        # volume_ttr = np.sum(np.abs(ssnr_ttr))
        # volume_ttr_a = noise_estimator.compute_analytic_total_ssnr()
        #
        # noise_estimator.illumination = illumination_two_squares_not_rotated
        # ssnr_tsnr = np.abs(noise_estimator.compute_ssnr())
        # ssnr_tsnr_ra = noise_estimator.ring_average_ssnr()
        # volume_tsnr = np.sum(np.abs(ssnr_tsnr))
        # volume_tsnr_a = noise_estimator.compute_analytic_total_ssnr()
        #
        # noise_estimator.illumination = illumination_two_squares_rotated
        # ssnr_tsr = np.abs(noise_estimator.compute_ssnr())
        # ssnr_tsr_ra = noise_estimator.ring_average_ssnr()
        # volume_tsr = np.sum(np.abs(ssnr_tsr))
        # volume_tsr_a = noise_estimator.compute_analytic_total_ssnr()
        #
        # noise_estimator.illumination = illumination_five_waves_two_angles
        ssnr_5w = np.abs(noise_estimator.compute_ssnr())
        ssnr_5w_ra = noise_estimator.ring_average_ssnr()
        volume_5w = np.sum(np.abs(ssnr_5w))
        volume_5w_a = noise_estimator.compute_analytic_total_ssnr()

        # noise_estimator.illumination = illumination_widefield
        # ssnr_widefield = np.abs(noise_estimator.compute_ssnr())
        # ssnr_widefield_ra = noise_estimator.ring_average_ssnr()
        # volume_widefield = np.sum(np.abs(ssnr_widefield))
        # volume_widefield_a = noise_estimator.compute_analytic_total_ssnr()

        # print("Volume ssnr widefield = ", volume_widefield)
        # print("Volume ssnr widefield_a = ", volume_widefield_a)
        # print("Volume ssnr ttnr = ", volume_ttnr)
        # print("Volume ssnr ttnr_a = ", volume_ttnr_a)
        # print("Volume ssnr ttr = ", volume_ttr)
        # print("Volume ssnr ttr_a = ", volume_ttr_a)
        # print("Volume ssnr tsnr = ", volume_tsnr)
        # print("Volume ssnr tsnr_a = ", volume_tsnr_a)
        # print("Volume ssnr tsr = ", volume_tsr)
        # print("Volume ssnr tsr_a = ", volume_tsr_a)
        # print("Volume ssnr 5w = ", volume_5w)
        # print("Volume ssnr 5w_a = ", volume_5w_a)

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
        ax1.set_ylabel(r"$ssnr_{ra}$", fontsize=20)
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
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_ttnr_ra[:, arg], label="two triangles not rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_ttr_ra[:, arg], label="two triangles rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_tsnr_ra[:, arg], label="two squares not rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_tsr_ra[:, arg], label="two squares rotated")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_5w_ra[:, arg], label="fiwe waves two angles")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_ttnr[:, int(N / 2), arg][fx >= 0], label="two triangles not rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_ttr[:, int(N / 2), arg][fx >= 0], label="two triangles rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_tsnr[:, int(N / 2), arg][fx >= 0], label="two squares not rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_tsr[:, int(N / 2), arg][fx >= 0], label="two squares rotated")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_5w[:, int(N / 2), arg][fx >= 0], label="five waves two angles")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("Ring averaged", fontsize=30)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$ssnr$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 3 * 10 ** 2)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_ttnr_ra[:, int(val)], label="two triangles not rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_ttr_ra[:, int(val)], label="two triangles rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_tsnr_ra[:, int(val)], label="two squares not rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_tsr_ra[:, int(val)], label="two squares rotated")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_5w_ra[:, int(val)], label="five waves two angles")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="widefield")

            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"ssnr")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10 ** 2)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_ttr[:, int(N / 2), int(val)][fx >= 0],
                     label="two triangles not rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_ttr[:, int(N / 2), int(val)][fx >= 0],
                     label="two triangles rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_tsnr[:, int(N / 2), int(val)][fx >= 0],
                     label="two squares not rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_tsr[:, int(N / 2), int(val)][fx >= 0],
                     label="two squares rotated")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_5w[:, int(N / 2), int(val)][fx >= 0],
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
        # fig.savefig(f'{path_to_figures}fz={:.2f}_compare_ssnr_conf_version.png'.format(two_NA_fz[arg]))
        # plt.show()

    def test_ssnr_color_maps(self):
        alpha = np.pi/4
        theta = np.pi/4
        NA = np.sin(alpha)
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 101
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

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , N)

        # print(fw2z_illumination)
        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.55, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        illumination_two_triangles_not_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1,  Mt=32, mutually_rotated=False)
        illumination_two_triangles_rotated = configurations.get_two_oblique_triangles_and_one_normal_wave(theta, 5/7, 1, 1, Mt=32, mutually_rotated=True)
        illumination_two_squares_not_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=64, mutually_rotated=False)
        illumination_two_squares_rotated = configurations.get_two_oblique_squares_and_one_normal_wave(theta, 1/2**0.5, 1, 1, Mt=1, mutually_rotated=True)
        illumination_five_waves_two_angles = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(theta, 5/7, 1, 1)


        illumination_list = {
            illumination_s_polarized : ("Square SIM, s-polarized waves", "squareL"),
            illumination_circular : ("Square SIM, circular", "squareC"),
            illumination_seven_waves : ("Hexagonal SIM", "hexagonal"),
            illumination_3waves : ("Conventional SIM", "conventional"),
            illumination_widefield : ("Widefield", "widefield"),
            # illumination_two_triangles_rotated : ("Two triangles crossed", "2tr"),
            # illumination_two_triangles_not_rotated : ("Two triangles parallel", "2tnr"),
            # illumination_two_squares_rotated : ("Two squares crossed", "2sr"),
            # illumination_two_squares_not_rotated : ("Two squares parallel", "2snr"),
            # illumination_five_waves_two_angles : ("State of Art SIM with 5 waves", "state_of_art_5")
        }

        optical_system = Lens3D(alpha=theta)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        for illumination in illumination_list:
            noise_estimator = SSNR3dSIM2dShifts(illumination, optical_system)

            ssnr = np.abs(noise_estimator.compute_ssnr())
            scaling_factor = 10**8
            ssnr_scaled = 1 + scaling_factor * ssnr
            ssnr_ring_averaged = noise_estimator.ring_average_ssnr()
            ssnr_ra_scaled = 1 + scaling_factor * ssnr_ring_averaged

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
            ax1.set_xlabel("$f_y \; [\\frac{2NA}{\lambda}]$", fontsize = 25)
            ax1.set_ylabel("$f_x \;  [\\frac{2NA}{\lambda}]$", fontsize = 25)
            mp1 = ax1.imshow(ssnr_scaled[:, :, N//2], extent=(-2, 2, -2, 2), norm=colors.LogNorm())
            # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
            # cb1.set_label("$1 + 10^8$ ssnr")
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2 = fig.add_subplot(122, sharey=ax1)
            ax2.set_xlabel("$f_z \; [\\frac{n - \sqrt{n^2 - NA^2}}{\lambda}]$", fontsize = 25)
            ax2.tick_params(labelsize = 20)
            ax2.tick_params('y', labelleft=False)
            # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
            mp2 = ax2.imshow(ssnr_scaled[N//2, :, :], extent=(-2, 2, -2, 2), norm=colors.LogNorm())
            # mp2 = ax2.imshow(ssnr_ra_scaled[:, :].T, extent=(0, fy[-1]/(2 * NA), fz[0]/(2 * NA), fz[-1]/(2 * NA)), norm=colors.LogNorm())
            cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=20)
            cb2.set_label("$1 + 10^8$ ssnr", fontsize = 25)
            ax2.set_aspect(1. / ax2.get_data_ratio())


            # fig.savefig(f'{path_to_figures}'
            #          + illumination_list[illumination][1] + '_ssnr.png')

            # def update1(val):
            #     ax1.set_title("ssnr, fy = {:.2f}, ".format(fy[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            #     ax1.set_xlabel("fz, $\lambda^{-1}$")
            #     ax1.set_ylabel("fx, $\lambda^{-1}$")
            #     Z = (ssnr_scaled[:, int(val), :])
            #     mp1.set_data(Z)
            #     mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            #     ax1.set_aspect(1. / ax1.get_data_ratio())
            #     plt.draw()
            #
            #     ax2.set_title("ssnr, fz = {:.2f}, ".format(fz[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            #     ax2.set_xlabel("fz, $\lambda^{-1}$")
            #     ax2.set_ylabel("fx, $\lambda^{-1}$")
            #     Z = (ssnr_scaled[:, :, int(val)])
            #     mp2.set_data(Z)
            #     mp2.set_clim(vmin=Z.min(), vmax=Z.max())
            #     ax2.set_aspect(1. / ax2.get_data_ratio())
            #     plt.draw()
            #
            # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
            # slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
            # slider_ssnr.on_changed(update1)

            # plt.show()

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
        scaled_fz = fz / (1 - np.cos(theta))

        optical_system = Lens3D()
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1 / 2 ** 0.5, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator = SSNR3dSIM2dShifts(illumination_s_polarized, optical_system)
        ssnr_s_polarized = np.abs(noise_estimator.compute_ssnr())
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnr()
        # volume_squareSP = noise_estimator.compute_ssnr_volume(ssnr_s_polarized, dV)
        # measure_s_polarized, threshold_s_polarized = noise_estimator.compute_ssnr_waterline_measure(ssnr_s_polarized)

        noise_estimator.illumination = illumination_circular
        ssnr_circular = np.abs(noise_estimator.compute_ssnr())
        ssnr_circular_ra = noise_estimator.ring_average_ssnr()
        # volume_circular = noise_estimator.compute_ssnr_volume(ssnr_circular, dV)
        # measure_circular, threshold_circular = noise_estimator.compute_ssnr_waterline_measure(ssnr_circular)

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnr()
        # volume_hexagonal = noise_estimator.compute_ssnr_volume(ssnr_seven_waves, dV)
        # measure_seven_waves, threshold7waves = noise_estimator.compute_ssnr_waterline_measure(ssnr_seven_waves)

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_3waves_ra = noise_estimator.ring_average_ssnr()
        # volume_conventional = noise_estimator.compute_ssnr_volume(ssnr_3waves, dV)
        # measure_3waves, threshold3waves = noise_estimator.compute_ssnr_waterline_measure(ssnr_3waves)

        noise_estimator.illumination = illumination_widefield
        ssnr_widefield = np.abs(noise_estimator.compute_ssnr())
        ssnr_widefield_ra = noise_estimator.ring_average_ssnr()
        # volume_widefield = noise_estimator.compute_ssnr_volume(ssnr_widefield, dV)
        # measure_widefield, _ = noise_estimator.compute_ssnr_waterline_measure(ssnr_widefield)

        # print("Volume ssnr widefield = ", volume_widefield)
        # print("Measure ssnr widefield = ", measure_widefield)
        # print("Volume ssnr s_polarized = ", volume_squareSP)
        # print("Measure ssnr s_polarized = ", measure_s_polarized, threshold_s_polarized)
        # print("Volume ssnr three_waves = ", volume_conventional)
        # print("Measure ssnr 3waves = ", measure_3waves, threshold3waves)
        # print("Volume ssnr seven_waves = ", volume_hexagonal)
        # print("Measure ssnr seven_waves = ", measure_seven_waves, threshold7waves)
        # print("Volume ssnr circular = ", volume_circular)
        # print("Measure ssnr circular = ", measure_circular, threshold_circular)

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

        ax1.set_title(r"Ring averaged, $f_z ={} \frac{1 - cos(\theta)}{\lambda}".format(scaled_fz[arg]), fontsize=20, pad=15)
        ax1.set_xlabel(r"$f_r$, $\frac{2NA}{\lambda}$", fontsize=20)
        ax1.set_ylabel(r"$ssnr_{ra}$", fontsize=20)
        ax1.set_yscale("log")
        ax1.set_ylim(1, 3 * 10 ** 3)
        ax1.set_xlim(0, fx[-1] / (2 * NA))
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=15)

        ax2.set_title("Slice $f_y$ = 0", fontsize=30)
        ax2.set_xlabel(r"$f_x$, $\frac{2NA}{\lambda}$", fontsize=30)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 3 * 10 ** 3)
        ax2.set_xlim(0, fx[-1] / (2 * NA))
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=25)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(ssnr_s_polarized_ra, axis=1), label="lattice SIM, 5 waves, s-polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(ssnr_circular_ra, axis=1), label="lattice SIM, 5 waves, circularly polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(ssnr_seven_waves_ra, axis=1), label="7 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(ssnr_3waves_ra, axis=1), label="Conventional SIM")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * np.sum(ssnr_widefield_ra, axis=1), label="Widefield")
        # ax1.hlines(y=(1 + multiplier * threshold3waves), xmin=0, xmax=2, linewidth=1, color='red')
        # ax1.hlines(y=(1 + multiplier * threshold7waves), xmin=0, xmax=2, linewidth=1, color='green')
        # ax1.hlines(y=(1 + multiplier * threshold_circular), xmin=0, xmax=2, linewidth=1, color='orange')
        # ax1.hlines(y=(1 + multiplier * threshold_s_polarized), xmin=0, xmax=2, linewidth=1, color='blue')

        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, s-polarized")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_circular[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, circularly polarized")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_seven_waves[:, int(N / 2), arg][fx >= 0], label="7 waves")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves[:, int(N / 2), arg][fx >= 0], label="Conventional SIM")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="Widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        # ax2.set_aspect(1. / ax2.get_data_ratio())

        # def update1(val):
        #     ax1.clear()
        #     ax1.set_title("Ring averaged", fontsize=30)
        #     ax1.set_xlabel(r"$f_r$", fontsize=25)
        #     ax1.set_ylabel(r"$ssnr$", fontsize=25)
        #     ax1.set_yscale("log")
        #     ax1.set_ylim(1, 3 * 10 ** 2)
        #     ax1.set_xlim(0, fx[-1] / (2 * NA))
        #     ax1.grid()
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, int(val)], label="S-polarized")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_circular_ra[:, int(val)], label="Circular")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, int(val)], label="7 waves")
        #     ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, int(val)], label="3 waves")
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
        #     # ax2.set_ylabel(r"ssnr")
        #
        #     ax2.set_yscale("log")
        #     ax2.set_ylim(1, 3 * 10 ** 2)
        #     ax2.set_xlim(0, fx[-1] / (2 * NA))
        #     ax2.grid()
        #
        #     # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(ssnr_s_polarized[:, :, int(val)])[q_axes[1] >= 0], label="S-polarized")
        #     # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(ssnr_circular[:, :, int(val)])[q_axes[1] >= 0], label="Circular")
        #
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized[:, int(N / 2), int(val)][fx >= 0], label="S-polarized")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_circular[:, int(N / 2), int(val)][fx >= 0], label="Circular")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_seven_waves[:, int(N / 2), int(val)][fx >= 0], label="7 waves")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3waves[:, int(N / 2), int(val)][fx >= 0], label="3 waves")
        #     ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0], label="Widefield")
        #     ax2.legend()
        #     ax2.set_aspect(1. / ax2.get_data_ratio())
        #
        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, 100)  # slider properties
        # slider_ssnr.on_changed(update1)

        ax1.legend()
        # ax2.legend()
        # fig.savefig(f'{path_to_figures}fz={:.2f}_compare_ssnr_conf_version.png'.format(two_NA_fz[arg]))
        plt.show()

    def testssnrFromNumericalSpacialWaves(self):
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

        optical_system = Lens3D(alpha = np.pi/3)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

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
        noise_estimator = SSNR3dSIM2dShifts(illumination_2z, optical_system)
        ssnr_2z = np.abs(noise_estimator.compute_ssnr())
        ssnr_2z_ra = noise_estimator.ring_average_ssnr()
        volume_2z = noise_estimator.compute_ssnr_volume(ssnr_2z, dV)
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
        noise_estimator = SSNR3dSIM2dShifts(illumination_3w, optical_system)
        ssnr_3w = np.abs(noise_estimator.compute_ssnr())
        ssnr_3w_ra = noise_estimator.ring_average_ssnr()
        volume_3w = noise_estimator.compute_ssnr_volume(ssnr_3w, dV)

        widefield = Illumination({
        (0, 0, 0) : Sources.IntensityPlaneWave(1, 0, np.array((0, 0, 0)))}, Mr=1)
        widefield.Mt = 1
        noise_estimator.illumination = widefield
        ssnr_widefield = np.abs(noise_estimator.compute_ssnr())
        ssnr_widefield_ra= noise_estimator.ring_average_ssnr()
        volume_w = noise_estimator.compute_ssnr_volume(ssnr_widefield, dV)
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
        ax1.set_ylabel(r"$ssnr_{ra}$", fontsize=20)
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
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_2z_ra[:, arg], label="2 triangles")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3w_ra[:, arg], label="3 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_2z[:, int(N / 2), arg][fx >= 0], label="5 waves, 2 z angles")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3w[:, int(N / 2), arg][fx >= 0], label="3 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("Ring averaged", fontsize=30)
            ax1.set_xlabel(r"$f_r$",  fontsize=25)
            ax1.set_ylabel(r"$ssnr$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 3 * 10 ** 2)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_2z_ra[:, int(val)], label=" 5 waves, 2 z angles")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3w_ra[:, int(val)], label=" 3 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="widefield")

            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"ssnr")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10 ** 2)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_2z[:, int(N / 2), int(val)][fx >= 0], label="5 waves, 2z angles")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_3w[:, int(N / 2), int(val)][fx >= 0], label="3 waves")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), int(val)][fx >= 0], label="widefield")

            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, 100)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        # fig.savefig(
        #     f'{path_to_figures}fz={:.2f}_compare_ssnr_conf_version.png'.format(
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
        optical_system = Lens3D(alpha= 2 * np.pi / 5)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        angle = 8 * np.pi / 20
        a = 0.9
        b, c = 1, 1

        k = 2 * np.pi
        il4 = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(angle, a, b, c, 0)
        noise4 = SSNR3dSIM2dShifts(il4, optical_system)
        ssnr4 = noise4.ssnr()

        il5 = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(angle, a, b, c, 1)
        noise5 = SSNR3dSIM2dShifts(il5, optical_system)
        ssnr5 = noise5.ssnr()
        ra4 = noise4.ring_average_ssnr()
        ra5 = noise5.ring_average_ssnr()
        volume4 = noise4.compute_ssnr_volume(ssnr4, 1)
        volume5 = noise5.compute_ssnr_volume(ssnr5, 1)

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
        ax1.set_ylabel(r"$ssnr_{ra}$", fontsize=20)
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

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr4[:, int(N / 2), arg][fx >= 0], label="4 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr5[:, int(N / 2), arg][fx >= 0], label="5 waves")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("fz = {:.2f}".format(two_NA_fz[int(val)]), fontsize=30)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$ssnr$", fontsize=25)
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
            # ax2.set_ylabel(r"ssnr")

            ax2.set_yscale("log")
            ax2.set_ylim(1, 3 * 10 ** 2)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr4[:, int(val), arg][fx >= 0], label="4 waves")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr5[:, int(val), arg][fx >= 0], label="5 waves")
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
        #     f'{path_to_figures}fz={:.2f}_compare_ssnr_conf_version.png'.format(
        #         fz[arg]))
        plt.show()

class TestApproximations(unittest.TestCase):
    def test_ring_avergaged_SSNR(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        r = np.sin(theta) / np.sin(alpha)
        NA = np.sin(alpha)
        max_r = 8
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
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = Lens3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.55, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_widefield   = SSNR3dSIM2dShifts(illumination_widefield, optical_system)
        # noise_estimator_confocal    = SSNRConfocal(optical_system)
        noise_estimator_s_polarized = SSNR3dSIM2dShifts(illumination_s_polarized, optical_system)
        noise_estimator_circular    = SSNR3dSIM2dShifts(illumination_circular, optical_system)
        noise_estimator_seven_waves = SSNR3dSIM2dShifts(illumination_seven_waves, optical_system)
        noise_estimator_3waves      = SSNR3dSIM2dShifts(illumination_3waves, optical_system)

        noise_estimators = {"widefield" : noise_estimator_widefield,
                            "s_polarized" : noise_estimator_s_polarized,
                            "circular" : noise_estimator_circular,
                            "seven waves" : noise_estimator_seven_waves,
                            "three waves" : noise_estimator_3waves}


        for name in noise_estimators:
            noise_estimator = noise_estimators[name]
            noise_estimator.compute_ssnr()
            dj_ra = stattools.average_rings3d(noise_estimator.dj, (fx, fy, fz))
            dj2_ra = stattools.average_rings3d(noise_estimator.dj**2, (fx, fy, fz))
            full_fraction_approximation = 1/2 * dj2_ra/dj_ra + dj_ra
            nom_and_denom_separated_approximation = (1/2 + 1) * dj_ra
            fig = plt.figure()
            ax = fig.gca()
            ax.set_title(name)
            ax.plot(fx[fx >= 0], np.log(1 + 10**8 * full_fraction_approximation[:, N//2]), label="<nom/denom>")
            ax.plot(fx[fx >= 0], np.log(1 + 10**8 * nom_and_denom_separated_approximation[:, N//2]), label="<nom>/<denom>")
            ax.legend()
        plt.show()