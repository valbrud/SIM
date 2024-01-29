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
class TestSSNR(unittest.TestCase):
    def test_SSNR_interpolations(self):
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
        SSNR_F = np.abs(noise_estimator.SSNR(q_axes))
        end_shifted = time.time()
        print("fourier time is ", end_shifted - begin_shifted)

        noise_estimator.optical_system = optical_system_linear
        begin_interpolation = time.time()
        SSNR_L = np.abs(noise_estimator.SSNR(q_axes))
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
        mp1 = ax1.imshow(np.log10(SSNR_F[:, :, int(N / 2)]))
        cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        mp2 = ax2.imshow(np.log10(SSNR_L[:, :, int(N / 2)]))
        cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)

        def update1(val):
            ax1.set_title("SSNR_F, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fx, $\lambda^{-1}$")
            ax1.set_ylabel("fy, $\lambda^{-1}$")
            Z = (np.log10(SSNR_F[:, :, int(val)]))
            mp1.set_data(Z)
            mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            plt.draw()
            print(np.amax(SSNR_F[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        def update2(val):
            ax2.set_title("SSNR_L, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax2.set_xlabel("fx, $\lambda^{-1}$")
            ax2.set_ylabel("fy, $\lambda^{-1}$")
            print(int(val))
            Z = (np.log10(SSNR_L[:, :, int(val)]))
            mp2.set_data(Z)
            min = Z.min()
            max = Z.max()
            mp2.set_clim(vmin=min, vmax=max)
            # plt.draw()

        slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnri = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnri.on_changed(update2)

        plt.show()

    def test_SSNR(self):
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
        two_triangles_illumination = "two_triangles"
        s_polarized_waves = "s_polarized"
        circular_waves = "circular"
        three_waves = "three_waves"
        widefield = "widefield"
        two_angles = "two_angles"
        seven_waves = "seven_waves"

        waves = two_triangles_illumination
        if waves == three_waves:
            Mr = 3
            title = "Three waves SIM"
            label = "three_waves"
        elif waves == s_polarized_waves:
            Mr = 1
            title = "Lattice SIM with polarized waves"
            label = "s_polarized"
        elif waves == circular_waves:
            Mr = 1
            title = "Lattice SIM with circular waves"
            label = "circular"
        elif waves == widefield:
            Mr = 1
            Mt = 1
            title = "Widefield"
            label = "widefield"
        elif waves == seven_waves:
            Mr = 1
            title = "Hexagonal lattice SIM with seven waves"
            label = "hexagonal"
        elif waves == two_angles:
            Mr = 3
            Mt = 15
            illumination = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(np.pi/4, 5/7, 1, 1, Mt=Mt)
            title = "5 waves 2 angles SIM"
            label = "5 waves 2 angles"
        elif waves == two_triangles_illumination:
            Mr = 1
            Mt = 128
            illumination = configurations.get_two_oblique_triangles_at_different_angles_and_one_normal_wave(np.pi / 4,
                                                                            1 / 2, 1, 2, Mt = Mt)
            title = "7 waves 2 triangles SIM"
            label = "2 triangles"

        optical_system = Lens(alpha=np.pi/4)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        noise_estimator = SNRCalculator(illumination, optical_system)

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        SSNR = np.abs(noise_estimator.SSNR(q_axes,))
        scaling_factor = 10**8
        SSNR_scaled = 1 + scaling_factor * SSNR
        SSNR_ring_averaged = noise_estimator.ring_average_SSNR(q_axes, SSNR)
        SSNR_ra_scaled = 1 + scaling_factor * SSNR_ring_averaged

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        fig.suptitle(title, fontsize = 30)
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
        mp1 = ax1.imshow(SSNR_scaled[:, int(N / 2), :], extent=(fz[0]/(2 * NA), fz[-1]/(2 * NA), fy[0]/(2 * NA), fy[-1]/(2 * NA)), norm=colors.LogNorm())
        # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        # cb1.set_label("$1 + 10^8$ SSNR")
        ax1.set_aspect(1. / ax1.get_data_ratio())

        ax2 = fig.add_subplot(122, sharey=ax1)
        ax2.set_xlabel("$f_x$", fontsize = 25)
        ax2.tick_params(labelsize = 20)
        ax2.tick_params('y', labelleft=False)
        # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
        mp2 = ax2.imshow(SSNR_scaled[:, :, int(N//2)].T, extent=(fy[0]/(2 * NA), fy[-1]/(2 * NA), fx[0]/(2 * NA), fx[-1]/(2 * NA)), norm=colors.LogNorm())
        cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
        cb2.ax.tick_params(labelsize=20)
        cb2.set_label("$1 + 10^8$ SSNR", fontsize = 25)
        ax2.set_aspect(1. / ax2.get_data_ratio())

        fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/'
                 + label + '_waves_SSNR.png')

        def update1(val):
            ax1.set_title("SSNR, fy = {:.2f}, ".format(fy[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            ax1.set_xlabel("fz, $\lambda^{-1}$")
            ax1.set_ylabel("fx, $\lambda^{-1}$")
            Z = (SSNR_scaled[:, int(val), :])
            mp1.set_data(Z)
            mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            ax1.set_aspect(1. / ax1.get_data_ratio())
            plt.draw()

            ax2.set_title("SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            ax2.set_xlabel("fz, $\lambda^{-1}$")
            ax2.set_ylabel("fx, $\lambda^{-1}$")
            Z = (SSNR_scaled[:, :, int(val)])
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

        two_triangles_illumination = "two_triangles"
        s_polarized_waves = "s_polarized"
        circular_waves = "circular"
        three_waves = "three_waves"
        widefield = "widefield"
        two_angles = "two_angles"
        seven_waves = "seven_waves"

        waves = two_triangles_illumination
        if waves == three_waves:
            Mr = 3
            title = "Three waves SIM"
            label = "three_waves"
        elif waves == s_polarized_waves:
            Mr = 1
            title = "Lattice SIM with polarized waves"
            label = "s_polarized"
        elif waves == circular_waves:
            Mr = 1
            title = "Lattice SIM with circular waves"
            label = "circular"
        elif waves == widefield:
            Mr = 1
            Mt = 1
            title = "Widefield"
            label = "widefield"
        elif waves == seven_waves:
            Mr = 1
            title = "Hexagonal lattice SIM with seven waves"
            label = "hexagonal"
        elif waves == two_angles:
            Mr = 3
            Mt = 15
            illumination = configurations.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(np.pi/4, 5/7, 1, 1, Mt=Mt)
            title = "5 waves 2 angles SIM"
            label = "5 waves 2 angles"
        elif waves == two_triangles_illumination:
            Mr = 1
            Mt = 128
            illumination = configurations.get_two_oblique_triangles_at_different_angles_and_one_normal_wave(np.pi / 4,
                                                                            1 / 2, 1, 2, Mt = Mt)
            title = "7 waves 2 triangles SIM"
            label = "2 triangles"

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        optical_system = Lens(alpha=np.pi/3)
        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        noise_estimator = SNRCalculator(illumination, optical_system)
        SSNR = np.abs(noise_estimator.SSNR(q_axes,))
        SSNR_scaled = np.log10(1 + 10**8 * np.abs(SSNR))

        noise_estimator.illumination = illumination
        SSNR_widefield = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_widefield_scaled = np.log10(1 + 10**8 * np.abs(SSNR_widefield))

        constant_value = 0.2
        verts, faces, _, _ = skimage.measure.marching_cubes(SSNR_scaled, level=constant_value)
        w_verts, w_faces, _, _ = skimage.measure.marching_cubes(SSNR_widefield_scaled, level=constant_value)
        Fx, Fy, Fz = np.meshgrid(fx, fy, fz)
        fig = plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.set_title(title, fontsize=20)
        ax1.set_xlabel(r"$f_x$", fontsize=20, labelpad=15)
        ax1.set_ylabel(r"$f_y$", fontsize=20, labelpad=15)
        ax1.set_zlabel(r"$f_z$", fontsize=20, labelpad=15)
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],  alpha=0.7)
        ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2],  alpha=1, color="red")

        # ax1.set_xlim(N // 4, 3 * N // 4)
        # ax1.set_ylim(N // 4, 3 * N // 4)
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
                 + label + '_waves_SSNR_isosurface.png')

        def update1(val):
            ax1.clear()
            ax1.set_title(title + "\n1 + $10^8 SSNR$ = {:.2f}".format(val/15), fontsize=18)
            ax1.set_xlabel(r"$f_x$", fontsize=18, labelpad=15)
            ax1.set_ylabel(r"$f_y$", fontsize=18, labelpad=15)
            ax1.set_zlabel(r"$f_z$", fontsize=18, labelpad=15)
            verts, faces, _, _ = skimage.measure.marching_cubes(SSNR_scaled, level=val/15)
            w_verts, w_faces, _, _ = skimage.measure.marching_cubes(SSNR_widefield_scaled, level=val/15)
            ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], alpha=0.5, color="blue")
            ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2], alpha=1, color="red")

            # ax1.set_xlim(N//4, 3 * N//4)
            # ax1.set_ylim(N//4, 3 * N//4)
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
        #          'Animation_' + label + '_waves_SSNR_isosurface.mp4', writer="ffmpeg")
        plt.show()

    def test_compare_SSNR(self):
        theta = np.pi/4
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
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 2**0.5, 3, Mt=1)
        illumination_widefield = configurations.widefield()

        noise_estimator = SNRCalculator(illumination_s_polarized, optical_system)
        SSNR_s_polarized = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_s_polarized_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_s_polarized)
        volume_s_polarized = noise_estimator.compute_SSNR_volume(SSNR_s_polarized, dV)

        noise_estimator.illumination = illumination_circular
        SSNR_circular = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_circular_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_circular)
        volume_circular = noise_estimator.compute_SSNR_volume(SSNR_circular, dV)

        noise_estimator.illumination = illumination_seven_waves
        SSNR_seven_waves = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_seven_waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_seven_waves)
        volume_seven_waves = noise_estimator.compute_SSNR_volume(SSNR_seven_waves, dV)

        noise_estimator.illumination = illumination_3waves
        SSNR_3waves = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_3waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_3waves)
        volume_3waves = noise_estimator.compute_SSNR_volume(SSNR_3waves, dV)

        noise_estimator.illumination = illumination_widefield
        SSNR_widefield = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_widefield_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_widefield)
        volume_widefield = noise_estimator.compute_SSNR_volume(SSNR_widefield, dV)

        print("Volume SSNR widefield = ", volume_widefield)
        print("Volume SSNR s_polarized = ", volume_s_polarized)
        print("Volume SSNR three_waves = ", volume_3waves)
        print("Volume SSNR seven_waves = ", volume_seven_waves)
        print("Volume SSNR circular = ", volume_circular)

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
        ax2.grid(which='minor', linestyle = '--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize = 15)

        multiplier = 10 ** 8
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, arg], label="lattice SIM, 5 waves, s-polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, arg],    label="lattice SIM, 5 waves, circularly polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_seven_waves_ra[:, arg],    label="7 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, arg],      label="Conventional SIM")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, arg],   label="Widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_s_polarized[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, s-polarized")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_circular[:, int(N / 2), arg][fx >= 0], label="lattice SIM, 5 waves, circularly polarized")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_seven_waves[:, int(N / 2), arg][fx >= 0], label="7 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[:, int(N / 2), arg][fx >= 0], label="Conventional SIM")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[:, int(N / 2), arg][fx >= 0], label="Widefield")
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
            # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_seven_waves_ra[:, int(val)],    label="7 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, int(val)],      label="3 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, int(val)],   label="Widefield")
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
            # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_seven_waves[:, int(N / 2),  int(val)][fx >= 0],    label="7 waves"   )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[:, int(N / 2), int(val)][fx >= 0],      label="3 waves"    )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[:, int(N / 2), int(val)][fx >= 0],   label="Widefield"  )
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, 50)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        # fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR_conf_version.png'.format(two_NA_fz[arg]))
        plt.show()

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
        sources = ttillum
        size = (2 * max_r, 2 * max_r, 2 * max_z)
        box = Box.Box(sources, size, N)
        box.compute_intensity_and_spacial_waves_numerically()
        iwaves = box.get_approximated_intensity_sources()
        illumination_2z = Illumination.init_from_list(iwaves, (k * np.sin(theta) / 14, k * np.sin(theta) * 3**0.5/ 14, k / 14), Mr = 1)
        illumination_2z.normalize_spacial_waves()
        noise_estimator = SNRCalculator(illumination_2z, optical_system)
        SSNR_2z = np.abs(noise_estimator.SSNR(q_axes))
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
        SSNR_3w = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_3w_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_3w)
        volume_3w = noise_estimator.compute_SSNR_volume(SSNR_3w, dV)

        widefield = Illumination({
        (0, 0, 0) : Sources.IntensityPlaneWave(1/Mt_widefield, 0, np.array((0, 0, 0)))}, Mr=1)
        widefield.Mt = 1
        noise_estimator.illumination = widefield
        SSNR_widefield = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_widefield_ra= noise_estimator.ring_average_SSNR(q_axes, SSNR_widefield)
        volume_w = noise_estimator.compute_SSNR_volume(SSNR_widefield, dV)
        print("volume 3 waves = ", volume_3w)
        print("volume 5 waves = ", volume_2z)
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
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_2z_ra[:, arg], label="5 waves, 2 z angles")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3w_ra[:, arg], label="3 waves")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_2z[:, int(N / 2), arg][fx >= 0], label="5 waves, 2 z angles")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3w[:, int(N / 2), arg][fx >= 0], label="3 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
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
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, int(val)], label="widefield")

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
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[:, int(N / 2), int(val)][fx >= 0], label="widefield")

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

        NA = np.sin(2 * np.pi / 5)
        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (2 * NA)
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = Lens(alpha= 2 * np.pi / 5)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        angle = 7 * np.pi / 20
        a = 0.9
        b, c = 2, 2

        k = 2 * np.pi
        k1 = k * np.sin(angle)
        k3 = k * a * np.sin(angle)
        k2 = k * (np.cos(angle) - 1)
        k4 = k * (np.cos(np.arcsin(a * np.sin(angle))) - 1)

        basevector_lengths = (0.01 * np.sin(angle) * k, 100 * k, 0.001 * k)
        four_waves = [
            IntensityPlaneWave(2 * b ** 2 + 2 * c ** 2, 0, np.array((0, 0, 0))),
            IntensityPlaneWave(-b ** 2, 0, np.array((2 * k1, 0, 0))),
            IntensityPlaneWave(-b ** 2, 0, np.array((-2 * k1, 0, 0))),
            IntensityPlaneWave(-c ** 2, 0, np.array((2 * k3, 0, 0))),
            IntensityPlaneWave(-c ** 2, 0, np.array((-2 * k3, 0, 0))),
            IntensityPlaneWave(b * c, 0, np.array((k3 - k1, 0, k4 - k2))),
            IntensityPlaneWave(b * c, 0, np.array((k3 - k1, 0, k2 - k4))),
            IntensityPlaneWave(b * c, 0, np.array((k1 - k3, 0, k4 - k2))),
            IntensityPlaneWave(b * c, 0, np.array((k1 - k3, 0, k2 - k4))),
            IntensityPlaneWave(-b * c, 0, np.array((k3 + k1, 0, k4 - k2))),
            IntensityPlaneWave(-b * c, 0, np.array((k3 + k1, 0, k2 - k4))),
            IntensityPlaneWave(-b * c, 0, np.array((-k1 - k3, 0, k4 - k2))),
            IntensityPlaneWave(-b * c, 0, np.array((-k1 - k3, 0, k2 - k4))),
        ]

        five_waves = [
            IntensityPlaneWave(1 + 2 * b ** 2 + 2 * c ** 2, 0, np.array((0, 0, 0))),
            IntensityPlaneWave(-b ** 2, 0, np.array((2 * k1, 0, 0))),
            IntensityPlaneWave(-b ** 2, 0, np.array((-2 * k1, 0, 0))),
            IntensityPlaneWave(-c ** 2, 0, np.array((2 * k3, 0, 0))),
            IntensityPlaneWave(-c ** 2, 0, np.array((-2 * k3, 0, 0))),
            IntensityPlaneWave(b * c, 0, np.array((k3 - k1, 0, k4 - k2))),
            IntensityPlaneWave(b * c, 0, np.array((k3 - k1, 0, k2 - k4))),
            IntensityPlaneWave(b * c, 0, np.array((k1 - k3, 0, k4 - k2))),
            IntensityPlaneWave(b * c, 0, np.array((k1 - k3, 0, k2 - k4))),
            IntensityPlaneWave(-b * c, 0, np.array((k3 + k1, 0, k4 - k2))),
            IntensityPlaneWave(-b * c, 0, np.array((k3 + k1, 0, k2 - k4))),
            IntensityPlaneWave(-b * c, 0, np.array((-k1 - k3, 0, k4 - k2))),
            IntensityPlaneWave(-b * c, 0, np.array((-k1 - k3, 0, k2 - k4))),
            IntensityPlaneWave(b, 0, np.array((k1, 0, k2))),
            IntensityPlaneWave(-b, 0, np.array((k1, 0, -k2))),
            IntensityPlaneWave(-b, 0, np.array((-k1, 0, k2))),
            IntensityPlaneWave(b, 0, np.array((-k1, 0, -k2))),
            IntensityPlaneWave(c, 0, np.array((k3, 0, k4))),
            IntensityPlaneWave(-c, 0, np.array((k3, 0, -k4))),
            IntensityPlaneWave(-c, 0, np.array((-k3, 0, k4))),
            IntensityPlaneWave(c, 0, np.array((-k3, 0, -k4))),
        ]
        il4 = Illumination.init_from_list(four_waves, basevector_lengths, Mr=3)
        il4.normalize_spacial_waves()
        il5 = Illumination.init_from_list(five_waves, basevector_lengths, Mr=3)
        il5.normalize_spacial_waves()
        noise4 = SNRCalculator(il4, optical_system)
        SSNR4 = noise4.SSNR(q_axes)
        noise5 = SNRCalculator(il5, optical_system)
        SSNR5 = noise5.SSNR(q_axes)
        ra4 = noise4.ring_average_SSNR(q_axes, SSNR4)
        ra5 = noise5.ring_average_SSNR(q_axes, SSNR5)
        volume4 = noise4.compute_SSNR_volume(SSNR4, dV)
        volume5 = noise5.compute_SSNR_volume(SSNR5, dV)

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
        # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, arg], label="widefield")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR4[:, int(N / 2), arg][fx >= 0], label="4 waves")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR5[:, int(N / 2), arg][fx >= 0], label="5 waves")
        # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[:, int(N / 2), arg][fx >= 0], label="widefield")
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
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ra4[:, int(val)], label="4 waves")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ra5[:, int(val)], label="5 waves")
            # ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, int(val)], label="widefield")

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

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR4[:, int(val), arg][fx >= 0], label="4 waves")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR5[:, int(val), arg][fx >= 0], label="5 waves")
            # ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[:, int(N / 2), int(val)][fx >= 0],
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