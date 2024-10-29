import sys
import matplotlib.pyplot as plt
import csv

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

import Box
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

from globvar import path_to_figures, path_to_animations
sys.path.append('../')

configurations = BFPConfiguration(refraction_index=1.5)
alpha = 2 * np.pi / 5
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

multiplier = 10 ** 5
ylim = 10 ** 2

optical_system = Lens3D(alpha=alpha, refractive_index_sample=nobject, refractive_index_medium=nmedium)
optical_system.compute_psf_and_otf((psf_size, N),
                                   apodization_filter=None)
squareL = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 1, Mt=1)
squareC = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.55, 1, Mt=1, phase_shift=0)
hexagonal = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1)
conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
widefield = configurations.get_widefield()

illumination_list = {
    widefield: "Widefield",
    conventional: "Conventional",
    squareL: "SquareL",
    squareC: "SquareC",
    hexagonal: "Hexagonal",
}


class TestArticlePlots(unittest.TestCase):
    def test_ring_averaged_ssnr(self):
        n_points = 51
        ax = np.linspace(two_NA_fx[N//2], two_NA_fx[-1], n_points)
        noise_estimator_widefield = SSNR3dSIM2dShifts(widefield, optical_system)
        noise_estimator_widefield.compute_ssnr()
        ssnr_widefield = noise_estimator_widefield.ssnr
        ssnr_widefield_ra = noise_estimator_widefield.ring_average_ssnr(number_of_samples=n_points)

        noise_estimator = SSNR3dSIM2dShifts(squareL, optical_system)
        ssnr_s_polarized = np.abs(noise_estimator.compute_ssnr())
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnr(number_of_samples=n_points)

        noise_estimator.illumination = squareC
        ssnr_circular = np.abs(noise_estimator.compute_ssnr())
        ssnr_circular_ra = noise_estimator.ring_average_ssnr(number_of_samples=n_points)

        noise_estimator.illumination = hexagonal
        ssnr_seven_waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnr(number_of_samples=n_points)

        noise_estimator.illumination = conventional
        ssnr_3waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_3waves_ra = noise_estimator.ring_average_ssnr(number_of_samples=n_points)

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
        ax1.set_title("$f_z = ${:.1f}".format(scaled_fz[arg]) + "$\; [\\frac{n - \sqrt{n^2 - NA^2}}{\lambda}]$", fontsize=30, pad=15)
        ax1.set_xlabel(r"$f_r \; [\frac{2NA}{\lambda}]$", fontsize=30)
        ax1.set_ylabel(r"$1 + 10^5 SSNR_{ra}$", fontsize=30)
        ax1.set_yscale("log")
        ax1.set_ylim(1, ylim)
        ax1.set_xlim(0, two_NA_fx[-1])
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=30)

        # ax2.set_title("Slice $f_y$ = 0", fontsize=25)
        ax2.set_title("$f_z = ${:.1f}".format(scaled_fz[arg // 2]) + "$\; [\\frac{n - \sqrt{n^2 - NA^2}}{\lambda}]$", fontsize=30, pad=15)
        ax2.set_xlabel(r"$f_r \; [\frac{2NA}{\lambda}]$", fontsize=30)
        ax2.set_ylabel(r"$1 + 10^5 SSNR_{ra}$", fontsize=30)
        ax2.set_yscale("log")
        ax2.set_ylim(1, ylim)
        ax2.set_xlim(0, two_NA_fx[-1])
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params('y', labelleft=False)
        ax2.tick_params(labelsize=30)

        ax1.plot(ax, 1 + multiplier / 10 * ssnr_3waves_ra[:, arg], label="Conventional")
        ax1.plot(ax, 1 + multiplier / 10 * ssnr_circular_ra[:, arg], label="SquareC")
        ax1.plot(ax, 1 + multiplier / 10 * ssnr_s_polarized_ra[:, arg], label="SquareL")
        ax1.plot(ax, 1 + multiplier / 10 * ssnr_seven_waves_ra[:, arg], label="Hexagonal")
        ax1.plot(ax, 1 + multiplier / 10 * ssnr_widefield_ra[:, arg], label="Widefield")

        ax2.plot(ax, 1 + multiplier * ssnr_3waves_ra[:, arg // 2], label="Conventional")
        ax2.plot(ax, 1 + multiplier * ssnr_circular_ra[:, arg // 2], label="SquareC")
        ax2.plot(ax, 1 + multiplier * ssnr_s_polarized_ra[:, arg // 2], label="SquareL")
        ax2.plot(ax, 1 + multiplier * ssnr_seven_waves_ra[:, arg // 2], label="Hexagonal")
        ax2.plot(ax, 1 + multiplier * ssnr_widefield_ra[:, arg // 2], label="Widefield")
        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        fig.savefig(f'{path_to_figures}ring_averaged_ssnr')
        plt.show()

    def test_ssnr_color_maps(self):
        for illumination in illumination_list:
            noise_estimator = SSNR3dSIM2dShifts(illumination, optical_system)

            ssnr = np.abs(noise_estimator.compute_ssnr())
            scaling_factor = 10 ** 8
            ssnr_scaled = 1 + scaling_factor * ssnr
            ssnr_ring_averaged = noise_estimator.ring_average_ssnr()
            ssnr_ra_scaled = 1 + scaling_factor * ssnr_ring_averaged

            Fx, Fy = np.meshgrid(fx, fy)
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            fig.suptitle(illumination_list[illumination], fontsize=30)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.1,
                                hspace=0)
            ax1 = fig.add_subplot(121)
            ax1.tick_params(labelsize=30)
            # ax1.set_title(title)
            ax1.set_xlabel("$f_y \; [\\frac{2NA}{\lambda}]$", fontsize=30)
            ax1.set_ylabel("$f_x \;  [\\frac{2NA}{\lambda}]$", fontsize=30)
            mp1 = ax1.imshow(ssnr_scaled[:, :, N // 2], extent=(-2, 2, -2, 2), norm=colors.LogNorm())
            # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
            # cb1.set_label("$1 + 10^8$ ssnr")
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2 = fig.add_subplot(122, sharey=ax1)
            ax2.set_xlabel("$f_z \; [\\frac{n - \sqrt{n^2 - NA^2}}{\lambda}]$", fontsize=30)
            ax2.tick_params(labelsize=30)
            ax2.tick_params('y', labelleft=False)
            # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
            mp2 = ax2.imshow(ssnr_scaled[N // 2, :, :], extent=(-2, 2, -2, 2), norm=colors.LogNorm())
            # mp2 = ax2.imshow(ssnr_ra_scaled[:, :].T, extent=(0, fy[-1]/(2 * NA), fz[0]/(2 * NA), fz[-1]/(2 * NA)), norm=colors.LogNorm())
            cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=30)
            cb2.set_label("$1 + 10^8$ ssnr", fontsize=30)
            ax2.set_aspect(1. / ax2.get_data_ratio())

            fig.savefig(f'{path_to_figures}'
                     + illumination_list[illumination] + '_ssnr.png')

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

            # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
            # slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
            # slider_ssnr.on_changed(update1)

            plt.show()

    def test_square_sim_anisotropy(self):
        # theta = alpha/2
        p = 0.55
        # illumination0 = configurations.get_4_oblique_s_waves_and_circular_normal(theta, p, 1, 1, 0)
        # illuminationPi4 = configurations.get_4_oblique_s_waves_and_circular_normal(theta, p, 1, 1, np.pi / 4)
        # illuminationPi2 = configurations.get_4_oblique_s_waves_and_circular_normal(theta, p, 1, 1, np.pi / 2)
        # illuminationPi = configurations.get_4_oblique_s_waves_and_circular_normal(theta, p, 1, 1, np.pi)
        #
        illumination0 = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, p, 1, 1, 0)
        illuminationPi4 = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, p, 1, 1, np.pi / 4)
        illuminationPi2 = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, p, 1, 1, np.pi / 2)
        illuminationPi = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, p, 1, 1, np.pi)

        illumination_list = {
            illumination0: "$0$",
            illuminationPi4: "0",
            illuminationPi2: "$\pi/2$",
            illuminationPi: "$\pi$"
        }
        noise_estimator_widefield = SSNR3dSIM2dShifts(widefield, optical_system)
        ssnr_widefield = noise_estimator_widefield.compute_ssnr()
        noise_estimator0 = SSNR3dSIM2dShifts(illumination0, optical_system)
        noise_estimatorPi4 = SSNR3dSIM2dShifts(illuminationPi4, optical_system)
        noise_estimatorPi2 = SSNR3dSIM2dShifts(illuminationPi2, optical_system)
        noise_estimatorPi = SSNR3dSIM2dShifts(illuminationPi, optical_system)

        ssnr0 = np.abs(noise_estimator0.compute_ssnr())
        ssnrPi4 = np.abs(noise_estimatorPi4.compute_ssnr())
        ssnrPi2 = np.abs(noise_estimatorPi2.compute_ssnr())
        ssnrPi = np.abs(noise_estimatorPi.compute_ssnr())
        scaling_factor = 10 ** 4

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        # fig.suptitle(f"Phase of the central beam = {illumination_list[illumination]}", fontsize=30)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.1,
                            hspace=0)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax1.set_title('Phase shift = 0', fontsize=30)
        ax2.set_title('Phase shift = $\pi/4$', fontsize=30)
        ax3.set_title('Phase shift = $\pi/2$', fontsize=30)
        ax4.set_title('Phase shift = $\pi$', fontsize=30)
        # ax1.set_xlabel('$k_x [\\frac{2NA}{\lambda}]$', fontsize = 30)
        # ax2.set_xlabel('$k_x [\\frac{2NA}{\lambda}]$', fontsize = 30)
        ax1.set_ylabel('$1 + 10^4 SSNR$', fontsize=30)
        ax3.set_ylabel('$1 + 10^4 SSNR$', fontsize=30)
        ax3.set_xlabel('$k_r [\\frac{2NA}{\lambda}]$', fontsize=30)
        ax4.set_xlabel('$k_r [\\frac{2NA}{\lambda}]$', fontsize=30)
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax4.set_yticklabels([])

        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax1.tick_params(labelsize=30)
        ax2.tick_params(labelsize=30)
        ax3.tick_params(labelsize=30)
        ax4.tick_params(labelsize=30)
        ax1.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        ax2.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        ax3.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        ax4.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        ax1.set_ylim(1, 2 * 10 ** 2)
        ax2.set_ylim(1, 2 * 10 ** 2)
        ax3.set_ylim(1, 2 * 10 ** 2)
        ax4.set_ylim(1, 2 * 10 ** 2)

        z_shift = 5
        ax1.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnr0[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        ax1.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnr0[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")

        ax2.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnrPi4[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        ax2.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnrPi4[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")

        ax3.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnrPi2[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        ax3.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnrPi2[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")

        ax4.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnrPi[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        ax4.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnrPi[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")

        ax1.legend(fontsize=30)
        ax2.legend(fontsize=30)
        ax3.legend(fontsize=30)
        ax4.legend(fontsize=30)

        fig.savefig(f'{path_to_figures}sim_anisotropy')
        plt.show()


    def test_sim_modalities_comparison(self):
        ...

    def test_illumination_animations(self):
        r_max = 2
        z_max = 1
        N = 200
        arg = N//2
        x = np.linspace(-r_max, r_max, N)
        y = np.linspace(-r_max, r_max, N)
        z = np.linspace(-z_max, z_max, N)
        psf_size = (2 * r_max, 2 * r_max, 2 * z_max)
        n_illum = len(illumination_list.keys()) - 1
        n_rows = int(n_illum**0.5)
        n_colums = n_rows
        fig, axes = plt.subplots(n_rows, n_colums, figsize=(12, 10), sharex=True, sharey=True)
        fig.suptitle(f'z = {round(z[arg], 2)} [wavelength]', fontsize=30)
        boxes = []
        i = 0
        for illumination in illumination_list:
            if illumination_list[illumination] == "Widefield":
                continue
            ax = axes[i//n_colums, i%n_colums]
            boxes.append(Box.BoxSIM(illumination, box_size=psf_size, point_number=N))
            boxes[i].compute_intensity_from_spacial_waves()
            ax.set_title(illumination_list[illumination], fontsize=30, pad=15)
            if i % n_colums == 0:
                ax.set_ylabel("y [wavelength]", fontsize=30)
            if i // n_colums == n_rows-1:
                ax.set_xlabel("x [wavelength]",  fontsize=30)
            ax.tick_params(labelsize=30)
            ax.imshow(boxes[i].intensity[:, :, arg].T, extent=(x[0], x[-1], y[0], y[-1]))
            i+=1

        plt.tight_layout()
        # plt.show()
        def update(val):
            i = 0
            fig.suptitle(f'z = {round(z[int(val)], 2)} [wavelength]', fontsize=30)
            for illumination in illumination_list:
                if illumination_list[illumination] == "Widefield":
                    continue
                ax = axes[i // n_colums, i % n_colums]
                ax.set_title(illumination_list[illumination], fontsize=30, pad=15)
                if i % n_colums == 0:
                    ax.set_ylabel("y [wavelength]", fontsize=30)
                if i // n_colums == n_rows - 1:
                    ax.set_xlabel("x [wavelength]", fontsize=30)
                ax.tick_params(labelsize=30)
                ax.imshow(boxes[i].intensity[:, :, int(val)].T, extent=(x[0], x[-1], y[0], y[-1]))
                i+=1
        ani = FuncAnimation(fig, update, frames=range(0, N), repeat=False, interval=40)
        ani.save(path_to_animations +
                 '3D_illumination.mp4', writer="ffmpeg")
        # plt.show()

    def test_plot_power_dependence(self):
        file_path = 'Power.csv'
        data = pd.read_csv(file_path)
        print(data['Configuration'].unique())
        fig1, axes1 = plt.subplots(figsize=(12, 10))
        fig2, axes2 = plt.subplots(figsize=(12, 10))

        # Filter data for the current combination
        configurations = data['Configuration'].unique()
        volume_widefield = int(data[data['Configuration'] == 'Widefield']['Volume'].iloc[0])
        entropy_widefield = int(data[data['Configuration'] == 'Widefield']['Entropy'].iloc[0])
        print(volume_widefield, entropy_widefield)
        # Plotting
        for configuration in configurations:
            # Convert analytical volume to rounded integer values
            if configuration == 'Widefield':
                continue
            print(configuration)
            plot_data = data[data['Configuration'] == configuration]
            print(data)
            bs = np.array(plot_data['Power'])
            analytical_volume = np.array(plot_data['Volume_a'])
            computed_volume = np.array(plot_data['Volume'])
            entropy = np.array(plot_data['Entropy'])
            axes1.plot(bs, (analytical_volume - volume_widefield) / volume_widefield, '--')
            color = axes1.lines[-1].get_color()

            ssnrvinc = (computed_volume - volume_widefield) / volume_widefield
            ssnrsinc = (entropy - entropy_widefield) / entropy_widefield
            axes1.plot(bs, ssnrvinc, label=configuration, color=color)
            axes2.plot(bs, ssnrsinc, label=configuration)
            if configuration == "SquareC":
                axes1.plot(bs[12], ssnrvinc[12], 'o', ms=10, color=color)
                axes2.plot(bs[12], ssnrsinc[12], 'o', ms=10, color=color)
            else:
                axes1.plot(bs[20], ssnrvinc[20], 'o', ms=10, color=color)
                axes2.plot(bs[20], ssnrsinc[20], 'o', ms=10, color=color)

            # axes1.set_title("SSNR volume increase \n for different intensity ratios", fontsize=30)
            # axes2.set_title("SSNR entropy increase \n for different intensity ratios", fontsize=30)

        axes1.grid()
        axes1.set_xlim(0, 2)
        axes1.set_ylim(0, 0.8)
        axes1.set_aspect(1 / axes1.get_data_ratio())
        axes1.tick_params(labelsize=25)
        axes1.set_xlabel(r"r", fontsize=30)
        axes1.set_ylabel("$SSNR^{INC}_V$", fontsize=30)
        axes1.legend(fontsize=20)

        axes2.grid()
        axes2.set_xlim(0, 2)
        axes2.set_ylim(0, 0.15)
        axes2.set_aspect(1 / axes2.get_data_ratio())
        axes2.tick_params(labelsize=25)
        axes2.set_xlabel(r'r', fontsize=30)
        axes2.set_ylabel("$SSNR^{INC}_S$", fontsize=30)

        axes1.legend(fontsize=20, loc="lower right")
        fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
        fig1.savefig(f'{path_to_figures}V_GAIN_b',bbox_inches='tight', pad_inches=0, dpi=300)

        axes2.legend(fontsize=20, loc="lower right")
        fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
        fig2.savefig(f'{path_to_figures}S_GAIN_b',bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()

    def test_plot_angle_dependence(self):
        file_path = 'Angles.csv'
        data = pd.read_csv(file_path)
        print(data['Configuration'].unique())
        fig1, axes1 = plt.subplots(figsize=(12, 10))
        fig2, axes2 = plt.subplots(figsize=(12, 10))

        # Filter data for the current combination
        configurations = data['Configuration'].unique()
        volume_widefield = int(data[data['Configuration'] == 'Widefield']['Volume'].iloc[0])
        entropy_widefield = int(data[data['Configuration'] == 'Widefield']['Entropy'].iloc[0])
        print(volume_widefield, entropy_widefield)
        # Plotting
        for configuration in configurations:
            # Convert analytical volume to rounded integer values
            if configuration == 'Widefield':
                continue
            print(configuration)
            plot_data = data[data['Configuration'] == configuration]
            print(data)
            bs = np.array(plot_data['NA_ratio'])
            analytical_volume = np.array(plot_data['Volume_a'])
            computed_volume = np.array(plot_data['Volume'])
            entropy = np.array(plot_data['Entropy'])
            axes1.plot(bs, (analytical_volume - volume_widefield) / volume_widefield, '--')
            color = axes1.lines[-1].get_color()

            ssnrvinc = (computed_volume - volume_widefield) / volume_widefield
            ssnrsinc = (entropy - entropy_widefield) / entropy_widefield
            axes1.plot(bs, ssnrvinc, label=configuration, color=color)
            axes2.plot(bs, ssnrsinc, label=configuration)

            # axes1.set_title("SSNR volume increase \n for different incident angles", fontsize=30)
            # axes2.set_title("SSNR entropy increase \n for different intensity ratios", fontsize=30)

        axes1.grid()
        axes1.set_xlim(0.2, 1)
        axes1.set_ylim(0, 1)
        axes1.set_aspect(1 / axes1.get_data_ratio())
        axes1.tick_params(labelsize=25)
        axes1.set_xlabel(r"r", fontsize=30)
        axes1.set_ylabel("$SSNR^{INC}_V$", fontsize=30)
        axes1.legend(fontsize=20)

        axes2.grid()
        axes2.set_xlim(0.2, 1)
        axes2.set_ylim(0, 0.2)
        axes2.set_aspect(1 / axes2.get_data_ratio())
        axes2.tick_params(labelsize=25)
        axes2.set_xlabel(r'r', fontsize=30)
        axes2.set_ylabel("$SSNR^{INC}_S$", fontsize=30)

        axes1.legend(fontsize=20, loc="lower right")
        fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
        fig1.savefig(f'{path_to_figures}V_GAIN_angles', bbox_inches='tight', pad_inches=0, dpi=300)

        axes2.legend(fontsize=20, loc="lower right")
        fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
        fig2.savefig(f'{path_to_figures}S_GAIN_angles', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()


class TestComputeVolumeEntropy(unittest.TestCase):
    def test_compute_angle_dependence(self):
        headers = ["Configuration", "NA_ratio", "Volume", "Volume_a", "Entropy"]

        squareL = lambda angle: configurations.get_4_oblique_s_waves_and_circular_normal(angle, 1, 1, Mt=1)
        squareC = lambda angle: configurations.get_4_circular_oblique_waves_and_circular_normal(angle, 0.55, 1, Mt=1)
        hexagonal = lambda angle: configurations.get_6_oblique_s_waves_and_circular_normal(angle, 1, 1, Mt=1)
        conventional = lambda angle: configurations.get_2_oblique_s_waves_and_s_normal(angle, 1, 1, Mt=1)

        config_methods = {
            "Conventional": conventional,
            "SquareL": squareL,
            "SquareC": squareC,
            "Hexagonal": hexagonal,
        }

        ratios = np.linspace(0.2, 1, 41)

        with open("Angles.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            widefield = configurations.get_widefield()
            ssnr_calc = SSNR3dSIM2dShifts(widefield, optical_system)
            ssnr = ssnr_calc.compute_ssnr()
            volume = ssnr_calc.compute_ssnr_volume()
            volume_a = ssnr_calc.compute_analytic_ssnr_volume()
            entropy = ssnr_calc.compute_true_ssnr_entropy()
            params = ["Widefield", 1, round(volume), round(volume_a), round(entropy)]
            print(params)
            writer.writerow(params)

            for configuration in config_methods:
                for ratio in ratios:
                    sin_theta = ratio * (NA / nmedium)
                    theta = np.arcsin(sin_theta)
                    illumination = config_methods[configuration](theta)
                    ssnr_calc = SSNR3dSIM2dShifts(illumination, optical_system)
                    ssnr = ssnr_calc.compute_ssnr()

                    volume = ssnr_calc.compute_ssnr_volume()
                    volume_a = ssnr_calc.compute_analytic_ssnr_volume()
                    entropy = ssnr_calc.compute_true_ssnr_entropy()

                    print("Volume ", round(volume))
                    print("Volume a", round(volume_a))
                    print("Entropy ", round(entropy))
                    params = [configuration, round(ratio, 2), round(volume), round(volume_a), round(entropy)]
                    print(params)

                    writer.writerow(params)

    def test_compute_power_dependence(self):
        headers = ["Configuration", "Power", "Volume", "Volume_a", "Entropy"]

        squareL = lambda power: configurations.get_4_oblique_s_waves_and_circular_normal(theta, power, 1, Mt=1)
        squareC = lambda power: configurations.get_4_circular_oblique_waves_and_circular_normal(theta, power, 1, Mt=1)
        hexagonal = lambda power: configurations.get_6_oblique_s_waves_and_circular_normal(theta, power, 1, Mt=1)
        conventional = lambda power: configurations.get_2_oblique_s_waves_and_s_normal(theta, power, 1, Mt=1)

        config_methods = {
            "Conventional": conventional,
            "SquareL": squareL,
            "SquareC": squareC,
            "Hexagonal": hexagonal,
        }

        ratios = np.linspace(0, 2, 41)

        with open("Power.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            widefield = configurations.get_widefield()
            ssnr_calc = SSNR3dSIM2dShifts(widefield, optical_system)
            ssnr = ssnr_calc.compute_ssnr()
            volume = ssnr_calc.compute_ssnr_volume()
            volume_a = ssnr_calc.compute_analytic_ssnr_volume()
            entropy = ssnr_calc.compute_true_ssnr_entropy()
            params = ["Widefield", 1, round(volume), round(volume_a), round(entropy)]
            print(params)
            writer.writerow(params)

            for configuration in config_methods:
                for ratio in ratios:
                    illumination = config_methods[configuration](ratio)
                    ssnr_calc = SSNR3dSIM2dShifts(illumination, optical_system)
                    ssnr = ssnr_calc.compute_ssnr()

                    volume = ssnr_calc.compute_ssnr_volume()
                    volume_a = ssnr_calc.compute_analytic_ssnr_volume()
                    entropy = ssnr_calc.compute_true_ssnr_entropy()

                    print("Volume ", round(volume))
                    print("Volume a", round(volume_a))
                    print("Entropy ", round(entropy))
                    params = [configuration, round(ratio, 2), round(volume), round(volume_a), round(entropy)]
                    print(params)

                    writer.writerow(params)
