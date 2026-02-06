import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)
import matplotlib.pyplot as plt
import csv

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d

import Box
from config.BFPConfigurations import *
import unittest
import time
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import Illumination
from SSNRCalculator import SSNRSIM3D, SSNRSIM2D
from OpticalSystems import System4f3D, System4f2D
import utils
from Sources import IntensityHarmonic3D, PlaneWave

path_to_figures = 'Figures/'
path_to_animations = 'Animations/'

if not os.path.exists('Figures/'):
    os.makedirs('Figures/')
if not os.path.exists('Animations/'):
    os.makedirs('Animations/')

sys.path.append('../../')

configurations = BFPConfiguration(refraction_index=1.5)
alpha = 2 * np.pi / 5
nmedium = 1.5
nobject = 1.5
NA = nmedium * np.sin(alpha)
theta = np.asin(0.9 * np.sin(alpha))
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
two_NA_fy = fy / (2 * NA)
scaled_fz = fz / fz_max_diff

multiplier = 10 ** 5
ylim = 10 ** 2

optical_system = System4f3D(alpha=alpha, refractive_index_sample=nobject, refractive_index_medium=nmedium, high_NA=True)
optical_system.compute_psf_and_otf((psf_size, N))

conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
squareL = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 1, Mt=1)
squareC = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.58, 1, Mt=1, phase_shift=0)
hexagonal = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1)

# box = Box.Box(conventional.waves.values(), box_size=psf_size, point_number=N)
# box.compute_intensity_from_spatial_waves()
# fig, ax = plt.subplots()
# ax.set_title("conventional", fontsize=30, pad=15)
# ax.set_ylabel("y [$\lambda$]", fontsize=30)
# ax.set_xlabel("x [$\lambda$]",  fontsize=30)
# ax.tick_params(labelsize=30)
# ax.imshow(box.intensity[:, :, arg].T, extent=(x[0], x[-1], y[0], y[-1]))
# plt.show()

widefield = configurations.get_widefield()

illumination_list = {
    widefield: "Widefield",
    conventional: "Conventional",
    squareL: "SquareL",
    # squareLNonLinear: "SquareLNonLinear",
    squareC: "SquareC",
    hexagonal: "Hexagonal",
}


class TestArticlePlots(unittest.TestCase):
    def test_ring_averaged_ssnr(self):
        n_points = 51
        r = np.linspace(two_NA_fx[N//2], two_NA_fx[-1], n_points)
        noise_estimator_widefield = SSNRSIM3D(widefield, optical_system)
        ssnr_widefield = noise_estimator_widefield.ssnri
        ssnr_widefield_ra = noise_estimator_widefield.ring_average_ssnri(number_of_samples=n_points)

        noise_estimator = SSNRSIM3D(squareL, optical_system)
        ssnr_s_polarized = noise_estimator.ssnri
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri(number_of_samples=n_points)

        noise_estimator.illumination = squareC
        ssnr_circular = noise_estimator.ssnri
        ssnr_circular_ra = noise_estimator.ring_average_ssnri(number_of_samples=n_points)

        noise_estimator.illumination = hexagonal
        ssnr_seven_waves = noise_estimator.ssnri
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri(number_of_samples=n_points)

        noise_estimator.illumination = conventional
        ssnr_3waves = noise_estimator.ssnri
        ssnr_3waves_ra = noise_estimator.ring_average_ssnri(number_of_samples=n_points)

        Fx, Fy = np.meshgrid(fx, fy)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        # fig.suptitle("Ring averaged SSNR for different configurations", fontsize=30)

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)

        ax.clear()
        ax.set_xlabel(r"$f_r \; [LCF]$", fontsize=20)
        ax.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=20)
        ax.set_yscale("log")
        ax.set_ylim(1, 2 * ylim)
        ax.set_xlim(0, two_NA_fx[-1])
        ax.grid(which='major')
        ax.grid(which='minor', linestyle='--')
        ax.tick_params(labelsize=20)

        ax.plot(r, 1 + multiplier / 10 * ssnr_widefield_ra[:, arg], color='black', label="Widefield")
        ax.plot(r, 1 + multiplier / 10 * ssnr_3waves_ra[:, arg], label="Conventional")
        ax.plot(r, 1 + multiplier / 10 * ssnr_s_polarized_ra[:, arg], label="SquareL")
        ax.plot(r, 1 + multiplier / 10 * ssnr_circular_ra[:, arg], label="SquareC")
        ax.plot(r, 1 + multiplier / 10 * ssnr_seven_waves_ra[:, arg], label="Hexagonal")
        ax.set_aspect(1. / ax.get_data_ratio())

        ax.legend(fontsize=18)
        # fig.savefig(f'{path_to_figures}ring_averaged_ssnr_fz={scaled_fz[arg]}.png', pad_inches=0, dpi=300)
        plt.show()

        ax.clear()
        ax.set_xlabel(r"$f_r \; [LCF]$", fontsize=20)
        ax.set_ylabel(r"$1 + 10^5 SSNR_{ra}$", fontsize=20)
        ax.set_yscale("log")
        ax.set_ylim(1, 2 * ylim)
        ax.set_xlim(0, two_NA_fx[-1])
        ax.grid(which='major')
        ax.grid(which='minor', linestyle='--')
        ax.tick_params(labelsize=20)

        ax.plot(r, 1 + multiplier * ssnr_widefield_ra[:, arg//2], color='black', label="Widefield")
        ax.plot(r, 1 + multiplier * ssnr_3waves_ra[:, arg//2], label="Conventional")
        ax.plot(r, 1 + multiplier * ssnr_s_polarized_ra[:, arg//2], label="SquareL")
        ax.plot(r, 1 + multiplier * ssnr_circular_ra[:, arg//2], label="SquareC")
        ax.plot(r, 1 + multiplier * ssnr_seven_waves_ra[:, arg//2], label="Hexagonal")
        ax.set_aspect(1. / ax.get_data_ratio())

        ax.legend(fontsize=18)
        # fig.savefig(f'{path_to_figures}ring_averaged_ssnr_fz={scaled_fz[arg//2]}.png', pad_inches=0, dpi=300)
        plt.show()

    def test_ssnr_color_maps(self):
        for illumination in illumination_list:
            noise_estimator = SSNRSIM3D(illumination, optical_system)

            ssnr = np.abs(noise_estimator.ssnri)
            scaling_factor = 10 ** 8
            ssnr_scaled = 1 + scaling_factor * ssnr
            ssnr_ring_averaged = noise_estimator.ring_average_ssnri()
            ssnr_ra_scaled = 1 + scaling_factor * ssnr_ring_averaged

            Fx, Fy = np.meshgrid(fx, fy)
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            # fig.suptitle(illumination_list[illumination], fontsize=30)
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.1,
                                hspace=0)
            ax1 = fig.add_subplot(121)
            ax1.tick_params(labelsize=30)
            # ax1.set_title(title)
            ax1.set_xlabel("$f_y \; [LCF]$", fontsize=30)
            ax1.set_ylabel("$f_x \;  [LCF]$", fontsize=30)
            mp1 = ax1.imshow(ssnr_scaled[:, :, N // 2], extent=(-2, 2, -2, 2), norm=colors.LogNorm())
            # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
            # cb1.set_label("$1 + 10^8$ ssnr")
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2 = fig.add_subplot(122, sharey=ax1)
            ax2.set_xlabel("$f_z \; [ACF]$", fontsize=30)
            ax2.tick_params(labelsize=30)
            ax2.tick_params('y', labelleft=False)
            # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
            mp2 = ax2.imshow(ssnr_scaled[:, N//2,  :], extent=(-2, 2, -2, 2), norm=colors.LogNorm())
            # mp2 = ax2.imshow(ssnr_ra_scaled[:, :].T, extent=(0, fy[-1]/(2 * NA), fz[0]/(2 * NA), fz[-1]/(2 * NA)), norm=colors.LogNorm())
            cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)
            cb2.ax.tick_params(labelsize=30)
            cb2.set_label("$1 + 10^8$ SSNR", fontsize=30)
            ax2.set_aspect(1. / ax2.get_data_ratio())

            # fig.savefig(f'{path_to_figures}'
            #          + illumination_list[illumination] + '_ssnr.png')

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
        noise_estimator_widefield = SSNRSIM3D(widefield, optical_system)
        ssnr_widefield = noise_estimator_widefield.ssnri
        noise_estimator0 = SSNRSIM3D(illumination0, optical_system)
        noise_estimatorPi4 = SSNRSIM3D(illuminationPi4, optical_system)
        noise_estimatorPi2 = SSNRSIM3D(illuminationPi2, optical_system)
        noise_estimatorPi = SSNRSIM3D(illuminationPi, optical_system)

        ssnr0 = np.abs(noise_estimator0.ssnri)
        ssnrPi4 = np.abs(noise_estimatorPi4.ssnri)
        ssnrPi2 = np.abs(noise_estimatorPi2.ssnri)
        ssnrPi = np.abs(noise_estimatorPi.ssnri)
        scaling_factor = 10 ** 4

        Fx, Fy = np.meshgrid(fx, fy)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        # fig.suptitle(f"Phase of the central beam = {illumination_list[illumination]}", fontsize=30)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.1,
                            hspace=0)
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)
        # ax3 = fig.add_subplot(223)
        # ax4 = fig.add_subplot(224)
        # ax1.set_yscale('log')
        # ax2.set_yscale('log')
        # ax3.set_yscale('log')
        # ax4.set_yscale('log')
        # ax1.set_title('Phase shift = 0', fontsize=30)
        # ax2.set_title('Phase shift = $\pi/4$', fontsize=30)
        # ax3.set_title('Phase shift = $\pi/2$', fontsize=30)
        # ax4.set_title('Phase shift = $\pi$', fontsize=30)
        # ax1.set_xlabel('$k_x [\\frac{2NA}{\lambda}]$', fontsize = 30)
        # ax2.set_xlabel('$k_x [\\frac{2NA}{\lambda}]$', fontsize = 30)
        # ax1.set_ylabel('$1 + 10^4 SSNR$', fontsize=30)
        # ax3.set_ylabel('$1 + 10^4 SSNR$', fontsize=30)
        # ax3.set_xlabel('$k_r [\\frac{2NA}{\lambda}]$', fontsize=30)
        # ax4.set_xlabel('$k_r [\\frac{2NA}{\lambda}]$', fontsize=30)
        # ax1.set_xticklabels([])
        # ax2.set_xticklabels([])
        # ax2.set_yticklabels([])
        # ax4.set_yticklabels([])
        #
        # ax1.grid()
        # ax2.grid()
        # ax3.grid()
        # ax4.grid()
        # ax1.tick_params(labelsize=30)
        # ax2.tick_params(labelsize=30)
        # ax3.tick_params(labelsize=30)
        # ax4.tick_params(labelsize=30)
        # ax1.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        # ax2.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        # ax3.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        # ax4.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
        # ax1.set_ylim(1, 2 * 10 ** 2)
        # ax2.set_ylim(1, 2 * 10 ** 2)
        # ax3.set_ylim(1, 2 * 10 ** 2)
        # ax4.set_ylim(1, 2 * 10 ** 2)
        #
        z_shift = 5
        for ssnr, phase in zip((ssnr0, ssnrPi4, ssnrPi2, ssnrPi), ('0', 'pi4', 'pi2', 'pi')):
            ax.clear()
            ax.grid()
            ax.tick_params(labelsize=30)
            ax.set_yscale('log')
            ax.set_xlabel('$k_r [\\frac{2NA}{\lambda}]$', fontsize=30)
            ax.set_ylabel('$1 + 10^4 SSNR$', fontsize=30)
            ax.plot(two_NA_fx[N //2:], 1 + scaling_factor * ssnr[:, N//2, N//2 + z_shift][N//2:], label='$k_x$')
            ax.plot(two_NA_fx[N //2:], 1 + scaling_factor * ssnr[N//2, :, N//2 + z_shift][N//2:], label='$k_y$')
            ax.legend(fontsize=30)
            ax.set_xlim(0, (2 * NA + 2 * nmedium * np.sin(theta)) / (2 * NA))
            ax.set_ylim(1, 10 ** 2)
            plt.show()
            # fig.savefig(f'{path_to_figures}sim_anisotropy-{phase}')


        # ax1.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnr0[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        # ax1.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnr0[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")
        #
        # ax2.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnrPi4[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        # ax2.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnrPi4[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")
        #
        # ax3.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnrPi2[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        # ax3.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnrPi2[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")
        #
        # ax4.plot(two_NA_fx[N // 2:], 1 + scaling_factor * ssnrPi[:, N // 2, N // 2 + z_shift][N // 2:], label="$k_x$")
        # ax4.plot(two_NA_fy[N // 2:], 1 + scaling_factor * ssnrPi[N // 2, :, N // 2 + z_shift][N // 2:], label="$k_y$")
        #
        # ax1.legend(fontsize=30)
        # ax2.legend(fontsize=30)
        # ax3.legend(fontsize=30)
        # ax4.legend(fontsize=30)

        # plt.show()

    def test_hexagonal_sim_anisotropy(self):
        # theta = alpha/2
        theta = np.pi/4
        p = 1
        k = 2 * np.pi * nmedium
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)

        vec_x = np.array((k * np.sin(theta), 0, k * np.cos(theta)))
        vec_mx = np.array((-k * np.sin(theta), 0, k * np.cos(theta)))
        ax_z = np.array((0, 0, 1))

        a0 = 2 + 6 * p ** 2
        sources = [

            PlaneWave(0, p / a0 ** 0.5, 0, 0, vec_x),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, vec_mx),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 2 * np.pi / 3)),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 2 * np.pi / 3)),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 4 * np.pi / 3)),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 4 * np.pi / 3)),

            PlaneWave(1 * np.exp(1j * np.pi) / a0 ** 0.5, 1j  * np.exp(1j * np.pi) / a0 ** 0.5, 0, 0, np.array((0, 0, k))),
        ]
        distorted1 =  IlluminationPlaneWaves3D.find_ipw_from_pw(sources)

        illumination_distorted1 =  IlluminationPlaneWaves3D.init_from_list(distorted1, (k1 / 2, 3 ** 0.5 / 2 * k1, k2))
        illumination_distorted1.Mt = 1
        illumination_distorted1.normalize_spatial_waves()

        sources = [

            PlaneWave(0, p / a0 ** 0.5, 0, 0, vec_x),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, vec_mx),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 2 * np.pi / 3)),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 2 * np.pi / 3)),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 4 * np.pi / 3)),
            PlaneWave(0, p / a0 ** 0.5, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 4 * np.pi / 3)),

            PlaneWave(1 / a0 ** 0.5, 1j / a0 ** 0.5, 0, 0, np.array((0, 0, k))),
        ]
        distorted0 =  IlluminationPlaneWaves3D.find_ipw_from_pw(sources)

        illumination_distorted0 = IlluminationPlaneWaves3D.init_from_list(distorted0, (k1 / 2, 3 ** 0.5 / 2 * k1, k2))
        illumination_distorted0.Mt = 1
        illumination_distorted0.normalize_spatial_waves()

        noise_estimatorD0 = SSNRSIM3D(illumination_distorted0, optical_system)
        noise_estimatorD1 = SSNRSIM3D(illumination_distorted1, optical_system)
        # noise_estimatorPi2 = SSNRSIM3D(illuminationPi2, optical_system)
        # noise_estimatorPi = SSNRSIM3D(illuminationPi, optical_system)

        ssnrD0 = np.abs(noise_estimatorD0.compute_ssnr())
        ssnrD1 = np.abs(noise_estimatorD1.compute_ssnr())
        entropy0 = noise_estimatorD0.compute_ssnri_entropy()
        entropy1 = noise_estimatorD1.compute_ssnri_entropy()
        print(entropy0)
        print(entropy1)
        # ssnrPi2 = np.abs(noise_estimatorPi2.compute_ssnr())
        # ssnrPi = np.abs(noise_estimatorPi.compute_ssnr())
        scaling_factor = 10 ** 5
        ssnrD0scaled = 1 + scaling_factor * ssnrD0
        ssnrD1scaled = 1 + scaling_factor * ssnrD1

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        # fig.suptitle(f"Phase of the central beam = {illumination_list[illumination]}", fontsize=30)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.1,
                            hspace=0)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # ax1.imshow(ssnr0scaled[:, :, N//2], norm=colors.LogNorm())
        # ax2.imshow(ssnrD1scaled[:, :, N//2], norm=colors.LogNorm())
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.plot(ssnrD0scaled[N//2 + 20, :, N//2])
        ax1.plot(ssnrD1scaled[N//2 + 20, :, N//2])
        # fig.savefig(f'{path_to_figures}sim_anisotropy')
        plt.show()

    def test_sim_modalities_comparison(self):
        ...

    def test_illumination_animations(self):
        r_max = 2
        z_max = 1
        N = 51
        arg = N//2
        x = np.linspace(-r_max, r_max, N)
        y = np.linspace(-r_max, r_max, N)
        z = np.linspace(-z_max, z_max, N)
        psf_size = (2 * r_max, 2 * r_max, 2 * z_max)
        n_illum = len(illumination_list.keys()) - 1
        n_rows = int(n_illum**0.5)
        n_colums = n_rows
        fig, axes = plt.subplots(n_rows, n_colums, figsize=(12, 10), sharex=True, sharey=True)
        fig.suptitle(f'z = {round(z[arg], 2)} [$\lambda$]', fontsize=30)
        boxes = []
        i = 0
        for illumination in illumination_list:
            if illumination_list[illumination] == "Widefield":
                continue
            ax = axes[i//n_colums, i%n_colums]
            boxes.append(Box.BoxSIM(illumination, box_size=psf_size, point_number=N))
            boxes[i].compute_intensity_from_spatial_waves()
            ax.set_title(illumination_list[illumination], fontsize=30, pad=15)
            if i % n_colums == 0:
                ax.set_ylabel("y [$\lambda$]", fontsize=30)
            if i // n_colums == n_rows-1:
                ax.set_xlabel("x [$\lambda$]",  fontsize=30)
            ax.tick_params(labelsize=30)
            ax.imshow(boxes[i].intensity[:, :, arg].T, extent=(x[0], x[-1], y[0], y[-1]))
            i+=1

        plt.tight_layout()
        # plt.show()
        def update(val):
            i = 0
            fig.suptitle(f'$z = {round(z[int(val)], 2)} \; [\lambda]$', fontsize=30)
            for illumination in illumination_list:
                if illumination_list[illumination] == "Widefield":
                    continue
                ax = axes[i // n_colums, i % n_colums]
                ax.set_title(illumination_list[illumination], fontsize=30, pad=15)
                if i % n_colums == 0:
                    ax.set_ylabel("$y \; [\lambda]$", fontsize=30)
                if i // n_colums == n_rows - 1:
                    ax.set_xlabel("$x \; [\lambda]$", fontsize=30)
                ax.tick_params(labelsize=30)
                ax.imshow(boxes[i].intensity[:, :, int(val)].T, extent=(x[0], x[-1], y[0], y[-1]))
                i+=1
        ani = FuncAnimation(fig, update, frames=range(0, N), repeat=False, interval=40)
        # ani.save(path_to_animations +
        #          '3D_illumination.mp4', writer="ffmpeg")
        plt.show()

    def test_plot_power_dependence(self):
        if not os.path.exists('Power_new.csv'):
            raise FileExistsError('Power.csv does not exist. Compute it first with test_compute_power_dependence')
        file_path = 'Power_new.csv'
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
        if not os.path.exists('Angles_new.csv'):
            raise FileExistsError('Angles.csv does not exist. Compute it first with test_compute_angle_dependence')
        file_path = 'Angles_new.csv'
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
        axes1.set_xlabel(r"$sin(\theta_{inc}) / sin(\alpha_{so})$", fontsize=30)
        axes1.set_ylabel("$SSNR^{INC}_V$", fontsize=30)
        axes1.legend(fontsize=20)

        axes2.grid()
        axes2.set_xlim(0.2, 1)
        axes2.set_ylim(0, 0.2)
        axes2.set_aspect(1 / axes2.get_data_ratio())
        axes2.tick_params(labelsize=25)
        axes2.set_xlabel(r'$sin(\theta_{inc}) / sin(\alpha_{so})$', fontsize=30)
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

        squareL = lambda angle: configurations.get_4_oblique_s_waves_and_s_normal_diagonal(angle, 1, 1, Mt=1)
        squareC = lambda angle: configurations.get_4_circular_oblique_waves_and_circular_normal(angle, 0.58, 1, Mt=1)
        hexagonal = lambda angle: configurations.get_6_oblique_s_waves_and_circular_normal(angle, 1, 1, Mt=1)
        conventional = lambda angle: configurations.get_2_oblique_s_waves_and_s_normal(angle, 1, 1, Mt=1)

        config_methods = {
            # "Conventional": conventional,
            "SquareL": squareL,
            # "SquareC": squareC,
            # "Hexagonal": hexagonal,
        }

        ratios = np.linspace(0.2, 1, 41)

        with open("Angles_new.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            widefield = configurations.get_widefield()
            ssnr_calc = SSNRSIM3D(widefield, optical_system)
            ssnr = ssnr_calc.ssnri
            volume = ssnr_calc.compute_ssnri_volume()
            volume_a = ssnr_calc.compute_analytic_ssnri_volume()
            entropy = ssnr_calc.compute_ssnri_entropy()
            params = ["Widefield", 1, round(volume), round(volume_a), round(entropy)]
            print(params)
            writer.writerow(params)

            for configuration in config_methods:
                for ratio in ratios:
                    sin_theta = ratio * (NA / nmedium)
                    theta = np.arcsin(sin_theta)
                    illumination = config_methods[configuration](theta)
                    ssnr_calc = SSNRSIM3D(illumination, optical_system)
                    ssnr = ssnr_calc.ssnri
                    # plt.imshow(np.log(1 + 10**8 * np.abs(ssnr[:, :, N // 2])))
                    # plt.show()

                    volume = ssnr_calc.compute_ssnri_volume()
                    volume_a = ssnr_calc.compute_analytic_ssnri_volume()
                    entropy = ssnr_calc.compute_ssnri_entropy()

                    print("Volume ", round(volume))
                    print("Volume a", round(volume_a))
                    print("Entropy ", round(entropy))
                    params = [configuration, round(ratio, 2), round(volume), round(volume_a), round(entropy)]
                    print(params)

                    writer.writerow(params)

    def test_compute_power_dependence(self):
        headers = ["Configuration", "Power", "Volume", "Volume_a", "Entropy"]

        squareL = lambda power: configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, power, 1, Mt=1)
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

        with open("Power_new.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            widefield = configurations.get_widefield()
            ssnr_calc = SSNRSIM3D(widefield, optical_system)
            ssnr = ssnr_calc.ssnri
            volume = ssnr_calc.compute_ssnri_volume()
            volume_a = ssnr_calc.compute_analytic_ssnri_volume()
            entropy = ssnr_calc.compute_ssnri_entropy()
            params = ["Widefield", 1, round(volume), round(volume_a), round(entropy)]
            print(params)
            writer.writerow(params)

            for configuration in config_methods:
                for ratio in ratios:
                    illumination = config_methods[configuration](ratio)
                    ssnr_calc = SSNRSIM3D(illumination, optical_system)
                    ssnr = ssnr_calc.ssnri

                    volume = ssnr_calc.compute_ssnri_volume()
                    volume_a = ssnr_calc.compute_analytic_ssnri_volume()
                    entropy = ssnr_calc.compute_ssnri_entropy()

                    print("Volume ", round(volume))
                    print("Volume a", round(volume_a))
                    print("Entropy ", round(entropy))
                    params = [configuration, round(ratio, 2), round(volume), round(volume_a), round(entropy)]
                    print(params)

                    writer.writerow(params)
