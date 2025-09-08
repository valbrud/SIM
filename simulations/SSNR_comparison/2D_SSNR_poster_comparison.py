import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config.BFPConfigurations import BFPConfiguration

configurations = BFPConfiguration()
from OpticalSystems import System4f2D
from SSNRCalculator import SSNRSIM2D
from kernels import psf_kernel2d

plt.rcParams['font.size'] = 50         # Sets default font size
plt.rcParams['axes.titlesize'] = 50     # Title of the axes
plt.rcParams['axes.labelsize'] = 50     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 50    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 50    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 25    # Font size for legend

theta = np.pi / 4
alpha = np.pi / 4
dx = 1 / (8 * np.sin(alpha))
dy = dx
N = 255
max_r = N//2 * dx
NA = np.sin(alpha)
psf_size = 2 * np.array((max_r, max_r))
dx = 2 * max_r / N
dy = 2 * max_r / N
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)

fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)

arg = N // 2

two_NA_fx = fx / (2 * NA)
two_NA_fy = fy / (2 * NA)

optical_system = System4f2D(alpha=alpha)
optical_system.compute_psf_and_otf((psf_size, N))

illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1, dimensionality=2)
illumination_widefield = configurations.get_widefield(dimensionality=2)


for kernel_size in [1, 5, 9]:
        kernel = psf_kernel2d(kernel_size, (dx, dy))

        #-----------finite kernel----------------#
        noise_estimator_finite = SSNRSIM2D(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.ssnri
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnri()

        # noise_estimator_finite.plot_effective_kernel_and_otf()
        # plt.show()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = noise_estimator_finite.ssnri
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnri()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = noise_estimator_finite.ssnri
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnri()

        #-----------FDR----------------#
        noise_estimator_ideal = SSNRSIM2D(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator_ideal.ssnri
        ssnr_widefield_ra = noise_estimator_ideal.ring_average_ssnri()

        noise_estimator_ideal.illumination = illumination_s_polarized
        ssnr_s_polarized = noise_estimator_ideal.ssnri
        ssnr_s_polarized_ra = noise_estimator_ideal.ring_average_ssnri()


        noise_estimator_ideal.illumination = illumination_3waves
        ssnr_3waves = noise_estimator_ideal.ssnri
        ssnr_3waves_ra = noise_estimator_ideal.ring_average_ssnri()

        Fx, Fy = np.meshgrid(fx, fy)
        fig1 = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig1.add_subplot(111)

        multiplier = 10**3

        ax1.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$")
        # ax1.set_title(r"SSNR", fontsize=25)
        ax1.set_yscale("log")
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.set_ylim(1, 1 + multiplier)
        ax1.set_xlim(0, 2)

        fig2 = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax2 = fig2.add_subplot(111)
        # ax2.set_title(f"Size = {kernel_r_size}", fontsize=25)
        ax2.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$")
        if kernel_size == 1:
            ax1.set_ylabel(r"$1 + 10^4 SSNR_{ra}$")
            ax2.set_ylabel(r"$f_r, \frac{2NA}{\lambda}$")

        fig3 = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax3 = fig3.add_subplot(111)

        ax3.set_ylim(10 ** -2, 1.1)
        # ax3.tick_params(labelsize=20)
        ax3.set_yscale("log")
        # ax3.set_title("Radially averaged ratio", fontsize=25)
        ax3.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$")
        ax3.set_ylabel(r"$SSNR_{fin}/SSNR_{id}$")
        ax3.set_xlim(0, fx[-1] / (2 * NA))
        ax3.grid()
        ax3.set_xlim(0, 2)

        ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier *ssnr_finite_widefield_ra), color='black', label="Widefield")
        ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier *ssnr_finite_s_polarized_ra), color='green', label="Lattice SIM")
        ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier *ssnr_finite_3waves_ra), color='red', label="Conventional SIM")
        ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier *ssnr_s_polarized_ra), color='green', linestyle='--', label="Lattice SIM ideal")
        ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier *ssnr_3waves_ra), color='red', linestyle='--', label="Conventional SIM ideal")
        ax1.set_aspect(1. / ax1.get_data_ratio())

        im = ax2.imshow(ssnr_finite_3waves / ssnr_3waves, vmin = 0., vmax = 1.0, extent=(two_NA_fx[0], two_NA_fx[-1], two_NA_fy[0], two_NA_fy[-1]))
        if kernel_size == 9:
            plt.colorbar(im, ax=ax2, extend = 'both', label=r"$SSNR_{finite}/SSNR_{ideal}$")

        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra / ssnr_widefield_ra, label="Widefield")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra/ ssnr_s_polarized_ra, label="Lattice SIM")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra / ssnr_3waves_ra, label="Conventional SIM")
        ax3.set_aspect(1. / ax3.get_data_ratio())

        ax1.legend(fontsize=20, loc="upper right")
        ax3.legend(fontsize=20, loc="lower right")
        # ax2.legend(fontsize=15, loc="lower left")
        plt.show()
        # fig1.savefig(f"./SSNR_finite_kernel_{kernel_size}.png", bbox_inches='tight')
        # fig2.savefig(f"./SSNR_ratio_{kernel_size}_ratio.png", bbox_inches='tight')
        # fig3.savefig(f"simulations/Figures/SSNR_ratio_ra{kernel_size}.png", bbox_inches='tight')

