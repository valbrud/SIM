import sys

import numpy as np

import WienerFiltering
import matplotlib.colors
import ShapesGenerator
import scipy
import matplotlib.pyplot as plt
import SIMulator
import stattools
import wrappers
from config.BFPConfigurations import *
import unittest
from matplotlib.widgets import Slider
import SSNRCalculator
from OpticalSystems import System4f3D, System4f2D
sys.path.append('../')
configurations = BFPConfiguration()
import kernels

class TestAgainstIdeal(unittest.TestCase):
    def test_compare_ssnr(self):
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

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , N)

        kernel_r_size = 7
        kernel_z_size = 7

        kernel = kernels.sinc_kernel(kernel_r_size, kernel_z_size)

        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_finite = SSNRCalculator.SSNRSIM3DFiniteKernel(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_widefield = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_widefield = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.plot_effective_kernel_and_otf()
        plt.show()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_s_polarized = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_s_polarized = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_seven_waves
        ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_seven_waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_seven_waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_3waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_3waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator.ssnri
        ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
        volume_widefield = noise_estimator.compute_ssnri_volume()
        entropy_widefield = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_s_polarized
        ssnr_s_polarized = np.abs(noise_estimator.ssnri)
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri()
        volume_squareSP = noise_estimator.compute_ssnri_volume()
        entropy_s_polarized = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.ssnri)
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri()
        volume_hexagonal = noise_estimator.compute_ssnri_volume()
        entropy_seven_waves = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.ssnri)
        ssnr_3waves_ra = noise_estimator.ring_average_ssnri()
        volume_conventional = noise_estimator.compute_ssnri_volume()
        entropy_3waves = noise_estimator.compute_ssnri_entropy()

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)
        print("Volume finite widefield = ", volume_finite_widefield)
        print("Entropy finite widefield = ", entropy_finite_widefield)

        print("Volume ssnr s_polarized = ", volume_squareSP)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)
        print("Volume finite s_polarized = ", volume_finite_s_polarized)
        print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

        print("Volume ssnr 3waves = ", volume_conventional)
        print("Entropy ssnr 3waves = ", entropy_3waves)
        print("Volume finite 3waves = ", volume_finite_3waves)
        print("Entropy finite 3waves = ", entropy_finite_3waves)

        print("Volume ssnr seven_waves = ", volume_hexagonal)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        print("Volume finite seven_waves = ", volume_finite_seven_waves)
        print("Entropy finite seven_waves = ", entropy_finite_seven_waves)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax1.set_title(r"Ideal SSNR", fontsize=25)
        ax1.set_yscale("log")
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=20)
        ax1.set_ylim(1, 10 ** 4)
        ax1.set_xlim(0, 2)

        ax2.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax2.set_title(r"Finite kernel", fontsize=25)
        ax2.set_yscale("log")
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params(labelsize=20)
        ax2.set_ylim(1, 10**4)
        ax2.set_xlim(0, 2)

        ax3.set_ylim(10 ** -2, 1.1)
        ax3.tick_params(labelsize=20)
        ax3.set_yscale("log")
        ax3.set_title("Ratio of two cases", fontsize=25)
        ax3.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_ylabel(r"$SSNR_{fin}/SSNR_{id}$", fontsize=25)
        ax3.set_xlim(0, fx[-1] / (2 * NA))
        ax3.grid()
        ax3.set_xlim(0, 2)

        multiplier = 10 ** 4
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="Widefield")
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, arg], label="SquareL")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, arg], label="Hexagonal")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, arg], label="Conventional")
        ax1.set_aspect(1 / ax1.get_data_ratio())

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_finite_widefield_ra[:, arg], label="Widefield")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_finite_s_polarized_ra[:, arg], label="SquareL")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_seven_waves_ra[:, arg], label="Hexagonal")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_3waves_ra[:, arg], label="Conventional")
        ax2.set_aspect(1. / ax2.get_data_ratio())

        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra[:, int(arg)] / ssnr_widefield_ra[:, int(arg)], label="Widefield")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra[:, int(arg)] / ssnr_s_polarized_ra[:, int(arg)], label="SquareL")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_seven_waves_ra[:, int(arg)] / ssnr_seven_waves_ra[:, int(arg)], label="Hexagonal")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra[:, int(arg)] / ssnr_3waves_ra[:, int(arg)], label="Conventional")
        ax3.set_aspect(1. / ax3.get_data_ratio())
        # ax3.plot(two_NA_fx, ssnr_finite_widefield[:, arg+3, arg] / ssnr_widefield[:, arg+5, arg], label="Widefield")
        # ax3.plot(two_NA_fx, ssnr_finite_s_polarized[:, arg+3, arg] / ssnr_3waves[:, arg+5, arg], label="SquareL")
        # ax3.plot(two_NA_fx, ssnr_finite_seven_waves[:, arg+3, arg] / ssnr_seven_waves[:, arg+5, arg], label="Hexagonal")
        # ax3.plot(two_NA_fx, ssnr_finite_3waves[:, arg+3, arg] / ssnr_3waves[:, arg+5, arg], label="Conventional")
        # ax3.set_aspect(1. / ax3.get_data_ratio())

        ax1.legend(fontsize=15, loc="lower left")
        ax2.legend(fontsize=15, loc="lower left")
        ax3.legend(fontsize=15, loc="lower left")

        def update1(val):
            ax1.clear()
            ax1.set_title("Comparison of 3D SIM modalities\n $f_z = ${:.1f}".format(fz[int(val)]/(1 - np.cos(NA))) + "$(\\frac{n - \sqrt{n^2 - NA^2}}{\lambda})$", fontsize=25, pad=15)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$ssnr$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 10 ** 4)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, int(val)], label="SquareL")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, int(val)], label="Conventional")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="Widefield")
            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_xlabel(r"$f_r$")
            ax2.set_yscale("log")
            ax2.set_ylim(1, 10 ** 4)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_s_polarized_ra[:, int(val)], label="SquareL")
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_3waves_ra[:, int(val)], label="Conventional")
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_widefield_ra[:, int(val)], label="Widefield")
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

            ax3.clear()
            ax3.set_ylim(10**-2, 1.1)
            ax3.set_yscale("log")
            ax3.set_xlim(0, fx[-1] / (2 * NA))
            ax3.grid()
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra[:, int(val)] / ssnr_widefield_ra[:, int(val)], label="Widefield")
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra[:, int(val)] / ssnr_s_polarized_ra[:, int(val)], label="SquareL")
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_seven_waves_ra[:, int(val)] / ssnr_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra[:, int(val)] / ssnr_3waves_ra[:, int(val)], label="Conventional")
            ax3.set_aspect(1. / ax3.get_data_ratio())
            ax3.legend()

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N)  # slider properties
        slider_ssnr.on_changed(update1)

        plt.show()
    def test_circularly_symmetric_kernel(self):
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

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , N)

        kernel_r_size = 9

        kernel = kernels.psf_kernel2d(kernel_r_size, (dx, dy))
        plt.plot(kernel[:, kernel_r_size//2, 0])
        plt.show()
        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=32)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_finite = SSNRCalculator.SSNRSIM3DFiniteKernel(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_widefield = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_widefield = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.plot_effective_kernel_and_otf()
        plt.show()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_s_polarized = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_s_polarized = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_seven_waves
        ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_seven_waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_seven_waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_3waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_3waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator.ssnri
        ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
        volume_widefield = noise_estimator.compute_ssnri_volume()
        entropy_widefield = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_s_polarized
        ssnr_s_polarized = np.abs(noise_estimator.ssnri)
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri()
        volume_squareSP = noise_estimator.compute_ssnri_volume()
        entropy_s_polarized = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.ssnri)
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri()
        volume_hexagonal = noise_estimator.compute_ssnri_volume()
        entropy_seven_waves = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.ssnri)
        ssnr_3waves_ra = noise_estimator.ring_average_ssnri()
        volume_conventional = noise_estimator.compute_ssnri_volume()
        entropy_3waves = noise_estimator.compute_ssnri_entropy()

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)
        print("Volume finite widefield = ", volume_finite_widefield)
        print("Entropy finite widefield = ", entropy_finite_widefield)

        print("Volume ssnr s_polarized = ", volume_squareSP)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)
        print("Volume finite s_polarized = ", volume_finite_s_polarized)
        print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

        print("Volume ssnr 3waves = ", volume_conventional)
        print("Entropy ssnr 3waves = ", entropy_3waves)
        print("Volume finite 3waves = ", volume_finite_3waves)
        print("Entropy finite 3waves = ", entropy_finite_3waves)

        print("Volume ssnr seven_waves = ", volume_hexagonal)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        print("Volume finite seven_waves = ", volume_finite_seven_waves)
        print("Entropy finite seven_waves = ", entropy_finite_seven_waves)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax1.set_title(r"Ideal SSNR", fontsize=25)
        ax1.set_yscale("log")
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=20)
        ax1.set_ylim(1, 10 ** 4)
        ax1.set_xlim(0, 2)

        ax2.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax2.set_title(r"Finite kernel", fontsize=25)
        ax2.set_yscale("log")
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params(labelsize=20)
        ax2.set_ylim(1, 10**4)
        ax2.set_xlim(0, 2)

        ax3.set_ylim(10 ** -2, 1.1)
        ax3.tick_params(labelsize=20)
        ax3.set_yscale("log")
        ax3.set_title("Ratio of two cases", fontsize=25)
        ax3.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_ylabel(r"$SSNR_{fin}/SSNR_{id}$", fontsize=25)
        ax3.set_xlim(0, fx[-1] / (2 * NA))
        ax3.grid()
        ax3.set_xlim(0, 2)

        multiplier = 10 ** 4
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="Widefield")
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, arg], label="SquareL")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, arg], label="Hexagonal")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, arg], label="Conventional")
        ax1.set_aspect(1 / ax1.get_data_ratio())

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_finite_widefield_ra[:, arg], label="Widefield")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_finite_s_polarized_ra[:, arg], label="SquareL")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_seven_waves_ra[:, arg], label="Hexagonal")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_3waves_ra[:, arg], label="Conventional")
        ax2.set_aspect(1. / ax2.get_data_ratio())

        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra[:, int(arg)] / ssnr_widefield_ra[:, int(arg)], label="Widefield")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra[:, int(arg)] / ssnr_s_polarized_ra[:, int(arg)], label="SquareL")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_seven_waves_ra[:, int(arg)] / ssnr_seven_waves_ra[:, int(arg)], label="Hexagonal")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra[:, int(arg)] / ssnr_3waves_ra[:, int(arg)], label="Conventional")
        ax3.set_aspect(1. / ax3.get_data_ratio())
        # ax3.plot(two_NA_fx, ssnr_finite_widefield[:, arg+3, arg] / ssnr_widefield[:, arg+5, arg], label="Widefield")
        # ax3.plot(two_NA_fx, ssnr_finite_s_polarized[:, arg+3, arg] / ssnr_3waves[:, arg+5, arg], label="SquareL")
        # ax3.plot(two_NA_fx, ssnr_finite_seven_waves[:, arg+3, arg] / ssnr_seven_waves[:, arg+5, arg], label="Hexagonal")
        # ax3.plot(two_NA_fx, ssnr_finite_3waves[:, arg+3, arg] / ssnr_3waves[:, arg+5, arg], label="Conventional")
        # ax3.set_aspect(1. / ax3.get_data_ratio())

        ax1.legend(fontsize=15, loc="lower left")
        ax2.legend(fontsize=15, loc="lower left")
        ax3.legend(fontsize=15, loc="lower left")

        def update1(val):
            ax1.clear()
            ax1.set_title("Comparison of 3D SIM modalities\n $f_z = ${:.1f}".format(fz[int(val)]/(1 - np.cos(NA))) + "$(\\frac{n - \sqrt{n^2 - NA^2}}{\lambda})$", fontsize=25, pad=15)
            ax1.set_xlabel(r"$f_r$", fontsize=25)
            ax1.set_ylabel(r"$ssnr$", fontsize=25)
            ax1.set_yscale("log")
            ax1.set_ylim(1, 10 ** 4)
            ax1.set_xlim(0, fx[-1] / (2 * NA))
            ax1.grid()
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, int(val)], label="SquareL")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, int(val)], label="Conventional")
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_widefield_ra[:, int(val)], label="Widefield")
            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())

            ax2.clear()
            ax2.set_xlabel(r"$f_r$")
            ax2.set_yscale("log")
            ax2.set_ylim(1, 10 ** 4)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_s_polarized_ra[:, int(val)], label="SquareL")
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_3waves_ra[:, int(val)], label="Conventional")
            ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_widefield_ra[:, int(val)], label="Widefield")
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

            ax3.clear()
            ax3.set_ylim(10**-2, 1.1)
            ax3.set_yscale("log")
            ax3.set_xlim(0, fx[-1] / (2 * NA))
            ax3.grid()
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra[:, int(val)] / ssnr_widefield_ra[:, int(val)], label="Widefield")
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra[:, int(val)] / ssnr_s_polarized_ra[:, int(val)], label="SquareL")
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_seven_waves_ra[:, int(val)] / ssnr_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra[:, int(val)] / ssnr_3waves_ra[:, int(val)], label="Conventional")
            ax3.set_aspect(1. / ax3.get_data_ratio())
            ax3.legend()

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N)  # slider properties
        slider_ssnr.on_changed(update1)

        plt.show()

    def test_compare_ssnr_colormaps(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N//2 * dx
        max_z = N//2 * dz

        kernel_r_size = 7
        kernel_z_size = 1
        kernel = np.zeros((kernel_r_size, kernel_r_size, kernel_z_size))
        func_r = np.zeros(kernel_r_size)
        func_r[0:kernel_r_size // 2 + 1] = np.linspace(0, 1, (kernel_r_size + 1) // 2 + 1)[1:]
        func_r[kernel_r_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_r_size + 1) // 2 + 1)[:-1]
        func_z = np.zeros(kernel_z_size)
        func_z[0:kernel_z_size // 2 + 1] = np.linspace(0, 1, (kernel_z_size + 1) // 2 + 1)[1:]
        func_z[kernel_z_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_z_size + 1) // 2 + 1)[:-1]
        func2d = func_r[:, None] * func_r[None, :]
        func3d = func_r[:, None, None] * func_r[None, :, None] * func_z[None, None, :]
        # kernel[0, 0, 0] = 1
        # kernel[:, :,  0] = func2d
        kernel = func3d

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

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_finite = SSNRCalculator.SSNRSIM3DFiniteKernel(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_widefield = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_widefield = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_s_polarized = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_s_polarized = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_seven_waves
        ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_seven_waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_seven_waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_3waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_3waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator.ssnri
        ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
        volume_widefield = noise_estimator.compute_ssnri_volume()
        entropy_widefield = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_s_polarized
        ssnr_s_polarized = np.abs(noise_estimator.ssnri)
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri()
        volume_squareSP = noise_estimator.compute_ssnri_volume()
        entropy_s_polarized = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.ssnri)
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri()
        volume_hexagonal = noise_estimator.compute_ssnri_volume()
        entropy_seven_waves = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.ssnri)
        ssnr_3waves_ra = noise_estimator.ring_average_ssnri()
        volume_conventional = noise_estimator.compute_ssnri_volume()
        entropy_3waves = noise_estimator.compute_ssnri_entropy()

        ssnr_diff_conventional = ssnr_3waves - ssnr_finite_3waves
        ssnr_diff_square = ssnr_s_polarized - ssnr_finite_s_polarized
        ssnr_diff_hexagonal = ssnr_seven_waves - ssnr_finite_seven_waves

        conventional_ratio = ssnr_finite_3waves/ssnr_3waves
        square_ratio = ssnr_finite_s_polarized/ssnr_s_polarized
        hexagonal_ratio = ssnr_finite_seven_waves/ssnr_seven_waves

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)
        print("Volume finite widefield = ", volume_finite_widefield)
        print("Entropy finite widefield = ", entropy_finite_widefield)

        print("Volume ssnr s_polarized = ", volume_squareSP)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)
        print("Volume finite s_polarized = ", volume_finite_s_polarized)
        print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

        print("Volume ssnr 3waves = ", volume_conventional)
        print("Entropy ssnr 3waves = ", entropy_3waves)
        print("Volume finite 3waves = ", volume_finite_3waves)
        print("Entropy finite 3waves = ", entropy_finite_3waves)

        print("Volume ssnr seven_waves = ", volume_hexagonal)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        print("Volume finite seven_waves = ", volume_finite_seven_waves)
        print("Entropy finite seven_waves = ", entropy_finite_seven_waves)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        scale = 10**4

        ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_title(r"Conventional", fontsize=25)
        ax1.imshow(1 + scale * ssnr_diff_conventional[:, :, N//2], extent=(-2, 2, -2, 2), cmap = 'viridis', norm=matplotlib.colors.LogNorm(vmin=1, vmax=scale))

        ax2.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_title(r"Square", fontsize=25)
        ax2.imshow(1 + scale * ssnr_diff_square[:, :, N//2], extent=(-2, 2, -2, 2), cmap = 'viridis', norm=matplotlib.colors.LogNorm(vmin=1, vmax=scale))

        ax3.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_title(r"Hexagonal", fontsize=25)
        ax3.imshow(1 + scale * ssnr_diff_hexagonal[:, :, N//2], extent=(-2, 2, -2, 2), cmap = 'viridis', norm=matplotlib.colors.LogNorm(vmin=1, vmax=scale))

        def update1(val):
            ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
            ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
            ax1.set_title(r"Conventional", fontsize=25)
            ax1.imshow(1 + scale * ssnr_diff_conventional[:, :, int(val)], extent=(-2, 2, -2, 2), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=1, vmax=scale))

            ax2.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
            ax2.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
            ax2.set_title(r"Square", fontsize=25)
            ax2.imshow(1 + scale * ssnr_diff_square[:, :, int(val)], extent=(-2, 2, -2, 2), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=1, vmax=scale))

            ax3.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
            ax3.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
            ax3.set_title(r"Hexagonal", fontsize=25)
            ax3.imshow(1 + scale * ssnr_diff_hexagonal[:, :, int(val)], extent=(-2, 2, -2, 2), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=1, vmax=scale))

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N)  # slider properties
        slider_ssnr.on_changed(update1)

        plt.show()

    def test_compare_ssnr_ratios(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N//2 * dx
        max_z = N//2 * dz

        kernel_r_size = 5
        kernel_z_size = 1
        kernel = np.zeros((kernel_r_size, kernel_r_size, kernel_z_size))
        func_r = np.zeros(kernel_r_size)
        func_r[0:kernel_r_size // 2 + 1] = np.linspace(0, 1, (kernel_r_size + 1) // 2 + 1)[1:]
        func_r[kernel_r_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_r_size + 1) // 2 + 1)[:-1]
        func_z = np.zeros(kernel_z_size)
        func_z[0:kernel_z_size // 2 + 1] = np.linspace(0, 1, (kernel_z_size + 1) // 2 + 1)[1:]
        func_z[kernel_z_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_z_size + 1) // 2 + 1)[:-1]
        func2d = func_r[:, None] * func_r[None, :]
        func3d = func_r[:, None, None] * func_r[None, :, None] * func_z[None, None, :]
        # kernel[0, 0, 0] = 1
        # kernel[:, :,  0] = func2d
        kernel = func3d

        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
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

        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_finite = SSNRCalculator.SSNRSIM3DFiniteKernel(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_widefield = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_widefield = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_s_polarized = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_s_polarized = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_seven_waves
        ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_seven_waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_seven_waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_3waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_3waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator.ssnri
        ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
        volume_widefield = noise_estimator.compute_ssnri_volume()
        entropy_widefield = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_s_polarized
        ssnr_s_polarized = np.abs(noise_estimator.ssnri)
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri()
        volume_squareSP = noise_estimator.compute_ssnri_volume()
        entropy_s_polarized = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.ssnri)
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri()
        volume_hexagonal = noise_estimator.compute_ssnri_volume()
        entropy_seven_waves = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.ssnri)
        ssnr_3waves_ra = noise_estimator.ring_average_ssnri()
        volume_conventional = noise_estimator.compute_ssnri_volume()
        entropy_3waves = noise_estimator.compute_ssnri_entropy()

        ssnr_diff_conventional = ssnr_3waves - ssnr_finite_3waves
        ssnr_diff_square = ssnr_s_polarized - ssnr_finite_s_polarized
        ssnr_diff_hexagonal = ssnr_seven_waves - ssnr_finite_seven_waves

        conventional_ratio = ssnr_finite_3waves/ssnr_3waves
        square_ratio = ssnr_finite_s_polarized/ssnr_s_polarized
        hexagonal_ratio = ssnr_finite_seven_waves/ssnr_seven_waves

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)
        print("Volume finite widefield = ", volume_finite_widefield)
        print("Entropy finite widefield = ", entropy_finite_widefield)

        print("Volume ssnr s_polarized = ", volume_squareSP)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)
        print("Volume finite s_polarized = ", volume_finite_s_polarized)
        print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

        print("Volume ssnr 3waves = ", volume_conventional)
        print("Entropy ssnr 3waves = ", entropy_3waves)
        print("Volume finite 3waves = ", volume_finite_3waves)
        print("Entropy finite 3waves = ", entropy_finite_3waves)

        print("Volume ssnr seven_waves = ", volume_hexagonal)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        print("Volume finite seven_waves = ", volume_finite_seven_waves)
        print("Entropy finite seven_waves = ", entropy_finite_seven_waves)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        scale = 10**4

        ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_title(r"Conventional", fontsize=25)
        ax1.imshow(conventional_ratio[:, :, N//2], extent=(-2, 2, -2, 2), cmap = 'viridis', norm=matplotlib.colors.LogNorm(vmin=10**-2, vmax=1))

        ax2.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_title(r"Square", fontsize=25)
        ax2.imshow(square_ratio[:, :, N//2], extent=(-2, 2, -2, 2), cmap = 'viridis', norm=matplotlib.colors.LogNorm(vmin=10**-2, vmax=1))

        ax3.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_title(r"Hexagonal", fontsize=25)
        ax3.imshow(hexagonal_ratio[:, :, N//2], extent=(-2, 2, -2, 2), cmap = 'viridis', norm=matplotlib.colors.LogNorm(vmin=10**-2, vmax=1))
        def update1(val):

            ax1.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
            ax1.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
            ax1.set_title(r"Conventional", fontsize=25)
            ax1.imshow(conventional_ratio[:, :, int(val)], extent=(-2, 2, -2, 2), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=10 ** -2, vmax=1))

            ax2.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
            ax2.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
            ax2.set_title(r"Square", fontsize=25)
            ax2.imshow(square_ratio[:, :, int(val)], extent=(-2, 2, -2, 2), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=10 ** -2, vmax=1))

            ax3.set_xlabel(r"$f_x, \frac{2NA}{\lambda}$", fontsize=25)
            ax3.set_ylabel(r"$f_y, \frac{2NA}{\lambda}$", fontsize=25)
            ax3.set_title(r"Hexagonal", fontsize=25)
            ax3.imshow(hexagonal_ratio[:, :, int(val)], extent=(-2, 2, -2, 2), cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=10 ** -2, vmax=1))

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N)  # slider properties
        slider_ssnr.on_changed(update1)

        plt.show()

    def test_compare_ssnr_2d(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        N = 51
        max_r = N//2 * dx
        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)

        kernel_r_size = 5

        kernel = kernels.psf_kernel2d(kernel_r_size, (dx, dy))[:, :, 0]

        arg = N // 2

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)

        optical_system = System4f2D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_finite = SSNRCalculator.SSNRSIM2DFiniteKernel(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_widefield = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_widefield = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.plot_effective_kernel_and_otf()
        plt.show()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_s_polarized = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_s_polarized = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_seven_waves
        ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_seven_waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_seven_waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnri()
        volume_finite_3waves = noise_estimator_finite.compute_ssnri_volume()
        entropy_finite_3waves = noise_estimator_finite.compute_ssnri_entropy()

        noise_estimator = SSNRCalculator.SSNRSIM2D(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator.ssnri
        ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
        volume_widefield = noise_estimator.compute_ssnri_volume()
        entropy_widefield = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_s_polarized
        ssnr_s_polarized = np.abs(noise_estimator.ssnri)
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnri()
        volume_squareSP = noise_estimator.compute_ssnri_volume()
        entropy_s_polarized = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.ssnri)
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnri()
        volume_hexagonal = noise_estimator.compute_ssnri_volume()
        entropy_seven_waves = noise_estimator.compute_ssnri_entropy()

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.ssnri)
        ssnr_3waves_ra = noise_estimator.ring_average_ssnri()
        volume_conventional = noise_estimator.compute_ssnri_volume()
        entropy_3waves = noise_estimator.compute_ssnri_entropy()

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)
        print("Volume finite widefield = ", volume_finite_widefield)
        print("Entropy finite widefield = ", entropy_finite_widefield)

        print("Volume ssnr s_polarized = ", volume_squareSP)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)
        print("Volume finite s_polarized = ", volume_finite_s_polarized)
        print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

        print("Volume ssnr 3waves = ", volume_conventional)
        print("Entropy ssnr 3waves = ", entropy_3waves)
        print("Volume finite 3waves = ", volume_finite_3waves)
        print("Entropy finite 3waves = ", entropy_finite_3waves)

        print("Volume ssnr seven_waves = ", volume_hexagonal)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        print("Volume finite seven_waves = ", volume_finite_seven_waves)
        print("Entropy finite seven_waves = ", entropy_finite_seven_waves)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(18, 8), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        multiplier = 10**3

        ax1.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax1.set_title(r"Ideal SSNR", fontsize=25)
        ax1.set_yscale("log")
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=20)
        ax1.set_ylim(1, 1 + multiplier)
        ax1.set_xlim(0, 2)

        ax2.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax2.set_title(r"Finite kernel", fontsize=25)
        ax2.set_yscale("log")
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params(labelsize=20)
        ax2.set_ylim(1, 1 + multiplier)
        ax2.set_xlim(0, 2)

        ax3.set_ylim(10 ** -2, 1.1)
        ax3.tick_params(labelsize=20)
        ax3.set_yscale("log")
        ax3.set_title("Ratio of two cases", fontsize=25)
        ax3.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax3.set_ylabel(r"$SSNR_{fin}/SSNR_{id}$", fontsize=25)
        ax3.set_xlim(0, fx[-1] / (2 * NA))
        ax3.grid()
        ax3.set_xlim(0, 2)

        ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier * ssnr_widefield_ra), label="Widefield")
        ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier *ssnr_s_polarized_ra), label="SquareL")
        ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier *ssnr_seven_waves_ra), label="Hexagonal")
        ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier *ssnr_3waves_ra), label="Conventional")
        ax1.set_aspect(1 / ax1.get_data_ratio())

        ax2.plot(two_NA_fy[fy >= 0], (1 + multiplier *ssnr_finite_widefield_ra), label="Widefield")
        ax2.plot(two_NA_fy[fy >= 0], (1 + multiplier *ssnr_finite_s_polarized_ra), label="SquareL")
        ax2.plot(two_NA_fx[fx >= 0], (1 + multiplier *ssnr_finite_seven_waves_ra), label="Hexagonal")
        ax2.plot(two_NA_fx[fx >= 0], (1 + multiplier *ssnr_finite_3waves_ra), label="Conventional")
        ax2.set_aspect(1. / ax2.get_data_ratio())

        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra / ssnr_widefield_ra, label="Widefield")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra/ ssnr_s_polarized_ra, label="SquareL")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_seven_waves_ra / ssnr_seven_waves_ra, label="Hexagonal")
        ax3.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra / ssnr_3waves_ra, label="Conventional")
        ax3.set_aspect(1. / ax3.get_data_ratio())

        ax1.legend(fontsize=15, loc="lower left")
        ax2.legend(fontsize=15, loc="lower left")
        plt.show()
class TestFinalFilter(unittest.TestCase):
    def test_finite_wiener3d(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz

        kernel_r_size = 7
        kernel_z_size = 1
        filter_size = 11
        kernel_grid = np.zeros((kernel_r_size, kernel_r_size, 2))
        for i in range(kernel_r_size):
            for j in range(kernel_r_size):
                kernel_grid[i, j] = np.array((dx * (i - kernel_r_size//2), dy * (j - kernel_r_size//2)))

        kernel = kernels.sinc_kernel(kernel_r_size, 1)
        anti_kernel = np.zeros(kernel.shape)
        # anti_kernel[kernel_r_size//2, kernel_r_size//2, 0] = np.sum(kernel)
        anti_kernel = -kernel
        # plt.plot(anti_kernel[:, kernel_r_size//2, 0])
        # print(anti_kernel)

        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r, max_z))
        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), N)

        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)
        two_NA_fz = fz / (1 - np.cos(alpha))

        optical_system = System4f2D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")


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

        illumination = illumination_2waves
        # illumination_3waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0)
        noise_estimator_finite = SSNRCalculator.SSNRSIM2D(illumination_2waves, optical_system, kernel[:, :, 0])

        ssnr_finite = np.abs(noise_estimator_finite.compute_ssnr())
        plt.imshow(ssnr_finite)
        plt.show()
        minssnr = np.amin(ssnr_finite[N//2, N//4:3 * N//4])
        maxssnr = np.amax(ssnr_finite[N//2,  N//4 : 3 * N//4])
        ratio = maxssnr/minssnr
        points = []
        values = []
        for r in range(illumination.Mr):
            angle = illumination.angles[r]
            point = np.array((2 * dx * np.cos(angle),  2 * dx * np.sin(angle)))
            points.append(point)
            points.append(-point)
            values.append(1/2 )
            values.append(1/2 )
        points = np.array(points)
        values = np.array(values)
        #
        tj_approximate = np.zeros((filter_size, filter_size, 1))
        tj_approximate = stattools.reverse_interpolation_nearest(x[N//2 - filter_size//2: N//2 + filter_size//2 + 1],
                                                              y[N//2 - filter_size//2: N//2 + filter_size//2 + 1], points, values)
        for r in range(illumination.Mr):
            angle = illumination.angles[r]

        apodization_size = 3
        apodization = kernels.psf_kernel2d(apodization_size, (dx, dy))
        apodization/= np.amax(apodization)
        # apodization += 1/ratio
        # plt.imshow(apodization)
        # plt.show()
        pad_value = (filter_size - apodization_size)//2
        apodization = np.pad(apodization, ((pad_value, pad_value), (pad_value, pad_value), (0, 0)))
        # tj_approximate = scipy.ndimage.convolve(tj_approximate, apodization, mode='constant', cval=0.0, origin=0)
        pad_value = N//2 - filter_size//2

        # tj = 1 / (noise_estimator_finite.dj + np.amax(noise_estimator_finite.dj)/10**4)
        # tj_approximate = tj[N//2 - filter_size//2: N//2 + filter_size//2 + 1,
        #                     N//2 - filter_size//2 : N//2 + filter_size//2 + 1,
        #                     N//2]
        tj_approximate = np.pad(tj_approximate, ((pad_value, pad_value), (pad_value, pad_value)))

        tj_ft = wrappers.wrapped_fftn(tj_approximate[:, :])
        # plt.plot(np.abs(tj_ft[:, N//2]) )
        plt.plot(tj_ft[:, N//2] * noise_estimator_finite.ssnr[:, N//2])
        plt.show()

        # plt.imshow(interpolated_values)
        # plt.show()
        image = np.zeros((N, N, N))
        image[N//2, N//2, N//2] = 10**9
        # image[(3*N+1)//4, N//2+4, N//2] = 10**9
        # image[(N+1)//4, N//2-3, N//2] = 10**9
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**9
        # image += 10**6
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.25, N=1000, I=10 ** 5)
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=10000, I=10 ** 4)
        # image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, r=0.1, N=10000, I=10 ** 10)
        image += 100
        arg = N // 2
        # wiener = SSNRBasedFiltering.WienerFilter3dModel(noise_estimator_finite)
        # ideal, _, _, _, tj = wiener.filter_object(image)
        # tj_real = wrappers.wrapped_fftn(tj)
        # plt.imshow(np.abs(tj_real[:, :, N//2]))
        # plt.show()
        # tj_real /= np.amax(tj_real)
        # filter_size = 51
        # tj_approximate = np.zeros((filter_size, filter_size, 1))
        # tj_approximate[:, :, 0] = tj_real[N//2-filter_size//2: N//2 + filter_size//2 + 1, N//2-filter_size//2: N//2 + filter_size//2 + 1, N//2]
        simulator = SIMulator.SIMulator(illumination, optical_system, psf_size, N)
        images = simulator.generate_sim_images(image)
        image_sr = simulator.reconstruct_real2d_finite_kernel(images, kernel)

        image_filtered = scipy.ndimage.convolve(image_sr, tj_approximate[:, :], mode='constant', cval=0.0, origin=0)
        # image_filtered = scipy.signal.convolve(image_sr, tj_real, mode='same')
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
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)
        ax1.set_title("Widefield", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("SR", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("Filtered", fontsize=25)
        ax3.tick_params(labelsize=20)

        ax1.imshow(image_widefield[:, :, N // 2], vmin=0)
        ax2.imshow(image_sr[:, :, N // 2], vmin=0)
        ax3.imshow(np.abs(image_filtered[:, :, N//2]), vmin=0)
        # ax4.imshow(np.abs(ideal[:, :, N//2]), vmin=0)

        image_widefield /= np.amax(image_widefield)
        image_sr /= np.amax(image_sr)
        image_filtered /= np.amax(image_filtered)
        ideal /= np.amax(ideal)

        image_widefield_ft = wrappers.wrapped_fftn(image_widefield)
        image_sr_ft = wrappers.wrapped_fftn(image_sr)
        image_filtered_ft = wrappers.wrapped_fftn(image_filtered)
        # image_ideal_ft = wrappers.wrapped_fftn(ideal)
        image_ft = wrappers.wrapped_fftn(image)

        image_widefield2d_ft = np.sum(image_widefield_ft, axis=2)
        image_sr2d_ft = np.sum(image_sr_ft, axis=2)
        image_filtered2d_ft = np.sum(image_filtered_ft, axis=2)
        # image_ideal2d_ft = np.sum(image_ideal_ft, axis=2)
        image2d_ft = np.sum(image_ft, axis=2)

        otf_widefield = image_widefield_ft / image_ft
        otf_sr = image_sr_ft / image_ft
        otf_filtered = image_filtered_ft / image_ft
        # otf_ideal = image_ideal_ft / image_ft

        otf_widefield2d = image_widefield2d_ft / image2d_ft
        otf_sr2d = image_sr2d_ft / image2d_ft
        otf_filtered2d = image_filtered2d_ft / image2d_ft
        # otf_ideal2d = image_idea/l2d_ft / image2d_ft
        def update1(val):
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # im1 = ax1.imshow(image_widefield[:, :, int(val)], vmin=0)
            # im2 = ax2.imshow(image_sr[:, :, int(val)], vmin=0)
            # im3 = ax3.imshow(image_filtered[:, :, int(val)], vmin=0)
            # im4 = ax4.imshow(filtered[:, :, int(val)], vmin=0)
            ax1.plot(image_widefield[:, N//2, int(val)], label = 'widefield')
            ax1.plot(image_sr[:, N//2, int(val)], label = 'sr')
            ax1.plot(image_filtered[:, N//2, int(val)], label = 'filtered')
            ax1.plot(ideal[:, N//2, int(val)], label = 'ideal')
            ax1.set_ylim(np.min(image_filtered), np.max(ideal))
            ax1.legend()

            ax2.plot(otf_widefield[:, N//2, int(val)], label = 'otf widefield')
            ax2.plot(otf_sr[:, N//2, int(val)], label = 'otf sr')
            ax2.plot(otf_filtered[:, N//2, int(val)], label = 'otf fitlered')
            ax2.plot(otf_ideal[:, N//2, int(val)], label = 'otf ideal')
            ax2.set_ylim(0, otf_widefield[N//2, N//2, N//2])
            ax2.legend()

            ax3.plot(otf_widefield2d[:, N//2], label = 'otf widefield 2d')
            ax3.plot(otf_sr2d[:, N//2], label = 'otf sr 2d')
            ax3.plot(otf_filtered2d[:, N//2], label = 'otf fitlered 2d')
            ax3.plot(otf_ideal2d[:, N//2], label = 'otf ideal 2d')
            # ax3.set_ylim(0, otf_widefield2d[N//2, N//2])
            ax3.legend()

            # ax3.set_ylim(np.min(image_filtered), np.max(image_filtered))
            # ax4.set_ylim(np.min(filtered), np.max(filtered))
            ax1.set_aspect(1 / ax1.get_data_ratio())
            ax2.set_aspect(1 / ax2.get_data_ratio())
            ax3.set_aspect(1 / ax3.get_data_ratio())
            # ax4.set_aspect(1 / ax4.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)
        plt.show()

    def test_finite_wiener2d(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N//2 * dz



        NA = np.sin(alpha)
        psf_size = 2 * np.array((max_r, max_r))
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)

        arg = N // 2

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)

        image = np.zeros((N, N, N))
        # image[N//2, N//2] = 10**6
        # image[(3*N+1)//4, N//2+4, N//2] = 10**9
        # image[(N+1)//4, N//2-3, N//2] = 10**9
        # image[(N+1)//4, (3 * N+1)//4, N//2] = 10**9
        # image += 10**6
        # image = ShapesGenerator.generate_sphere_slices((np.array((2 * max_r, 2 * max_r, 2 * max_z))), N, r=0.5, N=1000, I=20)
        image = ShapesGenerator.generate_random_lines(psf_size, N, 0.1, 100, 1000)
        # image += ShapesGenerator.generate_sphere_slices(N, )
        # plt.imshow(image)
        # plt.show()
        # image = ShapesGenerator.generate_random_spherical_particles((np.array((2 * max_r, 2 * max_r, 2 * max_z))), N, r=0.1, N=10000, I=100)
        image += 100
        image = image
        # plt.imshow(image)
        # plt.show()
        optical_system = System4f2D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_function="Sine")

        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination_2waves.spatial_shifts = spatial_shifts_conventional2d

        illumination = illumination_2waves
        kernel_size = 1
        kernel = kernels.psf_kernel2d(kernel_size, (dx, dx))[:, :, 0]
        noise_estimator_finite = SSNRCalculator.SSNRSIM2DFiniteKernel(illumination_2waves, optical_system, kernel)
        plt.imshow(np.abs(noise_estimator_finite.ssnr))
        plt.show()

        ssnr_finite = np.abs(noise_estimator_finite.compute_ssnr())
        minssnr = np.amin(ssnr_finite[N // 2, N // 4:3 * N // 4])
        maxssnr = np.amax(ssnr_finite[N // 2, N // 4: 3 * N // 4])
        ratio = maxssnr / minssnr
        points = []
        values = []
        for r in range(illumination.Mr):
            angle = illumination.angles[r]
            point = np.array((2 * dx * np.cos(angle), 2 * dx * np.sin(angle)))
            points.append(point)
            points.append(-point)
            values.append(1 / 2)
            values.append(1 / 2)
        points = np.array(points)
        values = np.array(values)

        filter_size = N

        tj_approximate = np.zeros((filter_size, filter_size))
        tj_approximate = stattools.reverse_interpolation_nearest(x[N // 2 - filter_size // 2: N // 2 + filter_size // 2 + 1],
                                                                          y[N // 2 - filter_size // 2: N // 2 + filter_size // 2 + 1], points, values)

        apodization_size = 3
        apodization = kernels.psf_kernel2d(apodization_size, (dx, dy))
        apodization /= np.amax(apodization)
        # apodization += 1/ratio
        # plt.imshow(apodization)
        # plt.show()
        pad_value = (filter_size - apodization_size) // 2
        apodization = np.pad(apodization, ((pad_value, pad_value), (pad_value, pad_value), (0, 0)))
        # tj_approximate = scipy.ndimage.convolve(tj_approximate, apodization, mode='constant', cval=0.0, origin=0)
        pad_value = N // 2 - filter_size // 2

        # tj = 1 / np.abs((noise_estimator_finite.kernel + np.amax(noise_estimator_finite.dj)/10**4))
        # plt.imshow(tj)
        # plt.show()
        # tj_approximate = np.zeros((filter_size, filter_size, 1))
        # tj_approximate[:, :, 0] = tj[N//2 - filter_size//2: N//2 + filter_size//2 + 1,
        #                     N//2 - filter_size//2 : N//2 + filter_size//2 + 1]
        # tj_approximate = np.pad(tj_approximate, ((pad_value, pad_value), (pad_value, pad_value), (0, 0)))
        #
        # tj_ft = wrappers.wrapped_fftn(tj_approximate[:, :])
        # plt.imshow((np.abs(1 + np.log(tj_ft))))
        # plt.plot(tj_ft[:, N//2] * noise_estimator_finite.ssnr[N//2, :,  N//2])
        # plt.show()

        simulator = SIMulator.SIMulator(illumination, optical_system, psf_size, N)
        images = simulator.generate_sim_images2d(image)
        # for rotation in images:
        #     for image in rotation:
        #         plt.imshow(image)
        #         plt.show()

        image_sr = simulator.reconstruct_real2d_finite_kernel(images, kernel)
        image_widefield = simulator.generate_widefield(images)
        # plt.imshow(image_widefield)
        # plt.show()
        # plt.imshow(image_sr)
        # plt.show()
        wiener = WienerFiltering.WienerFilter3dModel(noise_estimator_finite)
        ideal, _, _, _, tj = wiener.filter_object(image)
        # plt.gca().set_xlabel("fy")
        # plt.gca().set_ylabel("tj")
        # plt.plot(fx, np.abs(tj[N//2, :]))
        # plt.plot(fx, np.abs(tj[:, N//2]))
        # plt.show()
        ideal = wrappers.wrapped_ifftn(tj * wrappers.wrapped_fftn(image_sr))
        tj_real = wrappers.wrapped_fftn(tj)
        tj_real /= np.amax(tj_real)
        filter_size = 15
        tj_approximate = tj_real[N//2-filter_size//2: N//2 + filter_size//2 + 1, N//2-filter_size//2: N//2 + filter_size//2 + 1]
        plt.gca().set_xlabel("y")
        plt.gca().set_ylabel("tj real")
        plt.plot(x, np.real(tj_real[N//2, :]))
        # plt.plot(x, np.real(tj_real[:, N//2]))
        plt.show()
        image_filtered = scipy.ndimage.convolve(image_sr, tj_approximate, mode='constant', cval=0.0, origin=0)
        # image_filtered = scipy.signal.convolve(image_sr, tj_real, mode='same')
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
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)
        ax1.set_title("Widefield", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("SR", fontsize=25)
        ax2.tick_params(labelsize=20)

        ax3.set_title("Filtered", fontsize=25)
        ax3.tick_params(labelsize=20)

        ax4.set_title("Ideal", fontsize=25)
        ax4.tick_params(labelsize=20)

        ax1.imshow(image_widefield)
        ax2.imshow(image_sr)
        ax3.imshow(np.abs(image_filtered))
        ax4.imshow(np.abs(ideal))
        # ax4.plot(np.abs(ideal[:, N//2]))

        image_widefield /= np.amax(image_widefield)
        image_sr /= np.amax(image_sr)
        image_filtered /= np.amax(image_filtered)
        ideal /= np.amax(ideal)

        image_widefield_ft = wrappers.wrapped_fftn(image_widefield)
        image_sr_ft = wrappers.wrapped_fftn(image_sr)
        image_filtered_ft = wrappers.wrapped_fftn(image_filtered)
        image_ideal_ft = wrappers.wrapped_fftn(ideal)
        image_ft = wrappers.wrapped_fftn(image)

        image_widefield2d_ft = np.sum(image_widefield_ft)
        image_sr2d_ft = np.sum(image_sr_ft)
        image_filtered2d_ft = np.sum(image_filtered_ft)
        image_ideal2d_ft = np.sum(image_ideal_ft)
        image2d_ft = np.sum(image_ft)

        otf_widefield = image_widefield_ft / image_ft
        otf_sr = image_sr_ft / image_ft
        otf_filtered = image_filtered_ft / image_ft
        otf_ideal = image_ideal_ft / image_ft

        otf_widefield2d = image_widefield2d_ft / image2d_ft
        otf_sr2d = image_sr2d_ft / image2d_ft
        otf_filtered2d = image_filtered2d_ft / image2d_ft
        otf_ideal2d = image_ideal2d_ft / image2d_ft

        # def update1(val):
        #     ax1.clear()
        #     ax2.clear()
        #     ax3.clear()
        #     ax4.clear()
        #
        #     # im1 = ax1.imshow(image_widefield[:, :, int(val)], vmin=0)
        #     # im2 = ax2.imshow(image_sr[:, :, int(val)], vmin=0)
        #     # im3 = ax3.imshow(image_filtered[:, :, int(val)], vmin=0)
        #     # im4 = ax4.imshow(filtered[:, :, int(val)], vmin=0)
        #     ax1.plot(image_widefield[:, N // 2, int(val)], label='widefield')
        #     ax1.plot(image_sr[:, N // 2, int(val)], label='sr')
        #     ax1.plot(image_filtered[:, N // 2, int(val)], label='filtered')
        #     ax1.plot(ideal[:, N // 2, int(val)], label='ideal')
        #     # ax1.set_ylim()
        #     ax1.legend()
        #
        #     ax2.plot(otf_widefield[:, N // 2, int(val)], label='otf widefield')
        #     ax2.plot(otf_sr[:, N // 2, int(val)], label='otf sr')
        #     ax2.plot(otf_filtered[:, N // 2, int(val)], label='otf fitlered')
        #     ax2.plot(otf_ideal[:, N // 2, int(val)], label='otf ideal')
        #     ax2.set_ylim(0, otf_widefield[N // 2, N // 2, N // 2])
        #     ax2.legend()
        #
        #     ax3.plot(otf_widefield2d[:, N // 2], label='otf widefield 2d')
        #     ax3.plot(otf_sr2d[:, N // 2], label='otf sr 2d')
        #     ax3.plot(otf_filtered2d[:, N // 2], label='otf fitlered 2d')
        #     ax3.plot(otf_ideal2d[:, N // 2], label='otf ideal 2d')
        #     # ax3.set_ylim(0, otf_widefield2d[N//2, N//2])
        #     ax3.legend()
        #
        #     # ax3.set_ylim(np.min(image_filtered), np.max(image_filtered))
        #     # ax4.set_ylim(np.min(filtered), np.max(filtered))
        #     ax1.set_aspect(1 / ax1.get_data_ratio())
        #     ax2.set_aspect(1 / ax2.get_data_ratio())
        #     ax3.set_aspect(1 / ax3.get_data_ratio())
        #     # ax4.set_aspect(1 / ax4.get_data_ratio())
        #
        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        # slider_ssnr.on_changed(update1)
        plt.show()