import sys

import numpy as np

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
import SSNRCalculator
from OpticalSystems import Lens
import stattools
from Sources import IntensityPlaneWave
import tqdm
sys.path.append('../')
configurations = BFPConfiguration()
class TestAgainstFourier(unittest.TestCase):
    def test_compare_ssnr(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N//2 * dx
        max_z = N//2 * dz

        kernel_size = 1
        kernel = np.zeros((kernel_size, kernel_size, 1))
        func = np.zeros(kernel_size)
        func[0:kernel_size//2 + 1] = np.linspace(0, 1, kernel_size//2+1)
        func[kernel_size//2: kernel_size] = np.linspace(1, 0, kernel_size//2 + 1)
        func2d = func[:, None] * func[None, :]
        kernel[:, :,  0] = func
        # kernel[0, 0, 0] = 1

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

        optical_system = Lens(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        illumination_s_polarized = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=32)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        noise_estimator_finite = SSNRCalculator.SSNR3dSIM2dShiftsFiniteRealKernel(illumination_widefield, optical_system, kernel)
        ssnr_finite_widefield = noise_estimator_finite.compute_ssnr()
        ssnr_finite_widefield_ra = noise_estimator_finite.ring_average_ssnr()
        volume_finite_widefield = noise_estimator_finite.compute_ssnr_volume()
        entropy_finite_widefield = noise_estimator_finite.compute_true_ssnr_entropy()

        noise_estimator_finite.illumination = illumination_s_polarized
        ssnr_finite_s_polarized = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_s_polarized_ra = noise_estimator_finite.ring_average_ssnr()
        volume_finite_s_polarized = noise_estimator_finite.compute_ssnr_volume()
        entropy_finite_s_polarized = noise_estimator_finite.compute_true_ssnr_entropy()

        noise_estimator_finite.illumination = illumination_seven_waves
        ssnr_finite_seven_waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_seven_waves_ra = noise_estimator_finite.ring_average_ssnr()
        volume_finite_seven_waves = noise_estimator_finite.compute_ssnr_volume()
        entropy_finite_seven_waves = noise_estimator_finite.compute_true_ssnr_entropy()

        noise_estimator_finite.illumination = illumination_3waves
        ssnr_finite_3waves = np.abs(noise_estimator_finite.compute_ssnr())
        ssnr_finite_3waves_ra = noise_estimator_finite.ring_average_ssnr()
        volume_finite_3waves = noise_estimator_finite.compute_ssnr_volume()
        entropy_finite_3waves = noise_estimator_finite.compute_true_ssnr_entropy()

        noise_estimator = SSNRCalculator.SSNR3dSIM2dShifts(illumination_widefield, optical_system)
        ssnr_widefield = noise_estimator.compute_ssnr()
        ssnr_widefield_ra = noise_estimator.ring_average_ssnr()
        volume_widefield = noise_estimator.compute_ssnr_volume()
        entropy_widefield = noise_estimator.compute_true_ssnr_entropy()

        noise_estimator.illumination = illumination_s_polarized
        ssnr_s_polarized = np.abs(noise_estimator.compute_ssnr())
        ssnr_s_polarized_ra = noise_estimator.ring_average_ssnr()
        volume_s_polarized = noise_estimator.compute_ssnr_volume()
        entropy_s_polarized = noise_estimator.compute_true_ssnr_entropy()

        noise_estimator.illumination = illumination_seven_waves
        ssnr_seven_waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_seven_waves_ra = noise_estimator.ring_average_ssnr()
        volume_seven_waves = noise_estimator.compute_ssnr_volume()
        entropy_seven_waves = noise_estimator.compute_true_ssnr_entropy()

        noise_estimator.illumination = illumination_3waves
        ssnr_3waves = np.abs(noise_estimator.compute_ssnr())
        ssnr_3waves_ra = noise_estimator.ring_average_ssnr()
        volume_3waves = noise_estimator.compute_ssnr_volume()
        entropy_3waves = noise_estimator.compute_true_ssnr_entropy()

        print("Volume ssnr widefield = ", volume_widefield)
        print("Entropy ssnr widefield = ", entropy_widefield)
        print("Volume finite widefield = ", volume_finite_widefield)
        print("Entropy finite widefield = ", entropy_finite_widefield)

        print("Volume ssnr s_polarized = ", volume_s_polarized)
        print("Entropy ssnr s_polarized = ", entropy_s_polarized)
        print("Volume finite s_polarized = ", volume_finite_s_polarized)
        print("Entropy finite s_polarized = ", entropy_finite_s_polarized)

        print("Volume ssnr 3waves = ", volume_3waves)
        print("Entropy ssnr 3waves = ", entropy_3waves)
        print("Volume finite 3waves = ", volume_finite_3waves)
        print("Entropy finite 3waves = ", entropy_finite_3waves)

        print("Volume ssnr seven_waves = ", volume_seven_waves)
        print("Entropy ssnr seven_waves = ", entropy_seven_waves)
        print("Volume finite seven_waves = ", volume_finite_seven_waves)
        print("Entropy finite seven_waves = ", entropy_finite_seven_waves)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax1.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax1.set_yscale("log")
        ax1.grid(which='major')
        ax1.grid(which='minor', linestyle='--')
        ax1.tick_params(labelsize=20)
        ax1.set_ylim(1, 10 ** 4)

        ax2.set_xlabel(r"$f_r, \frac{2NA}{\lambda}$", fontsize=25)
        ax2.set_ylabel(r"$1 + 10^4 SSNR_{ra}$", fontsize=25)
        ax2.set_yscale("log")
        ax2.grid(which='major')
        ax2.grid(which='minor', linestyle='--')
        ax2.tick_params(labelsize=20)
        ax2.set_ylim(1, 10 ** 4)

        multiplier = 10 ** 4
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_widefield_ra[:, arg], label="Widefield")
        ax1.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_s_polarized_ra[:, arg], label="Square SIM, s-polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_seven_waves_ra[:, arg], label="Hexagonal SIM")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_3waves_ra[:, arg], label="Conventional SIM")

        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_finite_widefield_ra[:, arg], label="Widefield")
        ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * ssnr_finite_s_polarized_ra[:, arg], label="Square SIM, s-polarized")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_seven_waves_ra[:, arg], label="Hexagonal SIM")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_3waves_ra[:, arg], label="Conventional SIM")
        ax2.set_aspect(1. / ax2.get_data_ratio())


        def update1(val):
            ax1.clear()
            ax1.set_title("Comparison of 3D SIM modalities\n $f_z = ${:.1f}".format(two_NA_fz[arg]) + "$(\\frac{n - \sqrt{n^2 - NA^2}}{\lambda})$", fontsize=25, pad=15)
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
            # ax2.set_title("Slice $f_y$ = 0")
            ax2.set_xlabel(r"$f_x$")
            # ax2.set_ylabel(r"ssnr")

            ax2.set_yscale("log")
            # ax2.set_ylim(1,  10**4)
            ax2.set_xlim(0, fx[-1] / (2 * NA))
            ax2.grid()

            # ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_s_polarized_ra[:, int(val)], label="SquareL")
            # ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_seven_waves_ra[:, int(val)], label="Hexagonal")
            # ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_3waves_ra[:, int(val)], label="Conventional")
            # ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * ssnr_finite_widefield_ra[:, int(val)], label="Widefield")
            #
            ax2.plot(two_NA_fx[fx >= 0], ssnr_finite_s_polarized_ra[:, int(val)]/ssnr_3waves_ra[:, int(val)], label="SquareL")
            ax2.plot(two_NA_fx[fx >= 0], ssnr_finite_seven_waves_ra[:, int(val)]/ssnr_seven_waves_ra[:, int(val)], label="Hexagonal")
            ax2.plot(two_NA_fx[fx >= 0], ssnr_finite_3waves_ra[:, int(val)]/ssnr_3waves_ra[:, int(val)], label="Conventional")
            ax2.plot(two_NA_fx[fx >= 0], ssnr_finite_widefield_ra[:, int(val)]/ssnr_widefield_ra[:, int(val)], label="Widefield")

            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N)  # slider properties
        slider_ssnr.on_changed(update1)

        ax1.legend(fontsize=15)
        ax2.legend(fontsize=15)
        plt.show()