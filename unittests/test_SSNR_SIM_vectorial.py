import os
import sys
import unittest

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
sys.path.append(current_dir)

from config.BFPConfigurations import BFPConfiguration
from Illumination import IlluminationPlaneWaves2D
from OpticalSystems import System4f2D, System4f3D
import utils
from SSNRCalculator import SSNRSIM2D, SSNRSIM3D, SSNRSIMVectorial2D, SSNRSIMVectorial3D

configurations = BFPConfiguration(refraction_index=1.5)


class TestSSNRSIMVectorial(unittest.TestCase):

    def test_visualize_effective_kernels_conventional_and_hexagonal(self):
        theta = np.pi / 3
        NA = np.sin(theta)

        N = 101
        max_r = 5.0
        psf_size = (2 * max_r, 2 * max_r)

        optical_system = System4f2D(alpha=theta, refractive_index=1.5)
        optical_system.compute_psf_and_otf((psf_size, N))

        illum_conv = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1, dimensionality=2)
        illum_hex = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
        illum_widefield = configurations.get_widefield(dimensionality=2)

        ssnr_calc_conv_vectorial = SSNRSIMVectorial2D(illum_conv, optical_system)
        ssnr_calc_widefield_vectorial = SSNRSIMVectorial2D(illum_widefield, optical_system)
        fix, ax = plt.subplots(1, 2, figsize=(6, 5), constrained_layout=True)
        ax[0].imshow(np.log1p(10**4 * ssnr_calc_conv_vectorial.ssnri.real), cmap="viridis")
        ax[1].imshow(np.log1p(10**4 * ssnr_calc_widefield_vectorial.ssnri.real), cmap="viridis")
        plt.show()
        ssnr_calc_hex_vectorial = SSNRSIMVectorial2D(illum_hex, optical_system)
        fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

        for m in ((0, (-2, 0)), (0, (0, 0)), (0, (2, 0))):
            j = ssnr_calc_conv_vectorial.m_to_number_matrix[m]
            ax[0, j].imshow(np.abs(ssnr_calc_conv_vectorial.effective_kernels_ft[m]), cmap="gray")
            ax[0, j].set_title(f"conv {m}")
            ax[0, j].set_axis_off()

            ax[1, j].imshow(np.abs(ssnr_calc_hex_vectorial.effective_kernels_ft[m]), cmap="gray")
            ax[1, j].set_title(f"hex {m}")
            ax[1, j].set_axis_off()

        plt.show()

        

    def test_compare_ssnr_generalized_vs_standard(self):
        theta = 2 * np.pi / 5
        NA = np.sin(theta) * 1.5

        N = 101
        dx = 1 / (8 * NA)
        max_r = N // 2 * dx
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        psf_size = (2 * max_r, 2 * max_r)

        optical_system = System4f2D(alpha=theta, refractive_index=1.5, high_NA=True, vectorial=False)
        optical_system.compute_psf_and_otf((psf_size, N))

        illum_conv = configurations.get_2_oblique_s_waves_and_s_normal( 1 * theta, 1, 0, 3, Mt=1, dimensionality=2)
        illum_hex = configurations.get_6_oblique_s_waves_and_circular_normal( 1 * theta, 1, 0, Mt=1, dimensionality=2)

        ssnc_calc_conv_no_cc = SSNRSIM2D(illum_conv, optical_system)
        ssnc_calc_hex_no_cc = SSNRSIM2D(illum_hex, optical_system)

        ssnr_calc_conv_cc = SSNRSIMVectorial2D(illum_conv, optical_system)
        ssnr_calc_hex_cc = SSNRSIMVectorial2D(illum_hex, optical_system)

        ring_averaged_conv_no_cc = ssnc_calc_conv_no_cc.ring_average_ssnri()
        ring_averaged_hex_no_cc = ssnc_calc_hex_no_cc.ring_average_ssnri()
        ring_averaged_conv_cc = ssnr_calc_conv_cc.ring_average_ssnri()
        ring_averaged_hex_cc = ssnr_calc_hex_cc.ring_average_ssnri()
        plt.imshow(np.log1p(10**4 * ssnr_calc_hex_cc.ssnri.real).T, cmap="viridis", extent=(fx[0] / (2 * NA), fx[-1] / (2 * NA), fy[0] / (2 * NA), fy[-1] / (2 * NA)))
        for harmonic in illum_conv.harmonics.values():
            (kx, ky) = harmonic.wavevector /(2 * np.pi) / 2 / NA
            ax = plt.gca()
            ax.plot(kx, ky, 'r+', markersize=10, markeredgewidth=2)
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_conv_no_cc)), lw=1, label="conv no cc")
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_hex_no_cc)), lw=1, label="hex no cc")
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_conv_cc)), lw=1, ls="--", label="conv cc")
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_hex_cc)), lw=1, ls="--", label="hex cc")
        ax[0].set_title("Ring-averaged SSNRI (log(1e4*|SSNRI|))")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(ncols=2, fontsize=8)
        ratio_conv = ring_averaged_conv_cc / (ring_averaged_conv_no_cc + 1e-30)
        ratio_hex = ring_averaged_hex_cc / (ring_averaged_hex_no_cc + 1e-30)
        ax[1].plot(ratio_conv, lw=1, label="conv cc/no cc")
        ax[1].plot(ratio_hex, lw=1, label="hex cc/no cc")
        ax[1].set_title("Ratio of ring-averaged SSNRI (cc / no cc)")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(ncols=2, fontsize=8)
        plt.show()

    def test_compare_ssnr_generalized_vs_standard3d(self):
        theta = 2 * np.pi / 5
        NA = np.sin(theta) * 1.5

        N = 51
        dx = 1 / (8 * NA)
        dz = 1 / (4 * np.cos(theta) * 1.5)
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), N)
        psf_size = (2 * max_r, 2 * max_r, 2 * max_z)

        optical_system = System4f3D(alpha=theta, refractive_index_sample=1.5, refractive_index_medium=1.5, high_NA=True, vectorial=False)
        optical_system.compute_psf_and_otf((psf_size, N))

        illum_conv = configurations.get_2_oblique_s_waves_and_s_normal( 0.7 * theta, 1, 0, 3, Mt=1, dimensionality=3)
        illum_hex = configurations.get_6_oblique_s_waves_and_circular_normal( 0.7 * theta, 1, 0, Mt=1, dimensionality=3)

        ssnc_calc_conv_no_cc = SSNRSIM3D(illum_conv, optical_system)
        ssnc_calc_hex_no_cc = SSNRSIM3D(illum_hex, optical_system)

        ssnr_calc_conv_cc = SSNRSIMVectorial3D(illum_conv, optical_system)
        ssnr_calc_hex_cc = SSNRSIMVectorial3D(illum_hex, optical_system)

        ring_averaged_conv_no_cc = ssnc_calc_conv_no_cc.ring_average_ssnri()
        ring_averaged_hex_no_cc = ssnc_calc_hex_no_cc.ring_average_ssnri()
        ring_averaged_conv_cc = ssnr_calc_conv_cc.ring_average_ssnri()
        ring_averaged_hex_cc = ssnr_calc_hex_cc.ring_average_ssnri()
        # plt.imshow(np.log1p(10**4 * ssnr_calc_hex_cc.ssnri[..., N//2].real).T, cmap="viridis", extent=(fx[0] / (2 * NA), fx[-1] / (2 * NA), fy[0] / (2 * NA), fy[-1] / (2 * NA)))
        # for harmonic in illum_conv.harmonics.values():
        #     (kx, ky) = harmonic.wavevector[:2] /(2 * np.pi) / 2 / NA
        #     ax = plt.gca()
        #     ax.plot(kx, ky, 'r+', markersize=10, markeredgewidth=2)
        # plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_conv_no_cc[..., N//2])), lw=1, label="conv no cc")
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_hex_no_cc[..., N//2])), lw=1, label="hex no cc")
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_conv_cc[..., N//2])), lw=1, ls="--", label="conv cc")
        ax[0].plot(np.log1p(1e4 * np.abs(ring_averaged_hex_cc[..., N//2])), lw=1, ls="--", label="hex cc")
        ax[0].set_title("Ring-averaged SSNRI (log(1e4*|SSNRI|))")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(ncols=2, fontsize=8)
        ratio_conv = ring_averaged_conv_cc[..., N//2] / (ring_averaged_conv_no_cc[..., N//2 - 10] + 1e-30)
        ratio_hex = ring_averaged_hex_cc[..., N//2] / (ring_averaged_hex_no_cc[..., N//2 - 10] + 1e-30)
        ax[1].plot(ratio_conv, lw=1, label="conv cc/no cc")
        ax[1].plot(ratio_hex, lw=1, label="hex cc/no cc")
        ax[1].set_title("Ratio of ring-averaged SSNRI (cc / no cc)")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(ncols=2, fontsize=8)
        plt.show()

if __name__ == "__main__":
	unittest.main()
