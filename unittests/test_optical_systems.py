import unittest
import matplotlib.pyplot as plt
import numpy as np
import sys
import wrappers
from config.IlluminationConfigurations import *
from OpticalSystems import System4f3D, System4f2D
from matplotlib.widgets import Slider
sys.path.append('../')
from config.IlluminationConfigurations import BFPConfiguration
configurations = BFPConfiguration()
class TestOpticalSystems3D(unittest.TestCase):
    def test_OTF(self):
        alpha = np.pi/2
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 101
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))
        optical_system = System4f3D(alpha=alpha)
        low_NA_psf, low_NA_otf = optical_system.compute_psf_and_otf((psf_size, N))
        high_NA_psf, high_NA_otf = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True)
        print(np.sum(low_NA_psf), np.sum(high_NA_psf))
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(low_NA_psf[N//2, :, N//2], label="low_NA_model")
        axes[0, 0].plot(high_NA_psf[N//2, :, N//2], label="high_NA_model")
        plt.legend()
        axes[0, 1].plot(low_NA_psf[N//2, N//2, :], label="low_NA_model")
        axes[0, 1].plot(high_NA_psf[N//2, N//2, :], label="high_NA_model")
        plt.legend()
        axes[1, 0].plot(low_NA_otf[N//2, :, N//2], label="low_NA_model")
        axes[1, 0].plot(high_NA_otf[N//2, :, N//2], label = "high_NA_model")
        plt.legend()
        axes[1, 1].plot(low_NA_otf[N//2, N//2, :], label = "low_NA_model")
        axes[1, 1].plot(high_NA_otf[N//2, N//2, :], label = "high_NA_model")
        plt.legend()
        plt.show()

    def test_sim_otf(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 101
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)

        illumination = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3)
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination.spatial_shifts = spatial_shifts_conventional2d

        optical_system = System4f3D()
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))
        otf_sum = np.zeros((N, N, N), dtype=np.complex128)
        effective_otfs = optical_system.compute_effective_otfs_projective_3dSIM(illumination)
        for otf in effective_otfs:
            otf_sum += effective_otfs[otf]
        otf_sum /= np.amax(otf_sum)
        print(optical_system)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fy, np.abs(otf_sum[N//2, :, N//2]))
        ax.set_title("SIM OTF")
        ax.set_ylim(bottom=0)
        plt.show()

    def test_confocal_SSNRv(self):
        max_r = 4
        max_z = 10
        N = 61
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        dV = dx * dy * dz
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)
        ssnrv_widefield = []
        ssnrv_confocal = []
        NAs = np.linspace(0.1, 1, 10)
        for NA in NAs:
            alpha = np.asin(NA)
            optical_system = System4f3D(alpha)
            optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))
            # print(np.sum(optical_system.psf))
            ssnrv_widefield.append(np.sum(optical_system.psf**2))
            ssnrv_confocal.append(np.sum(optical_system.psf**4) / np.sum(optical_system.psf**2) / optical_system.psf[N//2, N//2, N//2])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Comparison of SSNR volumes")
        ax.set_xlabel("NA")
        ax.set_ylabel("SSNR_volume")
        ax.set_xlim(0.1, 1)
        # ax.set_ylim(bottom=0, top=0.05)
        plt.plot(NAs, np.array(ssnrv_widefield), label='Widefield')
        plt.plot(NAs, np.array(ssnrv_confocal), label='Confocal')
        plt.grid()
        plt.legend()
        plt.show()

    def test_cut_PSF(self):
        N = 101
        alpha = np.pi/4
        airy = 1.22 / (2 * np.sin(alpha))
        dx = 1 / (4 * np.sin(alpha))
        dy = dx
        dz = 1 / (2 * (1 - np.cos(alpha)))
        max_r = N//2 * dx
        max_z = N//2 * dz
        print(max_r, max_z)

        dV = dx * dy * dz
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        z = np.linspace(-max_z, max_z, N)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dx), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dx), N)
        optical_system = System4f3D(alpha)
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))
        psf = optical_system.psf
        # otf = optical_system.otf
        otf = wrappers.wrapped_ifftn(psf)
        otf/= np.amax(otf)
        # plt.plot(fx, otf[N//2, N//2, :], label='Full')
        psf_cut = np.zeros(psf.shape)
        size = 11
        psf_cut[N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1] \
            = psf[N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1]
        otf_cut = wrappers.wrapped_ifftn(psf_cut)
        otf_cut /= np.amax(otf_cut)

        fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set_ylim(bottom=0, top=0.05)
        # plt.imshow(psf[N//2, :, :])
        # plt.show()
        plt.plot(z, psf[N//2, N//2, :], label='Full')
        plt.plot(z, psf_cut[N//2, N//2, :], label='Cut')
        plt.legend()
        plt.show()
        plt.plot(x, psf[N//2, :, N//2], label='Full')
        plt.plot(x, psf_cut[N//2, :, N//2], label='Cut')
        plt.legend()
        plt.show()
        plt.plot(fz, otf[N//2, N//2, :], label='Full')
        plt.plot(fz, otf_cut[N//2, N//2, :], label='Cut')
        plt.legend()
        plt.show()
        plt.plot(fx, otf[N//2, :, N//2], label='Full')
        plt.plot(fx, otf_cut[N//2, :, N//2], label='Cut')
        plt.legend()
        plt.show()

    def test_energy_distribution(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 101
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        optical_system = System4f3D(alpha=alpha)
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))

        illumination_s_polarized = configurations.get_5_s_waves(theta, 0.5, 1, Mt=10)
        illumination_circular = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.55, 1, Mt=64)
        illumination_seven_waves = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=64)
        illumination_3waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        illumination_widefield = configurations.get_widefield()

        configs = {
            # "SquareL" : illumination_s_polarized,
            # "SquareC" : illumination_circular,
            # "Hexagonal" : illumination_seven_waves,
            "Conventional 3D" : illumination_3waves,
            "Conventional 2D" : illumination_2waves,
            "Widefield" : illumination_widefield,
        }
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Total energy in a cube of a given size (3D)", fontsize=25)
        ax.set_xlabel("Cube size (Pixels)", fontsize=25)
        ax.set_ylabel("Energy part (%)", fontsize=25)
        ax.tick_params(labelsize=20)
        ax.grid()
        for config in configs:
            illumination = configs[config]
            effective_otfs = optical_system.compute_effective_otfs_projective_3dSIM(illumination)
            otf_sim = np.zeros(optical_system.otf.shape, dtype=np.complex128)
            for otf in effective_otfs:
                otf_sim += effective_otfs[otf]
            psf_sim = np.abs(wrappers.wrapped_fftn(otf_sim))
            psf_sim /= np.sum(psf_sim)
            # plt.imshow(psf_sim[:, :, N//2])
            # plt.show()
            # plt.imshow(psf_sim[:, N//2, :])
            # plt.show()
            sizes = np.arange(1, N + 1, 2)
            percentage = np.zeros(sizes.size)
            for s in range(len(sizes)):
                size = sizes[s]
                energy_percentage = np.sum(psf_sim[N // 2 - size // 2: N // 2 + size // 2 + 1,
                                           N // 2 - size // 2 : N // 2 + size // 2 + 1,
                                           N // 2 - size // 2 : N // 2 + size // 2 + 1]) * 100
                percentage[s] = energy_percentage
                print(f"Configuration = {config}, size = {size}, energy = {energy_percentage} %")
            ax.plot(sizes, percentage, label=config)
        ax.legend(fontsize=20)
        plt.show()
class TestOpticalSystems2D(unittest.TestCase):
    def test_energy_distribution(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        N = 51
        max_r = N // 2 * dx
        optical_system = System4f2D(alpha=alpha)
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r)), N))

        illumination_2waves = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr=3, Mt=1)
        illumination_widefield = configurations.get_widefield()
        configs = {
            # "SquareL" : illumination_s_polarized,
            # "SquareC" : illumination_circular,
            # "Hexagonal" : illumination_seven_waves,
            # "Conventional 3D" : illumination_3waves,
            "Conventional 2D" : illumination_2waves,
            "Widefield" : illumination_widefield,
        }
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Total energy in a cube of a given size (2D)", fontsize=25)
        ax.set_xlabel("Cube size (Pixels)", fontsize=25)
        ax.set_ylabel("Energy part (%)", fontsize=25)
        ax.tick_params(labelsize =20)
        ax.grid()
        for config in configs:
            illumination = configs[config]
            effective_otfs = optical_system.compute_effective_otfs_2dSIM(illumination)
            otf_sim = np.zeros(optical_system.otf.shape, dtype = np.complex128)
            for otf in effective_otfs:
                otf_sim += effective_otfs[otf]
            psf_sim = np.abs(wrappers.wrapped_fftn(otf_sim))
            psf_sim /= np.sum(psf_sim)
            # plt.imshow(psf_sim[:, :, N//2])
            # plt.show()
            # plt.imshow(psf_sim[:, N//2, :])
            # plt.show()
            sizes = np.arange(1, N+1, 2)
            percentage = np.zeros(sizes.size)
            for s in range(len(sizes)):
                size = sizes[s]
                energy_percentage = np.sum(psf_sim[N//2 - size//2 : N//2 + size//2 + 1,
                                           N//2 - size//2 : N//2 + size//2 + 1]) * 100
                percentage[s] = energy_percentage
                print(f"Configuration = {config}, size = {size}, energy = {energy_percentage} %" )
            ax.plot(sizes, percentage, label = config)
        ax.legend(fontsize=20)
        plt.show()
    def test_shifted_otf(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        N = 51
        max_r = N // 2 * dx
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)

        illumination = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3)
        spatial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spatial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination.spatial_shifts = spatial_shifts_conventional2d

        optical_system = System4f2D()
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r)), N))
        otf_sum = np.zeros((N, N), dtype=np.complex128)
        effective_otfs = optical_system.compute_effective_otfs_2dSIM(illumination)
        for otf in effective_otfs:
            otf_sum += effective_otfs[otf]
        print(optical_system)
        fig = plt.figure()
        otf_sum /= np.amax(otf_sum)
        print(optical_system)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fy, np.abs(otf_sum[ :, N // 2]))
        ax.set_title("SIM OTF")
        ax.set_ylim(bottom=0)

        plt.show()
