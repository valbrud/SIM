import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import unittest
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy
from matplotlib import colors

import OpticalSystems
import hpc_utils
from config.BFPConfigurations import *
from OpticalSystems import System4f3D, System4f2D
from matplotlib.widgets import Slider
from config.BFPConfigurations import BFPConfiguration
configurations = BFPConfiguration()
class TestOpticalSystems3D(unittest.TestCase):
    def test_OTF(self):
        alpha = 2 * np.pi / 5
        dx = 1 / (4 * np.sin(alpha))
        dy = dx
        dz = 1 / (2 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))
        optical_system = System4f3D(alpha=alpha)
        low_NA_psf, low_NA_otf = optical_system.compute_psf_and_otf((psf_size, N))
        high_NA_psf, high_NA_otf = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=True)
        normalized_paraxial_psf = low_NA_psf / np.amax(low_NA_psf)
        normalized_high_NA_psf = high_NA_psf / np.amax(low_NA_psf)
        print(np.sum(low_NA_psf), np.sum(high_NA_psf))
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(normalized_paraxial_psf[N//2, :, N//2] , label="Paraxial approximation")
        axes[0, 0].plot(normalized_high_NA_psf[N//2, :, N//2], label="high_NA_model")
        plt.legend()
        axes[0, 1].plot(normalized_paraxial_psf[N//2, N//2, :], label="Paraxial approximation")
        axes[0, 1].plot(normalized_high_NA_psf[N//2, N//2, :], label="high_NA_model")
        plt.legend()
        axes[1, 0].plot(low_NA_otf[N//2, :, N//2], label="Paraxial approximation")
        axes[1, 0].plot(high_NA_otf[N//2, :, N//2], label = "high_NA_model")
        plt.legend()
        axes[1, 1].plot(low_NA_otf[N//2, N//4, :], label = "Paraxial approximation")
        axes[1, 1].plot(high_NA_otf[N//2, N//4, :], label = "high_NA_model")
        plt.legend()
        plt.show()

    def test_pixel_correction(self):
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 101
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))
        optical_system = System4f2D(alpha=alpha)
        optical_system.compute_psf_and_otf((psf_size, N), account_for_pixel_correction=False)
        otf_no_corr = np.copy(optical_system.otf)
        optical_system.compute_psf_and_otf((psf_size, N), account_for_pixel_correction=True)
        otf_corr = np.copy(optical_system.otf)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(np.abs(otf_no_corr), cmap='gray')
        axes[0].set_title("OTF without pixel correction")
        axes[1].imshow(np.abs(otf_corr), cmap='gray')
        axes[1].set_title("OTF with pixel correction")
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
        otf = hpc_utils.wrapped_fftn(psf)
        otf/= np.amax(otf)
        # plt.plot(fx, otf[N//2, N//2, :], label='Full')
        psf_cut = np.zeros(psf.shape)
        size = 11
        psf_cut[N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1] \
            = psf[N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1, N//2 - size//2:N//2 + size//2 + 1]
        otf_cut = hpc_utils.wrapped_fftn(psf_cut)
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
        theta = np.pi / 10
        alpha = np.pi / 10
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
            psf_sim = np.abs(hpc_utils.wrapped_ifftn(otf_sim))
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

    def test_angle_vs_pupil_plane_integration(self):
        alpha = np.pi / 12 * 5
        dx = 1 / (32 * np.sin(alpha))
        dy = dx
        dz = 1 / (16 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))
        optical_system = System4f3D(alpha=alpha)
        angle_psf, angle_otf = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=False)
        rho_psf, rho_otf = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=True)
        print(np.sum(angle_psf), np.sum(rho_psf))
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(angle_psf[N//2, :, N//2], label="integrated angle")
        axes[0, 0].plot(rho_psf[N//2, :, N//2], label="integrated pupil plane")
        plt.legend()
        axes[0, 1].plot(angle_psf[N//2, N//2, :], label="integrated angle")
        axes[0, 1].plot(rho_psf[N//2, N//2, :], label="integrated pupil plane")
        plt.legend()
        axes[1, 0].plot(angle_otf[N//2, :, N//2], label="integrated angle")
        axes[1, 0].plot(rho_otf[N//2, :, N//2], label = "integrated pupil plane")
        plt.legend()
        axes[1, 1].plot(angle_otf[N//2, N//2, :], label="integrated angle")
        axes[1, 1].plot(rho_otf[N//2, N//2, :], label="integrated pupil plane")
        plt.legend()
        plt.show()

class TestOpticalSystems2D(unittest.TestCase):
    def test_energy_distribution(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        N = 41
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
            psf_sim = np.abs(hpc_utils.wrapped_ifftn(otf_sim))
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
        illumination = IlluminationPlaneWaves2D.init_from_3D(illumination, dimensions=(1, 1))
        illumination.set_spatial_shifts_diagonally()

        optical_system = System4f2D()
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r)), N))
        otf_sum = np.zeros((N, N), dtype=np.complex128)
        _, effective_otfs = illumination.compute_effective_kernels(optical_system.psf, optical_system.psf_coordinates)
        for otf in effective_otfs:
            otf_sum += effective_otfs[otf]
        print(optical_system)
        otf_sum /= np.amax(otf_sum)
        print(optical_system)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fy / 2 / np.sin(theta), np.abs(otf_sum[ :, N // 2]))
        ax.set_title("SIM OTF")
        ax.set_ylim(bottom=0)
        plt.show()


class TestConstructPupilAberration(unittest.TestCase):

    def test_empty_dict(self):
        """With an empty dictionary, the aberration should be zero everywhere."""
        rho = np.linspace(0, 1, 10)
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        result = OpticalSystems.OpticalSystem.compute_pupil_plane_abberations({}, rho, phi)
        expected = np.zeros((len(rho), len(phi)))
        self.assertTrue(np.allclose(result, expected),
                        "Aberration with empty dict should be zero everywhere.")

    def test_constant_mode(self):
        """
        For the (0,0) mode, zernike(0,0) == 1, so the aberration should be a constant equal
        to the amplitude provided.
        """
        zernike_dict = {(0, 0): 5.0}
        rho = np.linspace(0, 1, 10)
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        result = OpticalSystems.OpticalSystem.compute_pupil_plane_abberations(zernike_dict, rho, phi)
        expected = np.full((len(rho), len(phi)), 5.0)
        self.assertTrue(np.allclose(result, expected),
                        "Aberration for mode (0,0) should be constant and equal to the amplitude.")

    def test_symmetry_mode(self):
        """
        The (2,2) Zernike polynomial is given by R₂₂(r)*cos(2φ). Since cos(2φ)
        is an even function, the resulting aberration should be symmetric in φ.
        """
        zernike_dict = {(4, 0): 1}
        rho = np.linspace(0, 1, 99)
        phi = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        result = OpticalSystems.OpticalSystem.compute_pupil_plane_abberations(zernike_dict, rho, phi)
        # plt.plot(result[5, :])
        # plt.show()
        RHO, PHI = np.meshgrid(rho, phi, indexing='ij')
        X, Y = RHO * np.cos(PHI), RHO * np.sin(PHI)
        x_cart, y_cart = np.linspace(-1, 1, 101), np.linspace(-1, 1, 101)
        dx, dy = 1/50, 1/50
        X_cart, Y_cart = np.meshgrid(x_cart, y_cart, indexing='ij')
        points = np.column_stack((X.ravel(), Y.ravel()))
        data_cart = scipy.interpolate.griddata(points, result.ravel(), (X_cart, Y_cart))
        data_cart = np.nan_to_num(data_cart)
        print(np.sum(data_cart[X_cart**2 + Y_cart**2 <= 1]**2) * dx * dy / np.pi)
        phase = np.exp(1j * 2 * np.pi * 0.072 * data_cart[X_cart**2 + Y_cart**2 <= 1])
        print(np.sum(phase) * dx * dy)
        I = (1 / np.pi)**2 * np.abs(np.sum(phase) * dx * dy)**2
        print(I)
        im = plt.imshow(data_cart.T, origin='lower')
        plt.colorbar(im)
        plt.show()
        # Flip along the φ axis (axis=1) and compare.
        # self.assertTrue(np.allclose(result, np.flip(result, axis=1), atol=1e-6),
        #                 "Aberration for (2,2) mode should be symmetric in phi.")

    def test_OTF_2D_aberrated(self):
        alpha = 67/57
        n = 1.518
        NA = n * np.sin(alpha)
        dx = 8 / 68
        dy = dx
        N = 61
        max_r = N // 2 * dx
        psf_size = 2 * np.array((max_r, max_r))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)

        # N = 61 
        # wavelength = 680e-9
        # px_scaled = 80e-9
        # dx = px_scaled / wavelength
        # NA = 1.4
        # nmedium = 1.518
        # alpha = np.arcsin(NA / nmedium)
        # print(alpha)

        # max_r = dx * N // 2
        # x = np.linspace(-max_r, max_r, N)
        # y = np.copy(x)
        # psf_size = 2 * np.array((max_r, max_r))

        # fx = np.linspace(-1/(2 * dx), 1/(2 * dx), N)
        # fy = np.copy(fx)
        # fr = np.linspace(0, 1 / (2 * dx), N//2 + 1)

        fxn = fx / (2 * NA)
        fyn = fy / (2 * NA)
        arg = N // 2
        # print(fz[arg])

        two_NA_fx = fx / (2 * NA)
        two_NA_fy = fy / (2 * NA)

        optical_system = System4f2D(alpha=alpha, refractive_index=n)
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        psf_size = np.array((2 * max_r, 2 * max_r))

        non_aberrated_psf, non_aberrated_otf = optical_system.compute_psf_and_otf((psf_size, N))
        aberrated_psf_spherical, aberrated_otf_spherical = optical_system.compute_psf_and_otf((psf_size, N),
                                                                          zernieke={(2, 0): 0.0, (4, 0): 0.0})
        
        plt.plot(non_aberrated_otf[N//2, N//2:], label='Non-aberrated OTF') 
        plt.plot(aberrated_otf_spherical[N//2, N//2:], label='Spherical Aberration OTF') 
        plt.show()
        aberrated_psf_comma, aberrated_otf_comma = optical_system.compute_psf_and_otf((psf_size, N),
                                                                          zernieke={(3, 1): 0.072})
        aberrated_psf_astigmatism, aberrated_otf_astigmatism = optical_system.compute_psf_and_otf((psf_size, N),
                                                                          zernieke={(2, 2): 0.072})
        aberrated_psfs = [aberrated_psf_spherical, aberrated_psf_comma, aberrated_psf_astigmatism]
        aberrated_otfs = [aberrated_otf_spherical, aberrated_otf_comma, aberrated_otf_astigmatism]
        i = 0
        for psf in aberrated_psfs:
            fig, axes = plt.subplots(figsize =(8, 6))
            im = axes.imshow(psf[:, :].T, origin='lower', extent=(two_NA_fy[0], two_NA_fy[-1], two_NA_fx[0], two_NA_fy[-1]))
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=30)
            cb.set_label("$PSF$", fontsize=30)
            axes.tick_params(labelsize=30)
            axes.set_xlabel("$f_y \; [LCF]$", fontsize=30)
            axes.set_ylabel("$f_x \;  [LCF]$", fontsize=30)
            # fig.savefig(f"aberrated_psf_{i}.png", bbox_inches='tight', pad_inches=0.1)
            i+=1

        i = 0
        for otf in aberrated_otfs:
            fig, axes = plt.subplots(figsize=(8, 6))
            im = axes.imshow(np.log(1 + 10*np.abs(otf[:, :].T)), origin='lower', extent=(two_NA_fy[0], two_NA_fy[-1], two_NA_fx[0], two_NA_fy[-1]), norm=colors.LogNorm())
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=30)
            axes.tick_params(labelsize=30)
            axes.set_xlabel("$f_y \; [LCF]$", fontsize=30)
            axes.set_ylabel("$f_x \;  [LCF]$", fontsize=30)
            cb.set_label("$1 + 10$ MTF", fontsize=30)
            # fig.savefig(f"aberrated_mtf_{i}.png", bbox_inches='tight', pad_inches=0.1)
            i+=1

        # axes[0, 0].imshow(non_aberrated_psf[N//2, :, :].T, origin='lower')
        # axes[0, 1].imshow(aberrated_psf_spherical[N//2, :, :].T, origin='lower')
        # axes[1, 0].imshow(non_aberrated_psf[:, :, N//2].T, origin='lower')
        # axes[1, 1].imshow(aberrated_psf_comma[:, :, N//2].T, origin='lower')
        # axes[2, 0].imshow(non_aberrated_psf[:, :, N//2].T, origin='lower')
        # axes[2, 1].imshow(aberrated_psf_astigmatism[:, :, N//2].T, origin='lower')
        # plt.show()

        # fig, axes = plt.subplots(3, 2)
        # axes[0, 0].imshow(np.log(1 + 10**8 * np.abs(non_aberrated_otf[:, :, N//2].T)), origin='lower')
        # axes[0, 1].imshow(np.log(1 + 10**8 * np.abs(aberrated_otf_spherical[:, :, N//2].T)), origin='lower')
        # axes[1, 0].imshow(np.log(1 + 10**8 * np.abs(non_aberrated_otf[:, :, N//2].T)), origin='lower')
        # axes[1, 1].imshow(np.log(1 + 10**8 * np.abs(aberrated_otf_comma[:, :, N//2].T)), origin='lower')
        # axes[2, 0].imshow(np.log(1 + 10**8 * np.abs(non_aberrated_otf[:, :, N//2].T)), origin='lower')
        # axes[2, 1].imshow(np.log(1 + 10**8 * np.abs(aberrated_otf_astigmatism[:, :, N//2].T)), origin='lower')
        #
        plt.show()

    def test_OTF_3D_aberrated(self):
        alpha = 2 * np.pi / 5
        nmedium = 1.5
        nobject = 1.5
        NA = nmedium * np.sin(alpha)
        theta = np.asin(0.9 * np.sin(alpha))
        fz_max_diff = nmedium * (1 - np.cos(alpha))
        dx = 1 / (4 * NA)
        dy = dx
        dz = 1 / (2 * fz_max_diff)
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

        optical_system = System4f3D(alpha=alpha, refractive_index_sample=nobject, refractive_index_medium=nmedium)
        x = np.linspace(-max_r, max_r, N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))

        non_aberrated_psf, non_aberrated_otf = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=True)
        aberrated_psf_spherical, aberrated_otf_spherical = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=True,
                                                                          zernieke={(4, 0): 0.072})
        aberrated_psf_comma, aberrated_otf_comma = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=True,
                                                                          zernieke={(3, 1): 0.072})
        aberrated_psf_astigmatism, aberrated_otf_astigmatism = optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, integrate_rho=True,
                                                                          zernieke={(2, 2): 0.072})
        # normalized_paraxial_psf = low_NA_psf / np.amax(low_NA_psf)
        # normalized_high_NA_psf = high_NA_psf / np.amax(high_NA_psf)
        # print(np.sum(low_NA_psf), np.sum(high_NA_psf))
        # fig, axes = plt.subplots(3, 2)
        aberrated_psfs = [aberrated_psf_spherical, aberrated_psf_comma, aberrated_psf_astigmatism]
        aberrated_otfs = [aberrated_otf_spherical, aberrated_otf_comma, aberrated_otf_astigmatism]
        i = 0
        # for psf in aberrated_psfs:
        #     fig, axes = plt.subplots(figsize =(8, 6))
        #     im = axes.imshow(psf[:, :, N//2].T, origin='lower', extent=(two_NA_fy[0], two_NA_fy[-1], two_NA_fx[0], two_NA_fy[-1]))
        #     cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        #     cb.ax.tick_params(labelsize=30)
        #     cb.set_label("$PSF$", fontsize=30)
        #     axes.tick_params(labelsize=30)
        #     axes.set_xlabel("$f_y \; [LCF]$", fontsize=30)
        #     axes.set_ylabel("$f_x \;  [LCF]$", fontsize=30)
        #     fig.savefig(f"aberrated_psf_{i}.png", bbox_inches='tight', pad_inches=0.1)
        #     i+=1

        i = 0
        for otf in aberrated_otfs:
            fig, axes = plt.subplots(figsize=(8, 6))
            im = axes.imshow(np.log(1 + 10*np.abs(otf[:, :, N // 2].T)), origin='lower', extent=(two_NA_fy[0], two_NA_fy[-1], two_NA_fx[0], two_NA_fy[-1]), norm=colors.LogNorm())
            cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=30)
            axes.tick_params(labelsize=30)
            axes.set_xlabel("$f_y \; [LCF]$", fontsize=30)
            axes.set_ylabel("$f_x \;  [LCF]$", fontsize=30)
            cb.set_label("$1 + 10$ MTF", fontsize=30)
            fig.savefig(f"aberrated_mtf_{i}.png", bbox_inches='tight', pad_inches=0.1)
            i+=1

                # axes[0, 0].imshow(non_aberrated_psf[N//2, :, :].T, origin='lower')
        # axes[0, 1].imshow(aberrated_psf_spherical[N//2, :, :].T, origin='lower')
        # axes[1, 0].imshow(non_aberrated_psf[:, :, N//2].T, origin='lower')
        # axes[1, 1].imshow(aberrated_psf_comma[:, :, N//2].T, origin='lower')
        # axes[2, 0].imshow(non_aberrated_psf[:, :, N//2].T, origin='lower')
        # axes[2, 1].imshow(aberrated_psf_astigmatism[:, :, N//2].T, origin='lower')
        # plt.show()

        # fig, axes = plt.subplots(3, 2)
        # axes[0, 0].imshow(np.log(1 + 10**8 * np.abs(non_aberrated_otf[:, :, N//2].T)), origin='lower')
        # axes[0, 1].imshow(np.log(1 + 10**8 * np.abs(aberrated_otf_spherical[:, :, N//2].T)), origin='lower')
        # axes[1, 0].imshow(np.log(1 + 10**8 * np.abs(non_aberrated_otf[:, :, N//2].T)), origin='lower')
        # axes[1, 1].imshow(np.log(1 + 10**8 * np.abs(aberrated_otf_comma[:, :, N//2].T)), origin='lower')
        # axes[2, 0].imshow(np.log(1 + 10**8 * np.abs(non_aberrated_otf[:, :, N//2].T)), origin='lower')
        # axes[2, 1].imshow(np.log(1 + 10**8 * np.abs(aberrated_otf_astigmatism[:, :, N//2].T)), origin='lower')
        #
        plt.show()

        # fig, ax = plt.subplots()
        # norm = np.max(non_aberrated_psf)
        # ax.plot(non_aberrated_psf[N//2, :, N//2] / norm , label="Ideal PSF")
        # ax.plot(aberrated_psf_spherical[N//2, :, N//2] / norm, label="Sperical")
        # ax.plot(aberrated_psf_comma[N//2, :, N//2] / norm, label="Comma")
        # ax.plot(aberrated_psf_astigmatism[N//2, :, N//2] / norm, label="Astigmatism")
        # plt.legend()
        # plt.show()