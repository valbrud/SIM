import unittest
import matplotlib.pyplot as plt
import numpy as np
import sys
import wrappers
from config.IlluminationConfigurations import *
from OpticalSystems import Lens, Lens2D
from matplotlib.widgets import Slider
sys.path.append('../')
from config.IlluminationConfigurations import BFPConfiguration
configurations = BFPConfiguration()
class TestOpticalSystems3D(unittest.TestCase):
    def test_shifted_otf(self):
        max_r = 10
        max_z = 25
        N = 80
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        illumination = Illumination(s_polarized_waves)

        # wavevectors = [np.array([ 4.44288294,  0.        , -1.84030237]), np.array([-4.44288294e+00,  5.44096237e-16, -1.84030237e+00]), np.array([ 2.72048118e-16,  4.44288294e+00, -1.84030237e+00]), np.array([ 2.72048118e-16, -4.44288294e+00, -1.84030237e+00]), np.array([4.44288294, 0.        , 1.84030237]), np.array([-4.44288294e+00,  5.44096237e-16,  1.84030237e+00]), np.array([2.72048118e-16, 4.44288294e+00, 1.84030237e+00]), np.array([ 2.72048118e-16, -4.44288294e+00,  1.84030237e+00])]
        wavevectors = [np.array([0.0 * np.pi,  0., 0.3 * np.pi])]
        # print(wavevectors)
        optical_system = Lens()
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))
        optical_system.prepare_Fourier_interpolation(wavevectors)
        otf_sum = np.zeros((len(x), len(y), len(z)), dtype=np.complex128)
        for otf in optical_system._effective_otfs:
            otf_sum += optical_system._shifted_otfs[otf]
        print(optical_system)
        # print(otf_sum[:, :, 10])
        fig = plt.figure()
        IM = otf_sum[:, int(N/2), :]

        ax = fig.add_subplot(111)
        mp1 = ax.imshow(np.abs(IM))

        plt.colorbar(mp1)
        def update(val):
            ax.clear()
            ax.set_title("otf_sum, fy = {:.2f}, ".format(fy[int(val)]) + "$\\lambda^{-1}$")
            ax.set_xlabel("fx, $\lambda^{-1}$")
            ax.set_ylabel("fy, $\lambda^{-1}$")
            # IM = abs(otf_sum[:, int(val), :])
            IM = abs(optical_system.psf[:, int(val), :])
            mp1.set_clim(vmin=IM.min(), vmax=IM.max())
            ax.imshow(np.abs(IM), extent=(x[0], x[-1], z[0], z[-1]))
            ax.set_aspect(1. / ax.get_data_ratio())


        slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update)
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
            optical_system = Lens(alpha)
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
        optical_system = Lens(alpha)
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

class TestOpticalSystems2D(unittest.TestCase):
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
        spacial_shifts_conventional2d = np.array(((0., 0., 0.), (1, 0, 0), (2, 0, 0)))
        spacial_shifts_conventional2d /= (3 * np.sin(theta))
        illumination.spacial_shifts = spacial_shifts_conventional2d

        optical_system = Lens2D()
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r)), N))
        otf_sum = np.zeros((N, N), dtype=np.complex128)
        effective_otfs = optical_system.compute_effective_otfs_2dSIM(illumination)
        for otf in effective_otfs:
            otf_sum += effective_otfs[otf]
        print(optical_system)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.abs(otf_sum[:, N//2]))

        def update(val):
            ax.clear()
            ax.set_title("otf_sum, fy = {:.2f}, ".format(fy[int(val)]) + "$\\lambda^{-1}$")
            ax.set_xlabel("fx, $\lambda^{-1}$")
            ax.set_ylabel("fy, $\lambda^{-1}$")
            # IM = abs(otf_sum[:, int(val), :])
            IM = abs(optical_system.psf)
            mp1.set_clim(vmin=IM.min(), vmax=IM.max())
            ax.imshow(np.abs(IM), extent=(x[0], x[-1], x[0], x[-1]))
            ax.set_aspect(1. / ax.get_data_ratio())

        slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, 2)  # slider properties
        slider_ssnr.on_changed(update)
        plt.show()
