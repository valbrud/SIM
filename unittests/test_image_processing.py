import sys

import numpy as np
import sys
sys.path.append('../')
import ApodizationFilters
import Sources
import ImageProcessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import unittest
import time
import skimage
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import tqdm

theta = np.pi / 4
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (np.cos(theta) - 1)

NA = np.sin(theta)

b = 1
Mt_s_polarized = 32
a0 = (2 + 4 * b ** 2)

norm = a0 * Mt_s_polarized
s_polarized_waves = {
    (0, 0, 0)  : Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

    (-2, 0, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((-2 * k1, 0, 0))),
    (2, 0, 0)  : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((2 * k1, 0, 0))),
    (0, 2, 0)  : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((0, -2 * k1, 0))),

    (1, 0, 1)  : Sources.IntensityPlaneWave((-1j * b) / norm, 0, np.array((k1, 0, k2))),
    (-1, 0, 1) : Sources.IntensityPlaneWave((1j * b) / norm, 0, np.array((-k1, 0, k2))),
    (0, 1, 1)  : Sources.IntensityPlaneWave((-1 * b) / norm, 0, np.array((0, k1, k2))),
    (0, -1, 1) : Sources.IntensityPlaneWave((1 * b) / norm, 0, np.array((0, -k1, k2))),

    (1, 0, -1) : Sources.IntensityPlaneWave((-1j * b) / norm, 0, np.array((k1, 0, -k2))),
    (-1, 0, -1): Sources.IntensityPlaneWave((1j * b) / norm, 0, np.array((-k1, 0, -k2))),
    (0, 1, -1) : Sources.IntensityPlaneWave((1 * b) / norm, 0, np.array((0, k1, -k2))),
    (0, -1, -1): Sources.IntensityPlaneWave((-1 * b) / norm, 0, np.array((0, -k1, -k2)))
}

Mt_circular = 32
b = 1/2**0.5
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (1 - np.cos(theta))
a0 = (2 + 8 * b ** 2)
circular_intensity_waves = {
    (1, 1, 0)  : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((k1, k1, 0))),
    (-1, 1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((-k1, k1, 0))),
    (1, -1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((k1, -k1, 0))),
    (-1, -1, 0): Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((-k1, -k1, 0))),

    (0, 2, 0)  : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((0, -2 * k1, 0))),
    (2, 0, 0)  : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((2 * k1, 0, 0))),
    (-2, 0, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((-2 * k1, 0, 0))),

    (1, 0, -1) : Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((k1, 0, -k2))),
    (-1, 0, 1) : Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((-k1, 0, k2))),
    (1, 0, 1)  : Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((k1, 0, k2))),
    (-1, 0, -1): Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((-k1, 0, -k2))),

    (0, 1, -1) : Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, k1, -k2))),
    (0, 1, 1)  : Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, k1, k2))),
    (0, -1, 1) : Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, -k1, k2))),
    (0, -1, -1): Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, -k1, -k2))),

    (0, 0, 0)  : Sources.IntensityPlaneWave(a0 / (Mt_circular * a0), 0, np.array((0, 0, 0)))
}

b = 1
a0 = 1 + 2 * b**2
Mr_tree_waves = 3
Mt_three_waves = 4
norm = a0 * Mr_tree_waves * Mt_three_waves
three_waves_illumination = {
    (0, 0, 0)  : Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

    (0, 2, 0)  : Sources.IntensityPlaneWave(b**2 / norm, 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0) : Sources.IntensityPlaneWave(b**2 / norm, 0, np.array((0, -2 * k1, 0))),

    (0, 1, 1)  : Sources.IntensityPlaneWave(b / norm, 0, np.array((0, k1, k2))),
    (0, -1, 1) :  Sources.IntensityPlaneWave(b / norm, 0, np.array((0, -k1, k2))),
    (0, 1, -1) : Sources.IntensityPlaneWave(b / norm, 0, np.array((0,  k1, -k2))),
    (0, -1, -1): Sources.IntensityPlaneWave(b / norm, 0, np.array((0, -k1, -k2))),
}


Mt_widefield = 1
widefield = {
    (0, 0, 0) : Sources.IntensityPlaneWave(1/Mt_widefield, 0, np.array((0, 0, 0)))
}

class TestOpticalSystems(unittest.TestCase):
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

        illumination = ImageProcessing.Illumination(s_polarized_waves)

        # wavevectors = [np.array([ 4.44288294,  0.        , -1.84030237]), np.array([-4.44288294e+00,  5.44096237e-16, -1.84030237e+00]), np.array([ 2.72048118e-16,  4.44288294e+00, -1.84030237e+00]), np.array([ 2.72048118e-16, -4.44288294e+00, -1.84030237e+00]), np.array([4.44288294, 0.        , 1.84030237]), np.array([-4.44288294e+00,  5.44096237e-16,  1.84030237e+00]), np.array([2.72048118e-16, 4.44288294e+00, 1.84030237e+00]), np.array([ 2.72048118e-16, -4.44288294e+00,  1.84030237e+00])]
        wavevectors = [np.array([0.0 * np.pi,  0., 0.3 *  np.pi])]
        # print(wavevectors)
        optical_system = ImageProcessing.Lens(interpolation_method="Fourier")
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))
        optical_system.prepare_Fourier_interpolation(wavevectors)
        otf_sum = np.zeros((len(x), len(y), len(z)), dtype=np.complex128)
        for otf in optical_system._shifted_otfs:
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

    def test_SSNR_old(self):
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
        waves = circular_intensity_waves

        illumination_polarized = ImageProcessing.Illumination(waves)
        optical_system = ImageProcessing.Lens()
        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        # wavevectors=illumination_polarized.get_wavevectors()
        # optical_system.prepare_Fourier_interpolation(wavevectors)

        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system)
        SSNR = np.abs(noise_estimator.SSNR(2 * np.pi * np.array((fx, fy, fz))))


        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        Fx, Fy = np.meshgrid(fx, fy)
        IM = SSNR[:, :, int(N / 2)].T
        ax.plot_wireframe(Fx, Fy, np.log(IM))

        def update(val):
            ax.clear()
            ax.set_title("SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax.set_xlabel("fx, $\lambda^{-1}$")
            ax.set_ylabel("fy, $\lambda^{-1}$")
            IM = SSNR[:, :, int(val)].T
            ax.plot_wireframe(Fx, Fy, np.log10(IM))

        slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update)

        ax1 = fig.add_subplot(122)
        pos1 = ax1.imshow(np.log10(SSNR[:, :, 10].T))

        def update_otf(val):
            ax1.clear()
            ax1.set_title("SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fx, $\lambda^{-1}$")
            ax1.set_ylabel("fy, $\lambda^{-1}$")
            ax1.imshow(np.log10(SSNR[:, :, int(val)].T), extent=(fx[0], fx[-1], fy[0], fy[-1]))
            # fig.colorbar(pos1, ax=ax1)

        slider_loc = plt.axes((0.2, 0.0, 0.65, 0.03))  # slider location and size
        slider_otf = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_otf.on_changed(update_otf)
        plt.show()

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

        waves = s_polarized_waves

        illumination_polarized = ImageProcessing.Illumination(waves)
        optical_system_fourier = ImageProcessing.Lens(interpolation_method="Fourier")
        optical_system_fourier.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N),
                                                   apodization_filter=None)
        optical_system_linear = ImageProcessing.Lens()
        optical_system_linear.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N),
                                                  apodization_filter=None)
        wavevectors = illumination_polarized.get_wavevectors()

        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system_fourier)

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

        waves = s_polarized_waves
        if waves == three_waves_illumination:
            Mr = 3
            Mt = Mt_three_waves
            title = "State of art SIM"
            label = "three_waves"
        elif waves == s_polarized_waves:
            Mr = 1
            Mt = Mt_s_polarized
            title = "Lattice SIM with polarized waves"
            label = "s_polarized"
        elif waves == circular_intensity_waves:
            Mr = 1
            Mt = Mt_circular
            title = "Lattice SIM with circular waves"
            label = "circular"
        elif waves == widefield:
            Mr = 1
            Mt = Mt_widefield
            title = "Widefield"
            label = "widefield"

        illumination_polarized = ImageProcessing.Illumination(waves, M_r=Mr)
        illumination_polarized.M_t = Mt
        optical_system = ImageProcessing.Lens(alpha=theta)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system)

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
        mp1 = ax1.imshow(SSNR_scaled[int(N / 2),:, :], extent=(fz[0]/(2 * NA), fz[-1]/(2 *NA), fy[0]/(2 * NA), fy[-1]/(2 * NA)), norm=colors.LogNorm())
        # cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        # cb1.set_label("$1 + 10^8$ SSNR")
        ax1.set_aspect(1. / ax1.get_data_ratio())

        ax2 = fig.add_subplot(122, sharey=ax1)
        ax2.set_xlabel("$f_x$", fontsize = 25)
        ax2.tick_params(labelsize = 20)
        ax2.tick_params('y', labelleft=False)
        # ax2.set_ylabel("fy, $\\frac{2NA}{\\lambda}$")
        mp2 = ax2.imshow(SSNR_scaled[:, :, int(N//2)], extent=(fy[0]/(2 * NA), fy[-1]/(2 *NA), fx[0]/(2 * NA), fx[-1]/(2 * NA)), norm=colors.LogNorm())
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

            # ax2.clear()
            # ax2.set_title("Ring averaged SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\frac{2NA}{\\lambda}$")
            # ax2.set_xlabel("fx, $\lambda^{-1}$")
            # ax2.set_ylabel("$1 + 10^8 \\langle SSNR \\rangle_{ra}$")
            # ssnr_r_sliced = (SSNR_ra_scaled[:, int(val)])
            # ax2.plot(q_axes[2][q_axes[2] >= 0] / (2 * np.pi), ssnr_r_sliced)
            # ax2.set_yscale("log")
            # ax2.set_aspect(1. / ax2.get_data_ratio())

        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        # slider_ssnr.on_changed(update1)


        plt.show()

    def test_isosurface_visualisation(self):
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

        waves = circular_intensity_waves

        if waves == three_waves_illumination:
            Mr = 3
            Mt = Mt_three_waves
            label = "three"
            title = "State of art SIM"
        elif waves == s_polarized_waves:
            Mr = 1
            Mt = Mt_s_polarized
            label = "s_polarized"
            title = "Lattice SIM with polarized waves"
        elif waves == circular_intensity_waves:
            Mr = 1
            Mt = Mt_circular
            label = "circular"
            title = "Lattice SIM with circular waves"

        illumination_polarized = ImageProcessing.Illumination(waves, M_r= Mr)
        illumination_widefield = ImageProcessing.Illumination(widefield, M_r =1)
        illumination_polarized.M_t = Mt
        optical_system = ImageProcessing.Lens(alpha=np.pi/4)

        optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2 * max_z), N))

        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system)

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        SSNR = np.abs(noise_estimator.SSNR(q_axes,))
        SSNR_scaled = np.log10(1 + 10**8 * np.abs(SSNR))
        noise_estimator.illumination = illumination_widefield
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
        ax1 = fig.add_subplot(111, projection = '3d')
        ax1.set_title(title, fontsize = 20)
        ax1.set_xlabel(r"$f_x$", fontsize = 20, labelpad = 15)
        ax1.set_ylabel(r"$f_y$", fontsize = 20, labelpad = 15)
        ax1.set_zlabel(r"$f_z$", fontsize = 20, labelpad = 15)
        ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],  alpha=0.7)
        ax1.plot_trisurf(w_verts[:, 0], w_verts[:, 1], w_faces, w_verts[:, 2],  alpha=1, color = "red")

        ax1.set_xlim(N // 4, 3 * N // 4)
        ax1.set_ylim(N // 4, 3 * N // 4)
        ax1.set_zlim(0, N)
        xticks = np.round((ax1.get_xticks() - N / 2) * dfx / NA, 2)
        yticks = np.round((ax1.get_yticks() - N / 2) * dfy / NA, 2)
        zticks = np.round((ax1.get_zticks() - N / 2) * dfz / NA, 2)
        ax1.set_xticklabels(xticks)
        ax1.set_yticklabels(yticks)
        ax1.set_zticklabels(zticks)
        ax1.tick_params(labelsize = 15)
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

            ax1.set_xlim(N//4, 3 * N//4)
            ax1.set_ylim(N//4, 3 * N//4)
            ax1.set_zlim(0, N)

            xticks = np.round((ax1.get_xticks() - N / 2) * dfx / NA, 2)
            yticks = np.round((ax1.get_yticks() - N / 2) * dfy / NA, 2)
            zticks = np.round((ax1.get_zticks() - N / 2) * dfz / NA, 2)
            ax1.set_xticklabels(xticks)
            ax1.set_yticklabels(yticks)
            ax1.set_zticklabels(zticks)
            plt.draw()

        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, 50)  # slider properties
        # slider_ssnr.on_changed(update1)

        ani = FuncAnimation(fig, update1, frames=range(2, 40), repeat=False, interval=100)
        ani.save('/home/valerii/Documents/projects/SIM/SSNR_article_1/Animations/'
                 'Animation_' + label + '_waves_SSNR_isosurface.mp4', writer="ffmpeg")
        plt.show()

    def test_compare_SSNR(self):
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

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        arg = N // 2
        print(fz[arg])

        two_NA_fx = fx / (2 * np.sin(np.pi/4))
        two_NA_fy = fy / (2 * np.sin(np.pi/4))
        two_NA_fz = fz / (2 * np.sin(np.pi/4))
        q_axes = 2 * np.pi * np.array((fx, fy, fz))

        optical_system = ImageProcessing.Lens()
        optical_system.compute_psf_and_otf((psf_size, N),
                                           apodization_filter=None)

        illumination_s_polarized = ImageProcessing.Illumination(s_polarized_waves)
        illumination_s_polarized.M_t = Mt_s_polarized

        illumination_circular = ImageProcessing.Illumination(circular_intensity_waves)
        illumination_circular.M_t = Mt_circular

        illumination_3waves = ImageProcessing.Illumination(three_waves_illumination, M_r=3)
        illumination_3waves.M_t = Mt_three_waves

        illumination_widefield = ImageProcessing.Illumination(widefield, M_r = 1)

        noise_estimator = ImageProcessing.NoiseEstimator(illumination_s_polarized, optical_system)
        SSNR_s_polarized = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_s_polarized_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_s_polarized)

        noise_estimator.illumination = illumination_circular
        SSNR_circular = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_circular_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_circular)

        noise_estimator.illumination = illumination_3waves
        SSNR_3waves = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_3waves_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_3waves)

        noise_estimator.illumination = illumination_widefield
        SSNR_widefield = np.abs(noise_estimator.SSNR(q_axes))
        SSNR_widefield_ra = noise_estimator.ring_average_SSNR(q_axes, SSNR_widefield)

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
        print(np.log10(1 + multiplier * SSNR_widefield))
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, arg], label="S-polarized")
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, arg],    label="Circular"   )
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, arg],      label="3 waves"    )
        ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, arg],   label="Widefield"  )

        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_s_polarized[int(N / 2), :, arg][fx >= 0], label="S-polarized")
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular[int(N / 2), :, arg][fx >= 0],    label="Circular"   )
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves[int(N / 2), :, arg][fx >= 0],      label="3 waves"    )
        ax2.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield[int(N / 2), :, arg][fx >= 0],   label="Widefield"  )

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
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, int(val)],    label="Circular"   )
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, int(val)],      label="3 waves"    )
            ax1.plot(two_NA_fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, int(val)],   label="Widefield"  )
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

            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_s_polarized[int(N / 2), :, int(val)][fx >= 0], label="S-polarized")
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_circular[int(N / 2), :, int(val)][fx >= 0],    label="Circular"   )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_3waves[int(N / 2), :, int(val)][fx >= 0],      label="3 waves"    )
            ax2.plot(two_NA_fy[fy >= 0], 1 + multiplier * SSNR_widefield[int(N / 2), :, int(val)][fx >= 0],   label="Widefield"  )
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        # slider_loc = plt.axes((0.2, 0.0, 0.3, 0.03))  # slider location and size
        # slider_ssnr = Slider(slider_loc, 'fz', 0, 50)  # slider properties
        # slider_ssnr.on_changed(update1)

        ax1.legend()
        ax2.legend()
        fig.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/fz={:.2f}_compare_SSNR.png'.format(fz[arg]))