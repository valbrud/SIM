import numpy as np
import Sources
import ImageProcessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import unittest
import time
from matplotlib.widgets import Slider
import tqdm

theta = np.pi / 4
b = 1/2**0.5
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (np.cos(theta) - 1)

Mt_s_polarized = 32
a0 = (2 + 4 * b ** 2)

norm = a0 * Mt_s_polarized
s_polarized_waves = {
    (0, 0, 0) : Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

    (-2, 0, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((-2 * k1, 0, 0))),
    (2, 0, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((2 * k1, 0, 0))),
    (0, 2, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((0, -2 * k1, 0))),

    (1, 0, 1) : Sources.IntensityPlaneWave((-1j * b) / norm, 0, np.array((k1, 0, k2))),
    (-1, 0, 1): Sources.IntensityPlaneWave((1j * b) / norm, 0, np.array((-k1, 0, k2))),
    (0, 1, 1) : Sources.IntensityPlaneWave((-1 * b) / norm, 0, np.array((0, k1, k2))),
    (0, -1, 1) : Sources.IntensityPlaneWave((1 * b) / norm, 0, np.array((0, -k1, k2))),

    (1, 0, -1) : Sources.IntensityPlaneWave((-1j * b) / norm, 0, np.array((k1, 0, -k2))),
    (-1, 0, -1) : Sources.IntensityPlaneWave((1j * b) / norm, 0, np.array((-k1, 0, -k2))),
    (0, 1, -1) : Sources.IntensityPlaneWave((1 * b) / norm, 0, np.array((0, k1, -k2))),
    (0, -1, -1) : Sources.IntensityPlaneWave((-1 * b) / norm, 0, np.array((0, -k1, -k2)))
}

Mt_circular = 32
theta = np.pi/4
b = 1 / 2
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (1 - np.cos(theta))
a0 = (2 + 8 * b ** 2)
circular_intensity_waves = {
    (1, 1, 0)  : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((k1, k1, 0))),
    (-1, 1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((-k1, k1, 0))),
    (1, -1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((k1, -k1, 0))),
    (-1, -1, 0): Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((-k1, -k1, 0))),

    (0, 2, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0): Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((0, -2 * k1, 0))),
    (2, 0, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((2 * k1, 0, 0))),
    (-2, 0, 0): Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((-2 * k1, 0, 0))),

    (1, 0, -1) : Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((k1, 0, -k2))),
    (-1, 0, 1) : Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((-k1, 0, k2))),
    (1, 0, 1): Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((k1, 0, k2))),
    (-1, 0, -1):Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((-k1, 0, -k2))),

    (0, 1, -1) : Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, k1, -k2))),
    (0, 1, 1)  : Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, k1, k2))),
    (0, -1, 1) : Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, -k1, k2))),
    (0, -1, -1): Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, -k1, -k2))),

    (0, 0, 0): Sources.IntensityPlaneWave(a0 / (Mt_circular * a0), 0, np.array((0, 0, 0)))
}

b = 1/2**0.5
a0 = 1 + 2 * b**2
Mr_tree_waves = 3
Mt_three_waves = 4
norm = a0 * Mr_tree_waves * Mt_three_waves
three_waves_illumination = {
    (0, 0, 0): Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

    (0, 2, 0): Sources.IntensityPlaneWave(b**2 / norm, 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0): Sources.IntensityPlaneWave(b**2 / norm, 0, np.array((0, -2 * k1, 0))),

    (0, 1, 1): Sources.IntensityPlaneWave(b / norm, 0, np.array((0, k1, k2))),
    (0, -1, 1):  Sources.IntensityPlaneWave(b / norm, 0, np.array((0, -k1, k2))),
    (0, 1, -1): Sources.IntensityPlaneWave(b / norm, 0, np.array((0,  k1, -k2))),
    (0, -1, -1): Sources.IntensityPlaneWave(b / norm, 0, np.array((0, -k1, -k2))),
}


Mt_widefield = 1
widefield = {
    (0, 0, 0) : Sources.IntensityPlaneWave(1/Mt_widefield, 0, np.array((0, 0, 0)))
}

class TestOpticalSystems(unittest.TestCase):
    def test_shifted_otf(self):
        max_r = 5
        max_z = 5
        N = 100
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        wavevectors = [wave.wavevector for wave in s_polarized_waves]

        optical_system = ImageProcessing.Lens()
        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system.compute_shifted_otf(wavevectors)
        otf_sum = np.zeros((len(x), len(y), len(z)), dtype=np.complex128)
        for otf in optical_system.shifted_otfs:
            otf_sum += optical_system.shifted_otfs[otf]
        print(otf_sum[:, :, 10])
        fig = plt.figure()
        IM = otf_sum[:, :, int(N / 2)]

        ax = fig.add_subplot(111)
        mp1 = ax.imshow(np.abs(IM))

        print(IM)
        plt.colorbar(mp1)
        def update(val):
            ax.clear()
            ax.set_title("otf_sum, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax.set_xlabel("fx, $\lambda^{-1}$")
            ax.set_ylabel("fy, $\lambda^{-1}$")
            IM = otf_sum[:, :, int(val)].real
            mp1.set_clim(vmin=IM.min(), vmax=IM.max())

            ax.imshow(np.abs(IM), extent=(fx[0], fx[-1], fy[0], fy[-1]))

        slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update)
        plt.show()

    def test_SSNR(self):
        max_r = 10
        max_z = 10
        N = 100
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        waves = circular_intensity_waves

        base_vector_x_shift = np.array((1 / (4 * np.sin(theta)), 0, 0))
        base_vector_y_shift = np.array((0, 1 / (4 * np.sin(theta)), 0))
        base_vector_z_shift = np.array((0, 0, 1 / (2 * (1 - np.cos(theta)))))
        spacial_shifts = np.zeros((4, 4, 2, 3))
        for i, j, k in [(i, j, k) for i in range(4) for j in range(4) for k in range(2)]:
            spacial_shifts[i, j, k] = i * base_vector_x_shift + j * base_vector_y_shift + k * base_vector_z_shift

        illumination_polarized = ImageProcessing.Illumination(waves, spacial_shifts)
        optical_system = ImageProcessing.Lens()
        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)

        wavevectors=[]
        for spacial_wave in illumination_polarized.waves:
            wavevectors.append(spacial_wave.wavevector)
        optical_system.compute_shifted_otf(wavevectors)
        optical_system.compute_wvdiff_otfs(wavevectors)

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
            IM = SSNR[:, :, int(val / 2)].T
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
        max_r = 10
        max_z = 10
        N = 30
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        waves = three_waves_illumination

        base_vector_x_shift = np.array((1 / (4 * np.sin(theta)), 0, 0))
        base_vector_y_shift = np.array((0, 1 / (4 * np.sin(theta)), 0))
        base_vector_z_shift = np.array((0, 0, 1 / (2 * (1 - np.cos(theta)))))
        spacial_shifts = np.zeros((4, 4, 2, 3))
        for i, j, k in [(i, j, k) for i in range(4) for j in range(4) for k in range(2)]:
            spacial_shifts[i, j, k] = i * base_vector_x_shift + j * base_vector_y_shift + k * base_vector_z_shift

        illumination_polarized = ImageProcessing.Illumination(waves, spacial_shifts)
        optical_system = ImageProcessing.Lens()
        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system)

        begin_shifted = time.time()
        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        wavevectors = []
        for spacial_wave in illumination_polarized.waves:
            wavevectors.append(spacial_wave.wavevector)
        optical_system.compute_shifted_otf(wavevectors)
        optical_system.compute_wvdiff_otfs(wavevectors)

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        SSNR = np.abs(noise_estimator.SSNR(q_axes, method="Fourier"))
        end_shifted = time.time()
        print(end_shifted - begin_shifted)

        # begin_interpolation = time.time()
        # SSNR_I = np.zeros((N, N, N))
        # SSNR_I = np.abs(noise_estimator.SSNR(q_axes, method="Scipy"))
        # end_interpolation = time.time()
        # print(end_interpolation - begin_interpolation)

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
        mp1 = ax1.imshow(np.log10(SSNR[:, :, int(N / 2)]))
        cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        # mp2 = ax2.imshow(np.log10(SSNR_I[:, :, int(N / 2)]))
        # cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)

        def update1(val):
            ax1.set_title("SSNR, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fx, $\lambda^{-1}$")
            ax1.set_ylabel("fy, $\lambda^{-1}$")
            Z = (np.log10(SSNR[:, :, int(val)]))
            mp1.set_data(Z)
            mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            plt.draw()
            print(np.amax(SSNR[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        # def update2(val):
        #     ax2.set_title("SSNR_I, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
        #     ax2.set_xlabel("fx, $\lambda^{-1}$")
        #     ax2.set_ylabel("fy, $\lambda^{-1}$")
        #     print(int(val))
        #     Z = (np.log10(SSNR_I[:, :, int(val)]))
        #     mp2.set_data(Z)
        #     min = Z.min()
        #     max = Z.max()
        #     mp2.set_clim(vmin=min, vmax=max)
        #     # plt.draw()
        #     print(np.amax(np.abs(SSNR[:, :, int(val)])))
        #
        # slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        # slider_ssnri = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        # slider_ssnri.on_changed(update2)

        plt.show()

    def test_SSNR_ring_averages(self):
        max_r = 3
        max_z = 3
        N = 50
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        waves = three_waves_illumination

        illumination_polarized = ImageProcessing.Illumination(waves, M_r=3)
        illumination_polarized.M_t = 32
        optical_system = ImageProcessing.Lens(alpha=np.pi/4)

        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system.compute_shifted_otf(illumination_polarized.get_wavevectors())
        optical_system.compute_wvdiff_otfs(illumination_polarized.get_wavevectors())

        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system)
        noise_estimator.compute_parameters_for_Vj()
        noise_estimator.compute_wfdiff_otfs_for_Vj()

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        SSNR = np.abs(noise_estimator.SSNR(q_axes, method="Scipy"))
        SSNR_ring_averaged = noise_estimator.ring_average_SSNR(q_axes, SSNR)
        print(SSNR_ring_averaged.shape)

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure()
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax1.set_title("log10(SSNR), fz = {:.2f}, ".format(fz[N//2]) + "$\\lambda^{-1}$")
        ax1.set_xlabel("fx, $\lambda^{-1}$")
        ax1.set_ylabel("fy, $\lambda^{-1}$")
        ax2 = fig.add_subplot(122)
        ax2.set_title("Ring averaged, fz = {:.2f}, ".format(fz[N//2]) + "$\\lambda^{-1}$")
        ax2.set_xlabel("fx, $\lambda^{-1}$")
        ax2.set_ylabel("fy, $\lambda^{-1}$")
        ax2.set_yscale("log")

        mp1 = ax1.imshow(np.log10(SSNR[:, :, int(N / 2)]), extent=(fx[0], fx[-1], fy[0], fy[-1]))
        cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        ax2.plot(q_axes[2][q_axes[2] >= 0], np.log10(SSNR_ring_averaged[:, N//2]))
        ax2.set_aspect(1. / ax2.get_data_ratio())


        def update1(val):
            ax1.set_title("log10(SSNR), fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fx, $\lambda^{-1}$")
            ax1.set_ylabel("fy, $\lambda^{-1}$")
            Z = (np.log10(SSNR[:, :, int(val)]))
            mp1.set_data(Z)
            mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            plt.draw()
            print(np.amax(SSNR[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        def update2(val):
            ax2.clear()
            ax2.set_title("Ring averaged, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax2.set_xlabel("fx, $\lambda^{-1}$")
            ax2.set_ylabel("SSNR, $\lambda^{-1}$")
            ssnr_r_sliced = (SSNR_ring_averaged[:, int(val)])
            ax2.plot(q_axes[2][q_axes[2] >= 0] / (2 * np.pi), ssnr_r_sliced)
            ax2.set_yscale("log")
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        slider_ra = Slider(slider_loc, 'fz', 0, len(SSNR_ring_averaged[0]))  # slider properties
        slider_ra.on_changed(update2)

        plt.show()

    def test_SSNR_projections(self):
        max_r = 6
        max_z = 6
        N = 50
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        waves = s_polarized_waves

        base_vector_x_shift = np.array((1 / (4 * np.sin(theta)), 0, 0))
        base_vector_y_shift = np.array((0, 1 / (4 * np.sin(theta)), 0))
        base_vector_z_shift = np.array((0, 0, 1 / (2 * (1 - np.cos(theta)))))
        spacial_shifts = np.zeros((4, 4, 2, 3))
        for i, j, k in [(i, j, k) for i in range(4) for j in range(4) for k in range(2)]:
            spacial_shifts[i, j, k] = i * base_vector_x_shift + j * base_vector_y_shift + k * base_vector_z_shift

        illumination_polarized = ImageProcessing.Illumination(waves, spacial_shifts)
        optical_system = ImageProcessing.Lens()
        noise_estimator = ImageProcessing.NoiseEstimator(illumination_polarized, optical_system)

        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        wavevectors = []
        for spacial_wave in illumination_polarized.waves:
            wavevectors.append(spacial_wave.wavevector)
        optical_system.compute_shifted_otf(wavevectors)
        optical_system.compute_wvdiff_otfs(wavevectors)

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        SSNR = np.abs(noise_estimator.SSNR(q_axes, method="Scipy"))

        Fx, Fy = np.meshgrid(fx, fy)
        fig = plt.figure()
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(121)
        ax1.set_title("SSNR, fz = {:.2f}, ".format(fz[N//2]) + "$\\lambda^{-1}$")
        ax1.set_xlabel("fx, $\lambda^{-1}$")
        ax1.set_ylabel("fy, $\lambda^{-1}$")
        ax2 = fig.add_subplot(122)
        ax2.set_title("Projection fx = {:.2f}, fz = {:.2f}, ".format(fx[N//2], fz[N//2]) + "$\\lambda^{-1}$")
        ax2.set_xlabel("fy, $\lambda^{-1}$")
        ax2.set_ylabel("SSNR, $\lambda^{-1}$")
        ax2.set_yscale("log")

        mp1 = ax1.imshow(np.log10(SSNR[:, :, int(N / 2)]), extent=(fx[0], fx[-1], fy[0], fy[-1]))
        cb1 = plt.colorbar(mp1, fraction=0.046, pad=0.04)
        ax2.plot(q_axes[1], SSNR[N//2, :, N//2])
        ax2.set_aspect(1. / ax2.get_data_ratio())


        def update1(val):
            ax1.set_title("log10(SSNR), fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fx, $\lambda^{-1}$")
            ax1.set_ylabel("fy, $\lambda^{-1}$")
            Z = (np.log10(SSNR[:, :, int(val)]))
            mp1.set_data(Z)
            mp1.set_clim(vmin=Z.min(), vmax=Z.max())
            plt.draw()
            print(np.amax(SSNR[:, :, int(val)]))

        slider_loc = plt.axes((0.2, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update1)

        def update2(val):
            ax2.clear()
            ax2.set_title("Projection fx = fy, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax2.set_xlabel("fy, $\lambda^{-1}$")
            ax2.set_ylabel("SSNR, $\lambda^{-1}$")
            ax2.plot(q_axes[1][q_axes[1] >= 0] / (2 * np.pi), np.diagonal(SSNR[:, :, int(val)])[q_axes[1] >= 0])
            ax2.set_yscale("log")
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        slider_proj = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_proj.on_changed(update2)

        plt.show()

    def test_compare_SSNR(self):
        max_r = 3
        max_z = 5
        N = 50
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        illumination_s_polarized = ImageProcessing.Illumination(s_polarized_waves)
        illumination_s_polarized.M_t = Mt_s_polarized
        optical_system_s_polarized = ImageProcessing.Lens()
        optical_system_s_polarized.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system_s_polarized.compute_shifted_otf(illumination_s_polarized.get_wavevectors())
        optical_system_s_polarized.compute_wvdiff_otfs(illumination_s_polarized.get_wavevectors())

        noise_estimator_polarized = ImageProcessing.NoiseEstimator(illumination_s_polarized, optical_system_s_polarized)
        noise_estimator_polarized.compute_parameters_for_Vj()
        noise_estimator_polarized.compute_wfdiff_otfs_for_Vj()

        illumination_circular = ImageProcessing.Illumination(circular_intensity_waves)
        illumination_circular.M_t = Mt_circular
        optical_system_circular = ImageProcessing.Lens()
        optical_system_circular.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system_s_polarized.compute_shifted_otf(illumination_circular.get_wavevectors())
        optical_system_s_polarized.compute_wvdiff_otfs(illumination_circular.get_wavevectors())
        noise_estimator_circular = ImageProcessing.NoiseEstimator(illumination_circular, optical_system_circular)
        noise_estimator_circular.compute_parameters_for_Vj()
        noise_estimator_circular.compute_wfdiff_otfs_for_Vj()

        illumination_3waves = ImageProcessing.Illumination(three_waves_illumination, M_r=3)
        illumination_3waves.M_t = Mt_three_waves
        optical_system_3waves = ImageProcessing.Lens()
        optical_system_3waves.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system_3waves.compute_shifted_otf(illumination_3waves.get_wavevectors())
        optical_system_3waves.compute_wvdiff_otfs(illumination_3waves.get_wavevectors())
        noise_estimator_3waves = ImageProcessing.NoiseEstimator(illumination_3waves, optical_system_3waves)
        noise_estimator_3waves.compute_parameters_for_Vj()
        noise_estimator_3waves.compute_wfdiff_otfs_for_Vj()

        illumination_widefield = ImageProcessing.Illumination(widefield)
        optical_system_widefield = ImageProcessing.Lens()
        optical_system_widefield.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system_widefield.compute_wvdiff_otfs(illumination_widefield.get_wavevectors())
        noise_estimator_widefield = ImageProcessing.NoiseEstimator(illumination_widefield, optical_system_widefield)
        noise_estimator_widefield.compute_parameters_for_Vj()
        noise_estimator_widefield.compute_wfdiff_otfs_for_Vj()

        q_axes = 2 * np.pi * np.array((fx, fy, fz))
        SSNR_s_polarized = np.abs(noise_estimator_polarized.SSNR(q_axes, method="Scipy"))
        SSNR_s_polarized_ra = noise_estimator_polarized.ring_average_SSNR(q_axes, SSNR_s_polarized)

        SSNR_circular = np.abs(noise_estimator_circular.SSNR(q_axes, method="Scipy"))
        SSNR_circular_ra = noise_estimator_circular.ring_average_SSNR(q_axes, SSNR_circular)

        SSNR_3waves = np.abs(noise_estimator_3waves.SSNR(q_axes, method="Scipy"))
        SSNR_3waves_ra = noise_estimator_3waves.ring_average_SSNR(q_axes, SSNR_3waves)

        SSNR_widefield = np.abs(noise_estimator_widefield.SSNR(q_axes, method="Scipy"))
        SSNR_widefield_ra = noise_estimator_widefield.ring_average_SSNR(q_axes, SSNR_widefield)

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


        ax1.set_title("Ring averaged SSNR, \n fz = {:.2f}, ".format(fz[int(N/2)]) + "$\\lambda^{-1}$")
        ax1.set_xlabel("fr, $\lambda^{-1}$")
        ax1.set_ylabel("$SSNR_{ra}$, $\lambda^{-1}$")
        ax1.set_yscale("log")
        ax1.set_ylim(1, 10**6)
        ax1.set_xlim(0, fx[-1])
        ax1.grid()

        ax2.set_title("SSNR projection fx = fy , \n fz = {:.2f}, ".format(fz[int(N/2)]) + "$\\lambda^{-1}$")
        ax2.set_xlabel("fx, $\lambda^{-1}$")
        ax2.set_ylabel("SSNR, $\lambda^{-1}$")
        ax2.set_yscale("log")
        ax2.set_ylim(1, 10**6)
        ax2.set_xlim(0, fx[-1])
        ax2.grid()

        multiplier = 10 ** 8
        ax1.plot(fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, N//2], label="S-polarized")
        ax1.plot(fx[fx >= 0], 1 + multiplier * SSNR_circular_ra[:, N//2], label="Circular")
        ax1.plot(fx[fx >= 0], 1 + multiplier * SSNR_3waves_ra[:, N//2], label="3 waves")
        ax1.plot(fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, N//2], label="Widefield")

        ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_s_polarized[:, :, N//2])[q_axes[1] >= 0], label="S-polarized")
        ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_circular[:, :, N//2])[q_axes[1] >= 0], label="Circular")
        ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_3waves[:, :, N//2])[q_axes[1] >= 0], label="3 waves")
        ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_widefield[:, :, N//2])[q_axes[1] >= 0], label="Widefield")

        ax1.set_aspect(1. / ax1.get_data_ratio())
        ax2.set_aspect(1. / ax2.get_data_ratio())

        def update1(val):
            ax1.clear()
            ax1.set_title("Ring averaged SSNR \n for s-polarized waves, \n fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax1.set_xlabel("fr, $\lambda^{-1}$")
            ax1.set_ylabel("$SSNR_{ra}$, $\lambda^{-1}$")
            ax1.set_yscale("log")
            ax1.set_ylim(1, 10**6)
            ax1.set_xlim(0, fx[-1])
            ax1.grid()
            ax1.plot(fx[fx >= 0], 1 + multiplier * SSNR_s_polarized_ra[:, int(val)], label="S-polarized")
            ax1.plot(fx[fx >= 0], (1 + multiplier * SSNR_circular_ra[:, int(val)]), label="Circular")
            ax1.plot(fx[fx >= 0], (1 + multiplier * SSNR_3waves_ra[:, int(val)]), label="3 waves")
            ax1.plot(fx[fx >= 0], 1 + multiplier * SSNR_widefield_ra[:, int(val)], label="Widefield")
            ax1.legend()
            ax1.set_aspect(1. / ax1.get_data_ratio())


        slider_loc = plt.axes((0.2, 0.1, 0.3, 0.03))  # slider location and size
        slider_ra_s = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ra_s.on_changed(update1)

        def update2(val):
            ax2.clear()
            ax2.set_title("SSNR projection fx = 0 \n for s-polarized waves, \n fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax2.set_xlabel("fx, $\lambda^{-1}$")
            ax2.set_ylabel("SSNR, $\lambda^{-1}$")
            ax2.set_yscale("log")
            ax2.set_ylim(1, 10**6)
            ax2.set_xlim(0, fx[-1])
            ax2.grid()
            # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_s_polarized[:, :, int(val)])[q_axes[1] >= 0], label="S-polarized")
            # ax2.plot(fx[fx >= 0], 1 + multiplier * np.diagonal(SSNR_circular[:, :, int(val)])[q_axes[1] >= 0], label="Circular")
            ax2.plot(fx[fx >= 0], 1 + multiplier * SSNR_s_polarized[int(N/2), :, int(val)][q_axes[1] >= 0],
                     label="S-polarized")
            ax2.plot(fx[fx >= 0], (1 + multiplier * SSNR_circular[int(N/2), :, int(val)][q_axes[1] >= 0]),
                     label="Circular")
            ax2.plot(fx[fx >= 0], (1 + multiplier * SSNR_3waves[int(N/2), :, int(val)][q_axes[1] >= 0]),
                     label="3 waves")
            ax2.plot(fx[fx >= 0], 1 + multiplier * SSNR_widefield[int(N/2), :, int(val)][q_axes[1] >= 0],
                     label="Widefield")
            ax2.legend()
            ax2.set_aspect(1. / ax2.get_data_ratio())

        slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        slider_proj_s = Slider(slider_loc, 'fz', 0, N-1)  # slider properties
        slider_proj_s.on_changed(update2)

        ax1.legend()
        ax2.legend()
        plt.show()
