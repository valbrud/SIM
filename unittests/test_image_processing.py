import numba
import numpy as np
import Sources
import ImageProcessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import unittest
import wrappers
import time
import scipy as sp
from matplotlib.widgets import Slider
import tqdm


class TestOpticalSystems(unittest.TestCase):
    def test_lens_otf(self):
        lens = ImageProcessing.Lens(regularization_parameter=0.0001)
        box_size = 100
        N = 40
        dx = box_size / N
        x = np.arange(-box_size / 2, box_size / 2, dx)
        y = np.copy(x)
        z = np.copy(x)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / box_size, N)
        fy = np.copy(fx)
        fz = np.copy(fx)
        otf = np.zeros((N, N, N))

        begin = time.time()
        for i, j, k in [(i, j, k) for i in range(len(fx)) for j in range(len(fy)) for k in range(len(fz))]:
            f_vector = np.array((fx[i], fy[j], fz[k]))
            otf[i, j, k] = lens.OTF(f_vector)
        end = time.time()
        print("time elapsed = ", end - begin)

        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection="3d")
        Fx, Fy = np.meshgrid(fx, fy)
        FZ = 20
        ax1.set_title("OTF, Fz = {}".format(FZ))
        ax1.set_xlabel("Fx")
        ax1.set_ylabel("Fy")
        ax1.set_zlim(0, 1)

        OTF = otf[:, :, FZ]
        ax1.plot_wireframe(Fx, Fy, OTF)

        psf = (wrappers.wrapped_fft(otf) / dx ** 3).real

        print(np.sum(psf[:, :, 20]))

        X, Y = np.meshgrid(x, y)
        Z = 20
        PSF = psf[:, :, Z]
        # for i in range(N):
        #     print(np.amax(PSF[:, i]))
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_wireframe(X, Y, PSF)
        ax2.set_title("PSF, z = {}".format(Z))
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlim(0, np.amax(psf))

        def update_freq(val):
            ax1.clear()
            ax1.set_xlabel("Fx")
            ax1.set_ylabel("Fy")
            ax1.set_title("OTF, z = {}".format(z[int(val)]))
            Z = otf[:, :, int(val)]
            ax1.set_zlim(0, 1)
            ax1.plot_wireframe(X, Y, Z)

        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        slider_freq = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_freq.on_changed(update_freq)

        def update_coord(val):
            ax2.clear()
            ax2.set_title("PSF, z = {}".format(z[int(val)]))
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            PSF = psf[:, :, int(val)]
            ax2.set_zlim(0, np.amax(psf))
            ax2.plot_wireframe(X, Y, PSF)
            print(sum(PSF))

        slider_loc = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider_coord = Slider(slider_loc, 'z', 0, N - 1)  # slider properties
        slider_coord.on_changed(update_coord)

        plt.show()

    def test_regularized_otf(self):
        e = 0.01
        lens = ImageProcessing.Lens(regularization_parameter=e)
        box_size = 100
        N = 30
        dx = box_size / N
        x = np.arange(-box_size / 2, box_size / 2, dx)
        y = np.copy(x)
        z = np.copy(x)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / box_size, N)
        fy = np.copy(fx)
        fz = np.copy(fx)
        s = fz / (4 * np.sin(0.1 / 2) ** 2)
        otf = np.zeros((N, N, N))

        begin = time.time()
        for i, j, k in [(i, j, k) for i in range(len(fx)) for j in range(len(fy)) for k in range(len(fz))]:
            f_vector = np.array((fx[i], fy[j], fz[k]))
            otf[i, j, k] = lens.regularized_analytic_OTF(f_vector)
        end = time.time()
        print("time elapsed = ", end - begin)

        otf[i, j, k] = otf[i, j, k] / np.amax(otf)
        norm = np.sum(otf[int(N / 2), int(N / 2), :])
        # print(otf[0, 0, :])
        otf = otf / norm
        # print(np.amax(otf))

        number = 19

        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection="3d")
        Fx, Fy = np.meshgrid(fx, fy)
        FZ = fx[number]
        ax1.set_title("OTF, Fz = {}".format(FZ))
        ax1.set_xlabel("Fx")
        ax1.set_ylabel("Fy")
        ax1.set_zlim(0, 1)

        OTF = otf[:, :, number]

        ax1.plot_wireframe(Fx, Fy, OTF)
        psf = (wrappers.wrapped_fft(otf) / dx ** 3).real

        E = np.zeros(N)
        for i in range(N):
            E[i] = np.sum(psf[:, :, i]) * dx
        ax3 = fig.add_subplot(133)
        ax3.plot(E)
        ax3.plot(E[int(N / 2)] * np.exp(- abs(z[np.arange(N)]) * e))

        X, Y = np.meshgrid(x, y)
        Z = x[number]
        PSF = psf[:, :, number]

        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot_wireframe(X, Y, PSF)
        ax2.set_title("PSF, z = {}".format(Z))
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlim(0, np.amax(psf))

        def update_freq(val):
            ax1.clear()
            ax1.set_xlabel("Fx")
            ax1.set_ylabel("Fy")
            ax1.set_title("OTF, fz = {}".format(fz[int(val)]))
            OTF = otf[:, :, int(val)]
            ax1.set_zlim(0, 1)
            ax1.plot_wireframe(X, Y, OTF)

        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        slider_freq = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_freq.on_changed(update_freq)

        def update_coord(val):
            ax2.clear()
            ax2.set_title("PSF, z = {}".format(z[int(val)]))
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            PSF = psf[:, :, int(val)]
            ax2.set_zlim(0, np.amax(psf))
            ax2.plot_wireframe(X, Y, PSF)
            print(np.sum(PSF))

        slider_loc = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        slider_coord = Slider(slider_loc, 'z', 0, N - 1)  # slider properties
        slider_coord.on_changed(update_coord)

        plt.show()

    def test_otf_integration(self):
        lx = np.linspace(-2, 2, 21)
        ly = np.linspace(-2, 2, 21)
        sz = np.linspace(-1, 1, 21)
        N = 21

        e = 0.1
        otf = np.zeros((len(lx), len(ly), len(sz)))
        for i, j, k in [(i, j, k) for i in range(20) for j in range(20) for k in range(20)]:
            l = (lx[i] ** 2 + ly[j] ** 2) ** 0.5
            s = sz[k]

            if l > 2:
                otf[i, j, k] = 0
                continue

            def p_max(theta):
                D = 4 - l ** 2 * (1 - np.cos(theta) ** 2)
                return (-l * np.cos(theta) + D ** 0.5) / 2

            def integrand(p, theta):
                denum = e ** 2 + (abs(s) - p * l * np.cos(theta)) ** 2
                return 8 * e * p / denum

            otf[i, j, k], _ = sp.integrate.dblquad(integrand, 0, np.pi / 2, lambda x: 0, p_max)

        otf = otf / np.amax(otf)
        number = 10
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection="3d")
        LX, LY = np.meshgrid(lx, ly)
        s = sz[number]
        ax1.set_title("OTF, s = {}".format(s))
        ax1.set_xlabel("lx")
        ax1.set_ylabel("ly")
        ax1.set_zlim(0, 1)
        OTF = otf[:, :, number]
        psf = (wrappers.wrapped_fft(otf)).real

        E = np.zeros(N)
        for i in range(N):
            E[i] = np.sum(psf[:, :, i])
        ax3 = fig.add_subplot(133)
        ax3.plot(E)

        ax1.plot_wireframe(LX, LY, OTF)

        def update_freq(val):
            ax1.clear()
            ax1.set_xlabel("lx")
            ax1.set_ylabel("ly")
            ax1.set_title("OTF, s = {}".format(sz[int(val)]))
            ax1.set_zlim(0, 1)
            OTF = otf[:, :, int(val)]
            ax1.plot_wireframe(LX, LY, OTF)

        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        slider_freq = Slider(slider_loc, 'fz', 0, 20)  # slider properties
        slider_freq.on_changed(update_freq)

        plt.show()

    def test_psf(self):
        lens = ImageProcessing.Lens(alpha=0.3, regularization_parameter=0.1)
        max_r = 10
        max_z = 100
        N = 50
        dx = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.copy(fx)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

        lens.compute_PSF_and_OTF((max_r, max_r, max_z), N)

        psf = lens.psf
        print(psf[0, 0, :])
        fig = plt.figure()
        ax_psf = fig.add_subplot(121)
        ax_otf = fig.add_subplot(122)

        X, Y = np.meshgrid(x, y)
        ax_psf.set_xlabel("z")
        ax_psf.set_ylabel("x")
        ax_psf.set_title("PSF, y = {:.2f}".format(y[int(N / 2) + 3]))
        ax_psf.imshow(psf[:, int(N / 2) + 3, :], aspect="auto", extent=(z[0], z[-1], x[0], x[-1]))

        otf = wrappers.wrapped_ifftn(psf)
        phase = np.unwrap(np.angle(otf))
        otf = np.abs(otf)
        # print(otf[int(N / 2) - 2:int(N / 2) + 2, int(N / 2) - 2:int(N / 2) + 2, [23,]])
        FX, FY = np.meshgrid(fx, fy)
        ax_otf.set_xlabel("fz")
        ax_otf.set_ylabel("fx")
        ax_otf.set_title("OTF, fy = {:.2f}".format(fy[int(N / 2) + 3]))
        ax_otf.imshow(otf[:, int(N / 2) + 3, :], aspect="auto", extent=(fz[0], fz[-1], fx[0], fx[-1]))

        # ax_otf_z = fig.add_subplot(223)

        # FS, FZ = np.meshgrid(fx, fz)
        # ax_otf_z.plot(fz, otf[int(N/2), int(N/2), :])
        def update_psf(val):
            ax_psf.clear()
            ax_psf.set_xlabel("x")
            ax_psf.set_ylabel("y")
            ax_psf.set_title("PSF, z = {:.2f}".format(z[int(val)]))
            Z = psf[:, :, int(val)]
            ax_psf.imshow(psf[:, :, int(val)], extent=(x[0], x[-1], y[0], y[-1]))
            print(np.sum(Z))

        # slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        # slider_freq = Slider(slider_loc, 'z', 0, N - 1)  # slider properties
        # slider_freq.on_changed(update_psf)
        def update_otf(val):
            ax_otf.clear()
            ax_otf.set_title("OTF, fz = {:.2f}".format(fz[int(val)]))
            ax_otf.set_xlabel("fx")
            ax_otf.set_ylabel("fy")
            FZ = otf[:, :, int(val)]
            ax_otf.imshow(otf[:, :, int(val)], extent=(fx[0], fx[-1], fy[0], fy[-1]))

        # slider_loc = plt.axes([0.2, 0.1, 0.65, 0.03])  # slider location and size
        # slider_coord = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        # slider_coord.on_changed(update_otf)

        plt.show()

    def test_get_oft(self):
        optical_system = ImageProcessing.Lens(alpha=0.3, regularization_parameter=0.1)
        max_r = 1
        max_z = 1
        N = 20
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        wavevector = 2 * np.pi * np.array((fx[10], fy[10], fz[10]))
        self.assertAlmostEqual(optical_system.get_otf(wavevector), optical_system.otf[10, 10, 10])

        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        wavevector = 2 * np.pi * np.array((fx[4], fy[10], fz[9]))
        self.assertAlmostEqual(optical_system.get_otf(wavevector), optical_system.otf[4, 10, 9])

        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        wavevector = 2 * np.pi * np.array((fx[5], fy[5], fz[5]))
        self.assertAlmostEqual(optical_system.get_otf(wavevector), optical_system.otf[5, 5, 5])

    def test_shifted_otf(self):
        max_r = 2.5
        max_z = 2.5
        N = 20
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        Mt = 4
        a0 = (1 + 2 * b ** 2)
        wavevectors = [np.array((0, 0, 0)), np.array((2 * k1, 0, 0)), np.array((- 2 * k1, 0, 0))]

        optical_system = ImageProcessing.Lens()
        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        optical_system.compute_shifted_otf(wavevectors)
        otf_sum = np.zeros((len(x), len(y), len(z)))
        for otf in optical_system.shifted_otfs:
            otf_sum += optical_system.shifted_otfs[otf]
        print(otf_sum[:, :, 10])
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        Fx, Fy = np.meshgrid(fx, fy)
        IM = otf_sum[:, :, int(N / 2)]
        # ax.imshow(IM, extent=(fx[0], fx[-1], fy[0], fy[-1]))

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.plot_wireframe(Fx, Fy, IM)
        print(IM)

        def update(val):
            ax.clear()
            ax.set_title("otf_sum, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax.set_xlabel("fx, $\lambda^{-1}$")
            ax.set_ylabel("fy, $\lambda^{-1}$")
            IM = otf_sum[:, :, int(val)]
            print(IM)
            ax.plot_wireframe(Fx, Fy, IM)
            # ax.imshow(IM, extent=(fx[0], fx[-1], fy[0], fy[-1]))

        slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update)
        plt.show()

    def test_SSNR(self):
        max_r = 2.5
        max_z = 2.5
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

        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        Mt = 4
        a0 = (1 + 2 * b ** 2)
        waves = [
            Sources.IntensityPlaneWave((1 + 2 * b ** 2) / (Mt * a0), 0, np.array((0, 0, 0))),
            Sources.IntensityPlaneWave((-b ** 2 / 2) / (Mt * a0), 0, np.array((-2 * k1, 0, 0))),
            Sources.IntensityPlaneWave((-b ** 2 / 2) / (Mt * a0), 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityPlaneWave((-b ** 2 / 2)/(Mt * a0), 0, np.array((0, 2 * k1, 0))),
            Sources.IntensityPlaneWave((-b ** 2 / 2)/(Mt * a0), 0, np.array((0, -2 * k1, 0))),
            Sources.IntensityPlaneWave((-1j * b / 2)/(Mt * a0), 0, np.array((k1, 0, k2))),
            Sources.IntensityPlaneWave((1j * b / 2)/(Mt * a0), 0, np.array((-k1, 0, k2))),
            Sources.IntensityPlaneWave((-1 * b / 2)/(Mt * a0), 0, np.array((0, k1, k2))),
            Sources.IntensityPlaneWave((1 * b / 2)/(Mt * a0), 0, np.array((0, -k1, k2))),
            Sources.IntensityPlaneWave((-1j * b / 2)/(Mt * a0), 0, np.array((k1, 0, -k2))),
            Sources.IntensityPlaneWave((1j * b / 2)/(Mt * a0), 0, np.array((-k1, 0, -k2))),
            Sources.IntensityPlaneWave((1 * b / 2)/(Mt * a0), 0, np.array((0, k1, -k2))),
            Sources.IntensityPlaneWave((-1 * b / 2)/(Mt * a0), 0, np.array((0, -k1, -k2)))
        ]

        base_vector_x_shift = np.array((1 / (4 * np.sin(theta)), 0, 0))
        base_vector_y_shift = np.array((0, 1 / (4 * np.sin(theta)), 0))
        base_vector_z_shift = np.array((0, 0, 1 / (2 * (1 - np.cos(theta)))))
        spacial_shifts = np.zeros((4, 4, 2, 3))
        for i, j, k in [(i, j, k) for i in range(4) for j in range(4) for k in range(2)]:
            spacial_shifts[i, j, k] = i * base_vector_x_shift + j * base_vector_y_shift + k * base_vector_z_shift

        IlluminationPolarized = ImageProcessing.Illumination(waves, spacial_shifts)
        optical_system = ImageProcessing.Lens()
        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)

        SSNR = np.zeros((N, N, N))
        for i, j, k in tqdm.tqdm([(i, j, k) for i in range(len(fx)) for j in range(len(fy)) for k in range(len(fz))]):
            f = np.array((fx[i], fy[j], fz[k]))
            q = f * 2 * np.pi
            SSNR[i, j, k] = np.abs(
                ImageProcessing.ImageProcessingFunctions.SSNR(q, IlluminationPolarized, optical_system))

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

        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        Mt = 32
        a0 = (2 + 8 * b ** 2)
        waves = [
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((k1, k1, 0))),
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((-k1, k1, 0))),
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((k1, -k1, 0))),
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((-k1, -k1, 0))),

            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((0, 2 * k1, 0))),
            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((0, -2 * k1, 0))),
            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((-2 * k1, 0, 0))),

            Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((k1, 0, -k2))),
            Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((-k1, 0, k2))),
            Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((k1, 0, k2))),
            Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((-k1, 0, -k2))),

            Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, k1, -k2))),
            Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, k1, k2))),
            Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, -k1, k2))),
            Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, -k1, -k2))),

            Sources.IntensityPlaneWave(a0 / (Mt * a0), 0, np.array((0, 0, 0)))
        ]

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

        begin_interpolation = time.time()
        SSNR_I = np.zeros((N, N, N))
        SSNR_I = np.abs(noise_estimator.SSNR(q_axes, method="Scipy"))
        end_interpolation = time.time()
        print(end_interpolation - begin_interpolation)

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
        mp2 = ax2.imshow(np.log10(SSNR_I[:, :, int(N / 2)]))
        cb2 = plt.colorbar(mp2, fraction=0.046, pad=0.04)

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

        def update2(val):
            ax2.set_title("SSNR_I, fz = {:.2f}, ".format(fz[int(val)]) + "$\\lambda^{-1}$")
            ax2.set_xlabel("fx, $\lambda^{-1}$")
            ax2.set_ylabel("fy, $\lambda^{-1}$")
            print(int(val))
            Z = (np.log10(SSNR_I[:, :, int(val)]))
            mp2.set_data(Z)
            min = Z.min()
            max = Z.max()
            mp2.set_clim(vmin=min, vmax=max)
            # plt.draw()
            print(np.amax(np.abs(SSNR[:, :, int(val)])))

        slider_loc = plt.axes((0.6, 0.1, 0.3, 0.03))  # slider location and size
        slider_ssnri = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnri.on_changed(update2)

        plt.show()

    def test_multiprocessing_acelleration(self):
        max_r = 3
        max_z = 3
        N = 20
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        Mt = 32
        a0 = (2 + 8 * b ** 2)
        waves = [
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((k1, k1, 0))),
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((-k1, k1, 0))),
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((k1, -k1, 0))),
            Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt * a0), 0, np.array((-k1, -k1, 0))),

            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((0, 2 * k1, 0))),
            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((0, -2 * k1, 0))),
            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt * a0), 0, np.array((-2 * k1, 0, 0))),

            Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((k1, 0, -k2))),
            Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((-k1, 0, k2))),
            Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((k1, 0, k2))),
            Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((-k1, 0, -k2))),

            Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, k1, -k2))),
            Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, k1, k2))),
            Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, -k1, k2))),
            Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt * a0), 0, np.array((0, -k1, -k2))),

            Sources.IntensityPlaneWave(a0 / (Mt * a0), 0, np.array((0, 0, 0)))
        ]

        base_vector_x_shift = np.array((1 / (4 * np.sin(theta)), 0, 0))
        base_vector_y_shift = np.array((0, 1 / (4 * np.sin(theta)), 0))
        base_vector_z_shift = np.array((0, 0, 1 / (2 * (1 - np.cos(theta)))))
        spacial_shifts = np.zeros((4, 4, 2, 3))
        for i, j, k in [(i, j, k) for i in range(4) for j in range(4) for k in range(2)]:
            spacial_shifts[i, j, k] = i * base_vector_x_shift + j * base_vector_y_shift + k * base_vector_z_shift

        illumination_circular = ImageProcessing.Illumination(waves, spacial_shifts)
        optical_system = ImageProcessing.Lens()

        optical_system.compute_PSF_and_OTF(np.array((2 * max_r, 2 * max_r, 2 * max_z)), N)
        noise_estimator = ImageProcessing.NoiseEstimator(illumination_circular, optical_system)
        wavevectors = []
        for spacial_wave in illumination_circular.waves:
            wavevectors.append(spacial_wave.wavevector)

        begin_single_cso = time.time()
        optical_system.compute_shifted_otf(wavevectors)
        print(optical_system.shifted_otfs.keys())
        end_single_cso = time.time()
        print("single cso = ", end_single_cso - begin_single_cso)

        # Cwo works fast and is not parallelized on CPU. In fact, it is hard to parallelize it efficiently.
        # Single is just for clarity

        begin_single_cwo = time.time()
        optical_system.compute_wvdiff_otfs(wavevectors)
        end_single_cwo = time.time()
        print("single cwo = ", end_single_cwo - begin_single_cwo)

        begin_single_ssnr = time.time()
        SSNR = noise_estimator.SSNR(2 * np.pi * np.array((fx, fy, fz)), method="Scipy")
        end_single_ssnr = time.time()
        print("single ssnr = ", end_single_ssnr - begin_single_ssnr)
        print(np.abs(SSNR[:, :, 10]))
        # optical_system.shifted_otfs = {}
        # pool = mp.Pool(mp.cpu_count())
        # begin_multi_cso = time.time()
        # otfs = [pool.apply_async(optical_system.compute_shifted_otf, args=([wavevector],)) for wavevector in wavevectors]
        # for otf in otfs:
        #     optical_system.shifted_otfs.update(otf.get())
        # pool.close()
        # pool.join()
        # print(optical_system.shifted_otfs.keys())
        # end_multi_cso = time.time()
        # print("multi cso = ", end_multi_cso - begin_multi_cso)
        #
        # begin_multi_ssnr = time.time()
        # f = np.array((fx, fy, fz))
        # SSNRmp = wrappers_multiprocessing.get_SSNR_multi(noise_estimator, 2 * np.pi * f)
        # end_multi_ssnr = time.time()
        # print("multi ssnr = ", end_multi_ssnr - begin_multi_ssnr)

        # self.assertEqual(np.sum(SSNR), np.sum(SSNRmp))