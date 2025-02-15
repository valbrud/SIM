import unittest
import numpy as np
import matplotlib.pyplot as plt
import wrappers
import sys
sys.path.append('../')

N = 100

class TestWrappersFT(unittest.TestCase):
    def test_1d(self):
        L = 250 / 10 ** 6
        b = 5 / 10 ** 6
        dx = L / N
        df = 1 / dx / N
        x = np.array([n * dx - L / 2 for n in range(N)])
        aperture = np.zeros(len(x))
        aperture[(x >= -b / 2) * (x <= b / 2)] = 1
        apertureFT = dx * wrappers.wrapped_fft(aperture)
        apertureIFT = 1 / dx * wrappers.wrapped_ifft(apertureFT)
        np.testing.assert_array_almost_equal(aperture, apertureIFT, 12)

    def test_nd(self):
        N = 512
        L = 250 / 10 ** 6
        b = 5 / 10 ** 6
        dx = L / N
        dy = L / N
        dfx = 1 / dx / N
        dfy = 1 / dy / N
        x = np.array([n * dx - L / 2 for n in range(N)])
        y = np.array([n * dy - L / 2 for n in range(N)])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        aperture = np.zeros((len(x), len(y)))
        aperture[(X >= -b / 2) * (X <= b / 2) * (Y >= -b/2) * (Y <= b/2)] = 1
        apertureFT = dx * dy * wrappers.wrapped_fftn(aperture)
        apertureIFT = 1 / dx / dy * wrappers.wrapped_ifftn(apertureFT)
        ax.plot_wireframe(X, Y, apertureIFT.real)
        plt.show()

    def test_fftw(self):
        L = 250 / 10 ** 6
        b = 5 / 10 ** 6
        dx = L / N
        df = 1 / dx / N
        x = np.array([n * dx - L / 2 for n in range(N)])
        aperture = np.zeros(len(x))
        aperture[(x >= -b / 2) * (x <= b / 2)] = 1
        apertureFT = dx * wrappers.wrapped_fft(aperture)
        apertureIFT = 1 / dx * wrappers.wrapped_ifft(apertureFT)
        np.testing.assert_array_almost_equal(aperture, apertureIFT, 12)
