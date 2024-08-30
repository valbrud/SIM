import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')


class TestPlottingPercularities(unittest.TestCase):
    def test_indexing(self):
        a = np.array([[[0, 1], [10, 11]], [[100, 101], [110, 111]]])
        print(a[:, :, 0])

    def test_axis(self):
        x = np.arange(10)
        y = np.arange(10)
        z = np.arange(10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((10, 10, 10))
        for i in range(10):
            Z[:, :, i][X % 2 == 0] = 1
        print(Z)
        plt.imshow(Z[:, :, 5])
        plt.show()


class TestFourierShift(unittest.TestCase):
    def test_first_shift_importance(self):
        N = 512
        L = 250 / 10 ** 6
        b = 5 / 10 ** 6
        dx = L / N
        df = 1 / dx / N
        x = np.array([n * dx - L / 2 for n in range(N)])
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / L, N)
        aperture = np.zeros(len(x))
        aperture[(x >= -b / 2) * (x <= b / 2)] = 1
        aperture_shifted = np.fft.fftshift(aperture)
        apertureFT = dx * np.fft.fft(aperture_shifted)
        print(apertureFT.real)
        plt.plot(fx, apertureFT.real)
        plt.show()
        apertureIFT = 1 / dx * np.fft.ifft(apertureFT)

class TestNumpyFeatures(unittest.TestCase):
    def test_meshgrid(self):
        x = np.linspace(1, 2, 2)
        y = np.copy(x)
        z = np.copy(x)
        unordered = np.array(np.meshgrid(x, y, z))
        combinations = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)
        sorted = combinations[np.lexsort((combinations[:, 2], combinations[:, 1], combinations[:, 0]))]
        vector = np.array((1, 0, -1))
        print(sorted)
        print(np.dot(sorted, vector))

    def test_array_indexing(self):
        a = np.array([[2, 3, 4], [5, 6, 7]])
        b = np.array([[2], [5]])
        print(a / b)
