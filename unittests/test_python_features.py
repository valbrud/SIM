import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import time


class TestGridLayoutPerformance(unittest.TestCase):
    def measure_einsum_time(self, arr, vec, subscripts, repeats=10):
        """
        Repeatedly performs the einsum operation and measures the time.
        """
        start = time.perf_counter()
        for _ in range(repeats):
            # We don't store the result because we just care about timing.
            _ = np.einsum(subscripts, arr, vec)
        end = time.perf_counter()
        return end - start

    def test_einsum_layout_performance(self):
        """
        Compares performance of einsum on two different memory layouts:
        (N, N, N, 3) vs. (3, N, N, N).
        Both will sum over the coordinate axis to produce (N, N, N).
        """
        N = 512

        # Create random arrays for shape (N, N, N, 3)
        arr_last = np.random.rand(N, N, 1, 3)
        vec_last = np.random.rand(N, N, 1, 3)

        # Move the last axis (3) to the front -> shape (3, N, N, N)
        arr_first = np.moveaxis(arr_last, -1, 0)
        vec_first = np.moveaxis(vec_last, -1, 0)

        # Warm-up calls (avoids timing the one-time overhead)
        # Shape (N, N, N, 3), sum over axis 'l' -> i j k l
        # -> result shape (i j k)
        _ = np.einsum('ijkl, ijkl -> ijk', arr_last, vec_last)
        # Shape (3, N, N, N), sum over axis 'l' -> l i j k
        # -> result shape (i j k)
        _ = np.einsum('lijk, lijk -> ijk', arr_first, vec_first)

        # Actually measure times
        repeats = 1000
        time_last = self.measure_einsum_time(arr_last, vec_last, 'ijkl, ijkl -> ijk', repeats)
        time_first = self.measure_einsum_time(arr_first, vec_first, 'lijk, lijk -> ijk', repeats)

        print(f"\nEinsum performance with shape (N, N, N, 3): {time_last:.6f} s")
        print(f"Einsum performance with shape (3, N, N, N): {time_first:.6f} s")

        # (Optionally) Assert or compare which one you expect to be faster
        # self.assertLess(time_last, time_first,
        #                 "Expected layout (N, N, N, 3) to be faster in this scenario.")


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
