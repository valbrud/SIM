import numpy as np
import matplotlib.pyplot as plt
import stattools
import unittest

class TestRingAveraging(unittest.TestCase):
    def test_averaging_over_uniform(self):
        array = np.ones((100, 100))
        averages = stattools.average_rings2d(array)
        assert np.allclose(averages, np.ones(50))

    def test_different_axes(self):
        x = np.arange(100)
        y = np.arange(0, 100, 2)
        array = np.ones((x.size, y.size))
        averages = stattools.average_rings2d(array, (x, y))
        assert np.allclose(averages, np.ones(50))

    def test_averaging_over_sine(self):
        x = np.arange(1000)
        y = np.arange(0, 1000, 2)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X**2 + Y**2)**0.5/100)
        averages = stattools.average_rings2d(sine_array.T, (x, y))
        plt.plot(averages)
        plt.plot(np.sin(y / 100))
        plt.show()


class TestRingExpansion(unittest.TestCase):
    def test_uniform_expansion(self):
        array = np.ones((100, 100))
        averages = stattools.average_rings2d(array)
        assert np.allclose(averages, np.ones(50))
        expanded = stattools.expand_ring_averages2d(averages)
        plt.imshow(expanded)
        plt.show()

    def test_sine_expansion(self):
        x = np.arange(1000)
        y = np.arange(0, 1000, 2)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X ** 2 + Y ** 2) ** 0.5 / 100)
        # plt.imshow(sine_array)
        # plt.show()
        averages = stattools.average_rings2d(sine_array.T, (x, y))
        plt.plot(averages)
        plt.plot(np.sin(y / 100))
        # plt.show()
        expanded = stattools.expand_ring_averages2d(averages, (x, y))
        plt.imshow(expanded)
        plt.show()

    def test_inhomogeneous_expansion(self):
        x = np.arange(-100, 100)
        y = np.arange(-100, 100)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X ** 2 + Y ** 2) ** 0.5 / 10)
        phi = np.arctan(Y/X)
        sine_array *= np.abs(np.sin(phi))
        plt.imshow(sine_array)
        plt.show()
        averages = stattools.average_rings2d(sine_array, (x, y))
        plt.plot(averages)
        # plt.plot(np.sin(y / 100))
        plt.show()
        expanded = stattools.expand_ring_averages2d(averages, (x, y))
        plt.plot(x, expanded[:, 100])
        # plt.imshow(expanded)
        plt.show()

class TestSurfaceLevels(unittest.TestCase):
    def test_split_spherically_symmetric(self):
        x = np.linspace(-100, 100, 201)
        y = np.copy(x)
        X, Y = np.meshgrid(x, y)
        R = (X**2 + Y**2)**0.5
        f = 1 / (1 + R)
        mask = stattools.find_decreasing_surface_levels2d(f)
        plt.imshow(mask)
        plt.show()

    def test_split_flower(self):
        x = np.linspace(-100, 100, 201)
        y = np.copy(x)
        X, Y = np.meshgrid(x, y)
        R1 = (X ** 2 + Y ** 2 / 10) ** 0.5
        R2 = (X**2 / 10 + Y**2)**0.5
        f = 1 / (1 + R1/10 + R2/10)
        # plt.imshow(f)
        # plt.show()
        mask = stattools.find_decreasing_surface_levels2d(f)
        plt.imshow(mask)
        plt.show()

    def test_split_3d(self):
        x = np.linspace(-100, 100, 201)
        y = np.copy(x)
        z = np.copy(y)
        X, Y, Z = np.meshgrid(x, y, z)
        R = (X ** 2 + Y ** 2 + Z**2) ** 0.5
        f = 1 / (1 + R)
        mask = stattools.find_decreasing_surface_levels3d(f, direction = 0)
        plt.imshow(mask[:, :, 100])
        plt.show()

