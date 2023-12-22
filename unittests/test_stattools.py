import numpy as np
import matplotlib.pyplot as plt
import stattools
import unittest

class TestRingAveraging(unittest.TestCase):
    def test_averaging_over_uniform(self):
        array = np.ones((100, 100))
        averages = stattools.average_ring(array)
        assert np.allclose(averages, np.ones(100))

    def test_different_axes(self):
        x = np.arange(100)
        y = np.arange(0, 100, 2)
        array = np.ones((x.size, y.size))
        averages = stattools.average_ring(array, (x, y))
        assert np.allclose(averages, np.ones(50))

    def test_averaging_over_sine(self):
        x = np.arange(1000)
        y = np.arange(0, 1000, 2)
        X, Y = np.meshgrid(x, y)
        sine_array = np.sin((X**2 + Y**2)**0.5/100)
        averages = stattools.average_ring(sine_array, (x, y))
        plt.plot(averages)
        plt.plot(np.sin(y / 100))
        plt.show()

class TestGridFitting(unittest.TestCase):
    y = np.array((1.01, 2.04, 1.98, 3.9, 4.1, 5.05, 6, 7.03, 6.9))
    base_vector, indices = stattools.find_optimal_base_vectors(y)
    print(base_vector)