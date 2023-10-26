import unittest
import numpy as np
import wrappers
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class TestFourierProperties(unittest.TestCase):
    def test_shift(self):
        x = np.arange(-10, 10, 0.1)
        f = np.arange(-100/20, 100/20, 1/20)
        fx = np.zeros(len(x))
        fx[abs(x) < 1] = 1
        fxk = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(fx)))

        fy = fx * np.exp(1j * 2 * np.pi * x)
        fyk = wrappers.wrapped_ifft(fy)
        # plt.plot(x)
        # plt.plot(y.real)
        plt.plot(f, fxk)
        plt.plot(f, fyk)
        plt.show()

class TestAnalyticResults(unittest.TestCase):
    def test_pattern(self):
        theta = np.pi / 4
        b = 1
        k = 2 ** 0.5
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        x = np.linspace(-5, 4.75, 40)
        y = np.linspace(-5, 4.75, 40)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(1j * k1 * X) + np.exp(-1j * k1 * X)
        print(Z)
        fig, ax = plt.subplots()
        maxValue = min(np.amax(Z), 100.0)
        print(maxValue)
        levels = np.linspace(0, maxValue + 1, 30)
        cf = ax.contourf(X, Y, Z, levels)
        plt.colorbar(cf)
        contour_axis = ax
        plt.show()

    def test_five_waves(self):
        theta = np.pi / 4
        b = 1
        k = 2 ** 0.5
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        x = np.linspace(-5, 4.75, 40)
        y = np.linspace(-5, 4.75, 40)
        z = np.linspace(-5, 4.75, 40)
        X, Y = np.meshgrid(x, y)
        I = np.zeros((40, 40, 40))
        for i, j, k in [(i, j, k) for i in x for j in y for k in z]:
            n1 = int((i + 5) * 4)
            n2 = int((j + 5) * 4)
            n3 = int((k + 5) * 4)
            I[n1, n2, n3] = 1 + 2 * b ** 2 - b ** 2 * (np.cos(2 * k1 * i) + np.cos(2 * k1 * j)) \
                            + 2 * b * (np.sin(k1 * i) * np.cos(k2 * k) + np.sin(k2 * k) * np.sin(k1 * j))
        fig, ax = plt.subplots()
        maxValue = min(np.amax(I), 100.0)
        print(maxValue)
        levels = np.linspace(0, maxValue + 1, 30)
        cf = ax.contourf(X, Y, I[:, :, 20], levels)
        plt.colorbar(cf)
        contour_axis = ax

        def update(val):
            contour_axis.clear()
            Z = I[:, :, int(val)]
            ax.clear()
            contour_axis.contourf(X, Y, Z, levels)

        slider_loc = plt.axes((0.2, 0.02, 0.65, 0.03))  # slider location and size
        slider = Slider(slider_loc, 'z', 0, 40 - 1)  # slider properties
        slider.on_changed(update)

        plt.show()
