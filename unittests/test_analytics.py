import unittest
import numpy as np
import wrappers
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
sys.path.append('../')


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

    def test_distribtuion_properties(self):
        import mpmath as mp
        import numpy as np
        import matplotlib.pyplot as plt

        def ER_sigma(sigma_b):
            # For sigma_b=0, R = sqrt(A1^2+A2^2) is Rayleigh with E[R]=sqrt(pi/2)
            if sigma_b == 0:
                return mp.sqrt(mp.pi/2)
            # Define the pdfs for u and v:
            f_u = lambda u: (u)*mp.e**(-u**2/2)
            print(mp.quad(lambda u: u * f_u(u), [0, mp.inf]))
            f_v = lambda v: (v/sigma_b**2)*mp.e**(-v**2/(2*sigma_b**2))
            print(mp.quad(lambda v: v * f_v(v), [0, mp.inf]))
            # Compute E[R] = ∫_0∞∫_0∞ sqrt(u+v) f_u(u) f_v(v) du dv.
            inner_integral = lambda u: mp.quad(lambda v: mp.sqrt(u**2+v**2) * f_v(v), [0, mp.inf])
            return mp.quad(lambda u: f_u(u)*inner_integral(u), [0, mp.inf])

        def ratio(sigma_b):
            ER = ER_sigma(sigma_b)
            ER2 = 2 + 2*sigma_b**2  # since E[R^2] = 2+2*sigma_b^2
            return ER2/ER

        # Compute for sigma_b from 0 to 1 (since sigma_a=1, symmetry implies maximum when sigma_b=1)
        sigma_bs = np.linspace(0, 2, 21)
        ratios = []
        for sb in sigma_bs:
            r_val = ratio(sb)
            ratios.append(r_val)
            print(f"sigma_b = {sb:4.2f}, E[R^2]/E[R] = {r_val}")

        # Plot the ratio vs sigma_b
        plt.figure()
        plt.plot(sigma_bs, ratios, marker='o')
        plt.xlabel("sigma_b")
        plt.ylabel("E[R^2] / E[R]")
        plt.title("Ratio E[R^2]/E[R] vs sigma_b (with sigma_a = 1)")
        plt.grid(True)
        plt.show()

    def test_distribtuion_properties_monte_carlo(self):
        sigma_a = 1
        sigma_b = np.linspace(0, 10, 101)
        n_points = 1000000
        for sb in sigma_b:
            a1, a2 = np.random.normal(0, sigma_a, n_points), np.random.normal(0, sigma_a, n_points)
            b1, b2 = np.random.normal(0, sb, n_points), np.random.normal(0, sb, n_points)
            R = np.sqrt(a1**2 + a2**2 + b1**2 + b2**2)
            R2 = a1**2 + a2**2 + b1**2 + b2**2
            R_mean = np.mean(R)
            R2_mean = np.mean(R2)
            ratio = R2_mean**0.5 / R_mean
            print(f"Sigma_b = {round(sb, 1)}, Monte Carlo Ratio: {ratio}")