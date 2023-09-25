import unittest
import numpy as np

import Sources
from VectorOperations import VectorOperations
from Sources import PlaneWave, PointSource
from Box import Box
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
# import parser

class TestRotations(unittest.TestCase):

    def test_rotate_vector3d(self):
        xvector = np.array((5, 0, 0))
        yvector = np.array((0, 5, 0))
        zvector = np.array((0, 0, 5))

        vector = VectorOperations.rotate_vector3d(xvector, yvector, np.pi)
        self.assertEqual(vector[0], -5)

        vector = VectorOperations.rotate_vector3d(vector, zvector, -np.pi / 2)
        self.assertEqual(vector[1], 5)

        vector = np.array((1, 1, 1))
        rot_vector = np.array((1, -1, 0))
        vector = VectorOperations.rotate_vector3d(vector, rot_vector, np.pi)
        self.assertEqual(vector[1], -1)


class TestPlaneWave(unittest.TestCase):
    def test_electric_field_polarizations_orthogonality(self):
        wavevector = np.array((0, 1, 1))
        phase1 = 0
        phase2 = np.pi
        ex = 1
        ey = 1
        plane_wave = PlaneWave(ex, ey, phase1, phase2, wavevector)
        self.assertAlmostEqual(np.dot(plane_wave.field_vectors[0], plane_wave.field_vectors[1]), 0)

    def test_electric_field(self):
        phase1 = 0
        ex = 1
        ey = 1
        wavevector = np.array((0, 0, 1))
        for phase2 in [0, np.pi / 2, np.pi]:
            plane_wave = PlaneWave(ex, ey, phase1, phase2, wavevector)
            E = plane_wave.get_electric_field(np.array((1, 1, 1)))
            print(np.vdot(E, E))


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

class TestSources(unittest.TestCase):
    ...
class TestBox(unittest.TestCase):
    def test_point_sources_initialization(self):
        source1 = PointSource((0, 0, 0), 10)
        source2 = PointSource((0, 2, 0), 5)
        source3 = PointSource((0, -2, 0), 5)
        sources = [source1, source2, source3]
        sources = {"PointSources": sources}
        box = Box(sources, 10, 40)
        box.compute_electric_field()
        box.compute_intensity_from_electric_field()
        box.plot_intensity_slices()
        # box.compute_intensity_fourier_space()
        # box.plot_intensity_fourier_space_slices()

    def test_plane_waves_initialisation(self):
        plane_wave1 = PlaneWave(0, 1, 0, 0, np.array((1, 0, 1)))
        plane_wave2 = PlaneWave(0, 1, 0, 0, np.array((-1, 0, 1)))
        plane_wave3 = PlaneWave(0, 1, 0, 0, np.array((0, 1, 1)))
        plane_wave4 = PlaneWave(0, 1, 0, 0, np.array((0, -1, 1)))
        plane_wave5 = PlaneWave(1, 1j, 0, 0, np.array((0, 0, 2 ** 0.5)))
        plane_waves = [plane_wave5, plane_wave1, plane_wave2, plane_wave3, plane_wave4]
        start = time.time()
        box = Box(plane_waves, 10, 40)
        box.compute_electric_field()
        box.compute_intensity_from_electric_field()
        box.plot_intensity_slices()
        box.compute_intensity_fourier_space()
        end = time.time()
        print(end - start)
        # box.plot_intensity_fourier_space_slices()

    def test_iplane_waves_initialization(self):
        theta = np.pi / 4
        b = 1
        k = 2 ** 0.5
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        waves = [
                 # Sources.IntensityPlaneWave(1 + 2 * b ** 2, 0, np.array((0, 0, 0))),
                 # Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((-2 * k1, 0, 0))),
                 # Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((2 * k1, 0, 0))),
                 # Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((0, 2 * k1, 0))),
                 # Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((0, -2 * k1, 0))),
                 Sources.IntensityPlaneWave(-1j * b / 2, 0, np.array((k1, 0, k2))),
                 Sources.IntensityPlaneWave(1j * b / 2, 0, np.array((-k1, 0, k2))),
                 # Sources.IntensityPlaneWave(-1 * b / 2, 0, np.array((0, k1, k2))),
                 # Sources.IntensityPlaneWave(1 * b / 2, 0, np.array((0, -k1, k2))),
                 Sources.IntensityPlaneWave(-1j * b / 2, 0, np.array((k1, 0, -k2))),
                 Sources.IntensityPlaneWave(1j * b / 2, 0, np.array((-k1, 0, -k2))),
                 # Sources.IntensityPlaneWave(1 * b / 2, 0, np.array((0, k1, -k2))),
                 # Sources.IntensityPlaneWave(-1 * b / 2, 0, np.array((0, -k1, -k2)))
                ]
        box = Box(waves, 10, 40)
        box.compute_intensity_from_spacial_waves()
        box.plot_intensity_slices()

    # Change order of parameters in Waves
    # def test_icos_isin_wave_initialization(self):
    #     theta = np.pi / 4
    #     b = 1
    #     k = 2 ** 0.5
    #     k1 = k * np.sin(theta)
    #     k2 = k * (np.cos(theta) - 1)
    #     waves = [Sources.IntensityCosineWave(np.array((0, 0, 0)), 2 + 4 * b ** 2),
    #              Sources.IntensityCosineWave(np.array((2 * k1, 0, 0)), -2 * b ** 2),
    #              Sources.IntensityCosineWave(np.array((0, 2 * k1, 0)), -2 * b ** 2),
    #              Sources.IntensitySineWave(np.array((0, 0, k2)), 4 * 2 ** 0.5, np.pi/4)]
    #     box = Box(waves, 10, 40)
    #     box.compute_intensity_from_spacial_waves()
    #     box.plot_intensity_slices()

    def test_adding_sources(self):
        box = Box({}, box_size=10, point_number=40)
        source1 = PointSource((0, 0, 0), 10)
        box.add_source(source1)
        box.compute_intensity_from_electric_field()
        box.plot_intensity_slices()


class TestParser(unittest.TestCase):


    def test_configurations_import(self):
        spec = spec_from_loader("example_config.conf",
                                SourceFileLoader("example_config.conf", "./config/example_config.conf"))
        conf = module_from_spec(spec)
        spec.loader.exec_module(conf)
        print(len(conf.sources))


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

        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_loc, 'z', 0, 40 - 1)  # slider properties
        slider.on_changed(update)

        plt.show()
