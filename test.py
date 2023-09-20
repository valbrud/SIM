import unittest
import numpy as np

import Sources
from VectorOperations import VectorOperations
from Sources import PlaneWave, PointSource
from Box import Box
import time
import matplotlib.pyplot as plt
class TestRotations(unittest.TestCase):

    def test_rotate_vector3d(self):
        xvector = np.array((5, 0, 0))
        yvector = np.array((0, 5, 0))
        zvector = np.array((0, 0, 5))

        vector = VectorOperations.rotate_vector3d(xvector, yvector, np.pi)
        self.assertEqual(vector[0], -5)

        vector = VectorOperations.rotate_vector3d(vector, zvector, -np.pi/2)
        self.assertEqual(vector[1], 5)

        vector = np.array((1, 1, 1))
        rot_vector = np.array((1, -1, 0))
        vector = VectorOperations.rotate_vector3d(vector, rot_vector, np.pi)
        self.assertEqual(vector[1], -1)

class TestPlaneWave(unittest.TestCase):
    def test_electric_field(self):
        wavevector = np.array((0, 1, 1))
        phase1 = 0
        phase2 = np.pi
        ex = 1
        ey = 1
        plane_wave = PlaneWave(ex, ey, phase1, phase2, wavevector)
        self.assertAlmostEqual(np.dot(plane_wave.field_vectors[0], plane_wave.field_vectors[1]), 0)


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


class TestBox(unittest.TestCase):
    def test_plane_waves_initialisation(self):
        plane_wave1 = PlaneWave(0, 1, 0, 0, np.array((1, 0, 1)))
        plane_wave2 = PlaneWave(0, 1, 0, 0, np.array((-1, 0, 1)))
        plane_wave3 = PlaneWave(0, 1, 0, 0, np.array((0, 1, 1)))
        plane_wave4 = PlaneWave(0, 1, 0, 0, np.array((0, -1, 1)))
        plane_wave5 = PlaneWave(1, 1, 0, np.pi / 2, np.array((0, 0, 1)))
        plane_waves = [plane_wave1, plane_wave2, plane_wave3, plane_wave4]
        sources = {"PlaneWaves": plane_waves}
        start = time.time()
        box = Box(sources, 10, 40)
        box.compute_field()
        box.plot_intensity_slices()
        box.compute_intensity_fourier_space()
        end = time.time()
        print(end - start)
        box.plot_intensity_fourier_space_slices()

class TestBox(unittest.TestCase):
    def test_plane_waves_initialisation(self):
        plane_wave1 = PlaneWave(0, 1, 0, 0, np.array((1, 0, 1)))
        plane_wave2 = PlaneWave(0, 1, 0, 0, np.array((-1, 0, 1)))
        plane_wave3 = PlaneWave(0, 1, 0, 0, np.array((0, 1, 1)))
        plane_wave4 = PlaneWave(0, 1, 0, 0, np.array((0, -1, 1)))
        plane_wave5 = PlaneWave(1, 1, 0, np.pi / 2, np.array((0, 0, 1)))
        plane_waves = [plane_wave1, plane_wave2, plane_wave3, plane_wave4]
        sources = {"PlaneWaves": plane_waves}
        start = time.time()
        box = Box(sources, 10, 40)
        box.compute_field()
        box.plot_intensity_slices()
        box.compute_intensity_fourier_space()
        end = time.time()
        print(end - start)
        box.plot_intensity_fourier_space_slices()

    def test_point_sources_initialization(self):
        source1 = PointSource((0, 0, 0), 10)
        source2 = PointSource((0, 2, 0), 5)
        source3 = PointSource((0, -2, 0), 5)
        sources = [source1, source2, source3]
        sources = {"PointSources": sources}
        box = Box(sources, 10, 40)
        box.compute_field()
        box.plot_intensity_slices()
        box.compute_intensity_fourier_space()
        box.plot_intensity_fourier_space_slices()

    def test_spacial_waves_initialization(self):
        theta = np.pi / 4
        b = 1 / 2**0.5
        k = 2 ** 0.5
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        waves = {'SpacialWave' :[Sources.SpacialWave(np.array((0,0,0)), 2 + 4 * b ** 2),
                 Sources.SpacialWave(np.array((-2 * k1, 0, 0)),  -b**2),
                 Sources.SpacialWave(np.array((2 * k1, 0, 0)), -b**2),
                 Sources.SpacialWave(np.array((0, 2 * k1, 0)), -b ** 2),
                 Sources.SpacialWave(np.array((0, -2 * k1, 0)), -b ** 2),
                 Sources.SpacialWave(np.array((0, 0, k2)), 2 - 2j),
                 Sources.SpacialWave(np.array((0, 0, -k2)), 2 + 2j)]}
        box = Box(waves, 10, 40)
        box.set_intensity_from_spacial_waves()
        box.plot_intensity_slices()