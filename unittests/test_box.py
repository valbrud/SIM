import unittest
import numpy as np
import Sources
from Sources import PlaneWave, PointSource
from Box import Box
import time
import sys
sys.path.append('../')


class TestBox(unittest.TestCase):
    def test_point_sources_initialization(self):
        source1 = PointSource((0, 0, 0), 10)
        source2 = PointSource((0, 2, 0), 5)
        source3 = PointSource((0, -2, 0), 5)
        sources = [source1, source2, source3]
        box = Box(sources, 10, 40)
        box.compute_electric_field()
        box.compute_intensity_from_electric_field()
        box.plot_intensity_slices()
        box.compute_intensity_fourier_space()
        box.plot_intensity_fourier_space_slices()

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
        box.plot_intensity_fourier_space_slices()

    def test_plane_waves_initialisation2d(self):
        plane_wave1 = PlaneWave(0, 1, 0, 0, np.array((1, 0, 1)))
        plane_wave2 = PlaneWave(0, 1, 0, 0, np.array((-1, 0, 1)))
        plane_wave3 = PlaneWave(0, 1, 0, 0, np.array((0, 1, 1)))
        plane_wave4 = PlaneWave(0, 1, 0, 0, np.array((0, -1, 1)))
        # plane_wave5 = PlaneWave(1, 1j, 0, 0, np.array((0, 0, 2 ** 0.5)))
        plane_waves = [ plane_wave1, plane_wave2, plane_wave3, plane_wave4]
        start = time.time()
        box = Box(plane_waves, (10, 10), 40)
        box.compute_electric_field()
        box.compute_intensity_from_electric_field()
        box.plot_intensity_slices()
        box.compute_intensity_fourier_space()
        end = time.time()
        print(end - start)
        box.plot_intensity_fourier_space_slices()

    def test_iplane_waves_initialization(self):
        theta = np.pi / 4
        b = 1
        k = 2 ** 0.5
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        waves = [
                 Sources.IntensityPlaneWave(1 + 2 * b ** 2, 0, np.array((0, 0, 0))),
                 Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((-2 * k1, 0, 0))),
                 Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((2 * k1, 0, 0))),
                 Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((0, 2 * k1, 0))),
                 Sources.IntensityPlaneWave(-b ** 2 / 2, 0, np.array((0, -2 * k1, 0))),
                 Sources.IntensityPlaneWave(-1j * b / 2, 0, np.array((k1, 0, k2))),
                 Sources.IntensityPlaneWave(1j * b / 2, 0, np.array((-k1, 0, k2))),
                 Sources.IntensityPlaneWave(-1 * b / 2, 0, np.array((0, k1, k2))),
                 Sources.IntensityPlaneWave(1 * b / 2, 0, np.array((0, -k1, k2))),
                 Sources.IntensityPlaneWave(-1j * b / 2, 0, np.array((k1, 0, -k2))),
                 Sources.IntensityPlaneWave(1j * b / 2, 0, np.array((-k1, 0, -k2))),
                 Sources.IntensityPlaneWave(1 * b / 2, 0, np.array((0, k1, -k2))),
                 Sources.IntensityPlaneWave(-1 * b / 2, 0, np.array((0, -k1, -k2)))
                ]
        box = Box(waves, 10, 40)
        box.compute_intensity_from_spatial_waves()
        box.plot_intensity_slices()

    def test_adding_sources(self):
        box = Box({}, box_size=10, point_number=40)
        source1 = PointSource((0, 0, 0), 10)
        box.add_source(source1)
        box.compute_intensity_from_electric_field()
        box.plot_intensity_slices()


