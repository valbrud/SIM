import unittest
from config.IlluminationConfigurations import *
from Box import Box
from Sources import PlaneWave, IntensityHarmonic3D
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools
import sys
sys.path.append('../')

class TestIllumination(unittest.TestCase):
    def test_index_waves(self):
        sources = [
            Sources.IntensityHarmonic3D(a0 / norm, 0, np.array((0, 0, 0))),

            Sources.IntensityHarmonic3D(b ** 2 / norm, 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityHarmonic3D(b ** 2 / norm, 0, np.array((-2 * k1, 0, 0))),

            Sources.IntensityHarmonic3D(b / norm, 0, np.array((k1, 0, k2))),
            Sources.IntensityHarmonic3D(b / norm, 0, np.array((-k1, 0, k2))),
            Sources.IntensityHarmonic3D(b / norm, 0, np.array((k1, 0, -k2))),
            Sources.IntensityHarmonic3D(b / norm, 0, np.array((-k1, 0, -k2))),
        ]
        three_waves_dict =  IlluminationPlaneWaves3D.index_frequencies(sources, (k1, k1, k2))
        assert (three_waves_dict.keys() == three_waves_illumination.keys())

        sources = [
            Sources.IntensityHarmonic3D(2 + 4 * b ** 2, 0, np.array((0, 0, 0))),
            Sources.IntensityHarmonic3D(-b ** 2, 0, np.array((-2 * k1, 0, 0))),
            Sources.IntensityHarmonic3D(-b ** 2, 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityHarmonic3D(-b ** 2, 0, np.array((0, 2 * k1, 0))),
            Sources.IntensityHarmonic3D(-b ** 2, 0, np.array((0, -2 * k1, 0))),
            Sources.IntensityHarmonic3D(-1j * b, 0, np.array((k1, 0, k2))),
            Sources.IntensityHarmonic3D(1j * b, 0, np.array((-k1, 0, k2))),
            Sources.IntensityHarmonic3D(-1 * b, 0, np.array((0, k1, k2))),
            Sources.IntensityHarmonic3D(1 * b, 0, np.array((0, -k1, k2))),
            Sources.IntensityHarmonic3D(-1j * b, 0, np.array((k1, 0, -k2))),
            Sources.IntensityHarmonic3D(1j * b, 0, np.array((-k1, 0, -k2))),
            Sources.IntensityHarmonic3D(1 * b, 0, np.array((0, k1, -k2))),
            Sources.IntensityHarmonic3D(-1 * b, 0, np.array((0, -k1, -k2))),
        ]
        five_waves_dict =  IlluminationPlaneWaves3D.index_frequencies(sources, base_vector_lengths=(k1, k1, k2))
        assert (five_waves_dict.keys() == s_polarized_waves.keys())

    def test_expanded_lattice(self):
        illumination = BFPConfiguration().get_6_oblique_s_waves_and_circular_normal(np.pi/4, 1)
        illumination.compute_expanded_lattice()

    def test_numerical_peaks_search(self):
        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)
        k2 = k * (np.cos(theta) - 1)
        vec_x = np.array((k/2 * np.sin(theta), 0, k * np.cos(theta)))
        vec_mx = np.array((-k * np.sin(theta), 0, k * np.cos(theta)))
        ax_z = np.array((0, 0, 1))

        sources = [

            PlaneWave(0, b, 0, 0, vec_x),
            PlaneWave(0, b, 0, 0, vec_mx),
            PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z,  2 * np.pi/3)),
            PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z,  2 * np.pi/3)),
            PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 4 * np.pi / 3)),
            PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 4 * np.pi / 3)),

            PlaneWave(1, 1j * 1, 0, 0, np.array((0, 0, k))),
        ]
        box = Box(sources, 15, 100)
        axes = (box.grid[:, 0, 0, 0], box.grid[0, :, 0, 1], box.grid[0, 0, :, 2])
        box.compute_electric_field()
        box.compute_intensity_from_electric_field()
        box.compute_intensity_fourier_space()
        fourier_peaks, amplitudes = stattools.estimate_localized_peaks(box.intensity_fourier_space, axes)
        print(len(fourier_peaks))
        intensity_sources_discrete = []
        for fourier_peak, amplitude in zip(fourier_peaks, amplitudes):
            intensity_sources_discrete.append(IntensityHarmonic3D(amplitude, 0, 2 * np.pi * np.array(fourier_peak)))
            print(fourier_peak*np.pi * 2)
        res = stattools.find_optimal_base_vectors(fourier_peaks)
        print(fourier_peaks[:10])
        print(res)
        box2 = Box(intensity_sources_discrete, (20, 20, 80), 100)
        box2.compute_intensity_from_spatial_waves()
        box2.compute_intensity_fourier_space()
        # plt.imshow(np.abs(box2.intensity_fourier_space[:, :, 20]))
        # plt.show()

