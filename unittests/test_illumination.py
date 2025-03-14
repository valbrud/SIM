import unittest

from scipy.special import factorial

from config.IlluminationConfigurations import *
from Box import Box
from Sources import PlaneWave, IntensityHarmonic3D
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools
import sys
sys.path.append('../')
from Illumination import IlluminationNonLinearSIM2D

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

    def test_non_linear_polynomial_illumination(self):
        configurations = BFPConfiguration(refraction_index=1.5)
        alpha = 2 * np.pi / 5
        nmedium = 1.5
        nobject = 1.5
        NA = nmedium * np.sin(alpha)
        theta = np.asin(0.9 * np.sin(alpha))
        fz_max_diff = nmedium * (1 - np.cos(alpha))
        dx = 1 / (64 * NA)
        dy = dx
        dz = 1 / (4 * fz_max_diff)

        N = 255
        max_r = N // 2 * dx

        psf_size = 2 * np.array((2 * max_r, 2 * max_r))
        x = np.linspace(-max_r, max_r, N)
        y = np.copy(x)
        dimensions = (1, 1)
        illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        illumination_3waves2d = IlluminationPlaneWaves2D.init_from_3D(illumination_3waves3d, dimensions)

        p = 10
        nonlinear_expansion_coefficients = [0, ]
        n=1
        while(p**n / factorial(n)) > 10**-2:
            nonlinear_expansion_coefficients.append(p**n / factorial(n) * (-1)**(n+1))
            n += 1

        illumination_3waves_non_linear = IlluminationNonLinearSIM2D.init_from_linear_illumination(illumination_3waves2d, tuple(nonlinear_expansion_coefficients))
        for wave in illumination_3waves_non_linear.waves:
            print(wave, illumination_3waves_non_linear.waves[wave].wavevector, illumination_3waves_non_linear.waves[wave].amplitude)
        illumination_density = illumination_3waves2d.get_illumination_density(coordinates=(x, y))
        emission_density = illumination_3waves_non_linear.get_illumination_density(coordinates=(x, y))


        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_title("Linear", fontsize=25, pad=15)
        ax1.tick_params(labelsize=20)

        ax2.set_title("Non-linear", fontsize=25)
        ax2.tick_params(labelsize=20)

        im1 = ax1.imshow(illumination_density, vmin=0)
        im2 = ax2.imshow(emission_density, vmin=0)
        plt.show()

        plt.plot(emission_density[127, :])
        plt.show()
