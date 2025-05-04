import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)


import unittest

from scipy.special import factorial

from config.BFPConfigurations import *
from Box import Box
from Sources import PlaneWave, IntensityHarmonic3D, IntensityHarmonic2D
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools
import sys

from Illumination_experimental import *
from config.SIM_N100_NA15 import *

class TestInitialization(unittest.TestCase):
    def setUp(self):
        self.Mr = 3
        self.Mt = 1
        self.dimensions2D = (1, 1)
        self.dimensions3D = (1, 1, 1)
        self.intensity_dict_2D = {
            (0, 0): IntensityHarmonic2D(wavevector=np.array((0, 0)), amplitude=1),
        }
        self.intensity_dict_3D = {
            (0, 0, 0): IntensityHarmonic3D(wavevector=np.array((0, 0, 0)), amplitude=1),
        }
        self.spatial_shifts_2D = np.array(((0., 0.),))
        self.spatial_shifts_3D = np.array(((0., 0., 0.), ))

    def test_cannot_instantiate_PlaneWavesSIM(self):
        # PlaneWavesSIM is abstract so instantiation should raise an error.
        with self.assertRaises(TypeError):
            _ = PlaneWavesSIM({}, self.dimensions2D, self.Mr, self.spatial_shifts_2D)

    def test_instantiate_IlluminationPlaneWaves2D(self):
        # Should instantiate without error.
        illum2D = IlluminationPlaneWaves2D(
            self.intensity_dict_2D,
            dimensions=self.dimensions2D,
            Mr=self.Mr,
            spatial_shifts=self.spatial_shifts_2D
        )
        self.assertIsNotNone(illum2D)
        # Check some property (for example, that angles attribute exists)
        self.assertTrue(hasattr(illum2D, 'angles'))

    def test_instantiate_IlluminationPlaneWaves3D(self):
        # Should instantiate without error.
        illum3D = IlluminationPlaneWaves3D(
            self.intensity_dict_3D,
            dimensions=self.dimensions3D,
            Mr=self.Mr,
            spatial_shifts=self.spatial_shifts_3D
        )

class TestIllumination(unittest.TestCase):
    def test_index_waves(self):
        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)
        k2 = k * (1 - np.cos(theta))
        a0 = 1 + 2 * b**2

        sources = [
            Sources.IntensityHarmonic3D(a0, 0, np.array((0, 0, 0))),

            Sources.IntensityHarmonic3D(b ** 2, 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityHarmonic3D(b ** 2, 0, np.array((-2 * k1, 0, 0))),

            Sources.IntensityHarmonic3D(b, 0, np.array((k1, 0, k2))),
            Sources.IntensityHarmonic3D(b, 0, np.array((-k1, 0, k2))),
            Sources.IntensityHarmonic3D(b, 0, np.array((k1, 0, -k2))),
            Sources.IntensityHarmonic3D(b, 0, np.array((-k1, 0, -k2))),
        ]
        three_waves_dict =  IlluminationPlaneWaves3D.index_frequencies(sources, (k1, k1, k2))
        test_illumination = BFPConfiguration().get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 1, Mt=1)
        indexed_three = set(three_waves_dict.keys())
        test_indices = set([key[1] for key in test_illumination.harmonics.keys() if key[0] == 0])
        assert (indexed_three == test_indices)

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
        test_illumination = BFPConfiguration().get_4_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1)
        indexed_five = set(five_waves_dict.keys())
        test_indices = set([key[1] for key in test_illumination.harmonics.keys() if key[0] == 0])
        assert (indexed_five == test_indices)

    def test_expanded_lattice(self):
        illumination = BFPConfiguration().get_6_oblique_s_waves_and_circular_normal(np.pi/4, 1)
        expanded_lattice = illumination.compute_expanded_lattice()
        print('number of peaks in expanded lattice is', len(expanded_lattice))
        assert(len(expanded_lattice) == 61)

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

        p = 2
        nonlinear_expansion_coefficients = [0, ]
        n=1
        while(p**n / factorial(n)) > 10**-7:
            nonlinear_expansion_coefficients.append(p**n / factorial(n) * (-1)**(n+1))
            n += 1

        illumination_3waves_non_linear = IlluminationNonLinearSIM2D.init_from_linear_illumination(illumination_3waves2d, tuple(nonlinear_expansion_coefficients))
        for harmonic in illumination_3waves_non_linear.harmonics:
            print(harmonic, illumination_3waves_non_linear.harmonics[harmonic].wavevector, illumination_3waves_non_linear.harmonics[harmonic].amplitude)
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

        plt.plot(emission_density[:, 127])
        plt.show()
