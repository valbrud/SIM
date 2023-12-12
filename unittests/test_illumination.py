import unittest
from config.IlluminationConfigurations import *
class TestIllumination(unittest.TestCase):
    def test_index_waves(self):
        sources = [
            Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

            Sources.IntensityPlaneWave(b ** 2 / norm, 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(b ** 2 / norm, 0, np.array((-2 * k1, 0, 0))),

            Sources.IntensityPlaneWave(b / norm, 0, np.array((k1, 0, k2))),
            Sources.IntensityPlaneWave(b / norm, 0, np.array((-k1, 0, k2))),
            Sources.IntensityPlaneWave(b / norm, 0, np.array((k1, 0, -k2))),
            Sources.IntensityPlaneWave(b / norm, 0, np.array((-k1, 0, -k2))),
        ]
        three_waves_dict = Illumination.index_frequencies(sources, (k1, k1, k2))
        assert (three_waves_dict.keys() == three_waves_illumination.keys())

        sources = [
            Sources.IntensityPlaneWave(2 + 4 * b ** 2, 0, np.array((0, 0, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((-2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((0, 2 * k1, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((0, -2 * k1, 0))),
            Sources.IntensityPlaneWave(-1j * b, 0, np.array((k1, 0, k2))),
            Sources.IntensityPlaneWave(1j * b, 0, np.array((-k1, 0, k2))),
            Sources.IntensityPlaneWave(-1 * b, 0, np.array((0, k1, k2))),
            Sources.IntensityPlaneWave(1 * b, 0, np.array((0, -k1, k2))),
            Sources.IntensityPlaneWave(-1j * b, 0, np.array((k1, 0, -k2))),
            Sources.IntensityPlaneWave(1j * b, 0, np.array((-k1, 0, -k2))),
            Sources.IntensityPlaneWave(1 * b, 0, np.array((0, k1, -k2))),
            Sources.IntensityPlaneWave(-1 * b, 0, np.array((0, -k1, -k2))),
        ]
        five_waves_dict = Illumination.index_frequencies(sources, base_vector_lengths=(k1, k1, k2))
        assert (five_waves_dict.keys() == s_polarized_waves.keys())

    def test_seven_waves_illumination(self):
        print(seven_waves_illumination)