import unittest
import numpy as np
from Sources import PlaneWave, PointSource


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
