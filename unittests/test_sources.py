import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import unittest
import matplotlib.pyplot as plt
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

    def test_get_square_pattern(self):
        theta = np.pi / 4
        b = 1
        k = 2 * np.pi
        k1 = k * np.sin(theta)

        a0 = 2 + 3 * b**2
        sources = [
            PlaneWave(0, b/a0**0.5, 0, 0, np.array((k1, 0, k1))),
            PlaneWave(0, -b/a0**0.5, 0, 0, np.array((-k1, 0, k1))),
            PlaneWave(0, b/a0**0.5, 0, 0, np.array((0, k1, k1))),
            PlaneWave(0, -b/a0**0.5, 0, 0, np.array((0, -k1, k1))),
            PlaneWave(1/a0**0.5, 1/a0**0.5, 0, 0, np.array((0, 0, k))),
        ]


        x = np.linspace(-5, 5, 100)
        y = np.copy(x)
        z = np.copy(x)

        grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        electric_field = np.zeros(grid.shape, dtype=np.complex128)

        for source in sources:
            electric_field += source.get_electric_field(grid)

        intensity = np.einsum('ijkl,ijkl->ijk', electric_field, np.conj(electric_field))
        plt.imshow(intensity[..., 50].real, vmin=0)
        plt.show()

        plt.imshow(intensity[:, 50, :].real, vmin=0)
        plt.show()
