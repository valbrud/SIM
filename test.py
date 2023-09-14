import unittest
import numpy as np
from VectorOperations import VectorOperations
from PlaneWave import PlaneWave
from Box import Box, Voxel
import time

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

class TestCommonSense(unittest.TestCase):
    def test_dot_product(self):
        wavevector1 = np.array((0, 1, 1))
        wavevector2 = np.array((0, -1, 1))
        for x in np.linspace(-5, 5, 1000):
            for y in np.linspace(-5, 5, 1000):
                r = np.array((x, y, 0))
                difference = np.exp(1j * wavevector1.dot(r)) + np.exp(1j * wavevector2.dot(r))
                print(difference)

class TestBox(unittest.TestCase):
    def test_initialization(self):
        plane_wave1 = PlaneWave(0, 1, 0, 0, np.array((0, 1, 1)))
        plane_wave2 = PlaneWave(0, 1, 0, 0, np.array((0, -1, 1)))
        plane_wave3 = PlaneWave(0, 1, 0, 0, np.array((1, 0, 1)))
        plane_wave4 = PlaneWave(0, 1, 0, 0, np.array((-1, 0, 1)))
        plane_wave5 = PlaneWave(1, 1, 0, 0, np.array((0, 1, 0)))
        plane_waves = [plane_wave1, plane_wave2, plane_wave3, plane_wave4, plane_wave5]
        start = time.time()
        box = Box(plane_waves, 10, 40)
        box.compute_field()
        end = time.time()
        print(end - start)
        box.plot_intensity_slice()