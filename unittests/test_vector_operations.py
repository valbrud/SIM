import unittest
import numpy as np
from VectorOperations import VectorOperations
import sys
sys.path.append('../')

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
