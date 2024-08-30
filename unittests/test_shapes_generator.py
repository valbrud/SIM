import sys
from config.IlluminationConfigurations import *
import unittest
from ShapesGenerator import generate_random_spheres
import matplotlib.pyplot as plt
sys.path.append('../')
configurations = BFPConfiguration()

class TestSpheres(unittest.TestCase):
    def test_random_overlapping_spheres(self):
        max_r = 4
        max_z = 4
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        spheres = generate_random_spheres(psf_size, N, r =0.5,  N=100)
        plt.imshow(spheres[:, :, N//2])
        plt.show()
