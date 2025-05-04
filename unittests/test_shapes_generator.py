import sys
from config.BFPConfigurations import *
import unittest
from ShapesGenerator import generate_random_spherical_particles, generate_sphere_slices
import matplotlib.pyplot as plt
sys.path.append('../')
configurations = BFPConfiguration()

class TestSpheres(unittest.TestCase):
    def test_random_overlapping_spheres(self):
        max_r = 4
        max_z = 4
        N = 100
        psf_size = 2 * np.array((max_r, max_r, max_z))
        spheres = generate_random_spherical_particles(psf_size, N, r =0.5,  N=100)
        plt.imshow(spheres[:, :, N//2])
        plt.show()

    def test_slices(self):
        theta = np.pi / 4
        alpha = np.pi / 4
        dx = 1 / (8 * np.sin(alpha))
        dy = dx
        dz = 1 / (4 * (1 - np.cos(alpha)))
        N = 51
        max_r = N // 2 * dx
        max_z = N // 2 * dz
        psf_size = 2 * np.array((max_r, max_r, max_z))
        spheres = generate_sphere_slices(psf_size, N, r =0.5,  N=100)
        plt.imshow(spheres[:, :, N//2])
        plt.show()

