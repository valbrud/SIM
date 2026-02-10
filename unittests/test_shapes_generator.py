import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from config.BFPConfiguration import *
import unittest
from ShapesGenerator import generate_random_spherical_particles, generate_sphere_slices
import matplotlib.pyplot as plt
configurations = BFPConfiguration()
import hpc_utils

class TestSpheres(unittest.TestCase):
    def test_random_overlapping_spheres(self):
        max_r = 4
        max_z = 4
        n = 101
        N = 100
        psf_size = 2 * np.array((max_r, max_r))
        np.random.seed(42)
        spheres_1 = generate_random_spherical_particles(psf_size, n, radius=0.5,  num_particles=N)
        np.random.seed(42)
        spheres_2 = generate_random_spherical_particles(psf_size, n, radius=1,  num_particles= N, intensity = 25000)

        spheres_1_ft = hpc_utils.wrapped_fftn(spheres_1)
        spheres_2_ft = hpc_utils.wrapped_fftn(spheres_2)
        plt.plot(np.log1p(np.abs(spheres_1_ft)[:,  n//2]))
        plt.plot(np.log1p(np.abs(spheres_2_ft)[:,  n//2]))
        plt.show()

        plt.imshow(np.log1p(np.abs(spheres_1_ft)[:, :]))
        plt.show()

        plt.imshow(np.log1p(np.abs(spheres_2_ft)[:, :]))
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

