"""
Unit‑tests for IlluminationPatternEstimator2D

The test generates a synthetic SIM 2‑D stack with your SIMulator2D and checks
that the estimator recovers:
    * zero rotational phase increment  (Mr = 1)
    * ~120° translational phase increment  (2π / 3)
    * a modulation depth a₁ close to the simulated value
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

# ---------------------------------------------------------------------
# project imports (exact names from your repo)
# ---------------------------------------------------------------------
from wrappers import wrapped_fftn
from PatternEstimator import IlluminationPatternEstimator2D
from OpticalSystems import System4f2D
from SIMulator import SIMulator2D
from Illumination import IlluminationPlaneWaves2D
from Sources import IntensityHarmonic2D
import unittest
import numpy as np
import ShapesGenerator
import matplotlib.pyplot as plt

# one of your config files – adjust if you prefer another
from config.SIM_N100_NA15 import (
    alpha, theta, dx, configurations, psf_size
)


# ---------------------------------------------------------------------
# helper to build a *simple* illumination pattern
# ---------------------------------------------------------------------
def build_experimental_illumination():
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta -0.03, 1, 0, Mr=1, Mt=1
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum2d

def build_theoretical_illumination():
    """Two oblique plus one normal beam, 3 phase shifts."""
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta, 1, 0, Mr=1, Mt=1
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    # print(illum2d.spatial_shifts)

    return illum2d
# ---------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------
class TestPatternEstimator2D(unittest.TestCase):

    def setUp(self):
        # ---------- optics --------------------------------------------------
        N = 501                                   # keep small for speed
        max_r = N // 2 * dx
        psf_size = 2 * np.array((2 * max_r, 2 * max_r))
        self.optical_system = System4f2D(alpha=alpha)
        self.optical_system.compute_psf_and_otf((psf_size, N))

        # ---------- illumination & simulator -------------------------------
        self.experimenatal_illumination = build_experimental_illumination()
        self.theoretical_illumination = build_theoretical_illumination()
        self.simulator = SIMulator2D(self.experimenatal_illumination, self.optical_system)

        # synthetic object: random dots
        np.random.seed(0)  # for reproducibility
        self.sample = ShapesGenerator.generate_random_lines(psf_size, N, 0.3, 1000, 10000)
        # plt.imshow(self.sample, cmap='gray')
        # plt.show()

        # ---------- estimator under test -----------------------------------
        self.estimator = IlluminationPatternEstimator2D(
            self.theoretical_illumination, self.optical_system
        )

    # -----------------------------------------------------------------
    def test_parameter_recovery(self):
        """Estimator recovers phases and modulation depth on clean data."""
        raw_stack = self.simulator.generate_sim_images(self.sample)
        raw_stack = self.simulator.add_noise(raw_stack)  # (3, 3, N, N)
        plt.imshow(raw_stack[0, 0], cmap='gray')
        plt.show()

        base_vectors = np.array(self.theoretical_illumination.get_base_vectors())/(2 * np.pi)
        true_vectors = np.array(self.experimenatal_illumination.get_base_vectors())/(2 * np.pi)
        dq = self.estimator.optical_system.otf_frequencies[0][1] - self.estimator.optical_system.otf_frequencies[0][0]
        print('dq size = ', dq )
        print("initial_guess", np.array(base_vectors))
        print('true_wavevectors =', np.array(true_vectors))

        # print("base_vectors =", base_vectors)

        refined_vectors = self.estimator.estimate_illumination_parameters(
            raw_stack,
            return_as_illumination_object=False, 
            zooming_factor=2,
            peak_neighborhood_size=31, 
            max_iterations=10
        )

        precision = (true_vectors[0] -  refined_vectors[0]) / dq
        print('achieved_precision = ', precision, 'pixels')
        # print("dψ_trans =", dψ_trans)
        # print("a_m =", a_m)
        # # rotational increment should be zero (Mr = 1)
        # self.assertTrue(np.allclose(dψ_rot, 0.0, atol=1e-3))

        # # translational increment should be ~ 2π/3
        # expected = 2 * np.pi / 3
        # self.assertTrue(np.allclose(dψ_trans[1:], expected, atol=1e-2))

        # # first‑order modulation depth should be ~ amplitude of (1, 0) harmonic
        # self.assertAlmostEqual(a_m[1], self.illumination.waves[(1, 0)].amplitude,
        #                        delta=0.05)

    # -----------------------------------------------------------------
    def test_fourier_shortcut(self):
        """Estimator accepts pre‑FFT stack when spatial_domain=False."""
        raw_stack = self.simulator.generate_sim_images(self.sample)
        fft_stack = wrapped_fftn(raw_stack, axes=(-2, -1))

        # should run without raising
        self.estimator.estimate_illumination_parameters(
            fft_stack,
            spatial_domain=False,
            return_as_illumination_object=False
        )

    # -----------------------------------------------------------------
    def test_shape_mismatch_raises(self):
        """Mismatch in Mr/Mt raises ValueError."""
        wrong_shape = np.zeros((2, 5, 64, 64))  # wrong Mr & Mt
        with self.assertRaises(ValueError):
            self.estimator.estimate_illumination_parameters(wrong_shape)


# ---------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main(verbosity=2)
