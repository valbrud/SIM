import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import copy
import kernels
import numpy as np
import Reconstructor
import SSNRCalculator
import hpc_utils
import matplotlib.pyplot as plt
import Apodization
import background_estimation
import Illumination

def compute_loss_function():
    pass

def estimate_aberrations(stack, psf_stack, illumination, optical_system, zernieke):
    for i in range(len(psf_stack)):
        stack_one_rotation = stack[i:i+1, ...]
        harmonics_one_rotation = {key[1]: illumination.harmonics[key] for key in illumination.harmonics if key[0] == i}
        illumination_one_rotation = Illumination.IlluminationPlaneWaves2D(harmonics_one_rotation, dimensions=illumination.dimensions, angles=illumination.angles[i:i+1])
        illumination_one_rotation.set_spatial_shifts_diagonally()
        estimated_object_vector_one_rotation = np.real(background_estimation.estimate_background(stack_one_rotation, psf_stack, illumination_one_rotation, optical_system))
    