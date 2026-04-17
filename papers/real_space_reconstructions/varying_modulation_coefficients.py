import os.path
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from config.BFPConfigurations import *
from matplotlib.widgets import Slider
import SSNRCalculator
from OpticalSystems import System4f3D, System4f2D
import kernels
import utils, hpc_utils
import pickle
import csv 

configurations = BFPConfiguration(refraction_index=1.5)


if __name__ == "__main__":
    alpha = 2 * np.pi / 5
    theta = np.arcsin(0.9 * np.sin(alpha))
    nmedium = 1.5
    nsample = 1.5

    dx = 1 / (8 * nmedium * np.sin(alpha))
    dy = dx
    dz = 1 / (4 * nmedium * (1 - np.cos(alpha)))
    Nl = 101
    Nz = 51
    max_r = Nl//2 * dx
    max_z = Nz//2 * dz


    NA = nmedium * np.sin(alpha)
    psf_size = 2 * np.array((max_r, max_r, max_z))
    dV = dx * dy * dz
    x = np.linspace(-max_r, max_r, Nl)
    y = np.copy(x)
    z = np.linspace(-max_z, max_z, Nz)

    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , Nl)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , Nl)
    fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , Nz)

    arg = Nl // 2
    print(fz[arg])

    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / nmedium / (1 - np.cos(alpha))
    
    if not os.path.exists(current_dir + '/Data/optical_system3D_vectorial.pkl'):
        optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nsample)
        optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Nz)), high_NA=True, vectorial=True)
        with open(current_dir + '/Data/optical_system3D_vectorial.pkl', 'wb') as f:
            pickle.dump(optical_system, f)
    else:
        with open(current_dir + '/Data/optical_system3D_vectorial.pkl', 'rb') as f:
            optical_system = pickle.load(f)

    illumination_conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 0.5, 1, Mr=3, Mt=1, dimensionality=3)
    
    cut_off_frequency_l = 1 / 2 / (2 * dx)

    illumination_conventional_projected = copy.deepcopy(illumination_conventional).project_in_quasi_2D()
    kernel = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency_l)[..., None], (31, 31, 31))

    noise_estimator_finite_conventional = SSNRCalculator.SSNRSIM3D(illumination_conventional, optical_system, kernel=kernel, illumination_reconstruction=illumination_conventional_projected)

# ...existing code...

import pandas as pd  # Add this import at the top

# ...existing code...

csv_file = current_dir + '/optimal_modulation_2d_kernel_3d_reconstruction_reduced_excitation_modulation.csv'

# Load existing data if file exists, else create empty DataFrame
if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
    # Create a set of existing (rounded m1, m2) pairs for quick lookup
    existing_pairs = set(zip(df_existing['m1'].round(2), df_existing['m2'].round(2)))
else:
    df_existing = pd.DataFrame(columns=['m1', 'm2', 'Volume', 'Entropy'])
    existing_pairs = set()

# List to collect new rows
new_rows = []

for m1 in np.arange(0.5, 20.1, 0.5):
    for m2 in np.arange(0.5, 20.1, 0.5):
        pair = (round(m1, 2), round(m2, 2))
        if pair not in existing_pairs:
            print(f"Computing for m1={m1}, m2={m2}")
            # ...existing computation code...
            illumination_new = copy.deepcopy(illumination_conventional_projected)
            for r in range(3):
                illumination_new.harmonics[(r, (1, 0, 0))].amplitude *= m1
                illumination_new.harmonics[(r, (-1, 0, 0))].amplitude *= m1
                illumination_new.harmonics[(r, (2, 0, 0))].amplitude *= m2
                illumination_new.harmonics[(r, (-2, 0, 0))].amplitude *= m2

            noise_estimator_finite_conventional.illumination_reconstruction = illumination_new

            ssnr_finite = noise_estimator_finite_conventional.ssnri
            ssnr_finite_ra = noise_estimator_finite_conventional.ring_average_ssnri()

            volume_finite = noise_estimator_finite_conventional.compute_ssnri_volume()
            entropy_finite = noise_estimator_finite_conventional.compute_ssnri_entropy()

            print(f"Volume finite {m1}, {m2} = ", volume_finite)
            print(f"Entropy finite {m1}, {m2} = ", entropy_finite)

            # Collect new row as dict
            new_rows.append({
                'm1': round(m1, 2),
                'm2': round(m2, 2),
                'Volume': volume_finite,
                'Entropy': entropy_finite
            })
            existing_pairs.add(pair)  # Prevent duplicates in the same run

# Append new rows to existing DataFrame and save
if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_file, index=False)
    print(f"Added {len(new_rows)} new rows to {csv_file}")
else:
    print("No new rows to add.")

# ...existing code...