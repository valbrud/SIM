import os.path
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import scipy
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
import pandas as pd  # Add this import at the top

configurations = BFPConfiguration(refraction_index=1.5)


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

optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nsample, high_NA=True, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Nz)))

illumination_conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 0.5, 1, Mr=3, Mt=1, dimensionality=3)

cut_off_frequency_l = 1 / 2 / (2 * dx)

illumination_conventional_projected = copy.deepcopy(illumination_conventional).project_in_quasi_2D()
kernel = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency_l)[..., None], (31, 31, 31))

noise_estimator_finite_conventional = SSNRCalculator.SSNRSIM3D(illumination_conventional, optical_system, kernel=kernel, illumination_reconstruction=illumination_conventional_projected)

# ...existing code...


# ...existing code...

csv_file = current_dir + '/Data/optimal_notch_kernel_filtering.csv'

# Load existing data if file exists, else create empty DataFrame
if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
    # Create a set of existing (rounded m1, m2) pairs for quick lookup
    existing_pairs = set(zip(df_existing['size_low_pass'].round(2), df_existing['size_notch'].round(2)))
else:
    df_existing = pd.DataFrame(columns=['size_low_pass', 'size_notch', 'Volume', 'Entropy'])
    existing_pairs = set()

# List to collect new rows
new_rows = []

for size_low_pass in np.arange(0.2, 5.1, 0.2):
    for size_notch in np.arange(0.2, 5.1, 0.2):
        cut_off_frequency_low_pass = 1 / 4 / (size_low_pass * dx)
        cut_off_frequency_notch = 1 / 4 / (size_notch * dx)
    
        pair = (round(size_low_pass, 2), round(size_notch, 2))
        if pair not in existing_pairs:
            print(f"Computing for size low pass={size_low_pass}, size notch ={size_notch}")
            kernel_low_pass = kernels.psf_kernel2d(0, (dx, dx), cut_off_frequency_low_pass)
            kernel_notch = kernels.finite_notch_kernel(0, (dx, dx), cut_off_frequency_notch)

            kernel_total = scipy.signal.convolve2d(kernel_low_pass, kernel_notch, mode='full')
            noise_estimator_finite_conventional.kernel = kernel_total[..., None]

            ssnr_finite = noise_estimator_finite_conventional.ssnri
            ssnr_finite_ra = noise_estimator_finite_conventional.ring_average_ssnri()

            volume_finite = noise_estimator_finite_conventional.compute_ssnri_volume()
            entropy_finite = noise_estimator_finite_conventional.compute_ssnri_entropy()
            
            new_rows.append({
                'size_low_pass': round(size_low_pass, 2),
                'size_notch': round(size_notch, 2),
                'Volume': volume_finite,
                'Entropy': entropy_finite
            })

            print(f"Volume finite {size_low_pass}, {size_notch} = ", volume_finite)
            print(f"Entropy finite {size_low_pass}, {size_notch} = ", entropy_finite)

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