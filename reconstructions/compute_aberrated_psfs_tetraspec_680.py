import os.path
import sys
import pickle
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)
import matplotlib.pyplot as plt
from OpticalSystems import System4f2D
import wrappers 

# Parameters from the original file
N = 61 
wavelength = 680e-9
px_scaled = 80e-9
dx = px_scaled / wavelength
NA = 1.18
nmedium = 1.518
alpha = np.arcsin(NA / nmedium)
print(alpha)

max_r = N //2 * dx
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
psf_size = 2 * np.array((max_r, max_r))

fx = np.linspace(-1/(2 * dx), 1/(2 * dx), N)
fy = np.copy(fx)
fr = np.linspace(0, 1 / (2 * dx), N//2 + 1)

fxn = fx / (2 * NA)
fyn = fy / (2 * NA)

# Aberration strengths
aberration_steps = np.arange(0, 0.108 + 0.0072, 0.0072)  # 15 steps: 0 to 0.108

# File to save/load
pickle_file = current_dir + '/aberrated_psf_dict3_tetraspec_680.pkl'

# Load existing dict if available
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        psf_dict = pickle.load(f)
    print(f"Loaded existing PSF dict with {len(psf_dict)} entries.")
else:
    psf_dict = {}
    print("Starting new PSF dict.")

fig, ax = plt.subplots()

# Loop over defocus and spherical
for defocus in aberration_steps:
    for spherical in aberration_steps:
        key = (defocus, spherical)
        if key in psf_dict:
            print(f"Skipping {key}, already computed.")
            continue
        
        print(f"Computing PSF for defocus={defocus}, spherical={spherical}")
        
        # Create optical system


        optical_system = System4f2D(alpha=alpha, refractive_index=nmedium)
        
        # Define zernike aberrations
        zernieke = {
            (2, 0): defocus,  # Defocus
            (4, 0): spherical  # Spherical
        }
        
        # Compute PSF
        optical_system.compute_psf_and_otf((psf_size, N), zernieke=zernieke)
        # ax.plot( optical_system.otf[N//2, N//2:], label=f'Defocus: {defocus:.3f}, Spherical: {spherical:.3f}')
        # plt.show()
        # Store in dict
        psf_dict[key] = optical_system.psf.copy()
        
        # Save after each computation
        with open(pickle_file, 'wb') as f:
            pickle.dump(psf_dict, f)
        
        print(f"Saved PSF for {key}")

print("Computation complete. PSF dict saved to aberrated_psf_dict2.pkl")
