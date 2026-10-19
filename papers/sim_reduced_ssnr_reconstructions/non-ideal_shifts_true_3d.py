import sys
import os

import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
sys.path.append(current_dir)

from config.BFPConfigurations import BFPConfiguration
from Illumination import IlluminationPlaneWaves2D
from OpticalSystems import System4f2D, System4f3D
import utils
from SSNRCalculator import SSNRSIM2D, SSNRSIM3D, SSNRSIMGLS2D, SSNRSIMGLS3D
import copy

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
NA = 1.5 * np.sin(alpha)
dx = 1 / (8 * NA)
dz = 1 / (4 * 1.5 * (1 - np.cos(alpha)))
dy = dx
N = 101
max_r = N//2 * dx
max_z = N//2 * dz

psf_size = 2 * np.array((max_r, max_r, max_z))

x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
z = np.linspace(-max_z, max_z, N)
print(x)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), N)
fr = fx[N//2:] / (2 * NA)

optical_system = System4f3D(alpha=alpha, refractive_index_medium=1.5, refractive_index_sample=1.5, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))

illumination = configurations.get_2_oblique_s_waves_and_s_normal(0.9 * alpha, 1, 1, Mt=5, dimensionality=3, dimensions=(1, 1, 1))

illumination_imperfect = copy.deepcopy(illumination)
illumination.set_spatial_shifts_diagonally()

# print(illumination_imperfect.phase_matrix)
# fx_normalized = fr / (2 * NA)
# fy_normalized = fr / (2 * NA)
# fx_normalized = fr / (1 - np.cos(theta))

phase_matrix = {}
for Mr in range(3):
    for n in range(5):
        phase_matrix[(Mr, n, (0, 0, 0))] = 1
        phase_matrix[(Mr, n, (2, 0, 0))] = np.exp(1j * 4* n * np.pi/5)
        phase_matrix[(Mr, n,  (-2, 0, 0))] = np.exp(1j * -4*n * np.pi/5)
        phase_matrix[(Mr, n,  (1, 0, 1))] = np.exp(1j * 2*n * np.pi/5)
        phase_matrix[(Mr, n, (1, 0, -1))] = np.exp(1j * 2*n * np.pi/5) 
        phase_matrix[(Mr, n, (-1, 0, 1))] = np.exp(1j * -2*n * np.pi/5)
        phase_matrix[(Mr, n, (-1, 0, -1))] = np.exp(1j * -2*n * np.pi/5) 
    
    phase_matrix[(Mr, 5, (0, 0, 0))] = 1
    phase_matrix[(Mr, 5, (2, 0, 0))] = np.exp(1j * 8 * np.pi  / 5)
    phase_matrix[(Mr, 5,  (-2, 0, 0))] = np.exp(1j * -8 * np.pi  / 5)
    phase_matrix[(Mr, 5,  (1, 0, 1))] = np.exp(1j *(4* np.pi  / 5 + np.pi / 2))
    phase_matrix[(Mr, 5, (1, 0, -1))] = np.exp(1j *(4* np.pi  / 5 - np.pi / 2))
    phase_matrix[(Mr, 5, (-1, 0, 1))] = np.exp(1j *(- 4* np.pi  / 5 + np.pi / 2))
    phase_matrix[(Mr, 5, (-1, 0, -1))] = np.exp(1j *(- 4* np.pi  / 5 -np.pi / 2))

    phase_matrix[(Mr, 6, (0, 0, 0))] = 1
    phase_matrix[(Mr, 6, (2, 0, 0))] = 1
    phase_matrix[(Mr, 6,  (-2, 0, 0))] = 1
    phase_matrix[(Mr, 6,  (1, 0, 1))] = np.exp(1j * np.pi / 2)
    phase_matrix[(Mr, 6, (1, 0, -1))] = np.exp(-1j * np.pi / 2)
    phase_matrix[(Mr, 6, (-1, 0, 1))] = np.exp(1j * np.pi / 2)
    phase_matrix[(Mr, 6, (-1, 0, -1))] = np.exp(-1j * np.pi / 2)

illumination_imperfect.phase_matrix = phase_matrix
print(illumination_imperfect.Mt)
print(np.linalg.cond(illumination_imperfect.phase_matrix_array_form))

noise_estimator_perfect= SSNRSIM3D(illumination, optical_system, imperfect_phase_shifts=False)
noise_estimator_imperfect = SSNRSIM3D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
noise_estimator_corrected = SSNRSIMGLS3D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
print(np.round(noise_estimator_corrected.mixing_matrix, 2))
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()
ssnr_corrected = noise_estimator_corrected.ring_average_ssnri()

print(noise_estimator_perfect.compute_ssnri_entropy(), noise_estimator_perfect.compute_ssnri_volume())
print(noise_estimator_imperfect.compute_ssnri_entropy(), noise_estimator_imperfect.compute_ssnri_volume())
print(noise_estimator_corrected.compute_ssnri_entropy(), noise_estimator_corrected.compute_ssnri_volume())

# im = plt.imshow(ssnr_imperfect/ssnr_corrected, vmin=0, vmax=2)
# plt.colorbar(im)
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlabel(r'$f_r [2NA/\lambda]$')
ax.set_ylabel(r'$1 + 10^4 SSNR$')
ax.set_xlim(0, 1.9)
ax.set_ylim(1, 10**4 * 1.5)
ax.plot(fr, 1 + 10**4 * np.sum(ssnr_perfect, axis=1), color='black', label="Optimal")
ax.plot(fr, 1 + 10**4 * np.sum(ssnr_imperfect, axis=1), color = 'green',  label="Sub-optimal")
ax.set_yscale('log')
# ax.plot(1 + 10**4 * ssnr_corrected[N//2, :], label="Corrected")
plt.legend()
fig.savefig(current_dir + f"/Figures/simulations/ssnr_3d.png", bbox_inches='tight', pad_inches=0.01)
fig2, ax2 = plt.subplots(1, 1, figsize=(9, 8))
ax2.set_xlabel(r'$f_r [2NA/\lambda]$')
ax2.set_ylabel(r'$f_z [\sqrt{n_{m}^2 - NA^2}/\lambda]$')
im = ax2.imshow((ssnr_perfect+ 1e-12).T / (ssnr_imperfect + 1e-9).T, extent=[0, 2, -2, 2], aspect='auto', origin='lower')
plt.colorbar(im, ax=ax2)
# ax2.plot(1 + 10**4 * ssnr_corrected[N//2, :], label="Corrected")
fig2.savefig(current_dir + f"/Figures/simulations/ssnr_ratios_3d.png", bbox_inches='tight', pad_inches=0.01)
plt.show()