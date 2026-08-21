import sys
import os

import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
sys.path.append(current_dir)

from kernels import psf_kernel2d

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
N = 301
dx = 1 / (8 * NA)
dy = dx
max_r = N//2 * dx
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
# print(x)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fr = fx[N//2:] / (2 * NA)

psf_size = 2 * np.array((max_r, max_r))

optical_system = System4f2D(alpha=NA, refractive_index=1)
optical_system.compute_psf_and_otf((psf_size, N))

illumination = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(alpha*0.9, 1, 0, Mt=5, dimensionality=2, dimensions=(1, 1, 1))

illumination_imperfect = copy.deepcopy(illumination)
illumination.set_spatial_shifts_diagonally()

# print(illumination_imperfect.phase_matrix)
# fx_normalized = fr / (2 * NA)
# fy_normalized = fr / (2 * NA)
# fx_normalized = fr / (1 - np.cos(theta))

# phi = np.array([
#     [0, 0],
#     [0, 2*np.pi/3],
#     [0, 4*np.pi/3],
#     [2*np.pi/3, 0],
#     [4*np.pi/3, 2*np.pi/3],
# ])
phi = np.array([
    [0, 0],
    [0, 2*np.pi/3],
    [0, 4*np.pi/3],
    [2*np.pi/3, 0],
    [4*np.pi/3, 2 * np.pi/3],
])

modes = [
    (0, 0),   # DC
    (2, 0),   # +s
    (-2, 0),  # -s
    (0, 2),   # +t
    (0, -2),  # -t
]

phase_matrix = {}

for ms, mt in modes:
    for n, (phi_s, phi_t) in enumerate(phi):
        value = ms * phi_s + mt * phi_t
        phase_matrix[(0, n, (ms, mt))] = np.exp(1j * value)

# print(phase_matrix)

phase_matrix_array_form = np.column_stack([
    np.ones(5),
    np.exp(1j * phi[:,0]),
    np.exp(-1j * phi[:,0]),
    np.exp(1j * phi[:,1]),
    np.exp(-1j * phi[:,1]),
])

print(np.linalg.cond(phase_matrix_array_form))
illumination_imperfect.phase_matrix = phase_matrix
# kernel= psf_kernel2d(1)
noise_estimator_perfect= SSNRSIM2D(illumination, optical_system,  imperfect_phase_shifts=True)
noise_estimator_perfect_cc= SSNRSIMGLS2D(illumination, optical_system,  imperfect_phase_shifts=False)
noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
noise_estimator_corrected = SSNRSIMGLS2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
ssnr_perfect_cc = noise_estimator_perfect_cc.ring_average_ssnri()
ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()
ssnr_corrected = noise_estimator_corrected.ring_average_ssnri()

# fig, ax = plt.subplots(1, 3)
# ax[0].plot(noise_estimator_perfect.dj[:, N//2], label='perfect')
# ax[0].plot(noise_estimator_imperfect.dj[:, N//2], label='imperfect')
# ax[0].plot(noise_estimator_corrected.dj[:, N//2], label='corrected')
# ax[1].plot(noise_estimator_perfect.vj[:, N//2], label='perfect')
# ax[1].plot(noise_estimator_imperfect.vj[:, N//2], label='imperfect')
# ax[1].plot(noise_estimator_corrected.vj[:, N//2], label='corrected')
# ax[2].plot(noise_estimator_perfect.vj[:, N//2] /  noise_estimator_perfect.dj[:, N//2]**2, label='perfect')
# ax[2].plot(noise_estimator_imperfect.vj[:, N//2] / noise_estimator_imperfect.dj[:, N//2]**2, label='imperfect')
# ax[2].plot(noise_estimator_corrected.vj[:, N//2] / noise_estimator_corrected.dj[:, N//2]**2 , label='corrected')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# plt.show()
# print(noise_estimator_corrected.mixing_matrix)
# ssnr_corrected.plot_effective_kernel_and_otf((0, (2, 0)))
# plt.show()

print(noise_estimator_perfect.compute_ssnri_entropy(), noise_estimator_perfect.compute_ssnri_volume())
print(noise_estimator_perfect_cc.compute_ssnri_entropy(), noise_estimator_perfect_cc.compute_ssnri_volume())
print(noise_estimator_imperfect.compute_ssnri_entropy(), noise_estimator_imperfect.compute_ssnri_volume())
print(noise_estimator_corrected.compute_ssnri_entropy(), noise_estimator_corrected.compute_ssnri_volume())

# im = plt.imshow(ssnr_imperfect/ssnr_corrected, vmin=0, vmax=2)
# plt.colorbar(im)
# plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel("$f_r$, [LCF]")
ax.set_ylabel("$1 + 10^3SSNR_{ra}$")
ax.set_yscale("log")
ax.plot(fr, 1 + 10**3 *ssnr_perfect, label="LS unitary")
ax.plot(fr, 1 + 10**3 *ssnr_imperfect, label="LS non-unitary")
ax.plot(fr, 1 + 10**3 * ssnr_corrected, label="GLS non-unitary")
ax.set_ylim(bottom=1)
ax.set_xlim(0, fr[-1])

ax.legend()

fig.savefig(current_dir + f"/Figures/simulations/ssnr_square_imperfect_phase_shifts.png", bbox_inches='tight', pad_inches=0.01)

plt.show()