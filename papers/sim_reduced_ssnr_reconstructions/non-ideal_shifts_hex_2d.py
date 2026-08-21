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
plt.rcParams['axes.titlesize'] = 40     # Title of the axes
plt.rcParams['axes.labelsize'] = 40     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 40    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 40    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
theta = 0.9 * alpha
NA = np.sin(alpha) * 1.5
dx = 1 / (8 * NA)
dy = dx
N = 301
max_r = N // 2 * dx
x = np.linspace(-max_r, max_r, N)
fx = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fr = fx[N//2:]/(2 * NA)
# illumination = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=19, dimensionality=2)
illumination = configurations.get_3_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=7, dimensionality=2)
illumination_imperfect = copy.deepcopy(illumination)
illumination.set_spatial_shifts_diagonally()
kx, ky = illumination.get_base_vectors(0)
print(kx, ky)
# exit()
spatial_shifts = np.zeros((19, 2))
periodx = 2 * np.pi / kx * 1.04
periody = 2 * 2 * np.pi / ky * 1.04
print(illumination.spatial_shifts)
# for i in range(19):
#     print(i, periodx/19 * i, 4  * periody/19 * i)
#     spatial_shifts[i] = np.array([1 * periodx/19 * i, 4 * periody/19 * i])

spatial_shifts = np.zeros((7, 2))
for i in range(7):
    print(i, periodx/7 * i, 3  * periody/7 * i)
    spatial_shifts[i] = np.array([1 * periodx/7 * i, 3 * periody/7 * i])

# illumination_imperfect.harmonics[(0, (2, 0))].wavevector[1] = -2
# illumination_imperfect.harmonics[(0, (0, 2))].wavevector[0] = 1
# illumination_imperfect.harmonics[(0, (-2, 0))].wavevector[1] = 2
# illumination_imperfect.harmonics[(0, (0, -2))].wavevector[0] = -1
illumination_imperfect.spatial_shifts = spatial_shifts
# print(illumination.phase_matrix)
# print(illumination_imperfect.phase_matrix)
print(np.linalg.cond(illumination_imperfect.phase_matrix_array_form))
fx_normalized = fr / (2 * NA)
fy_normalized = fr / (2 * NA)
# fx_normalized = fr / (1 - np.cos(theta))

optical_system = System4f2D(alpha=alpha, refractive_index=1.5)
optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r), N))


noise_estimator_perfect= SSNRSIM2D(illumination, optical_system, imperfect_phase_shifts=False)
noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
noise_estimator_corrected = SSNRSIMGLS2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()
ssnr_corrected = noise_estimator_corrected.ring_average_ssnri()

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel("$f_r$, [LCF]")
ax.set_ylabel("$1+10^3 SSNR$")
ax.set_yscale("log")
ax.plot(fr, 1 + 10**3 *ssnr_perfect, label="LS unitary")
ax.plot(fr, 1 + 10**3 *ssnr_imperfect, label="LS non-unitary")
ax.plot(fr, 1 + 10**3 * ssnr_corrected, label="GLS non-unitary")
ax.set_ylim(bottom=1)
ax.set_xlim(0, fr[-1])

ax.legend()

fig.savefig(current_dir + f"/Figures/simulations/ssnr_hexagonal_imperfect_phase_shifts.png", bbox_inches='tight', pad_inches=0.05)

plt.show()

