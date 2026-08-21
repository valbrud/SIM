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

theta = 4 * np.pi / 8
alpha = np.pi / 4
dx = 1 / (8 * np.sin(alpha))
dy = dx
N = 101
max_r = N // 2 * dx
x = np.linspace(-max_r, max_r, N)
fr = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
NA = np.sin(alpha)

illumination = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 0, Mt=19, dimensionality=2)
illumination_imperfect = copy.deepcopy(illumination)
illumination.set_spatial_shifts_diagonally()
spatial_shifts = np.zeros((19, 2))
periodx = 1 / (np.sin(theta) / 2)
periody = 1 / (np.sqrt(3) / 2 * np.sin(theta))
print(illumination.spatial_shifts)
# for i in range(19):
#     print(i, periodx/19 * i, 15  * periody/19 * i)
#     spatial_shifts[i] = np.array([  1.05 * periodx/19 * i, 4.05 * periody/19 * i])
print(illumination_imperfect.harmonics[(0, (2, 0))].wavevector)
illumination_imperfect.harmonics[(0, (2, 0))].wavevector[1] = -2
# illumination_imperfect.harmonics[(0, (0, 2))].wavevector[0] = 1
illumination_imperfect.harmonics[(0, (-2, 0))].wavevector[1] = 2
# illumination_imperfect.harmonics[(0, (0, -2))].wavevector[0] = -1
illumination_imperfect.set_spatial_shifts_diagonally()
print(illumination.phase_matrix)
print(illumination_imperfect.phase_matrix)
fx_normalized = fr / (2 * NA)
fy_normalized = fr / (2 * NA)
fx_normalized = fr / (1 - np.cos(theta))

optical_system = System4f2D(alpha=NA, refractive_index=1)
optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r), N))


noise_estimator_perfect= SSNRSIM2D(illumination, optical_system, imperfect_phase_shifts=True)
noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
noise_estimator_corrected = SSNRSIMGLS2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()
ssnr_corrected = noise_estimator_corrected.ring_average_ssnri()

plt.imshow(noise_estimator_imperfect.ssnri/noise_estimator_corrected.ssnri, vmin=0, vmax=2)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(ssnr_perfect, label="Perfect")
ax.plot(ssnr_imperfect, label="Imperfect")
ax.plot(ssnr_corrected, label="Corrected")
plt.legend()
plt.show()

