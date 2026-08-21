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
plt.rcParams['xtick.labelsize'] = 10    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 10    # Font size for y-tick labels
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
noise_estimator_perfect= SSNRSIM2D(illumination, optical_system,  imperfect_phase_shifts=True)

illumination_imperfect = copy.deepcopy(illumination)
illumination.set_spatial_shifts_diagonally()
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()

phi = np.array([
    [0, 0],
    [2 * np.pi/5, 4*np.pi/5],
    [4 * np.pi/5, 8*np.pi/5],
    [6 * np.pi/5, 12 * np.pi/5],
    [8 * np.pi/5, 16 * np.pi/5],
])
print(phi)
print(np.angle(illumination.phase_matrix_array_form[:, 1]), np.angle(illumination.phase_matrix_array_form[:, 2]))

# quit()


phase_matrix_array_form = np.column_stack([
    np.ones(5),
    np.exp(1j * phi[:,0]),
    np.exp(-1j * phi[:,0]),
    np.exp(1j * phi[:,1]),
    np.exp(-1j * phi[:,1]),
])

print(np.linalg.cond(phase_matrix_array_form))

modes = [
    (0, 0),   # DC
    (2, 0),   # +s
    (-2, 0),  # -s
    (0, 2),   # +t
    (0, -2),  # -t
]


n_realizations = 20

phase_error_strengths = [1, 2, 5, 10, 20]
ssnr_total_imperfect = np.zeros((len(phase_error_strengths), N, N))
ssnr_total_corrected = np.zeros((len(phase_error_strengths), N, N))
for i, phase_errors_strength in enumerate(phase_error_strengths):
    phase_matrix = {}
    print(phase_errors_strength)
    for j in range(n_realizations):
        for n, (phi_s, phi_t) in enumerate(phi):
            phase_error = np.random.normal(0, phase_error_strengths[i]) / 180. * np.pi
            e_s, e_t = phase_error, phase_error
            # e_s, e_t = 0, 0
            for ms, mt in modes:
                value = ms * (phi_s + e_s) + mt * (phi_t + e_t)
                phase_matrix[(0, n, (ms, mt))] = np.exp(1j * value)

        # print(phase_matrix)

        illumination_imperfect.phase_matrix = phase_matrix
        print(np.linalg.cond(illumination_imperfect.phase_matrix_array_form))
        # kernel= psf_kernel2d(1)
        # noise_estimator_perfect_cc= SSNRSIMGLS2D(illumination, optical_system,  imperfect_phase_shifts=False)
        noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
        noise_estimator_corrected = SSNRSIMGLS2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
        # ssnr_perfect_cc = noise_estimator_perfect_cc.ring_average_ssnri()
        ssnr_imperfect = noise_estimator_imperfect.ssnri
        ssnr_corrected = noise_estimator_corrected.ssnri
        ssnr_total_imperfect[i] += ssnr_imperfect
        ssnr_total_corrected[i] += ssnr_corrected

    # fig, ax = plt.subplots()
    # ax.plot(fr, 1 + 10**3 * ssnr_perfect * 2  , color='black', label="LS unitary")
    # ax.plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_imperfect[i], (fx, fx)), color='red', label="LS non-unitary")
    # ax.plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_corrected[i], (fx, fx)), color='green', label="GLS non-unitary")
    # plt.show()

ssnr_total_imperfect /= n_realizations
ssnr_total_corrected /= n_realizations
fig, ax = plt.subplots(1, 5, figsize=(30, 6))
for i in range(len(phase_error_strengths)):
    # if i != 0:
    #     ax[i].set_axis_off()
    ax[i].set_title(f'std = {phase_error_strengths[i]}')
    ax[i].plot(fr, 1 + 10**3 * ssnr_perfect / 5, color='black', label="LS unitary")
    ax[i].plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_imperfect[i], (fx, fx)), color='red', label="LS non-unitary")
    ax[i].plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_corrected[i], (fx, fx)), color='green', label="GLS non-unitary")
    ax[i].set_ylim(bottom=1)
    ax[i].set_xlim(0, fr[-1])
    # ax[i].legend()
    ax[i].set_aspect(1/ax[i].get_data_ratio())

# fig.savefig(current_dir + f"/Figures/simulations/ssnr_square_imperfect_phase_shifts.png", bbox_inches='tight', pad_inches=0.01)

plt.show()