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
from SSNRCalculator import SSNRSIM2D, SSNRSIMGLS2D
import copy

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 25    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 25    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 25    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

np.random.seed(0)

alpha = 2 * np.pi / 5
NA = 1.5 * np.sin(alpha)
N = 401
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

optical_system = System4f2D(alpha=alpha, refractive_index=1.5, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))

illumination = configurations.get_2_oblique_s_waves_and_s_normal(alpha*0.9, 1, 0, Mt=3, Mr=3, dimensionality=2, dimensions=(1, 1, 1))
illumination.set_spatial_shifts_diagonally()
noise_estimator_perfect= SSNRSIM2D(illumination, optical_system,  imperfect_phase_shifts=True)
# plt.imshow(np.log1p(10**8 * np.abs(noise_estimator_perfect.ssnri)), cmap='gray', origin='lower')
# plt.show()
illumination_imperfect = copy.deepcopy(illumination)
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()

phi = np.array([
    [0, 0],
    [2 * np.pi/3, 0],
    [4 * np.pi/3, 0],
])
print(phi)
print(np.angle(illumination.phase_matrix_array_form[:, 1]), np.angle(illumination.phase_matrix_array_form[:, 2]))

# quit()


phase_matrix_array_form = np.column_stack([
    np.ones(3),
    np.exp(1j * phi[:,0]),
    np.exp(-1j * phi[:,0]),
])

print(np.linalg.cond(phase_matrix_array_form))

modes = [
    (0, 0),   # DC
    (2, 0),   # +t
    (-2, 0),  # -t
]


n_realizations = 100

phase_error_strengths = [10, 20,]
# line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
colors = ['green', 'blue']
# labels = ['$5\\degree$', '$10\\degree$', '$20\\degree$']
ssnr_total_imperfect = np.zeros((len(phase_error_strengths), N, N))
ssnr_total_corrected = np.zeros((len(phase_error_strengths), N, N))
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel('$f_r [2NA/\\lambda]$')
ax.set_ylabel('$1 + 10^4 SSNR_K$')
ax.plot(fr, (1 + 10**4 * ssnr_perfect), color='black', label="$\\delta = 0\\degree$" )

for i, phase_errors_strength in enumerate(phase_error_strengths):
    phase_matrix = {}
    print(phase_errors_strength)
    for j in range(n_realizations):
        for n, (phi_s, phi_t) in enumerate(phi):
            for r in range(illumination.Mr):
                e_s, e_t = np.random.normal(0, (1/np.sqrt(2))* phase_error_strengths[i]) / 180. * np.pi, (1 / np.sqrt(2)) * np.random.normal(0, phase_error_strengths[i]) / 180. * np.pi
                for ms, mt in modes:
                    value = ms * (phi_s + e_s) + mt * (phi_t + e_t)
                    phase_matrix[(r, n, (ms, mt))] = np.exp(1j * value)

        # print(phase_matrix)

        illumination_imperfect.phase_matrix = phase_matrix
        print(np.linalg.cond(illumination_imperfect.phase_matrix_array_form))
        # kernel= psf_kernel2d(1)
        # noise_estimator_perfect_cc= SSNRSIMGLS2D(illumination, optical_system,  imperfect_phase_shifts=False)
        noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
        # if phase_errors_strength == 20:
        #     plt.imshow(np.log1p(10**4 * np.abs(noise_estimator_imperfect.ssnri)), cmap='gray', origin='lower')
        #     plt.show()
        noise_estimator_corrected = SSNRSIMGLS2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
        # ssnr_perfect_cc = noise_estimator_perfect_cc.ring_average_ssnri()
        ssnr_imperfect = noise_estimator_imperfect.ssnri
        ssnr_corrected = noise_estimator_corrected.ssnri
        ssnr_total_imperfect[i] += ssnr_imperfect
        ssnr_total_corrected[i] += ssnr_corrected

    # fig, ax = plt.subplots()
    ax.plot(fr, (1 + 10**4 * utils.average_rings2d(ssnr_total_imperfect[i] / n_realizations, (fx, fx))), color=colors[i], label="$\\delta$=" + str(phase_error_strengths[i]) + "$\\degree$")
    # ax.plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_corrected[i], (fx, fx)), color='green', label="GLS non-unitary")

ax.set_yscale('log')
ax.set_ylim(bottom=1)
ax.set_xlim(0, fr[-1])
ax.legend()
ax.set_aspect(1/ax.get_data_ratio())
fig.savefig(current_dir + f"/Figures/simulations/ssnr_conventional_imperfect_phase_shifts.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.show()

# ssnr_total_imperfect /= n_realizations
# ssnr_total_corrected /= n_realizations
# fig, ax = plt.subplots(1, 5, figsize=(30, 6))
# for i in range(len(phase_error_strengths)):
#     # if i != 0:
#     #     ax[i].set_axis_off()
#     ax[i].set_title(f'std = {phase_error_strengths[i]}')
#     ax[i].plot(fr, 1 + 10**3 * ssnr_perfect, color='black', label="LS unitary")
#     ax[i].plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_imperfect[i], (fx, fx)), color='red', label="LS non-unitary")
#     ax[i].plot(fr, 1 + 10**3 * utils.average_rings2d(ssnr_total_corrected[i], (fx, fx)), color='green', label="GLS non-unitary")


# fig.savefig(current_dir + f"/Figures/simulations/ssnr_square_imperfect_phase_shifts.png", bbox_inches='tight', pad_inches=0.01)

# plt.show()