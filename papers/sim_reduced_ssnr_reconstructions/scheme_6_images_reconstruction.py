import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
sys.path.append(current_dir)

from kernels import psf_kernel2d

from config.BFPConfigurations import BFPConfiguration
from Illumination import IlluminationPlaneWaves2D
from OpticalSystems import System4f2D
import utils
from SSNRCalculator import SSNRSIM2D, SSNRSIM2D6F
import copy

from SIMulator import SIMulator2D
import copy
import ShapesGenerator
import Reconstructor
from WienerFiltering import filter_true_wiener_sim
import hpc_utils
import Apodization
import resolution_measures

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 25    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 25    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 25    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
NA = 1.5 * np.sin(alpha)
N = 1001
dx = 1 / (8 * NA)
dy = dx
max_r = N//2 * dx
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
print(x)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fr = fx[N//2:] / (2 * NA)

psf_size = 2 * np.array((max_r, max_r))

k = 0.66
optical_system = System4f2D(alpha=alpha, refractive_index=1.5, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))
illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(
    np.asin(np.sin(alpha) * k), 1, 0, 3, Mt=1
)
illumination = IlluminationPlaneWaves2D.init_from_3D(
    illumination_3waves3d
)
illumination.set_spatial_shifts_diagonally(Mt=3)

noise_estimator = SSNRSIM2D(illumination, optical_system, imperfect_phase_shifts=False)
ssnr_conventional = noise_estimator.ring_average_ssnri()

noise_estimator_reduced = SSNRSIM2D6F(illumination, optical_system, imperfect_phase_shifts=False)
ssnr_reduced = noise_estimator_reduced.ring_average_ssnri()
# plt.imshow(np.log1p(10**5 *np.abs(noise_estimator_reduced.dj)), extent=(fx[0] / (2*NA), fx[-1] / (2*NA), fy[0] / (2*NA), fy[-1] / (2*NA)))
# plt.show()
# fig, ax = plt.subplots(1, 2, figsize=(10, 8))
# ax[0].plot(fr, ssnr_conventional, label='conventional')
# ax[0].plot(fr, ssnr_reduced, label='reduced')
# ax[0].legend()
# ax[0].set_xlim(0, 1.5)
# ax[1].plot(fr, ssnr_conventional / ssnr_reduced, label='advantage')
# ax[1].set_xlim(0, 1.5)
# ax[1].legend()
# plt.show()

# image = ShapesGenerator.generate_random_lines(
#     image_size=psf_size,
#     point_number=N,
#     line_width=0.4,
#     num_lines=100,
#     intensity=10**3
# )
scalebar = ScaleBar(
    dx= dx*488,        
    units='nm',        
    length_fraction=0.3,
    location='lower left',
    color='white',
    box_alpha=0,       
    label_loc='top',
    font_properties={'size': 25}
    )

# image = ShapesGenerator.generate_random_lines(psf_size, N, 0.1, 300, 2*10**4, generate_default=True)
# np.save(current_dir + f"/Figures/simulations/line_object.npy", image)
image_n1 = np.load(current_dir + f"/Figures/simulations/line_object.npy")
image_n2 = np.copy(image_n1)
print(np.average(image_n1))
# fig, ax = plt.subplots()
# ax.imshow(image_n1)
# ax.set_axis_off()
# ax.add_artist(scalebar)
# # fig.savefig(current_dir + f"/Figures/simulations/line_object.png", bbox_inches='tight', pad_inches=0.01)
# plt.show()


# exit()

simulator = SIMulator2D(illumination, optical_system)
illumination_reduced = copy.deepcopy(illumination)
phase_matrix_reduced = {}
for r in range(illumination.Mr):
    for n in range(2):
        phase_matrix_reduced[(r, n, (0, 0))] = 1 
        phase_matrix_reduced[(r, n, (2, 0))] = np.exp(1j * n * np.pi)
        phase_matrix_reduced[(r, n, (-2, 0))] = np.exp(1j * n * np.pi)
illumination_reduced.phase_matrix = phase_matrix_reduced

simulator_reduced = SIMulator2D(illumination_reduced, optical_system)

noiseless_images_conventional = simulator.generate_noiseless_sim_images(image_n1)

noisy_images_n1 = simulator.add_noise(noiseless_images_conventional)
noisy_images_n2 = simulator.add_noise(noiseless_images_conventional)

reconstructor_conventional = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator.effective_kernels_ft, return_ft=True, unitary=True)
reconstructor_reduced = Reconstructor.ReconstructorFourierDomain2D6F(illumination_reduced, optical_system, return_ft=True, unitary=True)

noiseless_images_reduced = simulator_reduced.generate_noiseless_sim_images(image_n1)
noisy_images_reduced_n1 = simulator_reduced.add_noise(noiseless_images_reduced)
noisy_images_reduced_n2 = simulator_reduced.add_noise(noiseless_images_reduced)
reconstructed_conventional_ft1 = reconstructor_conventional.reconstruct(noisy_images_n1)
reconstructed_reduced_ft1 = reconstructor_reduced.reconstruct(noisy_images_reduced_n1)
reconstructed_conventional_ft2 = reconstructor_conventional.reconstruct(noisy_images_n2)
reconstructed_reduced_ft2 = reconstructor_reduced.reconstruct(noisy_images_reduced_n2)
# plt.imshow(np.log1p(np.abs(reconstructed_conventional_ft1)))
# plt.show()
# plt.imshow(np.log1p(np.abs(reconstructed_reduced_ft1)))
# plt.show()

frc_ax, frc_conventional = resolution_measures.frc(reconstructed_conventional_ft1, reconstructed_conventional_ft2, is_fourier=True, num_bins=N//2 +1)
frc_ax, frc_reduced = resolution_measures.frc(reconstructed_reduced_ft1, reconstructed_reduced_ft2, is_fourier=True, num_bins=N//2 + 1)
fig1, ax1 = plt.subplots(figsize=(9, 8))
ax1.set_yscale('log')
ax1.plot(fr, 1 + (1/(1 - frc_conventional)), color='black', label='9f')
ax1.set_xlabel('$f_r [LCF]$')
ax1.set_ylabel('1 + SSNR')
ax1.plot(fr, 1 + (1/(1 - frc_reduced)), color='red', label='6f')
ax1.set_xlim(max(2*k - 1, 0), 1 + k)
ax1.set_ylim(bottom=1, top = 2* 10**4)
ax1.legend()
fig1.savefig(current_dir + f"/Figures/simulations/ssnr_6f.png", bbox_inches='tight', pad_inches=0.05)

fig2, ax2 = plt.subplots(figsize=(9, 8))
ax2.plot(fr, (1 - frc_reduced) / (1 - frc_conventional), color='red', label='measured')
ax2.plot(fr, ssnr_conventional / ssnr_reduced, color='black', label='theory')
ax2.legend()
ax2.set_xlabel('$f_r [LCF]$')
ax2.set_ylabel('SSNR ratio')
ax2.set_xlim(max(2*k-1, 0), 1 + k)
ax2.set_ylim(bottom=0, top = 6)
# ax[1].set_xlim(max(2*k-1, 0), 1 + k)
# ax[1].set_ylim(0, 100)
plt.show()
fig2.savefig(current_dir + f"/Figures/simulations/ssnr_ratios_6f.png", bbox_inches='tight', pad_inches=0.05)
# apodization = Apodization.AutocorrelationApodizationSIM2D(optical_system, illumination)
# apodized_perfect_ft = filtered_perfect_ft * apodization.ideal_otf
# apodized_imperfect_ft = filtered_imperfect_ft * apodization.ideal_otf

# apodized_perfect = hpc_utils.wrapped_ifftn(apodized_perfect_ft).real
# apodized_imperfect = hpc_utils.wrapped_ifftn(apodized_imperfect_ft).real


# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(apodized_perfect)
# axes[1].imshow(apodized_imperfect)
# plt.show()

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(fr, ssnr_imperfect_measured[N//2, N//2:]/ssnr_perfect_measured[N//2, N//2:], label='measured')
# ax.plot(fr, ssnr_imperfect/ssnr_perfect, label='theory')
# ax.legend()
# ax.set_xlim(0, 1.5)
# ax.set_ylim(0.4, 0.8)
# ax.set_xlabel('$f_r [LCF]$')
# ax.set_ylabel(r"$\langle SSNR_{n-u} \rangle /\langle SSNR_{u} \rangle$")
# plt.show()
# # fig.savefig(current_dir + f"/Figures/simulations/ssnr_ratios_square.png", bbox_inches='tight', pad_inches=0.01)
