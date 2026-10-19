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
from OpticalSystems import System4f2D, System4f3D
import utils
from SSNRCalculator import SSNRSIM2D, SSNRSIM3D, SSNRSIMGLS2D, SSNRSIMGLS3D
import copy

from SIMulator import SIMulator2D
import copy
import ShapesGenerator
import Reconstructor
from WienerFiltering import filter_true_wiener_sim
import hpc_utils
import Apodization
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
print(x)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fr = fx[N//2:] / (2 * NA)

psf_size = 2 * np.array((max_r, max_r))

optical_system = System4f2D(alpha=alpha, refractive_index=1.5, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))

illumination = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(alpha * 0.9, 1, 0, Mt=5, dimensionality=2, dimensions=(1, 1, 0))

illumination_imperfect = copy.deepcopy(illumination)
illumination.set_spatial_shifts_diagonally()

# print(illumination_imperfect.phase_matrix)
# fx_normalized = fr / (2 * NA)
# fy_normalized = fr / (2 * NA)
# fx_normalized = fr / (1 - np.cos(theta))

phi = np.array([
    [0, 0],
    [0, 2*np.pi/3],
    [0, 4*np.pi/3],
    [2*np.pi/3, 0],
    [4*np.pi/3, 2*np.pi/3],
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

print(phase_matrix)

phase_matrix_array_form = np.column_stack([
    np.ones(5),
    np.exp(1j * phi[:,0]),
    np.exp(-1j * phi[:,0]),
    np.exp(1j * phi[:,1]),
    np.exp(-1j * phi[:,1]),
])
np.random.seed(0)
print(np.linalg.cond(phase_matrix_array_form))
illumination_imperfect.phase_matrix = phase_matrix
noise_estimator_perfect= SSNRSIM2D(illumination, optical_system,  imperfect_phase_shifts=True)
noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()


# image= ShapesGenerator.generate_random_spherical_particles(psf_size,
#                                         (N, N),
#                                         radius = 0.2,
#                                         num_particles = 500,
#                                         intensity = 1000,
#                                         generate_default = True)

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
    
image = ShapesGenerator.generate_random_lines(psf_size, N, 0.1, 300, 2000, generate_default=True)
fig, ax = plt.subplots()
ax.imshow(image)
ax.set_axis_off()
ax.add_artist(scalebar)
fig.savefig(current_dir + f"/Figures/simulations/line_object.png", bbox_inches='tight', pad_inches=0.01)
plt.show()


# exit()

simulator = SIMulator2D(illumination, optical_system)
noiseless_images = simulator.generate_noiseless_sim_images(image)
noisy_images = simulator.add_noise(noiseless_images)
# for image in noisy_images[0]:
#     plt.imshow(np.log1p(10**8 * np.abs(hpc_utils.wrapped_fftn(image))))
#     plt.show()

simulator_imperfect = SIMulator2D(illumination_imperfect, optical_system)
noiseless_images_imperfect =simulator_imperfect.generate_noiseless_sim_images(image)
noisy_images_imperfect = simulator.add_noise(noiseless_images_imperfect)

reconstructor_perfect = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator_perfect.effective_kernels_ft, return_ft=True, unitary=True)
reconstructor_imperfect = Reconstructor.ReconstructorFourierDomain2D(illumination_imperfect, optical_system, effective_kernels=noise_estimator_imperfect.effective_kernels_ft, return_ft=True, unitary=False)

# ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
# ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()
ssnr_perfect = utils.average_rings2d(noise_estimator_perfect.dj, (fx, fx))/utils.average_rings2d(noise_estimator_perfect.vj+1e-9, (fx, fx))
ssnr_imperfect = utils.average_rings2d(noise_estimator_imperfect.dj, (fx, fx))/utils.average_rings2d(noise_estimator_imperfect.vj+1e-9, (fx, fx))

reconstructed_perfect_ft = reconstructor_perfect.reconstruct(noisy_images)
reconstructed_imperfect_ft = reconstructor_imperfect.reconstruct(noisy_images_imperfect)
# plt.imshow(np.log1p(np.abs(reconstructed_perfect_ft)))
# plt.show()
# plt.imshow(np.log1p(np.abs(reconstructed_imperfect_ft)))
# plt.show()

filtered_perfect_ft, _,ssnr_perfect_measured = filter_true_wiener_sim(reconstructed_perfect_ft, noise_estimator_perfect)
filtered_imperfect_ft, _, ssnr_imperfect_measured = filter_true_wiener_sim(reconstructed_imperfect_ft, noise_estimator_imperfect)

apodization = Apodization.AutocorrelationApodizationSIM2D(optical_system, illumination)
apodized_perfect_ft = filtered_perfect_ft * apodization.ideal_otf
apodized_imperfect_ft = filtered_imperfect_ft * apodization.ideal_otf

apodized_perfect = hpc_utils.wrapped_ifftn(apodized_perfect_ft).real
apodized_imperfect = hpc_utils.wrapped_ifftn(apodized_imperfect_ft).real


# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(apodized_perfect)
# axes[1].imshow(apodized_imperfect)
# plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fr, ssnr_imperfect_measured[N//2, N//2:]/ssnr_perfect_measured[N//2, N//2:], color = 'blue', label='numerical')
ax.plot(fr, ssnr_imperfect/ssnr_perfect, color='green', label='theory')
ax.legend()
ax.set_xlim(0, 1.5)
ax.set_ylim(0.4, 0.8)
ax.set_xlabel(r'$f_r [2NA/\lambda]$')
ax.set_ylabel(r"$\langle SSNR_{n-u} \rangle /\langle SSNR_{u} \rangle$")
plt.show()
fig.savefig(current_dir + f"/Figures/simulations/ssnr_ratios_square.png", bbox_inches='tight', pad_inches=0.01)
