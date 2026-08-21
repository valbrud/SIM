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
from SIMulator import SIMulator2D
import copy
import ShapesGenerator
import Reconstructor
from WienerFiltering import filter_true_wiener_sim
import hpc_utils
import Apodization
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
theta = 0.9 * alpha
NA = 1.5 * np.sin(alpha)
dx = 1 / (8 * NA)
dy = dx
N = 301
max_r = N // 2 * dx
x = np.linspace(-max_r, max_r, N)
fr = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)

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

# print(illumination_imperfect.harmonics[(0, (2, 0))].wavevector)
illumination_imperfect.spatial_shifts = spatial_shifts
print(illumination.phase_matrix)
print(illumination_imperfect.phase_matrix)
fx_normalized = fr / (2 * NA)
fy_normalized = fr / (2 * NA)
fx_normalized = fr / (1 - np.cos(theta))

psf_size = (2 * max_r, 2 * max_r)
optical_system = System4f2D(alpha=alpha, refractive_index=1.5)
optical_system.compute_psf_and_otf((psf_size, N))

image = ShapesGenerator.generate_random_lines(psf_size, N, 0.1, 300, 200, generate_default=True)
print("Photon count = ", np.sum(image)/N**2)

simulator = SIMulator2D(illumination, optical_system)
noiseless_images = simulator.generate_noiseless_sim_images(image)
noisy_images = simulator.add_noise(noiseless_images)

noise_estimator_perfect= SSNRSIM2D(illumination, optical_system, imperfect_phase_shifts=False)
noise_estimator_imperfect = SSNRSIM2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)
noise_estimator_corrected = SSNRSIMGLS2D(illumination_imperfect, optical_system, imperfect_phase_shifts=True)

reconstructor_perfect = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator_perfect.effective_kernels_ft, return_ft=True)
reconstructor_imperfect = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator_imperfect.effective_kernels_ft, return_ft=True)
reconstructor_corrected = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator_corrected.effective_kernels_ft, return_ft=True)

ssnr_perfect = noise_estimator_perfect.ring_average_ssnri()
ssnr_imperfect = noise_estimator_imperfect.ring_average_ssnri()
ssnr_corrected = noise_estimator_corrected.ring_average_ssnri()

reconstructed_perfect_ft = reconstructor_perfect.reconstruct(noisy_images)
reconstructed_imperfect_ft = reconstructor_imperfect.reconstruct(noisy_images)
reconstructed_corrected_ft = reconstructor_corrected.reconstruct(noisy_images)

filtered_perfect_ft, _,_ = filter_true_wiener_sim(reconstructed_perfect_ft, noise_estimator_perfect)
filtered_imperfect_ft, _, _ = filter_true_wiener_sim(reconstructed_imperfect_ft, noise_estimator_imperfect)
filtered_corrected_ft, _, _ = filter_true_wiener_sim(reconstructed_corrected_ft, noise_estimator_corrected)

apodization = Apodization.AutocorrelationApodizationSIM2D(optical_system, illumination)
apodized_perfect_ft = filtered_perfect_ft * apodization.ideal_otf
apodized_imperfect_ft = filtered_imperfect_ft * apodization.ideal_otf
apodized_corrected_ft = filtered_corrected_ft * apodization.ideal_otf

apodized_perfect = hpc_utils.wrapped_ifftn(apodized_perfect_ft).real
apodized_imperfect = hpc_utils.wrapped_ifftn(apodized_imperfect_ft).real
apodized_corrected = hpc_utils.wrapped_ifftn(apodized_corrected_ft).real

# fig, axes = plt.subplots(1, 3)
# axes[0].imshow(apodized_perfect)
# axes[1].imshow(apodized_imperfect)
# axes[2].imshow(apodized_corrected)
# plt.show()

for image, name in zip((apodized_perfect, apodized_imperfect, apodized_corrected), ("perfect", "imperfect", "corrected")):
    fig, ax = plt.subplots()
    if name=='perfect':
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
        ax.add_artist(scalebar)
    
    ax.imshow(image[100:200, 100:200])
    ax.set_axis_off()
    plt.show()
    # if name=='perfect':
        # fig.savefig(current_dir + f"/Figures/simulations/reconstruction_hexagonal_{name}.png", bbox_inches='tight', pad_inches=0.01)
    fig.savefig(current_dir + f"/Figures/simulations/reconstruction_hexagonal_{name}.png", bbox_inches='tight', pad_inches=0.01)
