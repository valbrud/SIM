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

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
theta = 0.5 * alpha
NA = 1.5 * np.sin(alpha)
dx = 1 / (8 * NA)
dy = dx
N = 301
max_r = N // 2 * dx
x = np.linspace(-max_r, max_r, N)
fx = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)

illumination = configurations.get_3_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=7, dimensionality=2)
illumination.set_spatial_shifts_diagonally()
fx_normalized = fx / (2 * NA)
fy_normalized = fx / (2 * NA)

psf_size = (2 * max_r, 2 * max_r)
optical_system = System4f2D(alpha=alpha, refractive_index=1.5, high_NA=True, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))

image = ShapesGenerator.generate_random_lines(psf_size, N, 0.1, 200, 100000, generate_default=True)
simulator = SIMulator2D(illumination, optical_system)
noiseless_images = simulator.generate_noiseless_sim_images(image)
noisy_images = simulator.add_noise(noiseless_images)

noise_estimator= SSNRSIM2D(illumination, optical_system)
noise_estimator_cc= SSNRSIMGLS2D(illumination, optical_system)

# plt.imshow((noise_estimator.ssnri / noise_estimator_cc.ssnri), vmin=0, vmax=2)
# plt.colorbar()
# plt.show()

reconstructor = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator.effective_kernels_ft, return_ft=True)
reconstructor_cc = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, effective_kernels=noise_estimator_cc.effective_kernels_ft, return_ft=True)

ssnr = utils.average_rings2d(noise_estimator.dj**2, (fx, fx))/utils.average_rings2d(noise_estimator.vj+1e-9, (fx, fx))
ssnr_cc = utils.average_rings2d(noise_estimator_cc.dj**2, (fx, fx))/utils.average_rings2d(noise_estimator_cc.vj+1e-9, (fx, fx))

reconstructed_ft = reconstructor.reconstruct(noisy_images)
reconstructed_cc_ft = reconstructor_cc.reconstruct(noisy_images)

filtered_ft, _,ssnr_measured = filter_true_wiener_sim(reconstructed_ft, noise_estimator)
filtered_cc_ft, _,ssnr_cc_measured = filter_true_wiener_sim(reconstructed_cc_ft, noise_estimator_cc)

apodization = Apodization.AutocorrelationApodizationSIM2D(optical_system, illumination)
apodized_ft = filtered_ft * apodization.ideal_otf
apodized_cc_ft = filtered_cc_ft * apodization.ideal_otf

apodized= hpc_utils.wrapped_ifftn(apodized_ft).real
apodized_cc = hpc_utils.wrapped_ifftn(apodized_cc_ft).real

plt.plot(fx[N//2:], ssnr_measured[N//2, N//2:]/ssnr_cc_measured[N//2, N//2:], label='measured')
plt.plot(fx[N//2:], ssnr/ssnr_cc, label='predicted')
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2)
axes[0].imshow(apodized)
axes[1].imshow(apodized_cc)

# axes[0].imshow(np.log1p(np.abs(filtered_perfect_ft)))
# axes[1].imshow(np.log1p(np.abs(filtered_imperfect_ft)))
# axes[2].imshow(np.log1p(np.abs(filtered_corrected_ft)))
plt.show()