import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)


import tifffile
import numpy as np
import matplotlib.pyplot as plt
import kernels 
import base64, xml.etree.ElementTree as ET, textwrap, binascii 
import re 

from OpticalSystems import System4f2D
from Illumination import IlluminationPlaneWaves2D
from PatternEstimator import IlluminationPatternEstimator2D, IlluminationPatternEstimator3D
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from config.BFPConfigurations import BFPConfiguration
from wrappers import wrapped_fftn, wrapped_ifftn
from WienerFiltering import filter_true_wiener, filter_flat_noise, filter_constant
from SSNRCalculator import SSNRSIM2D, SSNRWidefield2D
from Apodization import AutoconvolutuionApodizationSIM2D
import utils 
from ResolutionMeasures import frc_one_image
from otf_decoder import load_fairsim_otf


# reco_ft = wrapped_fftn(reconstructed_fairSIM[3, ...])
# plt.imshow(np.log1p(np.abs(reco_ft)).T, cmap='gray', origin='lower')
# plt.show()


N = 511
wavelength = 488e-9
px_scaled = 80e-9
dx = px_scaled / wavelength
NA = 1.2
nmedium = 1.518
alpha = np.arcsin(NA / nmedium)
max_r = dx * N // 2
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
psf_size = 2 * np.array((max_r, max_r))
# xml_file = "data/OMX-OTF-683nm-2d.xml"        
# otf_cube = load_fairsim_otf(xml_file)
# otf = otf_cube[2]  
# otf = np.fft.fftshift(otf)
# plt.plot(np.abs(otf[:, 256]), label='OTF_fairSIM')
# plt.show()
# psf = wrapped_ifftn(otf)
# plt.imshow(psf.T.real, cmap='gray', origin='lower')
# plt.show()
# psf = psf.reshape((256, 2,  256, 2))
# psf = np.sum(psf, axis=(1, 3))
# otf = wrapped_fftn(psf)
# plt.plot(otf[:, N//2], label='OTF_fairSIM')
# plt.show()
# otf = otf[2::2, 2::2]
optical_system = System4f2D(alpha = alpha, refractive_index=nmedium)
optical_system.compute_psf_and_otf((psf_size, N), account_for_pixel_size=False, save_pupil_function=True)
plt.plot(optical_system.otf_frequencies[0], optical_system.otf[:, N//2], label='OTF_simulated')
# plt.plot(optical_system.otf_frequencies[0], otf[:, N//2], label='OTF_fairSIM')
plt.legend()
plt.show()

# optical_system.otf = otf
# optical_system.otf /= np.amax(np.abs(optical_system.otf))
# optical_system.psf = wrapped_ifftn(optical_system.otf) 
# optical_system.psf /= np.sum(optical_system.psf)

plt.imshow(np.log1p(optical_system.otf.real), cmap='gray')
plt.show()

data = tifffile.imread('reconstructions/images/ActinSquareLatticeSIM.tif')
stack = data[:, 256:256+N, 256:256+N]
print(stack.shape)

image = stack[0, ...]
image_ft = wrapped_fftn(image)
noise = utils.average_rings2d(image_ft**2, optical_system.psf_coordinates)
# pure_noise_region = image_ft[otf < 10**-3]
# plt.plot(np.log1p(np.abs(noise)))
# plt.show()
# average_noise = np.mean(np.abs(pure_noise_region)**2)
# print('average_noise', average_noise)
# print('total_noise', np.sum(np.abs(pure_noise_region)**2))

# total_counts = np.sum(stack[0, 0, ...])
# print('total_counts', total_counts)

# exit()

offset = 90
gain = 6
stack = (stack - offset) // gain
stack[stack < 0] = 1
from windowing import make_mask_cosine_edge2d
mask = make_mask_cosine_edge2d(stack.shape[1:], 20)
stack = stack * mask[np.newaxis, ...]
print(stack.shape)
plt.imshow(stack[0, ...].T, cmap='gray', origin='lower')
plt.show()
plt.imshow(np.log1p(10 ** 8 * np.abs(wrapped_fftn(stack[0, ...]))).T, cmap='gray', origin='lower')
plt.show()
# plt.imshow(np.log1p(np.abs(wrapped_fftn(stack[1, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()
# plt.imshow(np.log1p(np.abs(wrapped_fftn(stack[2, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()


configurations = BFPConfiguration(refraction_index=nmedium)

illumination = configurations.get_4_circular_oblique_waves_and_circular_normal(
    alpha-0.2, 
    1, 0,
    Mt=5,
    dimensionality=2)

illumination.set_spatial_shifts_diagonally(number=13)

pattern_estimator = IlluminationPatternEstimator2D(
    illumination=illumination,
    optical_system=optical_system,
)

illumination = pattern_estimator.estimate_illumination_parameters(
    stack[None, ...], 
    peaks_estimation_method = 'interpolation',
    phase_estimation_method = 'autocorrelation',
    modulation_coefficients_method = 'peak_height_ratio',
    peak_search_area_size=31,
    zooming_factor=3, 
    max_iterations=10, 
    debug_info_level=3
)

# for r in range(illumination.Mr):
#     illumination.harmonics[(r, (2, 0))].amplitude = 1
#     illumination.harmonics[(r, (-2, 0))].amplitude = 1

illumination_widefield = configurations.get_widefield()
illumination_widefield = IlluminationPlaneWaves2D.init_from_3D(
    illumination_widefield, dimensions=(1, 1)
)

print(illumination.get_all_amplitudes())

# optical_system.upsample()
print(optical_system.psf.shape)
print(optical_system.psf_coordinates[0].size)

reconstructor_fourier = ReconstructorFourierDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    # kernel=kernels.psf_kernel2d(7, (dx, dx))
)

recontructor_spatial = ReconstructorSpatialDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=kernels.sinc_kernel2d(1)
)

recontructor_finite = ReconstructorSpatialDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=kernels.psf_kernel2d(7, (dx, dx))
)

reconstructor_widefield = ReconstructorFourierDomain2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
)

# rng = np.random.default_rng()
# for r in range(stack.shape[0]):
#     for n in range(stack.shape[1]):
#         stack[r, n] = rng.binomial(stack[r, n].astype(int), 0.5)

reconstructed_fourier = reconstructor_fourier.reconstruct(stack[None, ...], upsample=False)
reconstructed_spatial = recontructor_spatial.reconstruct(stack[None, ...], upsample=False)
reconstructed_finite = recontructor_finite.reconstruct(stack[None, ...], upsample=False) 
widefield  = reconstructor_fourier.get_widefield(stack[None, ...]) 
reconstructed_widefield = reconstructor_widefield.reconstruct(widefield[None, None, ...], upsample=False)
# np.save("reconstructions/images/reconstructed_fourier1.npy", reconstructed_fourier)
# np.save("reconstructions/images/reconstructed_spatial1.npy", reconstructed_spatial)
# np.save("reconstructions/images/reconstructed_finite1.npy", reconstructed_finite)
# np.save("reconstructions/images/reconstructed_widefield1.npy", reconstructed_widefield)

# print("Reconstructed images saved successfully.")

# frc_fourier, freq = frc_one_image(
#     reconstructed_fourier, 
#     num_bins=50, 
#     readout_noise=0
# )

# frc_spatial, freq = frc_one_image(
#     reconstructed_spatial, 
#     num_bins=50, 
#     readout_noise=0
# )

# frc_finite, freq = frc_one_image(
#     reconstructed_finite, 
#     num_bins=50, 
#     readout_noise=0
# )

# frc_widefield, freq = frc_one_image(
#     reconstructed_widefield, 
#     num_bins=50, 
#     readout_noise=0
# )

# fig, ax = plt.subplots(figsize=(15, 5))
# fig.suptitle('FRC curves')
# ax.plot(freq, frc_fourier, label='Fourier')
# ax.plot(freq, frc_spatial, label='Spatial')
# ax.plot(freq, frc_finite, label='Finite')
# ax.plot(freq, frc_widefield, label='Widefield')
# ax.legend()
# ax.hlines(0.143, 0, N//2, color='red', linestyle='--', label='0.143')
# plt.show()
 
# scaling_spatial = np.sum(np.abs(reconstructed_spatial)) / np.sum(np.abs(reconstructed_fourier))
# scaling_finite = np.sum(np.abs(reconstructed_finite)) / np.sum(np.abs(reconstructed_fourier))
# scaling_widefield = np.sum(np.abs(widefield)) / np.sum(np.abs(reconstructed_fourier))
# reconstructed_spatial /= scaling_spatial
# reconstructed_finite /= scaling_finite
# widefield /= scaling_widefield

fig, ax = plt.subplots(2, 4, figsize=(15, 5))
fig.suptitle('Reconstructed images')
ax[0, 0].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_fourier))).T, cmap='gray', origin='lower')
ax[0, 0].set_title('Fourier')
ax[0, 1].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_spatial))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('Spatial')
ax[0, 2].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_finite))).T, cmap='gray', origin='lower')
ax[0, 2].set_title('Finite')
ax[0, 3].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_widefield))).T, cmap='gray', origin='lower')
ax[0, 3].set_title('Widefield')
ax[1, 0].imshow(reconstructed_fourier.T, cmap='gray', origin='lower')
ax[1, 0].set_title('Fourier')
ax[1, 1].imshow(reconstructed_spatial.T, cmap='gray', origin='lower')
ax[1, 1].set_title('Spatial')
ax[1, 2].imshow(reconstructed_finite.T, cmap='gray', origin='lower')
ax[1, 2].set_title('Finite')
ax[1, 3].imshow(reconstructed_widefield.T, cmap='gray', origin='lower')
ax[1, 3].set_title('Widefield')
plt.show()


ssnr_fourier = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    # kernel=kernels.psf_kernel2d(7, (dx, dx)),
    readout_noise_variance=1.4,
)

ssnr_spatial = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    readout_noise_variance=1.4,
    kernel=kernels.sinc_kernel2d(1)
)

ssnr_finite = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    readout_noise_variance=1.4,
    kernel=kernels.psf_kernel2d(7, (dx, dx))
)

ssnr_widefield = SSNRSIM2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
    readout_noise_variance=1.4,
)

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
fig.suptitle('SSNR images')
ax[0, 0].imshow(np.log1p(10 ** 8 * np.abs(ssnr_fourier.ssnri)).T, cmap='gray', origin='lower')
ax[0, 0].set_title('Fourier')
ax[0, 1].imshow(np.log1p(10 ** 8 * (np.abs(ssnr_spatial.ssnri))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('Spatial')
ax[0, 2].imshow(np.log1p(10 ** 8 * np.abs(ssnr_finite.ssnri)).T, cmap='gray', origin='lower')
ax[0, 2].set_title('Finite')

ratio_spatial = np.where(ssnr_fourier.ring_average_ssnri_approximated(), ssnr_spatial.ring_average_ssnri_approximated()/ssnr_fourier.ring_average_ssnri(), 0)[:-10]
ratio_finite = np.where(ssnr_fourier.ring_average_ssnri_approximated(), ssnr_finite.ring_average_ssnri_approximated()/ssnr_fourier.ring_average_ssnri(), 0)[:-10]
ax[1, 0].plot(ratio_spatial)
ax[1, 0].plot(ratio_finite)

ax[1, 1].imshow(ssnr_spatial.ssnri/ssnr_fourier.ssnri, label = 'spatial', origin='lower')
ax[1, 2].imshow(ssnr_finite.ssnri/ssnr_fourier.ssnri, label = 'finite', origin='lower')
ax[1, 2].legend()
plt.show()

filtered_fourier, w_fourier, ssnr_fourier_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_fourier), 
    ssnr_fourier,
)

filtered_spatial, w_spatial, ssnr_spatial_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_spatial), 
    ssnr_spatial,
)

filtered_finite, w_finite, ssnr_finite_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_finite), 
    ssnr_finite,
)

filtered_widefield, w_widefield, ssnr_widefield_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_widefield),
    ssnr_widefield,
)

# filtered_fourier, w_fourier = filter_flat_noise(
#     wrapped_fftn(reconstructed_fourier), 
#     ssnr_fourier,
# )

# filtered_spatial, w_spatial = filter_flat_noise(
#     wrapped_fftn(reconstructed_spatial), 
#     ssnr_spatial,
# )

# filtered_finite, w_finite = filter_flat_noise(
#     wrapped_fftn(reconstructed_finite), 
#     ssnr_finite,
# )

# filtered_widefield, w_widefield = filter_flat_noise(
#     wrapped_fftn(reconstructed_widefield),
#     ssnr_widefield,
# )

# filtered_fourier, w_fourier = filter_constant(
#     wrapped_fftn(reconstructed_fourier), 
#     ssnr_fourier.dj,
#     1e-8,
# )

# filtered_spatial, w_spatial = filter_constant(
#     wrapped_fftn(reconstructed_spatial), 
#     ssnr_spatial.dj,
#     1e-7,
# )

# filtered_finite, w_finite = filter_constant(
#     wrapped_fftn(reconstructed_finite), 
#     ssnr_finite.dj,
#     1e-8,
# )

# filtered_widefield, w_widefield = filter_constant(
#     wrapped_fftn(reconstructed_widefield),
#     ssnr_widefield.dj,
#     1e-5,
# ) 
    
fig, ax = plt.subplots(3, 4, figsize=(15, 5))
fig.suptitle('Filtered images')
ax[0, 0].imshow(np.log1p(np.abs((filtered_fourier))).T, cmap='gray', origin='lower')
ax[0, 0].set_title('Fourier')
ax[0, 1].imshow(np.log1p(np.abs((filtered_spatial))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('Spatial')
ax[0, 2].imshow(np.log1p(np.abs((filtered_finite))).T, cmap='gray', origin='lower')
ax[0, 2].set_title('Finite')
ax[0, 3].imshow(np.log1p(np.abs((filtered_widefield))).T, cmap='gray', origin='lower')
ax[0, 3].set_title('Widefield')
ax[1, 0].imshow(np.abs(wrapped_ifftn(filtered_fourier)).T, cmap='gray', origin='lower')
ax[1, 0].set_title('Fourier')
ax[1, 1].imshow(np.abs(wrapped_ifftn(filtered_spatial)).T, cmap='gray', origin='lower')
ax[1, 1].set_title('Spatial')
ax[1, 2].imshow(np.abs(wrapped_ifftn(filtered_finite)).T, cmap='gray', origin='lower')
ax[1, 2].set_title('Finite')
ax[1, 3].imshow(np.abs(wrapped_ifftn(filtered_widefield)).T, cmap='gray', origin='lower')
ax[1, 3].set_title('Widefield')
ax[2, 0].imshow(np.log1p(ssnr_fourier_measured), cmap='gray', origin='lower')
ax[2, 0].set_title('Fourier')
ax[2, 1].imshow(np.log1p(ssnr_spatial_measured), cmap='gray', origin='lower')
ax[2, 1].set_title('Spatial')
ax[2, 2].imshow(np.log1p(ssnr_finite_measured), cmap='gray', origin='lower')
ax[2, 2].set_title('Finite')
ratio_spatial = utils.average_rings2d(np.where(ssnr_fourier_measured, ssnr_spatial_measured/ssnr_fourier_measured, 0), optical_system.otf_frequencies)[:-10]
ratio_finite = utils.average_rings2d(np.where(ssnr_fourier_measured, ssnr_finite_measured/ssnr_fourier_measured, 0), optical_system.otf_frequencies)[:-10]
ratio_widefield = utils.average_rings2d(np.where(ssnr_fourier_measured, ssnr_widefield_measured/ssnr_fourier_measured, 0), optical_system.otf_frequencies)[:-10]
ratio_spatial_theory = np.where(ssnr_fourier.ring_average_ssnri_approximated(), ssnr_spatial.ring_average_ssnri_approximated()/ssnr_fourier.ring_average_ssnri(), 0)[:-10]
ratio_finite_theory = np.where(ssnr_fourier.ring_average_ssnri_approximated(), ssnr_finite.ring_average_ssnri_approximated()/ssnr_fourier.ring_average_ssnri(), 0)[:-10]
ax[2, 3].plot(ratio_spatial, label='s/ff')
ax[2, 3].plot(ratio_finite, label='f/f')
# ax[2, 3].plot(ratio_widefield, label='widefield/fourier')
ax[2, 3].plot(ratio_spatial_theory, label = 's/f, t')
ax[2, 3].plot(ratio_finite_theory, label = 's/f, t')
# ax[2, 3].legend()
plt.show()

apodization = AutoconvolutuionApodizationSIM2D(
    optical_system=optical_system,
    illumination=illumination,
)

ideal_otf = apodization.ideal_otf
plt.imshow((np.abs(ideal_otf)), cmap='gray')
plt.title('Ideal OTF')
plt.show()

apodized_fourier = np.abs(wrapped_ifftn(ideal_otf * filtered_fourier))
apodized_spatial =  np.abs(wrapped_ifftn(ideal_otf * filtered_spatial))
apodized_finite = np.abs(wrapped_ifftn(ideal_otf * filtered_finite))
apodized_widefield = np.abs(wrapped_ifftn(optical_system.otf * filtered_widefield))

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
fig.suptitle('Apodized images')
ax[0].imshow(np.abs(apodized_fourier).T, cmap='gray', origin='lower')
ax[0].set_title('Fourier')
ax[1].imshow(np.abs(apodized_spatial).T, cmap='gray', origin='lower')
ax[1].set_title('Spatial')
ax[2].imshow(np.abs(apodized_finite).T, cmap='gray', origin='lower')
ax[2].set_title('Finite')
ax[3].imshow(np.abs(apodized_widefield).T, cmap='gray', origin='lower')
ax[3].set_title('Widefield')

plt.show()

slice_x = slice(N//2, N//2+100)
slice_y = slice(N//2-100, N//2)

reconstructions = {
    'fourier': apodized_fourier,
    'spatial': apodized_spatial,
    'finite': apodized_finite,
    'widefield': apodized_widefield
}
for name, reconstruction in reconstructions.items():
    fig, axes = plt.subplots(figsize=(12, 8))
    axes.imshow(reconstruction[slice_x, slice_y].T, cmap='gray', origin='lower', extent=(x[N//2 - 50], x[N//2 + 50], y[N//2 - 50], y[N//2 + 50]))
    # axes.axis('off')
    axes.set_xlabel('x, $\lambda$', fontsize=50)
    if name == 'widefield': 
        axes.set_ylabel('y, $\lambda$', fontsize=50)
    axes.tick_params(axis='both', which='major', labelsize=50)
    fig.savefig(f'reconstructions/images/{name}.png', bbox_inches='tight')

frc_fourier, freq = frc_one_image(
    apodized_fourier, 
    num_bins=50, 
    readout_noise=0
)

frc_spatial, freq = frc_one_image(
    apodized_spatial, 
    num_bins=50, 
    readout_noise=0
)

frc_finite, freq = frc_one_image(
    apodized_finite, 
    num_bins=50, 
    readout_noise=0
)

frc_widefield, freq = frc_one_image(
    apodized_widefield, 
    num_bins=50, 
    readout_noise=0
)

fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle('FRC curves')
ax.plot(freq, frc_fourier, label='Fourier')
ax.plot(freq, frc_spatial, label='Spatial')
ax.plot(freq, frc_finite, label='Finite')
ax.plot(freq, frc_widefield, label='Widefield')
ax.hlines(0.143, 0, 127, color='red', linestyle='--', label='0.143')
ax.legend()
plt.show()
