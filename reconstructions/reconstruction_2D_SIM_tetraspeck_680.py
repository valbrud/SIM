import os.path
import pickle
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
import copy 
import scipy 
# reconstructed_fairSIM = tifffile.imread('data/OMX_Tetraspeck200_680nm_fairSIM_reco.tiff')
# print(reconstructed_fairSIM.shape)

# reco_ft = wrapped_fftn(reconstructed_fairSIM[3, ...])
# plt.imshow(np.log1p(np.abs(reco_ft)).T, cmap='gray', origin='lower')
# plt.show()


N = 255
wavelength = 680e-9
px_scaled = 80e-9
dx = px_scaled / wavelength
NA = 1.4
NA_best_fit = 1.15
nmedium = 1.518
alpha = np.arcsin(NA / nmedium)
alpha_best_fit = np.arcsin(NA_best_fit / nmedium)

max_r = dx * N // 2
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
psf_size = 2 * np.array((max_r, max_r))
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fr = np.linspace(0, 1 / (2 * dx), N//2 + 1)
xml_file = project_root + "/data/OMX-OTF-683nm-2d.xml"        
otf_cube = load_fairsim_otf(xml_file)
otf = otf_cube[2]  
otf = np.fft.fftshift(otf)
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
otf = otf[2::2, 2::2]
optical_system = System4f2D(alpha = alpha_best_fit, refractive_index=nmedium)

aberrated_psf_dict = pickle.load(open(current_dir + "\\aberrated_psf_dict2_tetraspec_680.pkl", "rb"))
for key, value in aberrated_psf_dict.items():
    value = utils.expand_kernel(value, (N, N))
    aberrated_psf_dict[key] = value

optical_system.compute_psf_and_otf((psf_size, N), account_for_pixel_size=False, save_pupil_function=True)
# optical_system.psf = aberrated_psf_dict[(0.0864, 0.0288)]
plt.plot(optical_system.otf_frequencies[0], optical_system.otf[:, N//2], label='Estimated OTF')
plt.plot(optical_system.otf_frequencies[0], otf[:, N//2], label='FairSIM OTF')
plt.xlabel('q, $1/\lambda$')
plt.ylabel('Normalized value')
plt.legend()
# plt.savefig('reconstructions/images/OTF_comparison.png')
plt.show()

# optical_system.otf = otf
# optical_system.otf /= np.amax(np.abs(optical_system.otf))
# optical_system.psf = wrapped_ifftn(optical_system.otf) 
# optical_system.psf /= np.sum(optical_system.psf)

plt.imshow(np.log1p(optical_system.otf.real), cmap='gray')
plt.show()

data = tifffile.imread(project_root + '/data/OMX_Tetraspeck200_680nm.tiff')
print(data.shape)

stack = data.reshape((3, -1, 5, 512, 512))
stack = stack[:, 3, :, N//2+1:-N//2-1, N//2+1:-N//2-1]
image = stack[0, 0, ...]
image_ft = wrapped_fftn(stack[0, 0, ...])
noise = utils.average_rings2d(image_ft**2, optical_system.psf_coordinates)
pure_noise_region = image_ft[otf < 10**-3]
# plt.plot(np.log1p(np.abs(noise)))
# plt.show()
average_noise = np.mean(np.abs(pure_noise_region)**2)
print('average_noise', average_noise)
print('total_noise', np.sum(np.abs(pure_noise_region)**2))

total_counts = np.sum(stack[0, 0, ...])
print('total_counts', total_counts)

# exit()

offset = 90
gain = 6
stack = (stack - offset) // gain
stack[stack < 0] = 1
from windowing import make_mask_cosine_edge2d
mask = make_mask_cosine_edge2d(stack.shape[2:], 20)
stack = stack * mask[np.newaxis, np.newaxis, ...]
print(stack.shape)
plt.imshow(stack[0, 0, ...].T, cmap='gray', origin='lower')
plt.show()
plt.imshow(np.log1p(10 ** 8 * np.abs(wrapped_fftn(stack[0, 0, ...]))).T, cmap='gray', origin='lower')
plt.show()
# plt.imshow(np.log1p(np.abs(wrapped_fftn(stack[1, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()
# plt.imshow(np.log1p(np.abs(wrapped_fftn(stack[2, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()


configurations = BFPConfiguration(refraction_index=nmedium)

illumination3d = configurations.get_2_oblique_s_waves_and_s_normal(
    alpha_best_fit-0.4, 
    1, 0,
    Mr=3,
    Mt=5, 
    angles=(45 / 180 * np.pi , 165 / 180 * np.pi, 105 / 180 * np.pi),
) 

illumination = IlluminationPlaneWaves2D.init_from_3D(
    illumination3d, dimensions=(1, 1)
)
illumination.set_spatial_shifts_diagonally(number=5)

pattern_estimator = IlluminationPatternEstimator2D(
    illumination=illumination,
    optical_system=optical_system,
)

illumination = pattern_estimator.estimate_illumination_parameters(
    stack, 
    peaks_estimation_method = 'interpolation',
    phase_estimation_method = 'peak_phases',
    modulation_coefficients_method = 'peak_height_ratio',
    peak_search_area_size=11,
    zooming_factor=3, 
    max_iterations=10, 
    debug_info_level=3
)

illumination_experimental = copy.deepcopy(illumination)

# for harmonic in illumination.harmonics:
#     # wavevector, amplitude = illumination.harmonics[harmonic].wavevector, illumination.harmonics[harmonic].amplitude
#     # damping_factor = 2
#     # lukosz_factor = 2 * np.pi * 4 * NA / np.sqrt(np.sum(wavevector**2))
#     # lucosz_amplitude = np.cos(np.pi / (1 + lukosz_factor / damping_factor))
#     # print(harmonic, 'lucosz_factor=', lukosz_factor, 'lucosz_amplitude=', lucosz_amplitude)
#     # scaling_factor = (lucosz_amplitude / illumination.Mt / illumination.Mr) / np.abs(amplitude) 
#     if harmonic[1] != (0, 0):
#         illumination.harmonics[harmonic].amplitude = 0.016

illumination_widefield = configurations.get_widefield()
illumination_widefield = IlluminationPlaneWaves2D.init_from_3D(
    illumination_widefield, dimensions=(1, 1)
)

print(illumination.get_all_amplitudes())

# reconstructor_fourier = ReconstructorFourierDomain2D(
#     illumination=illumination,
#     optical_system=optical_system,
#     # kernel=kernels.psf_kernel2d(7, (dx, dx))

# )

reconstructor_spatial1 = ReconstructorSpatialDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=kernels.sinc_kernel2d(1)
)

reconstructor_spatial5 = ReconstructorSpatialDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=kernels.psf_kernel2d(5, (dx, dx))
)

reconstructor_spatial9 = ReconstructorSpatialDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=kernels.psf_kernel2d(9, (dx, dx))
)

reconstructor_widefield = ReconstructorFourierDomain2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
)


# rng = np.random.default_rng()
# for r in range(stack.shape[0]):
#     for n in range(stack.shape[1]):
#         stack[r, n] = rng.binomial(stack[r, n].astype(int), 0.5)

notch_kernel = kernels.angular_notch_kernel(11, 6, 0.25, theta0=np.pi/4)

# reconstructed_fourier = reconstructor_fourier.reconstruct(stack)
reconstructed_spatial1 = reconstructor_spatial1.reconstruct(stack)
reconstructed_spatial5 = reconstructor_spatial5.reconstruct(stack)
reconstructed_spatial9 = reconstructor_spatial9.reconstruct(stack)

# reconstucted_finite = scipy.signal.convolve(reconstructed_finite, notch_kernel, mode='same')

widefield  = reconstructor_spatial1.get_widefield(stack)
reconstructed_widefield = reconstructor_widefield.reconstruct(widefield[None, None, ...])
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
ax[0, 0].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_spatial1))).T, cmap='gray', origin='lower')
ax[0, 0].set_title('Fourier')
ax[0, 1].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_spatial5 ))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('Spatial')
ax[0, 2].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_spatial9))).T, cmap='gray', origin='lower')
ax[0, 2].set_title('Finite')
ax[0, 3].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_widefield))).T, cmap='gray', origin='lower')
ax[0, 3].set_title('Widefield')
ax[1, 0].imshow(reconstructed_spatial1.T, cmap='gray', origin='lower')
ax[1, 0].set_title('Fourier')
ax[1, 1].imshow(reconstructed_spatial5.T, cmap='gray', origin='lower')
ax[1, 1].set_title('Spatial')
ax[1, 2].imshow(reconstructed_spatial9.T, cmap='gray', origin='lower')
ax[1, 2].set_title('Finite')
ax[1, 3].imshow(reconstructed_widefield.T, cmap='gray', origin='lower')
ax[1, 3].set_title('Widefield')
plt.show()


# ssnr_spatial9 = SSNRSIM2D(
#     illumination=illumination_experimental,
#     illumination_reconstruction=illumination,
#     optical_system=optical_system,
#     # kernel=kernels.psf_kernel2d(7, (dx, dx)),
#     readout_noise_variance=1.4,
# )

ssnr_spatial1 = SSNRSIM2D(
    illumination=illumination,
    illumination_reconstruction=illumination,
    optical_system=optical_system,
    readout_noise_variance=1.4,
    kernel=kernels.sinc_kernel2d(1)
)

ssnr_spatial5 = SSNRSIM2D(
    illumination=illumination,
    illumination_reconstruction=illumination,
    optical_system=optical_system,
    readout_noise_variance=1.4,
    kernel=kernels.psf_kernel2d(5, (dx, dx))
)

ssnr_spatial9 = SSNRSIM2D(
    illumination=illumination,
    illumination_reconstruction=illumination,
    optical_system=optical_system,
    readout_noise_variance=1.4,
    kernel=kernels.psf_kernel2d(9, (dx, dx))
)

ssnr_widefield = SSNRSIM2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
    readout_noise_variance=1.4,
)

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
fig.suptitle('SSNR images')
# ax[0, 0].imshow(np.log1p(10 ** 8 * np.abs(ssnr_spatial9.ssnri)).T, cmap='gray', origin='lower')
# ax[0, 0].set_title('Fourier')
ax[0, 0].imshow(np.log1p(10 ** 8 * np.abs(ssnr_spatial1.ssnri)).T, cmap='gray', origin='lower')
ax[0, 0].set_title('K 1')
ax[0, 1].imshow(np.log1p(10 ** 8 * (np.abs(ssnr_spatial5.ssnri))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('K 5')
ax[0, 2].imshow(np.log1p(10 ** 8 * np.abs(ssnr_spatial9.ssnri)).T, cmap='gray', origin='lower')
ax[0, 2].set_title('K 9')

ratio15 = np.where(ssnr_spatial9.ring_average_ssnri_approximated(), ssnr_spatial1.ring_average_ssnri_approximated()/ssnr_spatial9.ring_average_ssnri(), 0)[:-20]
ratio59 = np.where(ssnr_spatial9.ring_average_ssnri_approximated(), ssnr_spatial5.ring_average_ssnri_approximated()/ssnr_spatial9.ring_average_ssnri(), 0)[:-20]
ax[1, 0].plot(ratio15)
ax[1, 0].plot(ratio59)

ax[1, 1].imshow((ssnr_spatial1.ssnri/ssnr_spatial9.ssnri)[20:-20, 20:-20], label = '15', origin='lower')
ax[1, 2].imshow((ssnr_spatial5.ssnri/ssnr_spatial9.ssnri)[20:-20, 20:-20], label = '59', origin='lower')
ax[1, 2].legend()
plt.show()

filtered_spatial1, w_spatial1, ssnr_spatial1_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_spatial1), 
    ssnr_spatial1,
)

filtered_spatial5, w_spatial5, ssnr_spatial5_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_spatial5), 
    ssnr_spatial5,
)

filtered_spatial9, w_spatial9, ssnr_spatial9_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_spatial9), 
    ssnr_spatial9,
)

filtered_widefield, w_widefield, ssnr_widefield_measured = filter_true_wiener(
    wrapped_fftn(reconstructed_widefield),
    ssnr_widefield,
)

# filtered_fourier, w_fourier = filter_flat_noise(
#     wrapped_fftn(reconstructed_fourier), 
#     ssnr_spatial9,
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
#     ssnr_spatial9.dj,
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
# ax[0, 0].imshow(np.log1p(np.abs((filtered_fourier))).T, cmap='gray', origin='lower')
# ax[0, 0].set_title('Fourier')
ax[0, 0].imshow(np.log1p(np.abs((filtered_spatial1))).T, cmap='gray', origin='lower')
ax[0, 0].set_title('Spatial 1')
ax[0, 1].imshow(np.log1p(np.abs((filtered_spatial5))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('Spatial 5')
ax[0, 2].imshow(np.log1p(np.abs((filtered_spatial9))).T, cmap='gray', origin='lower')
ax[0, 2].set_title('Widefield')
# ax[1, 0].imshow(np.abs(wrapped_ifftn(filtered_fourier)).T, cmap='gray', origin='lower')
# ax[1, 0].set_title('Fourier')
ax[1, 0].imshow(np.abs(wrapped_ifftn(filtered_spatial1)).T, cmap='gray', origin='lower')
ax[1, 0].set_title('Spatial 1')
ax[1, 1].imshow(np.abs(wrapped_ifftn(filtered_spatial5)).T, cmap='gray', origin='lower')
ax[1, 1].set_title('Spatial 5')
ax[1, 2].imshow(np.abs(wrapped_ifftn(filtered_spatial9)).T, cmap='gray', origin='lower')
ax[1, 2].set_title('Widefield')
ax[2, 0].imshow(np.log1p(ssnr_spatial9_measured), cmap='gray', origin='lower')
ax[2, 0].set_title('Fourier')
ax[2, 1].imshow(np.log1p(ssnr_spatial5_measured), cmap='gray', origin='lower')
ax[2, 1].set_title('Spatial')
ax[2, 2].imshow(np.log1p(ssnr_spatial9_measured), cmap='gray', origin='lower')
ax[2, 2].set_title('Finite')
ratio15 = utils.average_rings2d(np.where(ssnr_spatial9_measured, (ssnr_spatial1_measured - 1)/(ssnr_spatial9_measured-1), 0), optical_system.otf_frequencies)
ratio59 = utils.average_rings2d(np.where(ssnr_spatial9_measured, (ssnr_spatial5_measured - 1)/(ssnr_spatial9_measured-1), 0), optical_system.otf_frequencies)
ratio_widefield = utils.average_rings2d(np.where(ssnr_spatial9_measured, (ssnr_widefield_measured - 1)/(ssnr_spatial9_measured-1), 0), optical_system.otf_frequencies)
ratio15_theory = np.where(ssnr_spatial1.ring_average_ssnri_approximated(), ssnr_spatial1.ring_average_ssnri_approximated()/ssnr_spatial9.ring_average_ssnri_approximated(), 0)
ratio59_theory = np.where(ssnr_spatial5.ring_average_ssnri_approximated(), ssnr_spatial5.ring_average_ssnri_approximated()/ssnr_spatial9.ring_average_ssnri_approximated(), 0)
ax[2, 3].plot(fr, ratio15, label='s/f')
ax[2, 3].plot(fr, ratio59, label='f/f')
# ax[2, 3].plot(ratio_widefield, label='widefield/fourier')
ax[2, 3].plot(fr, ratio15_theory, label = 's/f, t')
ax[2, 3].plot(fr, ratio59_theory, label = 's/f, t')
# ax[2, 3].legend()
ax[2, 3].set_ylim(-0.1, 1.5)
plt.show()

fig, axes = plt.subplots()
axes.plot(fr, ratio15/ratio15_theory, label='$K_2/ K_1$')
axes.plot(fr, ratio59/ratio59_theory, label='$K_3/ K_1$')
# axes.plot(fr, ratio_widefield, label='$WF / K_1$')
axes.set_xlabel('q, $1/\lambda$')
axes.set_ylabel('$R^E/R^A$')
axes.set_xlim(0, 4)
axes.set_ylim(0, 10)
axes.legend()
plt.show() 

apodization = AutoconvolutuionApodizationSIM2D(
    optical_system=optical_system,
    illumination=illumination,
)

ideal_otf = apodization.ideal_otf
plt.imshow((np.abs(ideal_otf)), cmap='gray')
plt.title('Ideal OTF')
plt.show()

apodized_fourier   = np.abs(wrapped_ifftn(ideal_otf * filtered_spatial1))
apodized_spatial    = np.abs(wrapped_ifftn(ideal_otf * filtered_spatial5))
apodized_finite     = np.abs(wrapped_ifftn(ideal_otf * filtered_spatial9))
apodized_widefield  = np.abs(wrapped_ifftn(optical_system.otf * filtered_widefield))

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
