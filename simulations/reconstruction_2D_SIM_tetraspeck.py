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
from PatternEstimator import PatternEstimatorInterpolation2D
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from config.BFPConfigurations import BFPConfiguration
from wrappers import wrapped_fftn, wrapped_ifftn
from WienerFiltering import filter_true_wiener, filter_flat_noise, filter_constant
from SSNRCalculator import SSNRSIM2D, SSNRWidefield2D
from Apodization import AutoconvolutuionApodizationSIM2D
import stattools 
from ResolutionMeasures import frc_one_image

def _vec_from_band(text: str) -> np.ndarray:
    """Decode one <band-*> payload (big-endian float32)."""
    txt = text.strip()
    txt = txt.split(':END64')[0]                     # remove trailing tag
    txt = re.sub(r'(?i)^BASE64:\s*', '', txt)        # remove leading tag
    txt = re.sub(r'\s+', '', txt)                    # remove all whitespace
    txt += '=' * (-len(txt) % 4)                     # pad to /4 length
    raw = base64.b64decode(txt, validate=False)
    be  = np.frombuffer(raw, dtype='>f4')            # big-endian view
    return be.astype('<f4', copy=False)              # native order

def otf_radial_to_2d(xml_path: str, size: int = 512, out_tif: str | None = None):
    """
    Convert a fairSIM radial-OTF XML/TXT file to a square float32 magnitude image.
    Works on NumPy ≥ 2.0 (no .newbyteorder()).
    """
    root  = ET.parse(xml_path).getroot()
    data  = root.find('.//data')
    Δf    = float(data.find('cycles').text)          # cycles  ·  pixel⁻¹
    bands = sorted((b for b in data if b.tag.startswith('band-')),
                   key=lambda el: int(el.tag.split('-')[-1]), reverse=True)
    # concatenate high→mid→low, then flip each so that index 0 = DC
    radial = np.hstack([_vec_from_band(b.text)[::-1] for b in bands])

    # frequency axis for those samples
    freq_samples = np.arange(radial.size, dtype=np.float32) * Δf   # cy  ·  pix⁻¹

    # Cartesian grid in the same units (−0.5 … +0.5 cy/pix)
    fx, fy = np.meshgrid(np.linspace(-.5, .5, size, endpoint=False),
                         np.linspace(-.5, .5, size, endpoint=False))
    ρ = np.hypot(fx, fy)                                         # radial freq

    # interpolate magnitude; values outside last sample → 0
    H = np.interp(ρ, freq_samples, radial, left=0.0, right=0.0).astype(np.float32)

    if out_tif:
        tifffile.imwrite(out_tif, np.fft.ifftshift(H))           # Fiji-friendly
    return H
# quit()

reconstructed_fairSIM = tifffile.imread('data/OMX_Tetraspeck200_680nm_fairSIM_reco.tiff')
print(reconstructed_fairSIM.shape)

reco_ft = wrapped_fftn(reconstructed_fairSIM[3, ...])
plt.imshow(np.log1p(np.abs(reco_ft)).T, cmap='gray', origin='lower')
plt.show()


N = 255
wavelength = 680e-9
px_scaled = 80e-9
dx = px_scaled / wavelength
NA = 1.1
nmedium = 1.518
alpha = np.arcsin(NA / nmedium)
max_r = dx * N // 2
psf_size = 2 * np.array((max_r, max_r))
# otf = otf_radial_to_2d('data/OMX-OTF-683nm-2d.xml', out_tif='data/OMX_OTF_683nm.tiff')
# plt.imshow(otf, cmap='gray')
# plt.show()
optical_system = System4f2D(alpha = alpha, refractive_index=nmedium)
optical_system.compute_psf_and_otf((psf_size, N), account_for_pixel_size=False, save_pupil_function=True)

plt.imshow(np.log1p(optical_system.otf.real), cmap='gray')
plt.show()

data = tifffile.imread('data/OMX_Tetraspeck200_680nm.tiff')
print(data.shape)

stack = data.reshape((3, -1, 5, 512, 512))
stack = stack[:, 3, :, N//2+1:-N//2-1, N//2+1:-N//2-1]
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
    alpha-0.5, 
    1, 0,
    Mr=3,
    Mt=5, 
    angles=(45 / 180 * np.pi , 165 / 180 * np.pi, 105 / 180 * np.pi),
) 

illumination = IlluminationPlaneWaves2D.init_from_3D(
    illumination3d, dimensions=(1, 1)
)
# illumination.set_spatial_shifts_diagonally()

pattern_estimator = PatternEstimatorInterpolation2D(
    illumination=illumination,
    optical_system=optical_system,
)

illumination = pattern_estimator.estimate_illumination_parameters(
    stack, 
    interpolation_factor= 2,
    peak_search_area_size=15,
    estimate_modulation_coefficients=True,
    method_for_modulation_coefficients='peak_height_ratio',
    peak_interpolation_area_size=3, 
    iteration_number=10, 
    deconvolve_stacks=False,
    correct_peak_position=True, 
    ssnr_estimation_iters=100
)

# for r in range(illumination.Mr):
#     illumination.harmonics[(r, (2, 0))].amplitude = 1
#     illumination.harmonics[(r, (-2, 0))].amplitude = 1

illumination_widefield = configurations.get_widefield()
illumination_widefield = IlluminationPlaneWaves2D.init_from_3D(
    illumination_widefield, dimensions=(1, 1)
)

print(illumination.get_all_amplitudes())

reconstructor_fourier = ReconstructorFourierDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    # kernel=kernels.psf_kernel2d(7, (dx, dx))

)

recontructor_spatial = ReconstructorSpatialDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=kernels.sinc_kernel(1)[..., 0]
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


reconstructed_fourier = reconstructor_fourier.reconstruct(stack)
reconstructed_spatial = recontructor_spatial.reconstruct(stack)
reconstructed_finite = recontructor_finite.reconstruct(stack)
widefield  = reconstructor_fourier.get_widefield(stack)
reconstructed_widefield = reconstructor_widefield.reconstruct(widefield[None, None, ...])

frc_fourier, freq = frc_one_image(
    reconstructed_fourier, 
    num_bins=50, 
    readout_noise=0
)

frc_spatial, freq = frc_one_image(
    reconstructed_spatial, 
    num_bins=50, 
    readout_noise=0
)

frc_finite, freq = frc_one_image(
    reconstructed_finite, 
    num_bins=50, 
    readout_noise=0
)

frc_widefield, freq = frc_one_image(
    reconstructed_widefield, 
    num_bins=50, 
    readout_noise=0
)

fig, ax = plt.subplots(figsize=(15, 5))
fig.suptitle('FRC curves')
ax.plot(freq, frc_fourier, label='Fourier')
ax.plot(freq, frc_spatial, label='Spatial')
ax.plot(freq, frc_finite, label='Finite')
ax.plot(freq, frc_widefield, label='Widefield')
ax.legend()
ax.hlines(0.143, 0, N//2, color='red', linestyle='--', label='0.143')
plt.show()
 
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
    readout_noise_variance=10,
)

ssnr_spatial = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    readout_noise_variance=10,
    kernel=kernels.sinc_kernel(1)[..., 0]
)

ssnr_finite = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    readout_noise_variance=10,
    kernel=kernels.psf_kernel2d(7, (dx, dx))
)

ssnr_widefield = SSNRSIM2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
    readout_noise_variance=10,
)

fig, ax = plt.subplots(2, 3, figsize=(15, 5))
fig.suptitle('SSNR images')
ax[0, 0].imshow(np.log1p(10 ** 8 * np.abs(ssnr_fourier.ssnri)).T, cmap='gray', origin='lower')
ax[0, 0].set_title('Fourier')
ax[0, 1].imshow(np.log1p(10 ** 8 * (np.abs(ssnr_spatial.ssnri))).T, cmap='gray', origin='lower')
ax[0, 1].set_title('Spatial')
ax[0, 2].imshow(np.log1p(10 ** 8 * np.abs(ssnr_finite.ssnri)).T, cmap='gray', origin='lower')
ax[0, 2].set_title('Finite')

ratio_spatial = stattools.average_rings2d(np.where(ssnr_fourier.ssnri, ssnr_spatial.ssnri/ssnr_fourier.ssnri, 0), optical_system.otf_frequencies)[:-10]
ratio_finite = stattools.average_rings2d(np.where(ssnr_fourier.ssnri, ssnr_finite.ssnri/ssnr_fourier.ssnri, 0), optical_system.otf_frequencies)[:-10]
ax[1, 0].plot(ratio_spatial)
ax[1, 0].plot(ratio_finite)

ax[1, 1].imshow(ssnr_spatial.ssnri/ssnr_fourier.ssnri, label = 'spatial', origin='lower')
ax[1, 2].imshow(ssnr_finite.ssnri/ssnr_fourier.ssnri, label = 'spatial', origin='lower')
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
# ax[1, 3].set_title('Widefield')
# ax[2, 0].imshow(np.log1p(ssnr_fourier_measured), cmap='gray', origin='lower')
# ax[2, 0].set_title('Fourier')
# ax[2, 1].imshow(np.log1p(ssnr_spatial_measured), cmap='gray', origin='lower')
# ax[2, 1].set_title('Spatial')
# ax[2, 2].imshow(np.log1p(ssnr_finite_measured), cmap='gray', origin='lower')
# ax[2, 2].set_title('Finite')
# ratio_spatial = stattools.average_rings2d(np.where(ssnr_fourier_measured, ssnr_spatial_measured/ssnr_fourier_measured, 0), optical_system.otf_frequencies)[:-10]
# ratio_finite = stattools.average_rings2d(np.where(ssnr_fourier_measured, ssnr_finite_measured/ssnr_fourier_measured, 0), optical_system.otf_frequencies)[:-10]
# ratio_widefield = stattools.average_rings2d(np.where(ssnr_fourier_measured, ssnr_widefield_measured/ssnr_fourier_measured, 0), optical_system.otf_frequencies)[:-10]
# ax[2, 3].plot(ratio_spatial, label='spatial/fourier')
# ax[2, 3].plot(ratio_finite, label='finite/fourier')
# ax[2, 3].plot(ratio_widefield, label='widefield/fourier')
# ax[2, 3].set_ylim(0, 10)
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
