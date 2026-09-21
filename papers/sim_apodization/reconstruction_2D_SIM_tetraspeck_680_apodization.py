import os.path
import pickle
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import kernels 
import base64, xml.etree.ElementTree as ET, textwrap, binascii 
import re 

from OpticalSystems import System4f2D
from Illumination import IlluminationPlaneWaves2D, IlluminationNonLinearSIM2D
from PatternEstimation.PatternEstimator import IlluminationPatternEstimator2D, IlluminationPatternEstimator3D
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D, estimate_gain_and_offset
from config.BFPConfigurations import BFPConfiguration
from wrappers import wrapped_fftn, wrapped_ifftn
from WienerFiltering import filter_true_wiener_sim
from SSNRCalculator import SSNRSIM2D, SSNRWidefield2D
from Apodization import AutocorrelationApodizationSIM2D, TriangularApodizationSIM2D
import utils 
from resolution_measures import frc_one_image
from otf_decoder import load_fairsim_otf
import copy 
import scipy 

# reconstructed_fairSIM = tifffile.imread('data/OMX_Tetraspeck200_680nm_fairSIM_reco.tiff')
# print(reconstructed_fairSIM.shape)

# reco_ft = wrapped_fftn(reconstructed_fairSIM[3, ...])
# plt.imshow(np.log1p(np.abs(reco_ft)).T, cmap='gray', origin='lower')
# plt.show()

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

N = 255
wavelength = 680e-9
px_scaled = 80e-9
dx = px_scaled / wavelength
NA = 1.2
# NA_best_fit = 1.2
nmedium = 1.518
alpha = np.arcsin(NA / nmedium)
# alpha_best_fit = np.arcsin(NA_best_fit / nmedium)

max_r = dx * N // 2
x = np.linspace(-max_r, max_r, N)
y    = np.copy(x)
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
optical_system = System4f2D(alpha = alpha, refractive_index=nmedium, high_NA=True, vectorial=True)

# aberrated_psf_dict_file = current_dir + "\\psf_dicts\\aberrated_psf_vec_dict_expanded_tetraspec_680.pkl"
# aberrated_psf_dict = pickle.load(open(aberrated_psf_dict_file, "rb"))
# for key, value in aberrated_psf_dict.items():
#     value = utils.expand_kernel(value, (N, N))
#     aberrated_psf_dict[key] = value

optical_system.compute_psf_and_otf((psf_size, N))
# optical_system.psf = aberrated_psf_dict[(np.float64(1.17), np.float64(0.1224), np.float64(0.036))]
# optical_system.otf = otf
# plt.plot(optical_system.otf_frequencies[0], optical_system.otf[:, N//2], label='Estimated OTF')
# plt.plot(optical_system.otf_frequencies[0], otf[:, N//2], label='FairSIM OTF')
# plt.xlabel('q, $1/\lambda$')
# plt.ylabel('Normalized value')
# plt.legend()
# # plt.savefig('reconstructions/images/OTF_comparison.png')
# plt.show()

# optical_system.otf = otf
# optical_system.otf /= np.amax(np.abs(optical_system.otf))
# optical_system.psf = wrapped_ifftn(optical_system.otf) 
# optical_system.psf /= np.sum(optical_system.psf)

# plt.imshow(np.log1p(optical_system.otf.real), cmap='gray')
# plt.show()

data = tifffile.imread(project_root + '/data/OMX_Tetraspeck200_680nm.tif')
print(data.shape, data.dtype)


stack = data.reshape((3, -1, 5, 512, 512))
# offset = np.min(data)

stack = stack[:, 3, :, 100:100+N, 100:100+N]
from windowing import make_mask_cosine_edge2d
windowing_mask = make_mask_cosine_edge2d(stack.shape[2:], 50)

# plt.imshow(stack[0, 0, ...].T, cmap='gray', origin='lower')
# plt.show()
# plt.imshow(np.log1p(10 ** 8 * np.abs(wrapped_fftn(stack[0, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()

offset=80
readout_std = 0

image = np.copy(stack[0, 0, ...])
image = image * windowing_mask
# plt.imshow(image)
# plt.show()

# plt.hist(image.flatten(), bins=np.arange(0, 250, 1))
# plt.show()

frc1 = frc_one_image(
    image,
    num_bins=120,
    readout_noise=readout_std**2,
)

# plt.plot(frc1[1], frc1[0], label='Before correction')
# plt.xlabel('Spatial frequency, $1/\lambda$')
# plt.ylabel('FRC')
# plt.legend()
# plt.show()

image = np.copy(stack[0, 0, ...])
image_ft = wrapped_fftn(stack[0, 0, ...])
mask = np.zeros(image_ft.shape, dtype=bool)
edge = 70
mask[0:edge, 0:edge] = True
mask[-edge:, 0:edge] = True
mask[0:edge, -edge:] = True
mask[-edge:, -edge:] = True
# plt.imshow(mask)
# plt.show()
pure_noise_region = image_ft[mask]

average_noise = np.mean(np.abs(pure_noise_region)**2)
I0 = np.sum(image - offset)
In = I0**2 / average_noise 
f0 = (In + np.sqrt(In**2 + 4 * In * image.size * readout_std**2)) / 2
gain = I0 / f0
print("average noise:", round(average_noise), "I0:", I0, "In:", In, "f0:", f0)

noise = utils.average_rings2d(image_ft**2, optical_system.psf_coordinates)
# plt.hist(image.flatten(), bins=1000, range=(0, np.amax(image)))
# plt.show()

    # plt.plot(np.log1p(np.abs(noise)), label="average signal")
    # plt.plot(np.log1p(average_noise) * np.ones_like(noise), label='Measured noise level')
    # plt.plot(np.log1p(gain**2 * (f0 + image.size * readout_std**2)) * np.ones_like(noise), label='Estimated noise level')
    # plt.legend()
    # plt.show()

print('gain', np.mean(gain), 'offset', offset)
image -= offset
image = image / gain
image = image * windowing_mask
frc2 = frc_one_image(
    image,
    num_bins=120,
    readout_noise=readout_std**2,
)
# plt.plot(frc2[1], frc2[0], label='After offset correction')
# plt.xlabel('Spatial frequency, $1/\lambda$')
# plt.ylabel('FRC')
# plt.legend()
# plt.show()

stack = (stack - offset) / gain
stack[stack < 0] = 0
image = stack[0, 0, ...]

# plt.hist(image.flatten(), bins=1000, range=(0, np.amax(image)))
# plt.show()
total_counts = np.sum(stack[0, 0, ...])
print('total_counts', total_counts)


stack = stack * windowing_mask[np.newaxis, np.newaxis, ...]
print(stack.shape)
# plt.imshow(image.T, cmap='gray', origin='lower')
# plt.show()

# plt.imshow(np.log1p(np.abs(wrapped_fftn(stack[1, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()
# plt.imshow(np.log1p(np.abs(wrapped_fftn(stack[2, 0, ...]))).T, cmap='gray', origin='lower')
# plt.show()


configurations = BFPConfiguration(refraction_index=nmedium)

FORCE_RECOMPUTE = False
illumination_file = current_dir + '/illumination_vec_tetraspec.pkl'
if os.path.exists(illumination_file) and not FORCE_RECOMPUTE:
    with open(illumination_file, 'rb') as f:
        illumination = IlluminationPlaneWaves2D.load(illumination_file)
    print(f"Loaded existing illumination file.")

else:
    print("Estimating illumination pattern")
    illumination = IlluminationPlaneWaves2D.init_from_3D(configurations.get_2_oblique_s_waves_and_s_normal(
    alpha -0.7, 
    1, 0,
    Mr=3,
    Mt=5, 
    angles=(45 / 180 * np.pi , 165 / 180 * np.pi, 105 / 180 * np.pi),
    dimensionality=3
    ).project_in_quasi_2D(),
    )

    illumination.set_spatial_shifts_diagonally(Mt=5)

    pattern_estimator = IlluminationPatternEstimator2D(
        illumination=illumination,
        optical_system=optical_system,
    )

    illumination = pattern_estimator.estimate_illumination_parameters(
        stack, 
        peak_estimation_method = 'cross_correlation',
        phase_estimation_method = 'autocorrelation',
        modulation_coefficients_method = 'default',
        peak_search_area_size=11,
        zooming_factor=100, 
        max_iterations=2, 
        debug_info_level=1
    )

    if not os.path.exists(illumination_file):
        illumination.save(illumination_file)
        print(f"Saved estimated illumination pattern.")

# for r in illumination.Mr:
#     wavevector = illumination.get_wavevector((r, (2, 0)))
    
# illumination = IlluminationNonLinearSIM2D.init_from_linear_illumination(illumination, (0, 0, 1))
illumination.estimate_modulation_coefficients(stack, optical_system.psf, optical_system.x_grid, update=True)


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

optical_system.upsample(2)
image = utils.upsample(image, 2, True)
stack = np.array([np.array([utils.upsample(image, 2, True).real for image in one_rotation]) for one_rotation in stack])
print(illumination.get_all_amplitudes())

reconstructor_fourier = ReconstructorFourierDomain2D(
    illumination=illumination,
    optical_system=optical_system,
    # kernel=kernels.psf_kernel2d(7, (dx, dx))
    unitary=True

)

reconstructor_widefield = ReconstructorFourierDomain2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
    unitary=True
)

reconstructed_fourier = reconstructor_fourier.reconstruct(stack)

widefield  = reconstructor_fourier.get_widefield(stack)
reconstructed_widefield = reconstructor_widefield.reconstruct(widefield[None, None, ...])
pseudo_widefield = np.sum(stack, axis=(0, 1)) 
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Reconstructed images')
ax[0].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_fourier))).T, cmap='gray', origin='lower')
ax[0].set_title('Fourier')
ax[1].imshow(np.log1p(np.abs(wrapped_fftn(reconstructed_widefield))).T, cmap='gray', origin='lower')
ax[1].set_title('Widefield')
plt.show()


ssnr_fourier = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
)

ssnr_widefield = SSNRSIM2D(
    illumination=illumination_widefield,
    optical_system=optical_system,
)

filtered_fourier, w_fourier, ssnr_fourier_measured = filter_true_wiener_sim(
    wrapped_fftn(reconstructed_fourier), 
    ssnr_fourier,
)

filtered_widefield, w_widefield, ssnr_widefield_measured = filter_true_wiener_sim(
    wrapped_fftn(reconstructed_widefield),
    ssnr_widefield,
)

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
fig.suptitle('Filtered images')
ax.imshow(np.log1p(np.abs((filtered_fourier))).T, cmap='gray', origin='lower')
ax.set_title('Fourier')
    

apodization_auto = AutocorrelationApodizationSIM2D(
    optical_system=optical_system,
    illumination=illumination,
)
apodization_filter_auto = apodization_auto.apodization_function

apodization_triangular = TriangularApodizationSIM2D(optical_system, illumination)
apodization_filter1  = apodization_triangular.apodization_function
apodization_triangular.power = 0.4
apodization_filter04 = apodization_triangular.apodization_function


apodized_auto       = np.abs(wrapped_ifftn(apodization_filter_auto * filtered_fourier))
apodized1    = np.abs(wrapped_ifftn(apodization_filter1 * filtered_fourier))
apodized04    = np.abs(wrapped_ifftn(apodization_filter04 * filtered_fourier))

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# fig.suptitle('Apodized images')
ax[0].imshow(np.abs(pseudo_widefield[100:300, 40:240]).T, cmap='gray', origin='lower')
ax[0].set_title('a')
ax[1].imshow(np.abs(apodized1[100:300, 40:240]).T, cmap='gray', origin='lower')
ax[1].set_title('b')
im = ax[2].imshow(np.abs(apodized04[100:300, 40:240]).T, cmap='gray', origin='lower')
# plt.colorbar(im, ax=ax[2])
ax[2].set_title('c')
ax[3].imshow(np.abs(apodized_auto[100:300, 40:240]).T, cmap='gray', origin='lower')
ax[3].set_title('d')
ax[0].set_axis_off()
ax[1].set_axis_off()
ax[2].set_axis_off()
ax[3].set_axis_off()
fig.tight_layout()
scalebar = ScaleBar(
    dx= dx*680,           
    units='nm',        
    length_fraction=0.3,
    location='lower left',
    color='white',
    box_alpha=0,        # transparent background
    label_loc='top',
    font_properties={'size': 25}
    )
ax[0].add_artist(scalebar)
plt.show()
fig.savefig(os.path.join(current_dir, "Figures/experiments/apodization_comparison_tetraspec_680.png"), bbox_inches='tight')

