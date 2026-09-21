import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import czifile
import numpy as np
import matplotlib.pyplot as plt
import kernels
import scipy 

from OpticalSystems import System4f3D
from Illumination import IlluminationPlaneWaves3D, IlluminationPlaneWaves2D
from PatternEstimation.PatternEstimator import IlluminationPatternEstimator3D, IlluminationPatternEstimator2D
from Reconstructor import ReconstructorFourierDomain3D, ReconstructorSpatialDomain3D, ReconstructorSpatialDomain3DSliced, estimate_gain_and_offset
from config.BFPConfigurations import BFPConfiguration
from wrappers import wrapped_fftn, wrapped_ifftn
from WienerFiltering import filter_true_wiener_sim, filter_self_consistent_wiener
from SSNRCalculator import SSNRSIM3D
from Apodization import AutocorrelationApodizationSIM3D, TriangularApodizationSIM3D
import utils
import windowing
from matplotlib_scalebar.scalebar import ScaleBar   
import cupy
cupy.get_default_memory_pool().free_all_blocks()

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
Nl = 151          # lateral pixels (x, y)

wavelength = 488e-9
px_scaled  = 80e-9
dx         = px_scaled  / wavelength
NA         = 1.4
nmedium    = 1.518
alpha      = np.arcsin(NA / nmedium)
print(alpha)
dz         = 119.4e-9 / wavelength  # axial pixel size in lambda


# ---------------------------------------------------------------------------
# Read CZI file
# ---------------------------------------------------------------------------
raw = czifile.imread(project_root + '/reconstructions/actin_488_square_lattice/images/Actin-02.czi')
raw = raw.squeeze()   # remove singleton axes; expect shape like (Z, Y, X) or similar
print('CZI raw shape after squeeze:', raw.shape)

# CZI axes after squeeze are typically (Z, Y, X) – reorder to (X, Y, Z) to match SIM convention
# Adjust the transpose below if your squeeze gives a different order
               # e.g. (C, Z, Y, X)
data = raw.transpose(0, 2, 3, 1)

print('data shape (X, Y, Z):', data.shape)

Na = 9
max_r = Nl // 2 * dx
max_z = Na // 2 * dz
psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))

x = np.linspace(-max_r, max_r, Nl, endpoint=Nl % 2)
y = np.copy(x)
z = np.linspace(-max_z, max_z, Na, endpoint=Na % 2)

# ---------------------------------------------------------------------------
# Gain / offset estimation from edge pixels of one frame (z-centre)
# ---------------------------------------------------------------------------
mid_z = 17
image2d    = data[0, :, :, mid_z]

edge_pixels = Nl//25
H, W = data.shape[1], data.shape[2]
edge_mask = np.zeros((H, W), dtype=bool)
e = int(edge_pixels)
edge_mask[:e, :]  = True
edge_mask[-e:, :] = True
edge_mask[:, :e] |= True
edge_mask[:, -e:] |= True

image2d_ft       = wrapped_fftn(image2d)
pure_noise_region = image2d_ft[edge_mask]
average_noise     = np.mean(np.abs(pure_noise_region) ** 2)

offset = 100
gain   = average_noise / np.sum(image2d - offset)
print('gain', np.median(gain), 'offset', offset)

# ---------------------------------------------------------------------------
# Crop lateral region (Nl x Nl) and full Z stack, then calibrate
# ---------------------------------------------------------------------------
y0 = data.shape[1] // 2 - Nl // 2
x0 = data.shape[2] // 2 - Nl // 2
z0 = data.shape[3] // 2 - Na // 2
stack = data[:, y0:y0 + Nl, x0:x0 + Nl, z0: z0 + Na]          # (Nl, Nl, Na)
stack = (stack.astype(np.float64) - offset) / gain
print(np.sum(stack))
print('stack shape (Nl, Nl, Na):', stack.shape)    # expected ~(101, 101, 41)

# ---------------------------------------------------------------------------
# Cosine-edge windowing
# ---------------------------------------------------------------------------
mask2d = windowing.make_mask_cosine_edge2d((Nl, Nl), edge=20)
stack  = stack * mask2d[None, ..., None]

sum_stack = np.sum(stack, axis=3)
sum_stack = sum_stack.transpose(1, 2, 0)
# fig, ax, slider = utils.imshow3D(sum_stack)
# plt.show()

# fig, ax, slider = utils.imshow3D(stack[0, :, :, :], cmap='gray', origin='lower')
# plt.show()

# ---------------------------------------------------------------------------
# Optical system (3D)
# ---------------------------------------------------------------------------
optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nmedium, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Na)))
# plt.imshow(np.log1p(10**8 * optical_system.otf[N//2, :, :]), cmap='gray')
# plt.title("PSF")
# plt.show()
# image = 10**5 * np.ones(optical_system.psf.shape)
# plt.plot(optical_system.otf[Nl//2, :, Na//2])
# plt.show()
# fig, ax, slider =utils.imshow3D(optical_system.otf, vmin=0, vmax=np.amax(np.log1p(10**8)), mode='log1p', scaling=10**8, axis='y')
# ax.set_aspect(1/ax.get_data_ratio())
# plt.show()
# # ---------------------------------------------------------------------------
# Illumination (3D) – 4 circular oblique waves + circular normal
# ---------------------------------------------------------------------------
configurations = BFPConfiguration(refraction_index=nmedium)

illumination_file = current_dir + '/illumination_actin_3d.pkl'
FORCE_RECOMPUTE   = False

if os.path.exists(illumination_file) and not FORCE_RECOMPUTE:
    illumination = IlluminationPlaneWaves3D.load(illumination_file)
    print("Loaded existing illumination file.")
else:
    print("Estimating illumination pattern")
    illumination = configurations.get_4_circular_oblique_waves_and_circular_normal(
        alpha - 0.3,
        1, 1,
        Mt=5,
        alpha0=np.pi / 4,
        dimensionality=3
    )

    illumination.set_spatial_shifts_diagonally(Mt=13)

    pattern_estimator = IlluminationPatternEstimator3D(
        illumination=illumination,
        optical_system=optical_system,
    )

    # stack fed as (R, Mt, X, Y, Z); here R=1 rotation group
    illumination = pattern_estimator.estimate_illumination_parameters(
        stack[None, :, :, :, :],
        peak_estimation_method='cross_correlation',
        phase_estimation_method='peak_phases',  
        modulation_coefficients_method='default',
        peak_search_area_size=31,
        zooming_factor=100,
        max_iterations=2,
        debug_info_level=1,
        first_order_only=True
    )

    illumination.save(illumination_file)
    print("Saved estimated illumination pattern.")

illumination.estimate_modulation_coefficients(stack[None,...] , optical_system.psf, grid = optical_system.x_grid, method='peak_height_ratio', update=True)

# illumination.spatial_shifts = illumination.estimate_spatial_shifts_from_phase_matrix()
# print(illumination.spatial_shifts)

for harmonic in illumination.harmonics:
    illumination.harmonics[harmonic].amplitude = np.abs(illumination.harmonics[harmonic].amplitude)
    # if not harmonic[1] !=(0, 0, 0):
    #     if np.abs(harmonic[1][1]) == 1 and np.abs(harmonic[1][0]) == 1:
    #         illumination.harmonics[harmonic].amplitude = 0.3 * np.abs(illumination.harmonics[harmonic].amplitude)
    #     else:    
    #         illumination.harmonics[harmonic].amplitude = 0.9 * np.abs(illumination.harmonics[harmonic].amplitude)


# for n in range(illumination.Mt):
#     utils.imshow3D(illumination.get_illumination_density(optical_system.x_grid, r=0, n = n)[:10, :10, :10])
#     plt.show()

sim_input = stack[None, ...]                          # (1, Nl, Nl, Na)  [R, X, Y, Z]
upsample_factor = int(2)
optical_system.upsample(upsample_factor, naxes=2)
sim_input    = np.array([np.array([utils.upsample(image, factor=2, add_shot_noize=True, naxes=2) for image in  one_rotation]) for one_rotation in sim_input], dtype=np.float32)
print(sim_input.shape)

illumination_projected = illumination.project_in_quasi_2D()
for harmonic in illumination_projected.harmonics:
    if not harmonic[1] !=(0, 0, 0):
        illumination_projected.harmonics[harmonic].amplitude *=5

# illumination_projected.estimate

illumination_projected.Mt=13
# ---------------------------------------------------------------------------
# Reconstructors (3D)
# ---------------------------------------------------------------------------
illumination_widefield   = BFPConfiguration(refraction_index=nmedium).get_widefield(dimensionality=3)

flc = 1 / (4 * dx)

combined_kernel=kernels.combined_low_pass_notch_kernel(pixel_size=(dx, dx), first_zero_frequency_low_pass= flc, first_zero_frequency_notch=flc)

# combined_kernel = scipy.signal.convolve(kernels.psf_kernel2d(9), kernels.finite_notch_kernel(9))
# combined_kernel = kernels.psf_kernel2d(1)

apodization = AutocorrelationApodizationSIM3D(
    optical_system=optical_system,
    illumination=illumination,
)

apodization_widefield = AutocorrelationApodizationSIM3D(
    optical_system=optical_system,
    illumination=illumination_widefield,
    plane_wave_wavevectors=[np.array((0, 0, 0))]
)

optical_system.otf = optical_system.otf * np.where(apodization_widefield.ideal_otf > 10**-6, 1, 0)

reconstructor_fourier = ReconstructorFourierDomain3D(
    illumination=illumination,        
    optical_system=optical_system,
    unitary=True
)


reconstructor_widefield  = ReconstructorSpatialDomain3DSliced(
    illumination=illumination_widefield,
    optical_system=optical_system,
    kernel=kernels.psf_kernel2d(1)
)

# ---------------------------------------------------------------------------
# Reconstruct – stack shape (Nl, Nl, Na), passed as (R=1, Mt, Nl, Nl, Na)
# Expand dims: [None, ...] adds a rotation axis
# ---------------------------------------------------------------------------

reconstructed_fourier = reconstructor_fourier.reconstruct(sim_input)

print("Foruier reconstruction finished")

widefield_sum            = np.sum(sim_input, axis=(0, 1),) / 5  # (Nl, Nl, Na)
reconstructed_widefield  = reconstructor_widefield.reconstruct(
    widefield_sum[None, None, ...], backend='gpu'                    # (1, 1, Nl, Nl, Na)
)


print("Fourier", np.sum(reconstructed_fourier))
print("Widefield", np.sum(reconstructed_widefield))

print("Widefield reconsrtuction finished")


# ---------------------------------------------------------------------------
# Mid-slice visualisation
# ---------------------------------------------------------------------------
z_mid = Na // 2

fig, ax = plt.subplots(2, 2, figsize=(16, 6))
fig.suptitle('3D Reconstructions – middle z-slice')
for col, (name, img) in enumerate(zip(
        ['Widefield', 'Fourier'],
        [reconstructed_widefield, reconstructed_fourier])):
    ax[0, col].imshow(np.log1p(np.abs(wrapped_fftn(img[:, :, z_mid]))).T, cmap='gray', origin='lower')
    ax[0, col].set_title(name)
    ax[1, col].imshow(img[:, :, z_mid].T, cmap='gray', origin='lower')
    ax[1, col].set_title(name)
plt.show()


# fig, ax, slider =utils.imshow3D(optical_system.otf, vmin=0, vmax=np.amax(np.log1p(10**8)), mode='log1p', scaling=10**8, axis='y')
# ax.set_aspect(1/ax.get_data_ratio())
# plt.show()
# # ---------------------------------------------------------------------------
# SSNR (3D) & Wiener filtering
# ---------------------------------------------------------------------------

ssnr_fourier = SSNRSIM3D(
    illumination,
    optical_system=optical_system,
    readout_noise_variance=5,
)



ssnr_widefield = SSNRSIM3D(
    illumination=illumination_widefield,
    optical_system=optical_system,
    readout_noise_variance=5,
    kernel = kernels.psf_kernel2d(1)[..., None]
)

filtered_fourier_ft, w_fourier, ssnr_fourier_measured = filter_true_wiener_sim(
    wrapped_fftn(reconstructed_fourier), ssnr_fourier,
)


filtered_widefield_ft, w_widefield, ssnr_widefield_measured = filter_true_wiener_sim(
    wrapped_fftn(reconstructed_widefield), ssnr_widefield,
)

from matplotlib.widgets import Slider 

# fig, ax, slider_spatial= utils.imshow3D(ssnr_spatial_measured, mode='log1p')
# plt.show()
# fig, ax, slider_widefield = utils.imshow3D(ssnr_widefield_measured, mode='log1p')
# plt.show()
fig, ax1 = plt.subplots()
ax1.plot(np.log1p(ssnr_fourier_measured[:, Nl, Na//2]), color='blue', label="Fourier")
ax1.plot(np.log1p(ssnr_widefield_measured[:, Nl,  Na//2]), color='green', label='Widefield')
ax1.legend()
def update(val):
    val=int(val)
    ax1.clear()
    ax1.plot(np.log1p(ssnr_fourier_measured[:, Nl, val]), color='blue', label="Fourier")
    ax1.plot(np.log1p(ssnr_widefield_measured[:, Nl, val]), color='green', label="Widefield")
    ax1.legend()


slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
slider1 = Slider(slider_loc, 'z', 0, Na - 1)  # slider properties
slider1.on_changed(update)
plt.show()


slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
slider1 = Slider(slider_loc, 'z', 0, Na - 1)  # slider properties
slider1.on_changed(update)
plt.show()

filtered_fourier = wrapped_ifftn(filtered_fourier_ft).real
filtered_widefield= wrapped_ifftn(filtered_widefield_ft).real

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Filtered reconstructions – middle z-slice')

axes[0].imshow(filtered_widefield[:, :, z_mid].T, cmap='gray', origin='lower')
axes[0].set_title('Widefield')
axes[1].imshow(filtered_fourier[:, :, z_mid].T, cmap='gray', origin='lower')
axes[1].set_title('SIM Fourier')

slider_recon = utils.wrap_axes3d(
    axes,
    [filtered_widefield, filtered_fourier],
    axis='z', cmap='gray',
)
plt.show()
# ---------------------------------------------------------------------------
# Apodization (3D)
# ---------------------------------------------------------------------------
apodization_filter_auto = apodization.ideal_otf
apodized_fourier_auto  = np.abs(wrapped_ifftn(apodization_filter_auto * filtered_fourier_ft))
apodization = TriangularApodizationSIM3D(optical_system, illumination)
apodization_filter1 = apodization.apodization_function
apodized_fourier_triangular1  = np.abs(wrapped_ifftn(apodization_filter1 * filtered_fourier_ft))
apodization.power = 0.4
apodization_filter04 = apodization.apodization_function
apodized_fourier_triangular04  = np.abs(wrapped_ifftn(apodization_filter04 * filtered_fourier_ft))
# apodized_spatial  = np.abs(wrapped_ifftn(apodization_filter * filtered_spatial_ft))
# apodized_widefield = np.abs(wrapped_ifftn(apodization_filter.ideal_otf * filtered_widefield_ft))

y_mid = Nl
# ---------------------------------------------------------------------------
# Final visualisation with z-slider
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(10, 5))
# fig.suptitle('Apodized reconstructions – middle z-slice')
axes[0].imshow(widefield_sum[20:220, 20:220, z_mid].T, cmap='gray', origin='upper')
axes[0].set_title('a')
axes[1].imshow(apodized_fourier_triangular1[20:220, 20:220, z_mid].T, cmap='gray', origin='upper')
axes[1].set_title('b')
axes[2].imshow(apodized_fourier_triangular04[20:220, 20:220, z_mid].T, cmap='gray', origin='upper')
axes[2].set_title('c')
axes[3].imshow(apodized_fourier_auto[20:220, 20:220, z_mid].T, cmap='gray', origin='upper')
axes[3].set_title('d')
axes[0].set_axis_off()
axes[1].set_axis_off()
axes[2].set_axis_off()
axes[3].set_axis_off()
plt.tight_layout()

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
axes[0].add_artist(scalebar)

fig.savefig(current_dir + '/Figures/experiments/actin_3D_apodization_comparison.png', dpi=300, bbox_inches='tight')

# axes[1, 0].imshow(apodized_spatial[:, y_mid, :].T, cmap='gray', origin='lower')
# axes[1, 0].set_title('SIM Spatial')
# axes[1, 0].set_aspect(1/3)

# axes[1, 1].imshow(apodized_widefield[:, y_mid, :].T, cmap='gray', origin='lower')
# axes[1, 1].set_title('Widefield')
# axes[1, 1].set_aspect(1/3)

# axes[1, 2].imshow(apodized_fourier[:, y_mid, : ].T, cmap='gray', origin='lower')
# axes[1, 2].set_title('SIM Fourier')
# axes[1, 2].set_aspect(1/3)


slider_recon = utils.wrap_axes3d(
    axes,
    [widefield_sum[20:220, 20:220, :], apodized_fourier_triangular1[20:220, 20:220, :], apodized_fourier_triangular04[20:220, 20:220, :], apodized_fourier_auto[20:220, 20:220, :]],
    axis='z', cmap='gray',
)
plt.show()
