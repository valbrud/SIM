import tifffile
import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt

import scipy
import numpy as np
from config.BFPConfigurations import BFPConfiguration
import pickle 

configurations = BFPConfiguration(refraction_index=1.5)
from OpticalSystems import System4f3D
from SSNRCalculator import SSNRSIM3D
from kernels import psf_kernel2d, combined_low_pass_notch_kernel
from Reconstructor import ReconstructorSpatialDomain3DSliced, ReconstructorFourierDomain3D
from SIMulator import SIMulator3D
import skimage 
import tifffile
import windowing
import hpc_utils, utils
import Apodization
from WienerFiltering import filter_true_wiener_sim, filter_simulated_object_wiener
from matplotlib.widgets import Slider
# plt.rcParams['font.size'] = 50         # Sets default font size
# plt.rcParams['axes.titlesize'] = 50     # Title of the axes
# plt.rcParams['axes.labelsize'] = 50     # Labels on x and y axes
# plt.rcParams['xtick.labelsize'] = 50    # Font size for x-tick labels
# plt.rcParams['ytick.labelsize'] = 50    # Font size for y-tick labels
# plt.rcParams['legend.fontsize'] = 25    # Font size for legend


np.random.seed(1234)
# Set simulation parameters similar to the provided example.
Nl = 129
Na = 59

alpha = 2 * np.pi / 5
# theta = 2 * np.pi / 12
nmedium = 1.5
theta = np.arcsin(0.9 * np.sin(alpha))
dimensions = (1, 1, 0)
NA = nmedium * np.sin(alpha)
dx = 1 / (8 * NA)
dz = 1 / (4 * nmedium * (1 - np.cos(alpha)))
max_r = Nl // 2 * dx
max_z = Na // 2 * dz
psf_size = np.array((2 * max_r, 2 * max_r, 2 * max_z))

x = np.linspace(-max_r, max_r, Nl, endpoint=Nl%2)
z = np.linspace(-max_z, max_z, Na, endpoint=Na%2)
# print(z)

fx = np.linspace(-4 * NA, 4 * NA, Nl )
fr = fx[Nl//2:]

y = np.copy(x)
N_avg = 10**5
image = skimage.util.img_as_float(skimage.data.cells3d()[:, 1, :, :])
image = image.transpose(1, 2, 0)[:Nl, :Nl, 30 - Na//2:30 + (Na+1)//2]
image-=0.1
image = np.where(image>=0, image, 0)
# plt.hist(image.flatten(), bins=1000, range=(0, np.amax(image)))
# plt.show()
image *= N_avg
print(np.mean(image))
print(image.shape)
# plt.imshow(image[..., Na//2])
# plt.show()

optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nmedium, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Na)))
# plt.imshow(np.log1p(10**8 * optical_system.otf[N//2, :, :]), cmap='gray')
# plt.title("PSF")
# plt.show()
# image = 10**5 * np.ones(optical_system.psf.shape)
# plt.plot(optical_system.otf[Nl//2, :, Na//2])
# plt.show()
# fig, ax, slider =utils.imshow3D(optical_system.otf, vmin=0, vmax=np.amax(np.log1p(10**8)), mode='log1p', scaling=10**8, axis='y')
# plt.show()
widefield_noiseless = scipy.signal.convolve(image, optical_system.psf, mode='same')
# plt.title("Widefield image")
# plt.imshow(widefield)
# plt.show()

configurations = BFPConfiguration(refraction_index=1.5)
illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(
    theta, 1, 1 , 5 , Mt=1, dimensionality=3
)


illumination = illumination_3waves3d
illumination.set_spatial_shifts_diagonally()
print(illumination.angles)

# plt.imshow(illumination.get_illumination_density(coordinates=(x, x, z))[:, :, N//2])
# plt.show()

simulator = SIMulator3D(illumination, optical_system, readout_noise_variance=1)
regenerate = True
if not os.path.exists(current_dir + '/noisy_cells.pkl') or regenerate:
    print("Regenerating SIM images")
    sim_images = simulator.generate_noiseless_sim_images(image, debug=True)
    noisy_images = simulator.add_noise(sim_images)
    pickle.dump(noisy_images, open(current_dir + '/clean_cells.pkl', 'wb'))
    pickle.dump(noisy_images, open(current_dir + '/noisy_cells.pkl', 'wb'))
else:
    # sim_images = pickle.load(open(current_dir + '/clean_cells.pkl', 'rb'))
    # noisy_images = pickle.load(open(current_dir + '/noisy_cells.pkl', 'rb'))
    noisy_images = tifffile.imread(current_dir+'\\simulated_cell.tiff')
cut_off_frequency_l = 1 / (4 * dx)
import copy

# tifffile.imwrite("simulated_cell.tiff", noisy_images)

illumination_reconstruction_unmodulated = copy.deepcopy(illumination).project_in_quasi_2D()
illumination_reconstruction_modulated = copy.deepcopy(illumination_reconstruction_unmodulated)
illumination_ssnr_modulated = copy.deepcopy(illumination)
# illumination_reconstruction = IlluminationPlaneWaves2D.init_from_3D(illumination_reconstruction)
m1, m2 = 10, 5
for r in range(illumination_reconstruction_unmodulated.Mr):
    illumination_reconstruction_modulated.harmonics[(r, (1, 0, 0))].amplitude *= m1
    illumination_reconstruction_modulated.harmonics[(r, (-1, 0, 0))].amplitude *= m1
    illumination_reconstruction_modulated.harmonics[(r, (2, 0, 0))].amplitude *= m2
    illumination_reconstruction_modulated.harmonics[(r, (-2, 0, 0))].amplitude *= m2

for r in range(illumination.Mr):
    illumination_ssnr_modulated.harmonics[(r, (1, 0, 1))].amplitude *= m1
    illumination_ssnr_modulated.harmonics[(r, (-1, 0, -1))].amplitude *= m1
    illumination_ssnr_modulated.harmonics[(r, (1, 0, -1))].amplitude *= m1
    illumination_ssnr_modulated.harmonics[(r, (-1, 0, 1))].amplitude *= m1
    illumination_ssnr_modulated.harmonics[(r, (2, 0, 0))].amplitude *= m2
    illumination_ssnr_modulated.harmonics[(r, (-2, 0, 0))].amplitude *= m2

# kernel = psf_kernel2d(9)
kernel=combined_low_pass_notch_kernel(pixel_size=(dx, dx), first_zero_frequency_low_pass=cut_off_frequency_l, first_zero_frequency_notch=cut_off_frequency_l)
# kernel06=combined_low_pass_notch_kernel(pixel_size=(dx, dx), first_zero_frequency_low_pass=cut_off_frequency_l, first_zero_frequency_notch=0.6 * cut_off_frequency_l)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(kernel)    
# axes[1].imshow(kernel06)
# plt.show()
# plt.plot(kernel06[kernel06.shape[0]//2, :], label='11')
# plt.plot(kernel[kernel.shape[0]//2, :])
# plt.legend()
# plt.show()
# notch_kernel = scipy.signal.convolve2d(kernel, (1 - kernel), mode='full')

illumination_widefield = BFPConfiguration(refraction_index=1.5).get_widefield(3)

apodization = Apodization.AutoconvolutionApodizationSIM3D(optical_system, illumination)
apodization_filter = apodization.ideal_otf
apodization_widefield = Apodization.AutoconvolutionApodizationSIM3D(optical_system, illumination_widefield, plane_wave_wavevectors=[np.array((0, 0, 0))])

optical_system.otf = optical_system.otf * np.where(apodization_widefield.ideal_otf > 10**-3, 1, 0)

finite_reconstructor_unmodulated = ReconstructorSpatialDomain3DSliced(
    # illumination=illumination, 
    illumination=illumination_reconstruction_unmodulated,
    optical_system=optical_system,
    kernel=kernel, 
)

finite_reconstructor_modulated = ReconstructorSpatialDomain3DSliced(
    # illumination=illumination, 
    illumination=illumination_reconstruction_modulated,
    optical_system=optical_system,
    kernel=kernel, 
)


reconstructor_widefield = ReconstructorFourierDomain3D(
    illumination=illumination_widefield,
    optical_system=optical_system,
)

reconstructor_ideal = ReconstructorFourierDomain3D(
    illumination, 
    optical_system, 
    unitary=True
)

# Reconstruct the image.
# sim_images_slices = noisy_images[:, :, :, : N//2]
mask = windowing.make_mask_cosine_edge3d((Nl, Nl, Na), edge=20)
# sim_images *= mask[None, None, :, :, :]
noisy_images *= mask[None, None, :, :, :]
# plt.imshow(noisy_images[0, 0, ..., Na//2])
# plt.show()


widefield = np.sum(noisy_images, axis=(0,1))

# fix, ax, slider = utils.imshow3D(apodization_filter, mode='log1p', scaling=10**4, axis='y')
# ax.set_aspect(1/ax.get_data_ratio())
# plt.show()
# reconstructed_image = finite_reconstructor.reconstruct(noisy_images) 
reconstructed_image_unmodulated = finite_reconstructor_unmodulated.reconstruct(noisy_images, backend='gpu') 
reconstructed_image_modulated = finite_reconstructor_modulated.reconstruct(noisy_images, backend='gpu') 
reconstructed_image_unmodulated_ft = hpc_utils.wrapped_fftn(reconstructed_image_unmodulated)
reconstructed_image_modulated_ft = hpc_utils.wrapped_fftn(reconstructed_image_modulated)
plt.imshow(np.log1p(np.abs(reconstructed_image_unmodulated_ft[:, :, 29])))
plt.gca().set_aspect(1/plt.gca().get_data_ratio())
plt.title("Reconstructed image real")
plt.show()

# plt.imshow(np.where(np.abs(apodization_filter) > 10**-6, 1, 0)[:, :, N//2])
# plt.show()

ssnr_calc_unmodulated = SSNRSIM3D(   
    illumination = illumination, 
    optical_system = optical_system,
    kernel=kernel[..., None],
    illumination_reconstruction=illumination_reconstruction_unmodulated
    )

ssnr_calc_modulated = SSNRSIM3D(   
    illumination = illumination, 
    optical_system = optical_system,
    kernel=kernel[..., None],
    illumination_reconstruction=illumination_reconstruction_modulated
    )

ssnr_calc_widefield = SSNRSIM3D(
    illumination=illumination_widefield,
    optical_system=optical_system,
    )

ssnr_calc_ideal = SSNRSIM3D(   
    illumination = illumination, 
    optical_system = optical_system,
    )

# slider = ssnr_calc_ideal.plot_effective_kernel_and_otf((0, (1, 0, 0)))
# plt.show()

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

widefield = reconstructor_widefield.reconstruct(widefield[None, None, :, :, :])
fig, ax = plt.subplots(figsize=(12,10))
ax.set_xlabel('$f_r [LCF]$')
ax.set_ylabel('$f_z [ACF]$')
ax.set_aspect('equal', adjustable='box')
im0 = ax.imshow(np.real(ssnr_calc_unmodulated.ring_average_ssnri() / ssnr_calc_ideal.ring_average_ssnri()).T, extent=(0, 2, -2, 2), origin='lower', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im0, ax=ax, extend='both', shrink=0.8, label='$SSNR_{finite, ra} / SSNR_{ideal, ra}$')
fig.savefig(current_dir + f"/Figures/SSNR_ratio_3D_unmodulated_notch_{1}_{1}.png", bbox_inches='tight')

fig, ax = plt.subplots(figsize=(12,10))
ax.set_xlabel('$f_r [LCF]$')
ax.set_ylabel('$f_z [ACF]$')
ax.set_aspect('equal', adjustable='box')
im0 = ax.imshow(np.real(ssnr_calc_modulated.ring_average_ssnri() / ssnr_calc_ideal.ring_average_ssnri()).T, extent=(0, 2, -2, 2), origin='lower', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im0, ax=ax, extend='both', shrink=0.8, label='$SSNR_{finite, ra} / SSNR_{ideal, ra}$')
fig.savefig(current_dir + f"/Figures/SSNR_ratio_3D_modulated_notch_{10}_{5}.png", bbox_inches='tight')

ideal = reconstructor_ideal.reconstruct(noisy_images)
ideal_ft = hpc_utils.wrapped_fftn(ideal)
plt.imshow(np.log1p(np.abs(ideal_ft[:, :, 29])))
plt.gca().set_aspect(1/plt.gca().get_data_ratio())
plt.title("Reconstructed image ideal")
plt.show()

# reconstructed_image_ft *= np.where(np.abs(apodization_filter) > 10**-6, 1, 0)
widefield_ft = hpc_utils.wrapped_fftn(widefield) * np.where(np.abs(apodization_widefield.ideal_otf) > 10**-6, 1, 0)
# plt.imshow(np.where(np.abs(apodization_widefield.ideal_otf) > 10**-6, 1, 0)[:, N//2, :])
# plt.show()
image_ft = hpc_utils.wrapped_fftn(image)

filtered_unmodulated_ft, _, ssnr_unmodulated = filter_true_wiener_sim(
    reconstructed_image_unmodulated_ft, ssnr_calc_unmodulated,
    # image_ft
     )

filtered_modulated_ft, _, ssnr_modulated = filter_true_wiener_sim(
    reconstructed_image_modulated_ft, ssnr_calc_modulated,
    # image_ft
     )

filtered_widefield_ft, _, ssnr_widefield = filter_true_wiener_sim(
    widefield_ft, ssnr_calc_widefield,
    #  image_ft
     )

filtered_ideal_ft, _, ssnr_ideal = filter_true_wiener_sim(
    ideal_ft, ssnr_calc_ideal, 
    # image_ft
) 

# ssnr_unmodulated = ssnr_calc_unmodulated.ssnr_like_sectorial_average_from_image(reconstructed_image_unmodulated)
# ssnr_modulated = ssnr_calc_unmodulated.ssnr_like_sectorial_average_from_image(reconstructed_image_modulated)
# ssnr_ideal = ssnr_calc_unmodulated.ssnr_like_sectorial_average_from_image(reconstructed_image_ideal)

fig, ax1 = plt.subplots()
ax1.plot(np.log1p(ssnr_unmodulated[:, Nl//2, Na//2]), label="Unmodulated")
ax1.plot(np.log1p(ssnr_modulated[:, Nl//2, Na//2]), label="Modulated")
ax1.plot(np.log1p(ssnr_ideal[:, Nl//2,  Na//2]), label='Ideal')
ax1.legend()
def update(val):
    val=int(val)
    ax1.clear()
    ax1.plot(np.log1p(ssnr_unmodulated[:, Nl//2, val]), label="Unmodulated")
    ax1.plot(np.log1p(ssnr_modulated[:, Nl//2, val]), label="Modulated")
    ax1.plot(np.log1p(ssnr_ideal[:, Nl//2,  val]), label='Ideal')
    ax1.legend()

slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
slider1 = Slider(slider_loc, 'z', 0, Na - 1)  # slider properties
slider1.on_changed(update)

fig, ax = plt.subplots()
ssnr_theoretical_unmodulated = utils.average_rings3d(np.abs(ssnr_calc_unmodulated.dj * image_ft)**2) / utils.average_rings3d(np.abs(ssnr_calc_unmodulated.vj * np.amax(image_ft).real))
print(ssnr_theoretical_unmodulated.shape)
ssnr_theoretical_modulated = utils.average_rings3d(np.abs(ssnr_calc_modulated.dj * image_ft)**2) / utils.average_rings3d(np.abs(ssnr_calc_modulated.vj * np.amax(image_ft).real))
ssnr_theoretical_ideal = utils.average_rings3d(np.abs(ssnr_calc_ideal.dj * image_ft)**2) / utils.average_rings3d(np.abs(ssnr_calc_ideal.vj * np.amax(image_ft).real))

ax.plot(np.log1p(ssnr_theoretical_unmodulated[:, Na//2]), label="Unmodulated")
ax.plot(np.log1p(ssnr_theoretical_modulated[:, Na//2]), label="Modulated")
ax.plot(np.log1p(ssnr_theoretical_ideal[:, Na//2]), label='Ideal')
ax.legend()
def update3(val):
    val=int(val)
    ax.clear()
    ax.plot(np.log1p(ssnr_theoretical_unmodulated[:, val]), label="Unmodulated")
    ax.plot(np.log1p(ssnr_theoretical_modulated[:, val]), label="Modulated")
    ax.plot(np.log1p(ssnr_theoretical_ideal[:, val]), label='Ideal')
    ax.legend()

slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
slider = Slider(slider_loc, 'z', 0, Na - 1)  # slider properties
slider.on_changed(update3)

# fx, fy, fz = optical_system.otf_frequencies
# FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
# filtered_widefield_ft *= np.where(FX**2 + FY**2 < 1.5 * 1 / (2 * dx), 1, 0)
# filtered_widefield_ft *= np.where(FZ**2 < 1.5 * 1 / (2 * dz), 1, 0)
# filtered, _ = filter_constant(reconstructed_image_ft, ssnr_calc.dj, w=1e-6)
apodized_image_unmodulated_ft = filtered_unmodulated_ft * apodization_filter
apodized_image_modulated_ft = filtered_modulated_ft * apodization_filter
apodized_ideal_ft = filtered_ideal_ft * apodization_filter
apodized_widefield_ft = filtered_widefield_ft * apodization_widefield.ideal_otf

# apodized_image_unmodulated_ft[Nl//2, Nl//2, :Na//2] = 0; apodized_image_unmodulated_ft[Nl//2, Nl//2, Na//2 + 1:] = 0
# apodized_image_modulated_ft[Nl//2, Nl//2, :Na//2] = 0;   apodized_image_modulated_ft[Nl//2, Nl//2, Na//2 + 1:] = 0
# apodized_ideal_ft[Nl//2, Nl//2, :Na//2] = 0; apodized_ideal_ft[Nl//2, Nl//2, Na//2 + 1:] = 0

apodized_image_unmodulated = hpc_utils.wrapped_ifftn(apodized_image_unmodulated_ft).real
apodized_image_modulated = hpc_utils.wrapped_ifftn(apodized_image_modulated_ft).real
apodized_widefield = hpc_utils.wrapped_ifftn(apodized_widefield_ft).real

apodized_ideal = hpc_utils.wrapped_ifftn(apodized_ideal_ft).real

image_ft_ra = np.abs(utils.average_rings3d(hpc_utils.wrapped_fftn(image), (x, x, z)))
reconstructed_image_ft_ra =np.abs(utils.average_rings3d(reconstructed_image_unmodulated_ft, (x, x, z)))
filtered_ft_ra = np.abs(utils.average_rings3d(filtered_unmodulated_ft, (x, x, z)))
widefield_ft_ra = np.abs(utils.average_rings3d(apodized_widefield_ft, (x, x, z)))

# --- reconstructed-image comparison (2×3, z-slider) ---
fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(image[:, :, Na//2], cmap='gray')
axes[0, 0].set_title("Ground truth")
axes[0, 1].imshow(apodized_ideal[:, :, Na//2], cmap='gray')
axes[0, 1].set_title("Ideal")
axes[1, 0].imshow(apodized_image_unmodulated[:, :, Na//2], cmap='gray')
axes[1, 0].set_title("Unmodulated")
axes[1, 1].imshow(apodized_image_modulated[:, :, Na//2], cmap='gray')
axes[1, 1].set_title("Modulated")
axes[1, 2].imshow(apodized_widefield[:, :, Na//2], cmap='gray')
axes[1, 2].set_title("Filtered Widefield")
# axes[0, 2] is unused – pass None to skip it
slider_recon = utils.wrap_axes3d(
    axes,
    [image, apodized_ideal, None,
     apodized_image_unmodulated, apodized_image_modulated, apodized_widefield],
    axis='z', cmap='gray',
)

plt.show()

# # --- filter-response ratio plots (2×2, z-slider) ---
# ratio_ideal = filtered_ideal_ft / filtered_unmodulated_ft
# ratio_unmodulated = filtered_unmodulated_ft / filtered_unmodulated_ft
# ratio_modulated = filtered_modulated_ft / filtered_unmodulated_ft
# ratio_widefield = filtered_widefield_ft / filtered_unmodulated_ft

# fig, axes = plt.subplots(2, 2)
# axes[0, 0].imshow(np.abs(ratio_ideal[..., Na//2]), cmap='gray', vmin=0, vmax=5)
# axes[0, 0].set_title("Ideal FT")
# axes[0, 1].imshow(np.abs(ratio_unmodulated[..., Na//2]), cmap='gray', vmin=0, vmax=5)
# axes[0, 1].set_title("Unmodulated FT")
# axes[1, 0].imshow(np.abs(ratio_modulated[..., Na//2]), cmap='gray', vmin=0, vmax=5)
# axes[1, 0].set_title("Modulated FT")
# axes[1, 1].imshow(np.abs(ratio_widefield[..., Na//2]), cmap='gray', vmin=0, vmax=5)
# axes[1, 1].set_title("Filtered Widefield FT")

# slider_ratio = utils.wrap_axes3d(
#     axes,
#     [ratio_ideal, ratio_unmodulated, ratio_modulated, ratio_widefield],
#     mode='abs', axis='z',
# )
# plt.show()

# fig, axes = plt.subplots(1, 5)
# axes[0].imshow(widefield[:, :, Na//2], cmap='gray')
# axes[1].imshow(reconstructed_image[:, :, Na//2], cmap='gray')
# axes[2].imshow(np.abs(apodized_image[:, :, Na//2]), cmap='gray')
# axes[3].imshow(image[:, :, Na//2], cmap='gray')
# axes[4].imshow(np.abs(apodized_widefield[:, :, Na//2]), cmap='gray')
# plt.show()
