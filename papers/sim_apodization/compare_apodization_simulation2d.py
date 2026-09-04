import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
sys.path.append(current_dir)

from kernels import psf_kernel2d

from config.BFPConfigurations import BFPConfiguration
from Illumination import IlluminationPlaneWaves2D
from OpticalSystems import System4f2D, System4f3D
import utils
from SSNRCalculator import SSNRSIM2D
import copy

from SIMulator import SIMulator2D
import copy
import ShapesGenerator
import Reconstructor
from WienerFiltering import filter_true_wiener_sim
import hpc_utils
import Apodization
import skimage
from matplotlib_scalebar.scalebar import ScaleBar

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 20     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
NA = 1.5 * np.sin(alpha)
N = 301
dx = 1 / (12 * NA)
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

illumination = configurations.get_4_circular_oblique_waves_and_circular_normal(alpha * 0.9, 1, 0,  Mt=3, dimensionality=2, dimensions=(1, 1, 0))
illumination.set_spatial_shifts_diagonally()

noise_estimator= SSNRSIM2D(illumination, optical_system, imperfect_phase_shifts=False)
# noise_estimator= SSNRSIM2D(illumination, optical_system, kernel=psf_kernel2d(9), imperfect_phase_shifts=False)
# plt.imshow(np.log1p(10**3 * noise_estimator.ssnri))
# plt.show()

##################

path = current_dir + "\\image_2d.jpg"
image = utils.load_grayscale_image(path)
image = image[:N, 100:100+N]* 10**4


# image = skimage.util.img_as_float(skimage.data.cells3d()[30, 1, :, :])
# image -= np.min(image)
# image /= np.max(image)

# # image = np.ones_like(image) * 10**2
# print(image.shape)
# plt.imshow(image, cmap='gray')
# plt.show()
image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, 0.2, 500, 1000, generate_default=True) + ShapesGenerator.generate_random_lines(psf_size, N, 0.2, 200, 200, generate_default=True)
# fig, ax = plt.subplots()
# ax.imshow(image)
# ax.set_axis_off()
# fig.savefig(current_dir + f"/Figures/simulations/line_object.png", bbox_inches='tight', pad_inches=0.01)

print('average photon count per pixel = ', np.sum(image)/N**2)

apodization_autoconvolution = Apodization.AutocorrelationApodizationSIM2D(optical_system, illumination)
apodization_triangular1 = Apodization.TriangularApodizationSIM2D(optical_system, illumination)
apodization_triangular04 = Apodization.TriangularApodizationSIM2D(optical_system, illumination, power=0.4)


# effective_image_auto = scipy.signal.convolve2d(image, apodization_autoconvolution.ideal_psf, mode='same')
upper_bound = apodization_autoconvolution.ideal_otf * np.sum(image)
image_ft = hpc_utils.wrapped_fftn(image)
# real_bounded = np.where(image_ft.real < np.abs(upper_bound)/2, image_ft.real, upper_bound/2)
# imag_bounded = np.where(image_ft.imag < np.abs(upper_bound)/2, image_ft.imag, upper_bound/2)
# bounded = real_bounded + 1j * imag_bounded
bounded = np.where(np.abs(image_ft) < np.abs(upper_bound), image_ft, upper_bound * image_ft / np.abs(image_ft))

bound = utils.average_rings2d(upper_bound, (fx, fx))
image_ft_ra = utils.average_rings2d(np.abs(image_ft), (fx, fx))
bounded_ra = utils.average_rings2d(np.abs(bounded), (fx, fx))

# plt.plot(fr, np.log1p(bound))
# plt.plot(fr, np.log1p(image_ft_ra))
# plt.plot(fr, np.log1p(bounded_ra))
# plt.show()

psf1 = hpc_utils.wrapped_ifftn(apodization_triangular1.apodization_function).real
psf04 = hpc_utils.wrapped_ifftn(apodization_triangular04.apodization_function).real
effective_image_triangular1= scipy.signal.convolve2d(image, psf1, mode='same')
effective_image_triangular04 = scipy.signal.convolve2d(image, psf04, mode='same')
effective_image_bound = hpc_utils.wrapped_ifftn(bounded).real
effective_image_auto = scipy.signal.convolve2d(image, apodization_autoconvolution.ideal_psf, mode='same')
print(np.amin(effective_image_auto))
print(np.amin(effective_image_triangular1))
print(np.amin(effective_image_bound))
# fig, ax = plt.subplots(1, 3)
# ax[0].imshow(psf1)  
# ax[0].set_title('p=1')
# ax[1].imshow(psf04)
# ax[1].set_title('p=0.4')
# ax[2].imshow(apodization_autoconvolution.ideal_psf)
# ax[2].set_title('Autoconvolution')
# ax[0].set_axis_off()
# ax[1].set_axis_off()
# ax[2].set_axis_off()
# plt.show()

scalebar = ScaleBar(
    dx= dx*488,            # size of one pixel in physical units (e.g., µm/px)
    units='nm',         # 'um', 'nm', 'mm', etc.
    length_fraction=0.3,
    location='lower left',
    color='white',
    box_alpha=0,        # transparent background
    label_loc='top',
    font_properties={'size': 15}
    )
    


fig, ax = plt.subplots(1, 4, figsize=(20, 10))
ax[0].set_axis_off()
ax[0].add_artist(scalebar)
ax[1].set_axis_off()
ax[2].set_axis_off()
ax[3].set_axis_off()
ax[0].set_title("a")
ax[1].set_title("b")
ax[2].set_title("c")
ax[3].set_title("d")
ax[0].imshow(image[100:200, 180:280], cmap='gray')
ax[1].imshow(effective_image_triangular1[100:200, 180:280], cmap='gray')
ax[2].imshow(effective_image_triangular04[100:200, 180:280], cmap='gray')
ax[3].imshow(effective_image_auto[100:200, 180:280], cmap='gray')
fig.subplots_adjust(wspace=0.02)
# fig.savefig(current_dir + f"/Figures/simulations/compare_apodization_2d_cat.png", bbox_inches='tight', dpi=300, pad_inches=0.01)
fig.savefig(current_dir + f"/Figures/simulations/compare_apodization_2d_spherical.png", bbox_inches='tight', dpi=300, pad_inches=0.01)
plt.show()

quit()

simulator = SIMulator2D(illumination, optical_system)
noiseless_images = simulator.generate_noiseless_sim_images(image)
noisy_images = simulator.add_noise(noiseless_images)

widefield_image = np.sum(noisy_images, axis=(0, 1))
from windowing import make_mask_cosine_edge2d
mask = make_mask_cosine_edge2d(image.shape, 30)
noisy_images *= mask[None, None, ...]
reconstructor = Reconstructor.ReconstructorFourierDomain2D(illumination, optical_system, return_ft=False, unitary=True)
# reconstructor = Reconstructor.ReconstructorSpatialDomain2D(illumination, optical_system, kernel=psf_kernel2d(9), unitary=False)

# reconstructed_perfect_ft = reconstructor.reconstruct(noisy_images)
reconstructed_perfect = reconstructor.reconstruct(noisy_images)
reconstructed_perfect_ft = hpc_utils.wrapped_fftn(reconstructed_perfect)
plt.imshow(np.log1p(np.abs(reconstructed_perfect_ft)))
plt.imshow(reconstructed_perfect, cmap='gray')
plt.show()

filtered_ft, _,ssnr = filter_true_wiener_sim(reconstructed_perfect_ft, noise_estimator)
# plt.imshow(np.log1p(np.abs(ssnr)))
# plt.show()

image_ft = hpc_utils.wrapped_fftn(image)

ssnr_theory = utils.average_rings2d(noise_estimator.dj**2 * np.abs(image_ft)**2, (fx, fx) )/utils.average_rings2d(noise_estimator.vj, (fx, fx)) / np.amax(np.abs(image_ft))
filtered, _, ssnr = filter_true_wiener_sim(reconstructed_perfect_ft, noise_estimator)
plt.plot(np.log1p(ssnr_theory), label='theory')
plt.plot(np.log1p(ssnr[N//2, N//2:]), label='measured')
plt.legend()
plt.show()
plt.plot(ssnr_theory/ssnr[N//2, N//2:])
plt.show()

apodized_autoconvolution_ft = filtered_ft * apodization_autoconvolution.apodization_function
apodized_autoconvolution = hpc_utils.wrapped_ifftn(apodized_autoconvolution_ft).real
apodized_triangular1_ft = filtered_ft * apodization_triangular1.apodization_function
apodized_triangular1 = hpc_utils.wrapped_ifftn(apodized_triangular1_ft).real
apodized_triangular04_ft = filtered_ft * apodization_triangular04.apodization_function
apodized_triangular04 = hpc_utils.wrapped_ifftn(apodized_triangular04_ft).real

# scalebar = ScaleBar(
#     dx= dx*488,           
#     length_fraction=0.3,
#     location='lower left',
#     color='white',
#     box_alpha=0,
#     label_loc='top',
#     font_properties={'size': 25}
#     )
    
fig, ax = plt.subplots(1, 4, figsize=(20, 10))
ax[0].set_axis_off()
ax[1].set_axis_off()
ax[2].set_axis_off()
ax[3].set_axis_off()
ax[0].set_title("i")
ax[1].set_title("j")
ax[2].set_title("k")
ax[3].set_title("l")
ax[0].imshow(widefield_image[100:200, 180:280], cmap='gray')
ax[1].imshow(apodized_triangular1[100:200, 180:280], cmap='gray')
ax[2].imshow(apodized_triangular04[100:200, 180:280], cmap='gray')
ax[3].imshow(apodized_autoconvolution[100:200, 180:280], cmap='gray')
fig.subplots_adjust(wspace=0.02)
fig.savefig(current_dir + f"/Figures/simulations/compare_apodization_2d_cat_reconstructed.png", bbox_inches='tight', dpi=300, pad_inches=0.01)
plt.show()