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
from matplotlib_scalebar.scalebar import ScaleBar
np.random.seed(1234)
# Set simulation parameters similar to the provided example.
Nl = 129
Na = 51

alpha = 2 * np.pi / 5
# theta = 2 * np.pi / 12
nmedium = 1.5
theta = np.arcsin(0.9 * np.sin(alpha))
dimensions = (1, 1, 0)
NA = nmedium * np.sin(alpha)
dx = 1 / (8 * NA) / 3
dz = 1 / (4 * nmedium * (1 - np.cos(alpha))) / 3
# dz = dx
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
image_sparse = skimage.util.img_as_float(skimage.data.cells3d()[:, 0, :, :])
image_sparse = image_sparse.transpose(1, 2, 0)[:Nl, :Nl, 30 - Na//2:30 + (Na+1)//2]
image_sparse -= np.average(image_sparse)  
# image_sparse-=0.1
image_sparse = np.where(image_sparse>=0, image_sparse, 0)
image_sparse *= N_avg
print(np.mean(image_sparse), np.amax(image_sparse))
print(image_sparse.shape)

scalebar_1l = ScaleBar(
    dx= dx*488,  
    units='nm',        
    length_fraction=0.3,
    location='lower left',
    color='black',
    box_alpha=0,       
    scale_loc='bottom',
    font_properties={'size': 15}
    )

scalebar_1a = ScaleBar(
    dx= dz*488,  
    units='nm',        
    length_fraction=0.3,
    rotation='vertical',
    location='upper left',
    color='black',
    box_alpha=0,       
    scale_loc='right',
    font_properties={'size': 15}, 

    )

scalebar_2l = ScaleBar(
    dx= dx*488,  
    units='nm',        
    length_fraction=0.3,
    location='lower left',
    color='black',
    box_alpha=0, 
    scale_loc='bottom',      
    font_properties={'size': 15}
    )

scalebar_2a = ScaleBar(
    dx= dz*488,  
    units='nm',        
    length_fraction=0.3,
    location='upper left',
    rotation='vertical',
    color='black',
    box_alpha=0,       
    scale_loc='right',
    font_properties={'size': 15}
    )

image_dense = skimage.util.img_as_float(skimage.data.cells3d()[:, 1, :, :])
image_dense = image_dense.transpose(1, 2, 0)[:Nl, :Nl, 30 - Na//2:30 + (Na+1)//2]
image_dense -= np.average(image_dense)
# image_dense-=0.1
image_dense = np.where(image_dense>=0, image_dense, 0)
image_dense *= N_avg
print(np.mean(image_dense), np.amax(image_dense))
print(image_sparse.shape)


# plt.imshow(image_sparse[..., Na//2])
# plt.show()

optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nmedium, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Na)))
optical_system_quasi = System4f3D(alpha=np.pi/2, refractive_index_medium=nmedium, refractive_index_sample=nmedium, vectorial=True)
optical_system_quasi.compute_psf_and_otf((psf_size, (Nl, Nl, Na)))

widefield_noiseless_sparse = scipy.signal.convolve(image_sparse, optical_system.psf, mode='same')
widefield_noiseless_dense = scipy.signal.convolve(image_dense, optical_system.psf, mode='same')
# plt.title("Widefield image")
# plt.imshow(widefield)
# plt.show()

configurations = BFPConfiguration(refraction_index=1.5)
illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(
    theta, 1, 1, Mr=3, Mt=1, dimensionality=3
)

illumination = illumination_3waves3d
illumination.set_spatial_shifts_diagonally()
print(illumination.angles)

apodization_widefield = Apodization.AutocorrelationApodizationWidefield(optical_system)
optical_system.otf = optical_system.otf * np.where(apodization_widefield.ideal_otf > 10**-3, 1, 0)

apodization_triangular = Apodization.TriangularApodizationSIM3D(optical_system, illumination, power=1, Ndense=201)
apodization_filter1 = apodization_triangular.apodization_function
apodization_triangular.power = 0.4
apodization_filter04 = apodization_triangular.apodization_function

apodization_autoconvolution = Apodization.AutocorrelationApodizationSIM3D(optical_system, illumination, Ndense=201)
apodization_filter_auto = apodization_autoconvolution.ideal_otf
apodization_filter_auto = np.where(apodization_filter1 > 0, apodization_filter_auto, 0)

apodization_ISM = Apodization.AutocorrelationApodizationISM(optical_system_quasi, optical_system)
apodization_filter_ISM = apodization_ISM.ideal_otf
apodization_filter_ISM = np.where(apodization_filter1 > 0, apodization_filter_ISM, 0)

epsf_triangular1 = hpc_utils.wrapped_ifftn(apodization_filter1).real
epsf_triangular04 = hpc_utils.wrapped_ifftn(apodization_filter04).real
epsf_autoconvolution = hpc_utils.wrapped_ifftn(apodization_filter_auto).real
epsf_ISM = hpc_utils.wrapped_ifftn(apodization_filter_ISM).real

image_sparse1 = scipy.signal.convolve(image_sparse, epsf_triangular1, mode='same')
image_sparse04 = scipy.signal.convolve(image_sparse, epsf_triangular04, mode='same')
image_sparse_auto = scipy.signal.convolve(image_sparse, epsf_autoconvolution, mode='same')
image_sparse_ISM = scipy.signal.convolve(image_sparse, epsf_ISM, mode='same')

image_dense1 = scipy.signal.convolve(image_dense, epsf_triangular1, mode='same')
image_dense04 = scipy.signal.convolve(image_dense, epsf_triangular04, mode='same')
image_dense_auto = scipy.signal.convolve(image_dense, epsf_autoconvolution, mode='same')
image_dense_ISM = scipy.signal.convolve(image_dense, epsf_ISM, mode='same')

fig, axes = plt.subplots(2, 4, figsize = (16, 8))
axes[0, 0].imshow(image_dense[Nl//2, :, :].T, cmap='Reds', extent=[x[0], x[-1], z[0], z[-1]])
axes[0, 0].add_artist(scalebar_1l)
axes[0, 0].add_artist(scalebar_1a)
axes[0, 1].imshow(image_dense1[Nl//2, :, :].T, cmap='Reds', extent=[x[0], x[-1], z[0], z[-1]])
axes[0, 2].imshow(image_dense04[Nl//2, :, :].T, cmap='Reds', extent=[x[0], x[-1], z[0], z[-1]])
axes[0, 3].imshow(image_dense_ISM[Nl//2, :, :].T, cmap='Reds', extent=[x[0], x[-1], z[0], z[-1]])
axes[1, 0].imshow(image_sparse[Nl//2, :, :].T, cmap='Greens', extent=[x[0], x[-1], z[0], z[-1]])
axes[1, 0].add_artist(scalebar_2l)
axes[1, 0].add_artist(scalebar_2a)
axes[1, 1].imshow(image_sparse1[Nl//2, :, :].T, cmap='Greens', extent=[x[0], x[-1], z[0], z[-1]])
axes[1, 2].imshow(image_sparse04[Nl//2, :, :].T, cmap='Greens', extent=[x[0], x[-1], z[0], z[-1]])
axes[1, 3].imshow(image_sparse_ISM[Nl//2, :, :].T, cmap='Greens', extent=[x[0], x[-1], z[0], z[-1]])

# axes[0, 0].set_title("a")
# axes[0, 1].set_title("b")
# axes[0, 2].set_title("c")
# axes[0, 3].set_title("d")
# axes[1, 0].set_title("e")
# axes[1, 1].set_title("f")
# axes[1, 2].set_title("g")
# axes[1, 3].set_title("h")
axes[0, 0].set_title("i")
axes[0, 1].set_title("j")
axes[0, 2].set_title("k")
axes[0, 3].set_title("l")
axes[1, 0].set_title("m")
axes[1, 1].set_title("n")
axes[1, 2].set_title("o")
axes[1, 3].set_title("p")

axes[0, 0].set_axis_off()
axes[0, 1].set_axis_off()
axes[0, 2].set_axis_off()
axes[0, 3].set_axis_off()
axes[1, 0].set_axis_off()
axes[1, 1].set_axis_off()
axes[1, 2].set_axis_off()
axes[1, 3].set_axis_off()

# axes[0, 0].set_aspect(dz/dx)
# axes[0, 1].set_aspect(dz/dx)
# axes[0, 2].set_aspect(dz/dx)
# axes[0, 3].set_aspect(dz/dx)
# axes[1, 0].set_aspect(dz/dx)
# axes[1, 1].set_aspect(dz/dx)
# axes[1, 2].set_aspect(dz/dx)
# axes[1, 3].set_aspect(dz/dx)

plt.tight_layout()
fig.savefig(os.path.join(current_dir, "Figures/simulations/effective_apodization_cell_3d_oversampled.png"), bbox_inches='tight')
# fig.savefig(os.path.join(current_dir, "Figures/simulations/effective_apodization_cell_3d_Nyquist.png"), bbox_inches='tight')

slider_recon = utils.wrap_axes3d(
    axes,
    [np.transpose(image_dense, (0, 2, 1)), np.transpose(image_dense1, (0, 2, 1)),
     np.transpose(image_dense04, (0, 2, 1)), np.transpose(image_dense_ISM, (0, 2, 1)), 
     np.transpose(image_sparse, (0, 2, 1)), np.transpose(image_sparse1, (0, 2, 1)),
     np.transpose(image_sparse04, (0, 2, 1)), np.transpose(image_sparse_ISM, (0, 2, 1))],
    axis='x',
)

plt.show()
quit()

# plt.imshow(illumination.get_illumination_density(coordinates=(x, x, z))[:, :, N//2])
# plt.show()
simulator = SIMulator3D(illumination, optical_system, readout_noise_variance=1)
regenerate = True
if not os.path.exists(current_dir + '/noisy_sparse.pkl') or regenerate:
    print("Regenerating SIM images")
    sim_images = simulator.generate_noiseless_sim_images(image_sparse, debug=True)
    noisy_images = simulator.add_noise(sim_images)
    pickle.dump(noisy_images, open(current_dir + '/clean_sparse.pkl', 'wb'))
    pickle.dump(noisy_images, open(current_dir + '/noisy_sparse.pkl', 'wb'))
else:
    sim_images = pickle.load(open(current_dir + '/clean_sparse.pkl', 'rb'))
    noisy_images = pickle.load(open(current_dir + '/noisy_sparse.pkl', 'rb'))
    # noisy_images = tifffile.imread(current_dir+'\\simulated_cell.tiff')


cut_off_frequency_l = 1 / (4 * dx)
import copy

mask = windowing.make_mask_cosine_edge2d((Nl, Nl), 30)

noisy_images*= mask[None, None, ..., None]
illumination_reconstruction = illumination.project_in_quasi_2D()

m1, m2 = 10, 5
for r in range(illumination_reconstruction.Mr):
    illumination_reconstruction.harmonics[(r, (1, 0, 0))].amplitude *= m1
    illumination_reconstruction.harmonics[(r, (-1, 0, 0))].amplitude *= m1
    illumination_reconstruction.harmonics[(r, (2, 0, 0))].amplitude *= m2
    illumination_reconstruction.harmonics[(r, (-2, 0, 0))].amplitude *= m2


kernel=combined_low_pass_notch_kernel(pixel_size=(dx, dx), first_zero_frequency_low_pass=cut_off_frequency_l, first_zero_frequency_notch=cut_off_frequency_l)
illumination_widefield = BFPConfiguration(refraction_index=1.5).get_widefield(3)



# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(apodization_filter1[:, :, Na//2], cmap='gray')
# axes[0].set_title("apodization, p=1")
# axes[1].imshow(apodization_filter[:, :, Na//2], cmap='gray')
# axes[1].set_title("autoconvolution")

# slider_recon = utils.wrap_axes3d(
#     axes,
#     [apodization_filter1, apodization_filter],
#     axis='y', cmap='gray', mode='log1p', scaling=10**2
# )
# plt.show()

fig, axes = plt.subplots(1, 2)
axes[0].imshow(hpc_utils.wrapped_ifftn(apodization_filter1)[:, Nl//2, :].T.real)
axes[0].set_title("apodization, p=1")
axes[1].imshow(hpc_utils.wrapped_ifftn(apodization_filter)[:, Nl//2, :].T.real)
axes[1].set_title("autoconvolution")

# slider_recon = utils.wrap_axes3d(
#     axes,
#     [hpc_utils.wrapped_ifftn(apodization_filter1).real, hpc_utils.wrapped_ifftn(apodization_filter).real],
#     axis='y'
# )
plt.show()


# finite_reconstructor = ReconstructorSpatialDomain3DSliced(
#     # illumination=illumination, 
#     illumination=illumination_reconstruction,
#     optical_system=optical_system,
#     kernel=kernel, 
# )

# reconstructed= finite_reconstructor.reconstruct(noisy_images, backend='gpu') 
reconstructor_fourier = ReconstructorFourierDomain3D(
    illumination_reconstruction, 
    optical_system, 
    kernel=kernel[..., None],
    unitary=True, 
    return_ft=False
)
reconstructed = reconstructor_fourier.reconstruct(noisy_images)
reconstructed_ft = hpc_utils.wrapped_fftn(reconstructed)

# fig, ax, slider = utils.imshow3D(np.abs(reconstructed), cmap='gray')
# plt.imshow(reconstructed.real[:, :, 29])
# plt.show()

ssnr_calc = SSNRSIM3D(   
    illumination = illumination, 
    optical_system = optical_system,
    kernel=kernel[..., None],
    illumination_reconstruction=illumination_reconstruction
    )

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

image_ft = hpc_utils.wrapped_fftn(image)

filtered_ft, _, ssnr = filter_true_wiener_sim(
    reconstructed_ft, ssnr_calc,
     )

# filtered = hpc_utils.wrapped_ifftn(filtered_ft).real
# fig, ax, slider = utils.imshow3D(filtered, cmap='gray')
# plt.show()

apodized_autoconvolution_ft = filtered_ft * apodization_filter
apodized_triangular1_ft = filtered_ft * apodization_filter1
apodized_triangular04_ft = filtered_ft * apodization_filter04

apodized_autoconvolution = hpc_utils.wrapped_ifftn(apodized_autoconvolution_ft).real
apodized_triangular1 = hpc_utils.wrapped_ifftn(apodized_triangular1_ft).real
apodized_triangular04= hpc_utils.wrapped_ifftn(apodized_triangular04_ft).real

apodized_autoconvolution /= np.amax(apodized_autoconvolution)
apodized_triangular1 /= np.amax(apodized_triangular1)
apodized_triangular04 /= np.amax(apodized_triangular04)
image /= np.amax(image)

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(image[Nl//2, :, :].T, cmap='gray')
axes[0, 0].set_title("Ground truth")
axes[0, 1].imshow(apodized_triangular04[Nl//2, :, :].T, cmap='gray')
axes[0, 1].set_title(" p=0.4")
axes[1, 0].imshow(apodized_triangular1[Nl//2, :, :].T, cmap='gray')
axes[1, 0].set_title("p=1")
axes[1, 1].imshow(apodized_autoconvolution[Nl//2, :, :].T, cmap='gray')
axes[1, 1].set_title("Auto")
axes[0, 0].set_axis_off()
axes[0, 1].set_axis_off()
axes[1, 0].set_axis_off()
axes[1, 1].set_axis_off()

slider_recon = utils.wrap_axes3d(
    axes,
    [np.transpose(image, (0, 2, 1)), np.transpose(apodized_triangular04, (0, 2, 1)),
     np.transpose(apodized_triangular1, (0, 2, 1)), np.transpose(apodized_autoconvolution, (0, 2, 1))],
    axis='x', cmap='gray'
)

plt.show()  
