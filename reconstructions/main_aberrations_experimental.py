import os.path
import sys

import tifffile

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import pickle
import utils 
import windowing
from config.BFPConfigurations import BFPConfiguration
configurations = BFPConfiguration()

from OpticalSystems import System4f2D
from SSNRCalculator import SSNRSIM2D
from SIMulator import SIMulator2D
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from Illumination import IlluminationPlaneWaves2D
import ShapesGenerator
import wrappers
from WienerFiltering import filter_true_wiener
from kernels import psf_kernel2d
from PatternEstimator import IlluminationPatternEstimator2D
np.random.seed(1234)

N = 255
wavelength = 680e-9
px_scaled = 80e-9
dx = px_scaled / wavelength
NA = 1.4
NA_best_fit = 1.2
nmedium = 1.518
alpha = np.arcsin(NA / nmedium)
alpha_best_fit = np.arcsin(NA_best_fit / nmedium)

max_r =  N // 2 * dx
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
psf_size = 2 * np.array((max_r, max_r))
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fr = np.linspace(0, 1 / (2 * dx), N//2 + 1)
psf_size = np.array((2 * max_r, 2 * max_r))
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)

fx = np.linspace(-1/(2 * dx), 1/(2 * dx), N)
fy = np.copy(fx)

fxn = fx / (2 * NA)
fyn = fy / (2 * NA)

data = tifffile.imread(project_root + '/data/OMX_Tetraspeck200_680nm.tiff')
print(data.shape)

stack = data.reshape((3, -1, 5, 512, 512))
stack = stack[:, 3, :, N//2+1:-N//2-1, N//2+1:-N//2-1]
image = stack[0, 0]
print('total count', np.sum(image)/N**2)
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
plt.imshow(np.log1p(10 ** 8 * np.abs(wrappers.wrapped_fftn(stack[0, 0, ...]))).T, cmap='gray', origin='lower')
plt.show()

aberrated_psf_dict = pickle.load(open(current_dir + "\\aberrated_psf_dict3_tetraspec_680.pkl", "rb"))
for key, value in aberrated_psf_dict.items():
    value = utils.expand_kernel(value, (N, N))
    aberrated_psf_dict[key] = value

optical_system = System4f2D(alpha=alpha, refractive_index=nmedium)
optical_system.compute_psf_and_otf((psf_size, N))

otf = wrappers.wrapped_fftn(aberrated_psf_dict[(0.0, 0.0)])
plt.plot(otf[N//2, N//2:], label='Paraxial OTF')
plt.show()

configurations = BFPConfiguration(refraction_index=nmedium)

illumination = configurations.get_2_oblique_s_waves_and_s_normal(
    alpha_best_fit-0.5, 
    1, 0,
    Mr=3,
    Mt=5, 
    angles=(45 / 180 * np.pi , 165 / 180 * np.pi, 105 / 180 * np.pi),
    dimensionality=2
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
    debug_info_level=1
)


calc_reference7 = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=psf_kernel2d(7, (dx, dx))
)

calc_reference9 = SSNRSIM2D(
    illumination=illumination,
    optical_system=optical_system,
    kernel=psf_kernel2d(9, (dx, dx))
)

optical_system_reconstruction = System4f2D(alpha=alpha, refractive_index=nmedium)
optical_system_reconstruction.compute_psf_and_otf_coordinates(psf_size, N)

fig, axes = plt.subplots(1, 3)
min_loss_function = 10**6
estimated_key = 0 

color_idx = 0
fig, ax = plt.subplots()
for key, psf in aberrated_psf_dict.items():
    # if not key[0] == 0.0864:
    #     continue

    print("Evaluating key: {}".format(key))
    color = plt.cm.tab10(color_idx % 10)
    optical_system_reconstruction.psf = psf
    optical_system_reconstruction._otf /= optical_system_reconstruction._otf[N//2, N//2]

    illumination.estimate_modulation_coefficients(stack, optical_system_reconstruction.psf, optical_system_reconstruction.x_grid,
                                                                    method='peak_height_ratio', update=True)
    print("Estimated modulation coefficients: {}".format(illumination.get_amplitudes()[0])
          )
    
    spatial_reconstructor7 = ReconstructorSpatialDomain2D(
        illumination=illumination,
        optical_system=optical_system_reconstruction,
        kernel=psf_kernel2d(7, (dx, dx))
    )

    spatial_reconstructor9 = ReconstructorSpatialDomain2D(
        illumination=illumination,
        optical_system=optical_system_reconstruction,
        kernel=psf_kernel2d(9, (dx, dx))
    )


    reconstructed_image7 = spatial_reconstructor7.reconstruct(stack)
    reconstructed_image9 = spatial_reconstructor9.reconstruct(stack)

    calc7 = SSNRSIM2D(
        illumination=illumination,
        optical_system=optical_system_reconstruction,
        kernel = psf_kernel2d(7, (dx, dx))
    )

    calc9 = SSNRSIM2D(
        illumination=illumination,
        optical_system=optical_system_reconstruction,
        kernel = psf_kernel2d(9, (dx, dx))
    )

    
    filtered7, _, ssnr7 = filter_true_wiener(wrappers.wrapped_fftn(reconstructed_image7), calc7)
    filtered9, _, ssnr9 = filter_true_wiener(wrappers.wrapped_fftn(reconstructed_image9), calc9)

    r = fxn[N//2:]

    ratio79_experimental = (ssnr7[N//2, N//2:])/(ssnr9[N//2, N//2:])
    ratio79_theoretical = calc7.ring_average_ssnri_approximated()/calc9.ring_average_ssnri_approximated()

    R = np.where(ssnr7[N//2, N//2:] >=9,  ratio79_experimental/ratio79_theoretical, 0)
    R = np.where(r <= 1.5, R, 0)

    loss_function = np.sum((R - 1)**2)
    print("Loss function value for key {}: {}".format(key, loss_function))

    if loss_function < min_loss_function:
        min_loss_function = loss_function
        estimated_key = key

    if key[0] == 0.0864:
        if color_idx == 0:
            axes[0].plot(r, ratio79_experimental, color='black')

        axes[0].plot(r, ratio79_theoretical, color=color, label='{}'.format(key))

        # axes[1].plot(calc1.ssnri[N//2, N//2:])
        axes[0].set_title("Ratio 7/9")
        axes[0].set_ylim(0, 2)

        axes[1].plot(r, np.log1p(ssnr7[N//2, N//2:]), color=color, label='{}'.format(key))
        axes[1].plot(r, np.log1p(ssnr9[N//2, N//2:]), color=color, linestyle='-')
        if color_idx == 0:
            axes[1].plot(r, np.log(2) * np.ones_like(r), color='black', linestyle='-', label='Reference')
        # axes[1].plot(calc9.ssnri[N//2, N//2:])
        axes[1].set_title("SSNR")
        # axes[1].set_ylim(0, 2)
        # plt.show()

        # R = ratio19_experimental/ratio19_theoretical
        axes[2].plot(r, R, color=color, label='{}'.format(key))
        if color_idx == 0:
            axes[2].plot(r, np.ones_like(r), color='black', linestyle='--', label='Reference')

        # axes[3].plot(r[:-N//8], ratio19_theoretical[:-N//8], label='theoretical')
        axes[2].set_title("Ratio exp/theor")
        axes[2].set_ylabel("SSNR ratio")
        axes[2].set_xlabel("Spatial frequency (1/px)")
        axes[2].set_ylim(0, 2)

        color_idx += 1

# axes_test.legend()
# axes[0].legend()
# axes[1].legend()
# axes[2].legend()

print("estimated_key: {}".format(estimated_key))
# print("True key: {}".format(true_key))
print("Min loss function: {}".format(min_loss_function))

fig_comparison, axes_comparison = plt.subplots(1, 2)
axes_comparison[0].plot(x[N//2:], optical_system.psf[N//2, N//2:], label='Initial PSF')
axes_comparison[0].plot(x[N//2:], aberrated_psf_dict[estimated_key][N//2, N//2:], label='Estimated PSF')
axes_comparison[1].plot(r, optical_system.otf[N//2, N//2:], label='Initial OTF')
axes_comparison[1].plot(r, wrappers.wrapped_fftn(aberrated_psf_dict[estimated_key])[N//2, N//2:], label='Estimated OTF')
axes_comparison[0].legend()
axes_comparison[1].legend()

plt.show()