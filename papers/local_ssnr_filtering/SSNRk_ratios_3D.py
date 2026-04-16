import os.path
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from config.BFPConfigurations import *
from matplotlib.widgets import Slider
import SSNRCalculator
from OpticalSystems import System4f3D, System4f2D
import kernels
import utils, hpc_utils
import pickle
import Apodization
import scipy

configurations = BFPConfiguration(refraction_index=1.5)


plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

if __name__ == "__main__":
    alpha = 2 * np.pi / 5
    theta = np.arcsin(0.9 * np.sin(alpha))

    print(alpha, theta)
    nmedium = 1.5
    nsample = 1.5

    dx = 1 / (8 * nmedium * np.sin(alpha))
    dy = dx
    dz = 1 / (4 * nmedium * (1 - np.cos(alpha)))
    Nl = 201
    Nz = 101
    max_r = Nl//2 * dx
    max_z = Nz//2 * dz


    NA = nmedium * np.sin(alpha)
    psf_size = 2 * np.array((max_r, max_r, max_z))
    dV = dx * dy * dz
    x = np.linspace(-max_r, max_r, Nl)
    y = np.copy(x)
    z = np.linspace(-max_z, max_z, Nz)

    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , Nl)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , Nl)
    fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) , Nz)

    arg = Nl // 2
    print(fz[arg])

    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / nmedium / (1 - np.cos(alpha))
    
    optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nsample, high_NA=True, vectorial=True)
    optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Nz)))
    illumination = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, Mr=3, Mt=1, dimensionality=3)

    illumination_projected = copy.deepcopy(illumination).project_in_quasi_2D()
    m1 = 10
    m2 = 5
    illumination_projected_amplified = copy.deepcopy(illumination_projected)
    for r in range(3):
        illumination_projected_amplified.harmonics[(r, (1, 0, 0))].amplitude *= m1
        illumination_projected_amplified.harmonics[(r, (-1, 0, 0))].amplitude *= m1
        illumination_projected_amplified.harmonics[(r, (2, 0, 0))].amplitude *= m2
        illumination_projected_amplified.harmonics[(r, (-2, 0, 0))].amplitude *= m2

    f0 = 1 / 4 / dx / 0.9
    airy = kernels.psf_kernel2d(0, (dx, dx), f0)[:, :, None]
    airy_notch = kernels.combined_low_pass_notch_kernel(0, 0, (dx, dx), f0, f0)[:, :, None]

    apodization = Apodization.AutoconvolutionApodizationSIM3D(optical_system, illumination)
    apodization_function_ra = utils.average_rings3d(apodization.apodization_function, (fx, fx, fz))
    apodization_function_ra = np.where(apodization_function_ra.T > 1e-10, 1, 0)
    interior_mask = scipy.ndimage.binary_dilation(apodization_function_ra, iterations=1)
    print(interior_mask.shape)
    mask = interior_mask & ~apodization_function_ra
    support = np.zeros((*interior_mask.shape, 4))
    support[mask!=0] = [1, 0, 0, 1]
    # support = support.reshape(Nl//2+1, Nz, 4)
    print(support.shape)
    # support = np.ma.masked_where(apodization_function_ra & ~interior_mask, apodization_function_ra & ~interior_mask)
    # plt.imshow(support.T, cmap='gray')    
    # plt.show()

    noise_estimator_ideal = SSNRCalculator.SSNRSIM3D(illumination, optical_system)
    noise_estimator_airy = SSNRCalculator.SSNRSIM3D(illumination, optical_system, kernel=airy, illumination_reconstruction=illumination_projected)
    noise_estimator_airy_notch = SSNRCalculator.SSNRSIM3D(illumination, optical_system, kernel=airy_notch, illumination_reconstruction=illumination_projected)
    noise_estimator_airy_amplified = SSNRCalculator.SSNRSIM3D(illumination, optical_system, kernel=airy, illumination_reconstruction=illumination_projected_amplified)
    noise_estimator_airy_notch_amplified = SSNRCalculator.SSNRSIM3D(illumination, optical_system, kernel=airy_notch, illumination_reconstruction=illumination_projected_amplified)

    ssnr_ideal = noise_estimator_ideal.ring_average_ssnri()
    ssnr_airy = noise_estimator_airy.ring_average_ssnri() 
    ssnr_airy_amplified = noise_estimator_airy_amplified.ring_average_ssnri()
    ssnr_airy_notch = noise_estimator_airy_notch.ring_average_ssnri()
    ssnr_airy_notch_amplified = noise_estimator_airy_notch_amplified.ring_average_ssnri()
    
    print("Volume ideal airy = ", noise_estimator_airy.compute_ssnri_volume())
    print("Entropy ideal airy = ", noise_estimator_airy.compute_ssnri_entropy())

    print("Volume ideal airy amplified = ", noise_estimator_airy_amplified.compute_ssnri_volume())
    print("Entropy ideal airy amplified = ", noise_estimator_airy_amplified.compute_ssnri_entropy())

    print("Volume ideal airy notch = ", noise_estimator_airy_notch.compute_ssnri_volume())
    print("Entropy ideal airy notch = ", noise_estimator_airy_notch.compute_ssnri_entropy())

    print("Volume ideal airy notch amplified = ", noise_estimator_airy_notch_amplified.compute_ssnri_volume())
    print("Entropy ideal airy notch amplified = ", noise_estimator_airy_notch_amplified.compute_ssnri_entropy())

    ssnr_dict = {"airy": ssnr_airy, 
            "airy_amplified": ssnr_airy_amplified, 
            "airy_notch": ssnr_airy_notch, 
            "airy_notch_amplified": ssnr_airy_notch_amplified}

    for name, ssnr in ssnr_dict.items():
        fig, ax = plt.subplots(figsize=(12,10))
        ax.set_xlabel('$f_r [LCF]$')
        ax.set_ylabel('$f_z [ACF]$')
        ax.set_aspect('equal', adjustable='box')
        # im0 = axes[0].imshow(np.log1p(10**8 * ssnr_finite_conventional_ra.T), extent=(two_NA_fx[0], two_NA_fx[-1], two_NA_fz[0], two_NA_fz[-1]), origin='lower', aspect='auto')
        im0 = ax.imshow(ssnr.T / ssnr_ideal.T, extent=(0, two_NA_fx[-1], two_NA_fz[0], two_NA_fz[-1]), origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.imshow(support, extent=(0, two_NA_fx[-1], two_NA_fz[0], two_NA_fz[-1]), cmap='gray')
        ax.set_aspect(1. / ax.get_data_ratio())
        plt.colorbar(im0, ax=ax, extend='both', shrink=0.8, label='$SSNR_{ratio}$')

        # if labels[i] == 'Conventional':
        fig.savefig(current_dir + f"/Figures/ssnr/SSNR_ratio_3D_{name}.png", bbox_inches='tight')


    plt.show()