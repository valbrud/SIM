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


plt.rcParams['font.size'] = 50         # Sets default font size
plt.rcParams['axes.titlesize'] = 50     # Title of the axes
plt.rcParams['axes.labelsize'] = 50     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 50    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 50    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 20    # Font size for legend

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
    
    optical_system = System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nsample, high_NA=True, vectorial=False)
    optical_system.compute_psf_and_otf((psf_size, (Nl, Nl, Nz)))
    illumination_square = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 1, Mt=1, dimensionality=3)
    illumination_hexagonal = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1, dimensionality=3)
    illumination_conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, Mr=3, Mt=1, dimensionality=3)
    illumination_widefield = configurations.get_widefield(dimensionality=3)

    configurations = {"Conventional" : illumination_conventional,
                     "Square" : illumination_square,
                     "Hexagonal" : illumination_hexagonal}

    illumination_conventional_projected = copy.deepcopy(illumination_conventional).project_in_quasi_2D()
    m1 = 20
    m2 = 7
    for r in range(3):
        illumination_conventional_projected.harmonics[(r, (1, 0, 0))].amplitude *= m1
        illumination_conventional_projected.harmonics[(r, (-1, 0, 0))].amplitude *= m1
        illumination_conventional_projected.harmonics[(r, (2, 0, 0))].amplitude *= m2
        illumination_conventional_projected.harmonics[(r, (-2, 0, 0))].amplitude *= m2

    illumination_square_projected = copy.deepcopy(illumination_square).project_in_quasi_2D()
    illumination_hexagonal_projected = copy.deepcopy(illumination_hexagonal).project_in_quasi_2D()

    noise_estimator = SSNRCalculator.SSNRSIM3D(illumination_widefield, optical_system)
    ssnr_widefield = noise_estimator.ssnri
    ssnr_widefield_ra = noise_estimator.ring_average_ssnri()
    volume_widefield = noise_estimator.compute_ssnri_volume()
    entropy_widefield = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_square
    ssnr_square = np.abs(noise_estimator.ssnri)
    ssnr_square_ra = noise_estimator.ring_average_ssnri()
    volume_squareSP = noise_estimator.compute_ssnri_volume()
    entropy_square = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_hexagonal
    ssnr_hexagonal = np.abs(noise_estimator.ssnri)
    ssnr_hexagonal_ra = noise_estimator.ring_average_ssnri()
    volume_hexagonal = noise_estimator.compute_ssnri_volume()
    entropy_hexagonal = noise_estimator.compute_ssnri_entropy()

    noise_estimator.illumination = illumination_conventional
    ssnr_conventional = np.abs(noise_estimator.ssnri)
    ssnr_conventional_ra = noise_estimator.ring_average_ssnri()
    volume_conventional = noise_estimator.compute_ssnri_volume()
    entropy_conventional = noise_estimator.compute_ssnri_entropy()

    print("Volume ideal widefield = ", volume_widefield)
    print("Entropy ideal widefield = ", entropy_widefield)


    print("Volume ideal square = ", volume_squareSP)
    print("Entropy ideal square = ", entropy_square)


    print("Volume ideal conventional = ", volume_conventional)
    print("Entropy ideal conventional = ", entropy_conventional)


    print("Volume ideal hexagonal = ", volume_hexagonal)
    print("Entropy ideal hexagonal = ", entropy_hexagonal)
    kernel = np.zeros((1,1,1))
    kernel[0,0,0] = 1

    factor_l = 2
    factor_z = 2


    correction_factor_l = 2 / (1 + np.sin(theta)/np.sin(alpha))
    correction_factor_z = 2 / (1 + np.cos(theta)/np.cos(alpha))

    cut_off_frequency_l = 1 / 2 / (factor_l * dx)
    cut_off_frequency_z = 1 / 2 / (factor_z * dz)

    kernel = utils.expand_kernel(kernels.sinc_kernel3d(pixel_size=(dx, dy, dz), first_zero_frequency_r=cut_off_frequency_l, first_zero_frequency_z=cut_off_frequency_z), (Nl, Nl, Nz))
    # kernel = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(dx, dy), first_zero_frequency=cut_off_frequency_l)[..., None] * kernels.sinc_kernel1d(kernel_size=1, pixel_size=dz, first_zero_frequency=cut_off_frequency_z)[None, None, :], (Nl, Nl, Nz))
    # kernel = optical_system.psf
    # plt.plot(two_NA_fx, kernel_ft[Nl//2, :, Nz//2])
    # plt.imshow(kernel_ft[Nl//2, Nl//2:, :].T, extent=(0, two_NA_fz[-1], two_NA_fy[0], two_NA_fy[-1]), origin='lower', aspect='auto')
    # plt.show()
    
    noise_estimator_finite_conventional = SSNRCalculator.SSNRSIM3D(illumination_conventional, optical_system, kernel)
    noise_estimator_finite_square = SSNRCalculator.SSNRSIM3D(illumination_square, optical_system, kernel)
    noise_estimator_finite_hexagonal = SSNRCalculator.SSNRSIM3D(illumination_hexagonal, optical_system, kernel)

    apodization = Apodization.AutoconvolutionApodizationSIM3D(optical_system, illumination_conventional)
    apodization_function_ra = utils.average_rings3d(apodization.apodization_function, (fx, fx, fz))
    apodization_function_ra = np.where(apodization_function_ra.T > 1e-10, 1, 0)
    interior_mask = scipy.ndimage.binary_dilation(apodization_function_ra, iterations=1)
    print(interior_mask.shape)
    mask = interior_mask & ~apodization_function_ra
    support = np.zeros((*interior_mask.shape, 4))
    support[mask!=0] = [1, 0, 0, 1]
    # noise_estimator_finite_conventional = SSNRCalculator.SSNRSIM3D(illumination_conventional, optical_system, kernel, illumination_reconstruction=illumination_conventional_projected)
    # noise_estimator_finite_square = SSNRCalculator.SSNRSIM3D(illumination_square, optical_system, kernel, illumination_reconstruction=illumination_square_projected)
    # noise_estimator_finite_hexagonal = SSNRCalculator.SSNRSIM3D(illumination_hexagonal, optical_system, kernel, illumination_reconstruction=illumination_hexagonal_projected)

    ssnr_finite_conventional = noise_estimator_finite_conventional.ssnri
    ssnr_finite_conventional_ra = noise_estimator_finite_conventional.ring_average_ssnri()


    volume_finite_conventional = noise_estimator_finite_conventional.compute_ssnri_volume()
    entropy_finite_conventional = noise_estimator_finite_conventional.compute_ssnri_entropy()


    # fig = plt.figure(figsize=(12,12))
    # ax = fig.add_subplot(111, projection='3d')
    # X, Z = np.meshgrid(two_NA_fx[Nl//2:], two_NA_fz)
    # # ax.plot_wireframe(X, Z, np.log1p(10**6 * ssnr_conventional_ra), color='black', rstride=5, cstride=5)
    # ax.plot_surface(X, Z, np.log1p(10**6 * ssnr_finite_conventional_ra))
    # ax.set_xlabel('$f_r$ [LCF]', labelpad=20)  
    # ax.set_ylabel('$f_z$ [ACF]', labelpad=20)
    # ax.set_zlabel('$\log(1 + 10^6 \\, SSNR_K)$', labelpad=12, rotation=-90)
    # ax.set_xlim(0, two_NA_fx[-1] + 0.1)
    # ax.set_ylim(-0.1 + two_NA_fz[0], two_NA_fz[-1])
    # ax.set_zlim(0, 14)
    # ax.view_init(elev=30, azim=-30)  
    # fig.savefig(current_dir + f"/Figures/SSNR_3D_Conventional_{m1}_{m2}.png", bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    # quit()
    # print(f"Volume finite conventional = ", volume_finite_conventional)
    # print(f"Entropy finite conventional = ", entropy_finite_conventional)

    ssnr_finite_square = noise_estimator_finite_square.ssnri
    ssnr_finite_square_ra = noise_estimator_finite_square.ring_average_ssnri()

    volume_finite_square = noise_estimator_finite_square.compute_ssnri_volume()
    entropy_finite_square = noise_estimator_finite_square.compute_ssnri_entropy()

    print(f"Volume finite square = ", volume_finite_square)
    print(f"Entropy finite square = ", entropy_finite_square)

    ssnr_finite_hexagonal = noise_estimator_finite_hexagonal.ssnri
    ssnr_finite_hexagonal_ra = noise_estimator_finite_hexagonal.ring_average_ssnri()
    volume_finite_hexagonal = noise_estimator_finite_hexagonal.compute_ssnri_volume()
    entropy_finite_hexagonal = noise_estimator_finite_hexagonal.compute_ssnri_entropy()

    print(f"Volume finite hexagonal = ", volume_finite_hexagonal)
    print(f"Entropy finite hexagonal = ", entropy_finite_hexagonal)

    ssnr_conventional_ra[ssnr_conventional_ra < 10e-10] = 1e-10
    ssnr_square_ra[ssnr_square_ra < 10e-10] = 1e-10
    ssnr_hexagonal_ra[ssnr_hexagonal_ra < 10e-10] = 1e-10

    ssnr_finite_conventional_ra[ssnr_conventional_ra < 10e-10] = 0
    ssnr_finite_square_ra[ssnr_square_ra < 10e-10] = 0
    ssnr_finite_hexagonal_ra[ssnr_hexagonal_ra < 10e-10] = 0

    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_widefield_ra[0, :]), label='Widefield')
    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_conventional_ra[0, :]), label='Conventional', color='blue')
    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_square_ra[0, :]), label='Square', color='red')
    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_hexagonal_ra[0, :]), label='Hexagonal', color='green')

    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_finite_square_ra[0, :]), '--', color='red')
    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_finite_conventional_ra[0, :]), '--', color='blue')
    plt.plot(two_NA_fz, np.log1p(10**8 * ssnr_finite_hexagonal_ra[0, :]), '--', color='green')
    plt.legend()
    plt.show()

    ssnr_list = [ssnr_conventional_ra, ssnr_square_ra, ssnr_hexagonal_ra]
    ssnr_finite_list = [ssnr_finite_conventional_ra, ssnr_finite_square_ra, ssnr_finite_hexagonal_ra]
    labels = ['Conventional', 'Square', 'Hexagonal']
    for i in range(3):
        fig, ax = plt.subplots(figsize=(12,10))
        ax.set_xlabel('$f_r [LCF]$')
        if labels[i] == 'Conventional':
            ax.set_ylabel('$f_z [ACF]$')
        ax.set_aspect('equal', adjustable='box')
        # im0 = axes[0].imshow(np.log1p(10**8 * ssnr_finite_conventional_ra.T), extent=(two_NA_fx[0], two_NA_fx[-1], two_NA_fz[0], two_NA_fz[-1]), origin='lower', aspect='auto')
        im0 = ax.imshow(ssnr_finite_list[i].T / ssnr_list[i].T, extent=(0, two_NA_fx[-1], two_NA_fz[0], two_NA_fz[-1]), origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.imshow(support, extent=(0, two_NA_fx[-1], two_NA_fz[0], two_NA_fz[-1]), origin='lower')
        ax.set_aspect(1. / ax.get_data_ratio())
        if labels[i] == 'Hexagonal':
            plt.colorbar(im0, ax=ax, extend='both', shrink=0.8, label='$SSNR_{ratio}$')

        # if labels[i] == 'Conventional':
        fig.savefig(current_dir + f"/Figures/ssnr/SSNR_ratio_3D_triangular_{labels[i]}.png", bbox_inches='tight')


    plt.show()