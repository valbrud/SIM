import IPython.core.inputtransformer2
import IPython.core.formatters
import IPython.core.crashhandler
import IPython.core.crashhandler
import IPython.core.crashhandler
import IPython.core.crashhandler
import IPython.core.crashhandler
import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config.BFPConfigurations import BFPConfiguration

configurations = BFPConfiguration(refraction_index=1.5)
from OpticalSystems import System4f2D
from SSNRCalculator import SSNRSIM2D
from kernels import psf_kernel2d

plt.rcParams['font.size'] = 50         # Sets default font size
plt.rcParams['axes.titlesize'] = 50     # Title of the axes
plt.rcParams['axes.labelsize'] = 50     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 50    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 50    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 20    # Font size for legend

alpha = 2 * np.pi / 5
theta = np.arcsin(0.9 * np.sin(alpha))
n = 1.5
dx = 1 / (8 * n * np.sin(alpha))
dy = dx
N = 255
max_r = N//2 * dx
NA = n * np.sin(alpha)
psf_size = 2 * np.array((max_r, max_r))
dx = 2 * max_r / N
dy = 2 * max_r / N
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)

fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx)  , N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy)  , N)

arg = N // 2

two_NA_fx = fx / (2 * NA)
two_NA_fy = fy / (2 * NA)

optical_system = System4f2D(alpha=alpha, refractive_index=n, high_NA=True, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))

illumination_conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1, dimensionality=2)
ssnr_calc_ideal_filter_ideal = SSNRSIM2D(illumination_conventional, optical_system)
ssnr_ideal__fitlered_ideal_ra = ssnr_calc_ideal_filter_ideal.ring_average_ssnri()

for aberration_strength in (0.072, 2 * 0.072):
    optical_system_aberrated = System4f2D(alpha=alpha, refractive_index=n, high_NA=True, vectorial=True)
    optical_system_aberrated.compute_psf_and_otf((psf_size, N), zernieke={(4, 0):aberration_strength})
    
    ssnr_calc_ideal_filter_aberrated = SSNRSIM2D(illumination_conventional, optical_system, optical_system_aberrated.psf)
    ssnr_ideal_filtered_aberrated_ra = ssnr_calc_ideal_filter_aberrated.ring_average_ssnri()

    kernel = psf_kernel2d(0, (dx, dx), 1/(4 * dx))

    ssnr_calc_ideal_filtered_kernel = SSNRSIM2D(illumination_conventional, optical_system, kernel=kernel)
    ssnr_ideal_filtered_kernel_ra = ssnr_calc_ideal_filtered_kernel.ring_average_ssnri()

    ssnr_calc_aberrated_filtered_aberrated = SSNRSIM2D(illumination_conventional, optical_system_aberrated)
    ssnr_aberrated_filtered_aberrated_ra = ssnr_calc_aberrated_filtered_aberrated.ring_average_ssnri()

    ssnr_calc_aberrated_filtered_ideal = SSNRSIM2D(illumination_conventional, optical_system_aberrated, kernel=optical_system.psf)
    ssnr_aberrated_filtered_ideal_ra = ssnr_calc_aberrated_filtered_ideal.ring_average_ssnri()

    ssnr_calc_aberrated_filtered_kernel = SSNRSIM2D(illumination_conventional, optical_system_aberrated, kernel=kernel)
    ssnr_aberrated_filtered_kernel_ra = ssnr_calc_aberrated_filtered_kernel.ring_average_ssnri()

    Fx, Fy = np.meshgrid(fx, fy)
    fig1 = plt.figure(figsize=(18, 8), constrained_layout=True)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    ax1 = fig1.add_subplot(111)

    multiplier = 10**3

    ax1.set_xlabel(r"$f_r [LCF]$")
    ax1.set_ylabel(r"$1 + 10^3 SSNR_{ra}$")
    # ax1.set_title(rf"{aberration}", fontsize=25)
    ax1.set_yscale("log")
    ax1.grid(which='major')
    ax1.grid(which='minor', linestyle='--')
    # ax1.set_ylim(1, 1 + multiplier)
    # ax1.set_ylim(1, 7)
    ax1.set_xlim(0, 2)

    ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier * ssnr_ideal__fitlered_ideal_ra), color='black', label="II")
    # ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier * ssnr_ideal_filtered_aberrated_ra), color='red', linestyle='--', label="IA")
    # ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier * ssnr_ideal_filtered_kernel_ra), color='purple', linestyle='-.', label="IK")
    ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier * ssnr_aberrated_filtered_aberrated_ra), color='green', label="AA")
    ax1.plot(two_NA_fy[fy >= 0], (1 + multiplier * ssnr_aberrated_filtered_ideal_ra), color='blue', linestyle='--', label="AI")
    ax1.plot(two_NA_fx[fx >= 0], (1 + multiplier * ssnr_aberrated_filtered_kernel_ra), color='orange', linestyle='-.',  label="AK")
    ax1.set_aspect(1. / ax1.get_data_ratio())

    ax1.set_ylim(bottom=1)

    ax1.legend(fontsize=30, loc="upper right", ncols=1)
    plt.show()

    fig1.savefig(current_dir + f"/Figures/ssnr/SSNR_aberrated_spherical_{aberration_strength}.png", bbox_inches='tight')

