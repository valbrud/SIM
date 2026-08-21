import sys
import os

import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)
sys.path.append(current_dir)

from config.BFPConfigurations import BFPConfiguration
from Illumination import IlluminationPlaneWaves2D
from OpticalSystems import System4f2D, System4f3D
import utils
from SSNRCalculator import SSNRSIM2D, SSNRSIM3D, SSNRSIMGLS2D, SSNRSIMGLS3D


plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
NA = np.sin(alpha) * 1.5

N = 301
dx = 1 / (8 * NA)
max_r = N // 2 * dx
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fr = fx[N//2:]
psf_size = (2 * max_r, 2 * max_r)

optical_system = System4f2D(alpha=alpha, refractive_index=1.5, high_NA=True, vectorial=True)
optical_system.compute_psf_and_otf((psf_size, N))

for multiplier in (0.4, 0.5, 0.51, 0.6, 0.8):
    theta = alpha * multiplier
    illum_conv = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1, dimensionality=2)
    illum_hex = configurations.get_3_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
    illum_dict = {"Conventional": illum_conv, 
                "Hexagonal": illum_hex}
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    # ax.plot(np.log1p(1e4 * np.abs(ring_averaged_conv_no_cc)), lw=1, ls='--', label="Conventional LS")
    # ax.plot(np.log1p(1e4 * np.abs(ring_averaged_hex_no_cc)), lw=1, ls='--', label="Hexagonal LS")
    ratio_max = 0
    ax.set_xlim(0, fr[-1]/(2 * NA))
    ax.set_xlabel("$f_r$, [LCF]")
    ax.set_ylabel("$SSNR^{GLS}_{ra}/SSNR^{LS}_{ra}$")
    # ax.set_title(illumination_key)
    for illumination_key in illum_dict:
        illumination = illum_dict[illumination_key]
        ssnr_calc = SSNRSIM2D(illumination, optical_system)
        ssnr_calc_cc = SSNRSIMGLS2D(illumination, optical_system)
        fig, ax = plt.subplots()
        # im = ax.imshow(np.abs(ssnr_calc.dj)/np.abs(ssnr_calc.vj), vmin=0, vmax=2)
        # plt.colorbar(im)
        # plt.show()
        ring_averaged_no_cc = ssnr_calc.ring_average_ssnri()
        ring_averaged_cc = ssnr_calc_cc.ring_average_ssnri()
        ratio = ring_averaged_cc / (ring_averaged_no_cc + 1e-9)
        ax.plot(fr/(2 * NA), ratio, lw=1, label=illumination_key)
        if ratio_max < 1 + 1.1 * (np.amax(ratio)-1):
            ratio_max = 1 + 1.1 * (np.amax(ratio)-1)
        ax.set_ylim(bottom=1, top = ratio_max)
    ax.legend()
    plt.show()
    fig.savefig(current_dir + f"/Figures/simulations/GLS_LS_ratio_{multiplier}.png", bbox_inches='tight', pad_inches=0.5)



