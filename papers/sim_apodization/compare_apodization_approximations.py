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
from Apodization import TriangularApodizationSIM3D, AutocorrelationApodizationSIM3D, AutocorrelationApodizationWidefield
import hpc_utils, utils
from matplotlib_scalebar.scalebar import ScaleBar
import copy 

plt.rcParams['font.size'] = 30         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 30    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 30    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 20    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
theta = 0.9 * alpha
NA = 1.5 * np.sin(alpha)
dx = 1 / (8 * NA)
# dz= dx
dz = 1 / (4 * 1.5 * (1 - np.cos(alpha)))
dy = dx
Nl = 101
Nz = 101
max_r = Nl // 2 * dx
max_z = Nz // 2 * dz
x = np.linspace(-max_r, max_r, Nl)
airy_l = x * 2 * NA / 1.22 
z = np.linspace(-max_z, max_z, Nz)
airy_z = z * 2 * NA / 1.22
fx = np.linspace(-1 / (2 * dy), 1 / (2 * dy), Nl)
fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), Nz)
fr = fx[Nl//2:]/(2 * NA)


optical_system = System4f3D(alpha=alpha, refractive_index_medium=1.5, refractive_index_sample=1.5, high_NA=True, vectorial=True)
optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2* max_z), (Nl, Nl, Nz)))

widefield = configurations.get_widefield(dimensionality=3)
conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, Mr = 3, Mt=1, dimensionality=3)
hexagonal = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1, dimensionality=3)
apodization_triangular_conventional = TriangularApodizationSIM3D(optical_system, conventional, power=1, Ndense=201)
apodization_triangular_hexagonal = TriangularApodizationSIM3D(optical_system, hexagonal, power=1, Ndense=201)
support_triangular_conventional = np.where(apodization_triangular_conventional.apodization_function >= 1e-9* np.max(apodization_triangular_conventional.apodization_function), 1, 0)
support_triangular_hexagonal = np.where(apodization_triangular_hexagonal.apodization_function >= 1e-9* np.max(apodization_triangular_hexagonal.apodization_function), 1, 0)
apodization_autoconvolution = AutocorrelationApodizationSIM3D(optical_system, conventional, Ndense=201)
support_autocorrelation = np.where(apodization_autoconvolution.apodization_function >= 1e-9 * np.max(apodization_autoconvolution.apodization_function), 1, 0)
fig, axes = plt.subplots(2, 3)

axes[0, 0].imshow(support_triangular_conventional[:, :, Nz//2], cmap='gray')
axes[0, 0].set_title("a")
axes[0, 1].imshow(support_triangular_hexagonal[:, :, Nz//2], cmap='gray')
axes[0, 1].set_title("b")
axes[0, 2].imshow(support_autocorrelation[:, :, Nz//2], cmap='gray')
axes[0, 2].set_title("c")
axes[1, 0].imshow(support_triangular_conventional[:, Nl//2, :].T, cmap='gray')
axes[1, 0].set_title("d")
axes[1, 1].imshow(support_triangular_hexagonal[:, Nl//2, :].T, cmap='gray')
axes[1, 1].set_title("e")
axes[1, 2].imshow(support_autocorrelation[:, Nl//2, :].T, cmap='gray')
axes[1, 2].set_title("f")
axes[0, 0].set_axis_off()
axes[0, 1].set_axis_off()
axes[0, 2].set_axis_off()
axes[1, 0].set_axis_off()
axes[1, 1].set_axis_off()
axes[1, 2].set_axis_off()
fig.tight_layout()
fig.savefig(current_dir + "/Figures/simulations/compare_apodization_approximations.png", dpi=300, bbox_inches='tight')
# slider_recon = utils.wrap_axes3d(
#     axes,
#     [support_triangular_conventional, support_triangular_hexagonal, support_autocorrelation],
#     axis='y', cmap='gray', vmin=0, vmax=1, extent=[x[0], x[-1], x[0], x[-1], z[0], z[-1]]
# )
plt.show()