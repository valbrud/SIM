import sys
import os

import numpy as np
import matplotlib.pyplot as plt

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
from WienerFiltering import filter_true_wiener_sim
import hpc_utils
import Apodization

plt.rcParams['font.size'] = 20         # Sets default font size
plt.rcParams['axes.titlesize'] = 30     # Title of the axes
plt.rcParams['axes.labelsize'] = 30     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 15    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 15    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 30    # Font size for legend

configurations = BFPConfiguration(refraction_index=1.5)

alpha = 2 * np.pi / 5
NA = 1.5 * np.sin(alpha)
Nl = 101
Nz = 101
dx = 1 / (8 * NA)
dz = 1 / (4 * 1.5 * (1 - np.cos(alpha)))
dy = dx
max_r = Nl//2 * dx
max_z = Nz//2 * dz
x = np.linspace(-max_r, max_r, Nl)
z = np.linspace(-max_z, max_z, Nz)
y = np.copy(x)
print(x)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nl)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), Nl)
fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), Nz)
fr = fx[Nl//2:] / (2 * NA)

psf_size2d = 2 * np.array((max_r, max_r))

optical_system2d= System4f2D(alpha=alpha, refractive_index=1.5, vectorial=True)
optical_system2d.compute_psf_and_otf((psf_size2d, Nl))

hexagonal2d = configurations.get_3_oblique_s_waves_and_circular_normal(alpha * 0.9, 1, 0, Mt=3, dimensionality=2, dimensions=(1, 1, 0))
square2d = configurations.get_4_circular_oblique_waves_and_circular_normal(alpha * 0.9, 1, 0, Mt=3, dimensionality=2, dimensions=(1, 1, 0))

apodization_hexagonal2d = Apodization.AutocorrelationApodizationSIM2D(optical_system2d, hexagonal2d)
apodization_square2d = Apodization.AutocorrelationApodizationSIM2D(optical_system2d, square2d)

psf_size3d = 2 * np.array((max_r, max_r, max_z))
optical_system3d= System4f3D(alpha=alpha, refractive_index_medium=1.5, refractive_index_sample=1.5, vectorial=True)
optical_system3d.compute_psf_and_otf((psf_size3d, (Nl, Nl, Nz)))

hexagonal3d = configurations.get_6_oblique_s_waves_and_circular_normal(alpha * 0.9, 1, 1, Mt=3, dimensionality=3, dimensions=(1, 1, 0))
square3d = configurations.get_4_circular_oblique_waves_and_circular_normal(alpha * 0.9, 1, 1, Mt=3, dimensionality=3, dimensions=(1, 1, 0))

apodization_hexagonal3d = Apodization.AutocorrelationApodizationSIM3D(optical_system3d, hexagonal3d)
apodization_square3d = Apodization.AutocorrelationApodizationSIM3D(optical_system3d, square3d)

# fig, ax, slider = utils.imshow3D(apodization_hexagonal3d.ideal_ctf, cmap='gray', axis='z', vmin=0, vmax=1)
# plt.show()

fig, axes = plt.subplots(2, 4)
axes[0, 0].imshow(apodization_square2d.ideal_ctf)
axes[0, 1].imshow(apodization_hexagonal2d.ideal_ctf)
axes[0, 2].imshow(apodization_square3d.ideal_ctf[:, :, Nz + 45])
axes[0, 3].imshow(apodization_hexagonal3d.ideal_ctf[:, :, Nz + 45])
c1 = axes[1, 0].imshow(apodization_square2d.ideal_otf)
c2 = axes[1, 1].imshow(apodization_hexagonal2d.ideal_otf)
c3 = axes[1, 2].imshow(np.log1p(10 **3 *apodization_square3d.ideal_otf[:, :, Nz]))
c4 = axes[1, 3].imshow(np.log1p(10 **3 * apodization_hexagonal3d.ideal_otf[ :, :, Nz]))
plt.colorbar(c1, ax=axes[1, 0], fraction=0.046, pad=0.04)
plt.colorbar(c2, ax=axes[1, 1], fraction=0.046, pad=0.04)
plt.colorbar(c3, ax=axes[1, 2], fraction=0.046, pad=0.04)
plt.colorbar(c4, ax=axes[1, 3], fraction=0.046, pad=0.04)
axes[0, 0].set_title('a')
axes[0, 1].set_title('b')
axes[0, 2].set_title('c')
axes[0, 3].set_title('d')
axes[1, 0].set_title('e')
axes[1, 1].set_title('f')
axes[1, 2].set_title('g')
axes[1, 3].set_title('h')
axes[0, 0].set_axis_off()
axes[0, 1].set_axis_off()   
axes[0, 2].set_axis_off()
axes[0, 3].set_axis_off()   
axes[1, 0].set_axis_off()
axes[1, 1].set_axis_off()
axes[1, 2].set_axis_off()
axes[1, 3].set_axis_off()
# plt.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.03)
fig.savefig(current_dir + f"/Figures/simulations/ideal_kernels.png", bbox_inches='tight', pad_inches=0.0)
plt.show()


