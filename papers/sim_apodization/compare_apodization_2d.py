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
from Apodization import TriangularApodizationSIM2D, AutocorrelationApodizationSIM2D, AutocorrelationApodizationWidefield
import hpc_utils
from matplotlib_scalebar.scalebar import ScaleBar

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
dx = 1 / (128 * NA)
dy = dx
N = 301
max_r = N // 2 * dx
x = np.linspace(-max_r, max_r, N)
airy = x * 2 * NA / 1.22 
fx = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fr = fx[N//2:]/(2 * NA)


optical_system = System4f2D(alpha=alpha, refractive_index=1.5)
optical_system.compute_psf_and_otf(((2 * max_r, 2 * max_r), N))

power = 1
conventional = configurations.get_3_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
apodization_triangular = TriangularApodizationSIM2D(optical_system, conventional, power=power)
apodization_filter1 = apodization_triangular.apodization_function
apodization_triangular.power = 0.4 
apodization_filter04 = apodization_triangular.apodization_function

apodization_widefield = AutocorrelationApodizationWidefield(optical_system)
apodization_filter_widefield = apodization_widefield.apodization_function   

apodization_autoconvolution = AutocorrelationApodizationSIM2D(optical_system, conventional)
apodization_filter_autoconvolution = apodization_autoconvolution.apodization_function

scalebar = ScaleBar(
    dx= dx*488,           
    units='nm',        
    length_fraction=0.3,
    location='lower left',
    color='white',
    box_alpha=0,        # transparent background
    label_loc='top',
    font_properties={'size': 25}
    )
    

epsf1 = hpc_utils.wrapped_ifftn(apodization_filter1).real
epsf04 = hpc_utils.wrapped_ifftn(apodization_filter04).real
epsfA = hpc_utils.wrapped_ifftn(apodization_filter_autoconvolution).real

epsf1 /= np.amax(epsf1)
epsf04 /= np.amax(epsf04)
epsfA /= np.amax(epsfA)
ewfA = hpc_utils.wrapped_ifftn(apodization_filter_widefield).real 
ewfA /= np.amax(np.float32(ewfA))

a = 100

airy_cut = airy[N//2-a:N//2+a+1]

epsf1z= epsf1[N//2-a:N//2+a+1, N//2-a:N//2+a+1]
epsf04z= epsf04[N//2-a:N//2+a+1, N//2-a:N//2+a+1]
epsfAz= epsfA[N//2-a:N//2+a+1, N//2-a:N//2+a+1]
ewfAz= ewfA[N//2-a:N//2+a+1, N//2-a:N//2+a+1]

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
im00 = axes[0, 0].imshow(epsf1z,)
axes[0, 0].set_title('a')
axes[0, 0].add_artist(scalebar)
im01 =axes[0, 1].imshow(epsf04z,)
axes[0, 1].set_title('b')
im02 =axes[0, 2].imshow(epsfAz,)
axes[0, 2].set_title('c')
axes[1, 0].set_title('d')
axes[1, 1].set_title('e')
axes[1, 2].set_title('f')
axes[2, 0].set_title('g')
axes[2, 1].set_title('h')
axes[2, 2].set_title('i')
axes[0, 0].set_axis_off()
axes[0, 1].set_axis_off()
axes[0, 2].set_axis_off()
axes[1, 0].set_axis_off()
axes[1, 1].set_axis_off()
axes[1, 2].set_axis_off()
# axes[2, 0].set_axis_off()
# axes[2, 1].set_axis_off()
# axes[2, 2].set_axis_off()
# plt.show()
# fig.savefig(os.path.join(current_dir, "effective_PSF_hexagonal2D.png"), bbox_inches='tight')

# fig, axes=plt.subplots(figsize=(10, 8))
axes[2, 0].plot(airy_cut, epsf1[N//2, N//2-a:N//2+a+1], label= 'p=1')
axes[2, 0].plot(airy_cut, epsf04[N//2, N//2-a:N//2+a+1], label= 'p=0.4')
axes[2, 0].plot(airy_cut, epsfA[N//2, N//2-a:N//2+a+1], label='auto')
axes[2, 0].plot(airy_cut, ewfA[N//2, N//2-a:N//2+a+1], label='wf')
axes[2, 0].legend()
axes[2, 0].set_aspect(1/axes[2, 0].get_data_ratio())
axes[2, 0].set_xlabel('r [Airy units]')
axes[2, 0].set_ylabel('Intensity [a.u.]')
# fig.savefig(os.path.join(current_dir, "effective_PSF_hexagonal_slice_2D.png"), bbox_inches='tight')

square = configurations.get_4_oblique_s_waves_and_circular_normal(theta, 1, 0, Mt=1, dimensionality=2)
apodization_triangular = TriangularApodizationSIM2D(optical_system, square, power=power)
apodization_filter1 = apodization_triangular.apodization_function
apodization_triangular.power = 0.4 
apodization_filter04 = apodization_triangular.apodization_function

apodization_autoconvolution = AutocorrelationApodizationSIM2D(optical_system, square)
apodization_filter_autoconvolution = apodization_autoconvolution.apodization_function

epsf1 = hpc_utils.wrapped_ifftn(apodization_filter1).real
epsf04 = hpc_utils.wrapped_ifftn(apodization_filter04).real
epsfA = hpc_utils.wrapped_ifftn(apodization_filter_autoconvolution).real
epsf1 /= np.amax(epsf1)
epsf04 /= np.amax(epsf04)
epsfA /= np.amax(epsfA)

epsf1z= epsf1[N//2-a:N//2+a+1, N//2-a:N//2+a+1]
epsf04z= epsf04[N//2-a:N//2+a+1, N//2-a:N//2+a+1]
epsfAz= epsfA[N//2-a:N//2+a+1, N//2-a:N//2+a+1]

print("t1 sum", np.sum(epsf1))
print("t1 sum", np.sum(epsf1))
print("t1 sum", np.sum(epsf1))

# fig, axes = plt.subplots(1, 3, figsize=(10, 5))
im10 = axes[1, 0].imshow(epsf1z)
# axes[0].set_title('a')
im11 = axes[1, 1].imshow(epsf04z)
# axes[1].set_title('b')
im12 = axes[1, 2].imshow(epsfAz)
# axes[2].set_title('c')
# axes[2].set_axis_off()
# axes[1].set_axis_off()
# axes[2].set_axis_off()
# plt.show()
# fig.savefig(os.path.join(current_dir, "effective_PSF_square2D.png"), bbox_inches='tight')

# fig, axes=plt.subplots(1, 2, figsize=(10, 8))
axes[2, 1].plot(airy_cut, epsf1[N//2, N//2-a:N//2+a+1], label= 'p=1')
axes[2, 1].plot(airy_cut, epsf04[N//2, N//2-a:N//2+a+1], label= 'p=0.4')
axes[2, 1].plot(airy_cut, epsfA[N//2, N//2-a:N//2+a+1], label='auto')
axes[2, 1].plot(airy_cut, ewfA[N//2, N//2-a:N//2+a+1], label='wf')
axes[2, 1].legend()
axes[2, 1].set_aspect(1/axes[2, 1].get_data_ratio())
axes[2, 1].set_xlabel('r [Airy units]')
# axes[2, 1].set_ylabel('PSF intensity [a.u.]')
# axes[2, 1].set_title("x=0")
axes[2, 2].plot(airy_cut * 2 **0.5, np.diag(epsf1)[N//2-a:N//2+a+1], label= 'p=1')
axes[2, 2].plot(airy_cut * 2 **0.5, np.diag(epsf04)[N//2-a:N//2+a+1], label= 'p=0.4')
axes[2, 2].plot(airy_cut * 2**0.5, np.diag(epsfA)[N//2-a:N//2+a+1], label='auto')
axes[2, 2].plot(airy_cut * 2**0.5, np.diag(ewfA)[N//2-a:N//2+a+1], label='wf')
axes[2, 2].legend()
axes[2, 2].set_xlabel('r [Airy units]')
# axes[2, 2].set_ylabel('PSF intensity [a.u.]')
axes[2, 2].set_aspect(1/axes[2, 2].get_data_ratio())
plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04, format='%.2f')
plt.tight_layout()
# axes[2, 2].set_title("x=y")
# fig.savefig(os.path.join(current_dir, "Figures/simulations/effective_PSF2D.png"), bbox_inches='tight')
plt.show()