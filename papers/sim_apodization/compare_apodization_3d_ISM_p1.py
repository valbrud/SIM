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
from Apodization import TriangularApodizationSIM3D, AutocorrelationApodizationSIM3D
from Apodization import AutocorrelationApodizationWidefield, AutocorrelationApodizationISM
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
dx = 1 / (24 * NA)
dz= dx
# dz = 1 / (4 * 1.5 * (1 - np.cos(alpha)))
dy = dx
Nl = 201
Nz = 301
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
optical_system_quasi = System4f3D(alpha=np.pi/2, refractive_index_medium=1.5, refractive_index_sample=1.5, high_NA=True, vectorial=True)
optical_system_quasi.compute_psf_and_otf(((2 * max_r, 2 * max_r, 2* max_z), (Nl, Nl, Nz)))

widefield = configurations.get_widefield(dimensionality=3)
conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, Mr = 3, Mt=1, dimensionality=3)
conventional.clean_plane_waves()
apodization_triangular = TriangularApodizationSIM3D(optical_system, conventional, power=1, Ndense=201)


apodization_filter1 = apodization_triangular.apodization_function
apodization_triangular.power = 0.4 
apodization_filter04 = apodization_triangular.apodization_function

apodization_widefield = AutocorrelationApodizationWidefield(optical_system)

apodization_ISM = AutocorrelationApodizationISM(optical_system_quasi, optical_system)
apodization_filter_ISM = apodization_ISM.apodization_function

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(apodization_filter1[:, :, Nz//2], cmap='gray')
axes[0, 0].set_title("apodization, p=1")
axes[0, 1].imshow(apodization_filter_ISM[:, :, Nz//2], cmap='gray')
axes[0, 1].set_title("ISM")
axes[1, 0].imshow(apodization_filter1[:, Nl//2, :], cmap='gray')
axes[1, 0].set_title("apodization, p=1")
axes[1, 1].imshow(apodization_filter_ISM[:, Nl//2, :], cmap='gray')
axes[1, 1].set_title("ISM")

slider_recon = utils.wrap_axes3d(
    axes,
    [apodization_filter1, apodization_filter_ISM, np.transpose(apodization_filter1, (0, 2, 1)), np.transpose(apodization_filter_ISM, (0, 2, 1))],
    axis='z', cmap='gray', mode='log1p', scaling=10**2
)
plt.show()

apodization_filter_ISM_co = np.where(apodization_filter1, apodization_filter_ISM, 0)

epsf1 = hpc_utils.wrapped_ifftn(apodization_filter1).real
epsfWf = hpc_utils.wrapped_ifftn(apodization_widefield.apodization_function).real
epsfIsm_co = hpc_utils.wrapped_ifftn(apodization_filter_ISM_co).real

epsf1 /= np.amax(epsf1)
epsfWf /= np.amax(epsfWf)
epsfIsm_co /= np.amax(epsfIsm_co)

al = 20
az = 70
airy_cut_l = airy_l[Nl//2-al:Nl//2+al+1]
airy_cut_z = airy_z[Nz//2-az:Nz//2+az+1]
epsf1z= epsf1[Nl//2-az:Nl//2+az+1, Nl//2-az:Nl//2+az+1, Nz//2-az:Nz//2+az+1]
epsfWfz = epsfWf[Nl//2-az:Nl//2+az+1, Nl//2-az:Nl//2+az+1, Nz//2-az:Nz//2+az+1]
epsfIsm_coz = epsfIsm_co[Nl//2-az:Nl//2+az+1, Nl//2-az:Nl//2+az+1, Nz//2-az:Nz//2+az+1]

scalebar_1 = ScaleBar(
    dx= dx*488,  
    units='nm',        
    length_fraction=0.3,
    location='lower left',
    color='white',
    box_alpha=0,       
    label_loc='top',
    font_properties={'size': 25}
    )

scalebar_2 = copy.deepcopy(scalebar_1)
    
fig, axes = plt.subplots(2, 2, figsize=(22, 20))
im00 = axes[0,0].imshow(epsf1z[az - al: az + al + 1, az - al: az + al + 1, az].T, )
axes[0, 0].add_artist(scalebar_1)
axes[0,0].set_title('tPSF, x=1')
im01 = axes[0,1].imshow(epsf04z[az-al: az + al + 1, az - al: az + al + 1, az].T, )
axes[0,1].set_title('tPSF, x=0.4')
im02 = axes[1,0].imshow(epsfIsm_coz[az-al: az + al + 1, az - al: az + al + 1, az].T, )
axes[1,0].set_title('ISM_PSF')
im10 = axes[1,1].imshow(epsf1z[az, ...].T, )
axes[1,1].set_title('tPSF, x=1')
axes[1, 1].add_artist(scalebar_2)
axes[0, 0].set_axis_off()
axes[0, 1].set_axis_off()
axes[0, 1].set_axis_off()
axes[1, 0].set_axis_off()
axes[1, 1].set_axis_off()
axes[0, 0].set_title('a')
axes[0, 1].set_title('b')
axes[0, 2].set_title('c')
axes[1, 0].set_title('d')
axes[1, 1].set_title('e')
axes[1, 2].set_title('f')
axes[2, 0].set_title('g')
axes[2, 1].set_title('h')
axes[2, 2].set_title('i')
# plt.show()
plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04, format='%.2f')
plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04, format='%.2f')

# fix, axes=plt.subplots(1, 2, figsize=(10, 8))
axes[2, 0].plot(airy_cut_l, epsf1[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label= 'p=1')
axes[2, 0].plot(airy_cut_l, epsf04[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label= 'p=0.4')
axes[2, 0].plot(airy_cut_l, epsfA[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label='auto')
axes[2, 0].plot(airy_cut_l, epsfIsm[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label='ISM')
axes[2, 0].plot(airy_cut_l, epsfIsm_co[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label='ISM_co')
axes[2, 0].plot(airy_cut_l, epsfWf[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label='wf')
axes[2, 0].legend()
axes[2, 0].set_ylabel("Intensity [a.u.]")
axes[2, 0].set_xlabel("r [Airy units]")
# axes[2, 0].set_title("")
axes[2,1].plot(airy_cut_l * 2**0.5, np.diag(epsf1[..., Nz//2])[Nl//2-al:Nl//2+al+1], label= 'p=1')
axes[2, 1].plot(airy_cut_l * 2**0.5, np.diag(epsf04[..., Nz//2])[Nl//2-al:Nl//2+al+1], label= 'p=0.4')
axes[2, 1].plot(airy_cut_l * 2**0.5, np.diag(epsfA[..., Nz//2])[Nl//2-al:Nl//2+al+1], label='auto')
axes[2, 1].plot(airy_cut_l * 2**0.5, np.diag(epsfIsm[..., Nz//2])[Nl//2-al:Nl//2+al+1], label='ISM')
axes[2, 1].plot(airy_cut_l * 2**0.5, np.diag(epsfIsm_co[..., Nz//2])[Nl//2-al:Nl//2+al+1], label='ISM_co')
axes[2, 1].plot(airy_cut_l * 2**0.5, np.diag(epsfWf[..., Nz//2])[Nl//2-al:Nl//2+al+1], label='wf')
axes[2, 1].legend()
axes[2, 1].set_xlabel("r [Airy units]")
# axes[2, 1].set_title("x=y, z=0")
axes[2, 2].plot(airy_cut_z, epsf1z[az, az, ...], label= 'p=1')
axes[2, 2].plot(airy_cut_z, epsf04z[az, az, ...], label= 'p=0.4')
axes[2, 2].plot(airy_cut_z, epsfAz[az, az, ...], label='auto')
axes[2, 2].plot(airy_cut_z, epsfIsmz[az, az, ...], label='ISM')
axes[2, 2].plot(airy_cut_z, epsfIsm_coz[az, az, ...], label='ISM_co')
axes[2, 2].plot(airy_cut_z, epsfWfz[az, az, ...], label='wf')
axes[2, 2].legend()
axes[2, 2].set_xlabel("z [Airy units]")
# axes[2, 2].set_title("x=y=0")
fig.savefig(os.path.join(current_dir, "Figures/simulations/effective_PSF3D.png"), bbox_inches='tight')
plt.show()

# square = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 1, 1, Mt=1, dimensionality=3)
# apodization_triangular = TriangularApodizationSIM3D(optical_system, square, power=1)
# apodization_filter1 = apodization_triangular.apodization_function
# apodization_triangular.power = 0.4 
# apodization_filter04 = apodization_triangular.apodization_function
# apodization_autoconvolution = AutocorrelationApodizationSIM3D(optical_system, square)
# apodization_filter_autoconvolution = apodization_autoconvolution.apodization_function

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(apodization_filter1[:, :, Nz//2], cmap='gray')
# axes[0].set_title("apodization, p=1")
# axes[1].imshow(apodization_filter_autoconvolution[:, :, Nz//2], cmap='gray')
# axes[1].set_title("autoconvolution")

# slider_recon = utils.wrap_axes3d(
#     axes,
#     [apodization_filter1, apodization_filter_autoconvolution],
#     axis='y', cmap='gray', mode='log1p', scaling=10**4
# )
# plt.show()

# apodization_filter_autoconvolution = np.where(apodization_filter1, apodization_filter_autoconvolution, 0)

# epsf1 = hpc_utils.wrapped_ifftn(apodization_filter1).real
# epsf04 = hpc_utils.wrapped_ifftn(apodization_filter04).real
# epsfA = hpc_utils.wrapped_ifftn(apodization_filter_autoconvolution).real
# # epsfWf = hpc_utils.wrapped_ifftn(apodization_widefield.apodization_function).real
# epsf1z= epsf1[Nl//2-al:Nl//2+al+1, Nl//2-al:Nl//2+al+1, Nz//2-al:Nz//2+al+1]
# epsf04z= epsf04[Nl//2-al:Nl//2+al+1, Nl//2-al:Nl//2+al+1, Nz//2-al:Nz//2+al+1]
# epsfAz= epsfA[Nl//2-al:Nl//2+al+1, Nl//2-al:Nl//2+al+1, Nz//2-al:Nz//2+al+1]
# epsfWfz= epsfWf[Nl//2-al:Nl//2+al+1, Nl//2-al:Nl//2+al+1, Nz//2-al:Nz//2+al+1]

# epsf1z /= np.amax(epsf1z)
# epsf04z /= np.amax(epsf04z)
# epsfAz /= np.amax(epsfAz)
# epsfWfz /= np.amax(epsfWfz)

# print("t1 sum", np.sum(epsf1))
# print("t1 sum", np.sum(epsf1))
# print("t1 sum", np.sum(epsf1))

# fig, axes = plt.subplots(2, 3, figsize=(10, 5))
# axes[0,0].imshow(epsf1z[..., az].T, )
# axes[0,0].set_title('tPSF, x=1')
# axes[0,1].imshow(epsf04z[..., az].T, )
# axes[0,1].set_title('tPSF, x=0.4')
# axes[0,2].imshow(epsfAz[..., az].T, )
# axes[0,2].set_title('aPSF')
# axes[1,0].imshow(epsf1z[al, ...].T, )
# axes[1,0].set_title('tPSF, x=1')
# # axes[1, 0].set_aspect(1/axes[1,0].get_data_ratio()) 
# axes[1,1].imshow(epsf04z[al, ...].T, )
# # axes[1,1].set_aspect(1/axes[1,1].get_data_ratio())
# axes[1,1].set_title('tPSF, x=0.4')
# axes[1,2].imshow(epsfAz[al, ...].T, )
# # axes[1,2].set_aspect(1/axes[1,2].get_data_ratio())
# axes[1,2].set_title('aPSF')
# axes[0, 0].set_axis_off()
# axes[0, 1].set_axis_off()
# axes[0, 2].set_axis_off()
# axes[1, 0].set_axis_off()
# axes[1, 1].set_axis_off()
# axes[1, 2].set_axis_off()
# axes[0, 0].set_title('a')
# axes[0, 1].set_title('b')
# axes[0, 2].set_title('c')
# axes[1, 0].set_title('d')
# axes[1, 1].set_title('e')
# axes[1, 2].set_title('f')
# # plt.show()

# fix, axes=plt.subplots(1, 2, figsize=(10, 8))
# axes[0].plot(airy_cut_l, epsf1[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label= 'tPSF, p=1')
# axes[0].plot(airy_cut_l, epsf04[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label= 'tPSF, p=0.4')
# axes[0].plot(airy_cut_l, epsfA[Nl//2, Nl//2-al:Nl//2+al+1, Nz//2], label='aPSF')
# axes[0].legend()
# axes[0].set_title("x=0, z=0")
# # axes[1].plot(np.diag(epsf1)[Nl//2-al:Nl//2+al+1, Nz//2], label= 'tPSF, p=1')
# # axes[1].plot(np.diag(epsf04)[Nl//2-al:Nl//2+al+1, Nz//2], label= 'tPSF, p=0.4')
# # axes[1].plot(np.diag(epsfA)[Nl//2-al:Nl//2+al+1, Nz//2], label='aPSF')
# # axes[1].legend()
# # axes[1].set_title("x=y, z=0")
# axes[1].plot(airy_cut_z, epsf1z[al, al, ...], label= 'tPSF, p=1')
# axes[1].plot(airy_cut_z, epsf04z[al, al, ...], label= 'tPSF, p=0.4')
# axes[1].plot(airy_cut_z, epsfAz[al, al, ...], label='aPSF')
# axes[1].legend()
# axes[1].set_title("x=y=0")

# plt.show()