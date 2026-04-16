import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from config.BFPConfigurations import *
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import colors
from ShapesGenerator import generate_random_lines
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from SSNRCalculator import SSNRSIM2D, SSNRWidefield2D
from OpticalSystems import System4f2D
from kernels import psf_kernel2d
from SIMulator import SIMulator2D
from WienerFiltering import filter_true_wiener_sim
path_to_figures = os.path.join(current_dir, 'Figures/simulations/')
path_to_animations = os.path.join(current_dir,'Animations/')
import hpc_utils

np.random.seed(0)

if not os.path.exists(path_to_figures):
    os.makedirs(path_to_figures)
if not os.path.exists(path_to_animations):
    os.makedirs(path_to_animations)

sys.path.append('../../')
plt.rcParams['font.size'] = 40         # Sets default font size
plt.rcParams['axes.titlesize'] = 40     # Title of the axes
plt.rcParams['axes.labelsize'] = 40     # Labels on x and y axes
plt.rcParams['xtick.labelsize'] = 40    # Font size for x-tick labels
plt.rcParams['ytick.labelsize'] = 40    # Font size for y-tick labels
plt.rcParams['legend.fontsize'] = 40    # Font size for legend


if __name__=="__main__": 
    from config.SIM_N100_NA15 import *
    airy_unit = 1.22 / (2 * NA )
    optical_system = System4f2D(alpha=alpha, refractive_index=nobject)
    optical_system.compute_psf_and_otf(((psf_size[0], psf_size[1]), N))
    conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
    conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)
    conventional.set_spatial_shifts_diagonally()
    
    fourier_reconstructor = ReconstructorFourierDomain2D(
        illumination=conventional,
        optical_system=optical_system,
        unitary=False
    )

    spatial_reconstructor1 = ReconstructorSpatialDomain2D(
        illumination=conventional,
        optical_system=optical_system,
        kernel=psf_kernel2d(1, (dx, dx))
    )

    spatial_reconstructor5 = ReconstructorSpatialDomain2D(
        illumination=conventional,
        optical_system=optical_system,
        kernel=psf_kernel2d(5, (dx, dx))
    )

    image = generate_random_lines((10, 10), N, 0.2, 100, 1000)
    print("Generated photon number is ", np.sum(image)/N**2)
    simulator = SIMulator2D(
        illumination=conventional,
        optical_system=optical_system,
        readout_noise_variance=1
    )
 
    ssnr_calc = SSNRSIM2D(
        illumination=conventional,
        optical_system=optical_system,
    )

    simulator = SIMulator2D(conventional, optical_system)
    sim_images = simulator.generate_noiseless_sim_images(image)
    sim_images = simulator.add_noise(sim_images)
    print("Generated photon number in sim images is ", np.sum(sim_images))

    widefield = fourier_reconstructor.get_widefield(sim_images)
    print("Generated photon number in widefield image is ", np.sum(widefield))
    print("Photon number per pixel", np.sum(widefield)/widefield.size)
    reconstructed_fdr = fourier_reconstructor.reconstruct(sim_images)
    reconstructed_sdr = spatial_reconstructor1.reconstruct(sim_images)
    reconstructed_fk = spatial_reconstructor5.reconstruct(sim_images)
    filtered, _, _ = filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed_fdr), ssnr_calc)
    filtered = np.abs(hpc_utils.wrapped_ifftn(filtered))

    images = [
        image,
        widefield,
        reconstructed_fdr,
        reconstructed_sdr,
        reconstructed_fk, 
        filtered
    ]
    titles = [
        'Ground_truth',
        'Widefield image',
        'FDR',
        'SDR',
        'FK', 
        'filtered'
    ]
    print(dx)
    scalebar = ScaleBar(
    dx= 0.0876*488,            # size of one pixel in physical units (e.g., µm/px)
    units='nm',         # 'um', 'nm', 'mm', etc.
    length_fraction=0.3,
    location='lower left',
    color='white',
    box_alpha=0,        # transparent background
    font_properties={'size': 25}
    )
    
    for im, title in zip(images, titles):
        fig, axes = plt.subplots(figsize=(12, 8))
        if title=='Ground_truth': 
            axes.add_artist(scalebar)

        axes.imshow(np.abs(im), cmap='gray', 
        # extent=(x[0] / airy_unit, x[-1] / airy_unit, y[0] / airy_unit, y[-1] / airy_unit),
        )
        axes.set_axis_off()
        # axes.set_xlabel('x $[A. u.]$')
        # axes.axis('off')
        # if title == 'FDR' or title == 'Ground_truth':
            # axes.set_ylabel('y $[A. u.]$')
        # axes.set_xticks('off')
        # axes.set_yticks('off')
        fig.savefig(os.path.join(path_to_figures, title), bbox_inches='tight', dpi=300)
    # plt.show()
