import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from matplotlib import image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from config.BFPConfigurations import BFPConfiguration
configurations = BFPConfiguration()

from OpticalSystems import System4f2D
from SSNRCalculator import SSNRSIM2D
from SIMulator import SIMulator2D
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D
from Illumination import IlluminationPlaneWaves2D
import ShapesGenerator
import hpc_utils
from WienerFiltering import filter_true_wiener_sim
from kernels import psf_kernel2d
from PatternEstimation.PatternEstimator import IlluminationPatternEstimator2D
np.random.seed(1234)
from PIL import Image
from Apodization import AutocorrelationApodizationSIM2D

N = (301, 636)

alpha = np.pi/4
# theta = 2 * np.pi / 12
nmedium = 1.5
theta = np.arcsin(0.9 * np.sin(alpha))
dimensions = (1, 1)
NA = nmedium * np.sin(alpha)
dx = 1 / (8 * NA)
max_x = N[0] // 2 * dx
max_y = N[1] // 2 * dx

psf_size = np.array((2 * max_x, 2 * max_y))
x = np.linspace(-max_y, max_y, N[0], endpoint = N[0]%2)
y = np.linspace(-max_y, max_y, N[1], endpoint = N[1]%2)

fx = np.linspace(-1/(2 * dx), 1/(2 * dx), N[0], endpoint = N[0]%2)
fy = np.linspace(-1/(2 * dx), 1/(2 * dx), N[1], endpoint = N[1]%2)

fxn = fx / (2 * NA)
fyn = fy / (2 * NA)

image_proper = Image.open(current_dir + '/ProperOTF.png').convert('L')
image_proper = np.array(image_proper, dtype=np.float32) 
print(image_proper.shape)
print(np.sum(image_proper)/N[0]/N[1])
# plt.title("Image")
image_proper = (np.amax(image_proper) - image_proper)
image_proper*=100

im = plt.imshow(image_proper)
plt.colorbar(im)
plt.show()

optical_system = System4f2D(alpha=alpha, refractive_index=nmedium)
optical_system.compute_psf_and_otf((psf_size, N), zernike={(2, 0):0.1, (2, 2):0.1, (3, 1):-0.05, (4, 0):-0.05})


# plt.imshow(optical_system.psf[N[0]//2-10:N[0]//2+10, N[1]//2-10:N[1]//2+10].real)
# plt.show()


widefield = scipy.signal.convolve(image_proper, optical_system.psf, mode='same')

configurations = BFPConfiguration(refraction_index=1.5)
illumination = configurations.get_2_oblique_s_waves_and_s_normal(
    theta, 1, 0, 3, Mt=1, dimensionality=2
)


illumination.set_spatial_shifts_diagonally()
# plt.imshow(illumination.get_illumination_density(coordinates=(x, x)))
# plt.show()

simulator = SIMulator2D(illumination, optical_system, readout_noise_variance=1)
sim_images = simulator.generate_noiseless_sim_images(image_proper)

noisy_images = simulator.add_noise(sim_images)

reconstructor = ReconstructorFourierDomain2D(illumination, optical_system)
reconstructed = reconstructor.reconstruct(noisy_images)

plt.imshow(np.real(reconstructed))
plt.title("Reconstructed image")
plt.show()

calc = SSNRSIM2D(illumination, optical_system)

filtered, _, ssnr = filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed), calc)

apodization = AutocorrelationApodizationSIM2D(optical_system, illumination)
apodized_ft = filtered * apodization.ideal_otf
apodized = hpc_utils.wrapped_ifftn(apodized_ft)

plt.imshow(np.real(apodized), cmap='gray')
plt.show()


image_improper = Image.open(current_dir + '/ImproperOTF.png').convert('L')
image_improper = np.array(image_improper, dtype=np.float32) 
print(image_improper.shape)
print(np.sum(image_proper)/N[0]/N[1])
# plt.title("Image")
image_improper = (np.amax(image_improper) - image_improper)*100

im = plt.imshow(image_improper, vmin=0)
plt.colorbar(im)
plt.show()

optical_system_reconstruction = System4f2D(alpha=alpha * 1, refractive_index=nmedium )
optical_system_reconstruction.compute_psf_and_otf((psf_size, N))

plt.imshow(optical_system.psf[N[0]//2-10:N[0]//2+10, N[1]//2-10:N[1]//2+10].real)
plt.show()


widefield = scipy.signal.convolve(image_improper, optical_system.psf, mode='same')

configurations = BFPConfiguration(refraction_index=1.5)
illumination = configurations.get_2_oblique_s_waves_and_s_normal(
    theta, 1, 0, 3, Mt=1, dimensionality=2
)


illumination.set_spatial_shifts_diagonally()
# plt.imshow(illumination.get_illumination_density(coordinates=(x, x)))
# plt.show()

simulator = SIMulator2D(illumination, optical_system, readout_noise_variance=1)
sim_images = simulator.generate_noiseless_sim_images(image_improper)

noisy_images = simulator.add_noise(sim_images)

reconstructor = ReconstructorFourierDomain2D(illumination, optical_system_reconstruction)
reconstructed = reconstructor.reconstruct(noisy_images)

plt.imshow(np.real(reconstructed))
plt.title("Reconstructed image")
plt.show()

calc = SSNRSIM2D(illumination, optical_system_reconstruction)

filtered, _, ssnr = filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed), calc)

apodization = AutocorrelationApodizationSIM2D(optical_system_reconstruction, illumination)
apodized_ft = filtered * apodization.ideal_otf
apodized = hpc_utils.wrapped_ifftn(apodized_ft)

plt.imshow(np.real(apodized), cmap='gray')
plt.show()

