import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import ShapesGenerator
from OpticalSystems import System4f2D
import utils, hpc_utils
import windowing

N = 301
alpha = np.pi / 3
NA = np.sin(alpha)
dx = 1 / (4 * NA)
rmax = dx * N//2
psf_size= (2 * rmax, 2 * rmax)
mask = windowing.make_mask_cosine_edge2d((N, N), 100)
image = ShapesGenerator.generate_random_lines(psf_size, N, 1, 100, 1000, True) + ShapesGenerator.generate_random_spherical_particles(psf_size, N, 0.5, 100, 1000)
# image = ShapesGenerator.generate_random_spherical_particles(psf_size, N, 0.5, 1000, 1000)
# image = utils.load_grayscale_image(current_dir + "\\image_2d.jpg")[:N, 100:100+N]
image = image * mask
# plt.imshow(image)
# plt.show()

optical_system = System4f2D(alpha)
optical_system.compute_psf_and_otf((psf_size, N))
otf = optical_system.otf
otf_projected = np.sum(otf, axis=1)[:, None] * np.ones(N)[None, :] 
otf_projected /= np.amax(otf_projected)
# plt.imshow(otf_projected.real)
# plt.show()


# plt.plot(otf_projected)
# plt.show()
image_convolved = scipy.signal.convolve(image, optical_system.psf, mode='same')

image_deconvolved = hpc_utils.wrapped_ifftn(hpc_utils.wrapped_fftn(image_convolved) / (otf + 1e-3)).real * mask
image_deconvolved_projected = hpc_utils.wrapped_ifftn(hpc_utils.wrapped_fftn(image_convolved) / (otf_projected + 1e-3)).real * mask
image_deconvolved_projectedT = hpc_utils.wrapped_ifftn(hpc_utils.wrapped_fftn(image_convolved) / (otf_projected.T + 1e-3)).real * mask

dif1 = image_deconvolved - image_deconvolved_projected
dif2 = image_deconvolved - image_deconvolved_projectedT
fig, ax = plt.subplots(1, 4)
ax[0].imshow(image_deconvolved)
# ax[1].imshow(image_deconvolved_projected)
ax[1].imshow(dif1)
ax[2].imshow(dif2)
ax[3].imshow(dif1 * dif2)
plt.show()