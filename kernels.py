"""
kernels.py

This module contains functions for generating finite size real space kernels for the SSNR calculations.

Functions
    sinc_kernel: Generate a 2D/3D triangular kernel, resulting in :math: `sinc^2` in Fourier space.
    psf_kernel2d: Generate a 2D kernel that has the shape of PSF in the Fourier domain (and hence the shape of OTF in the real space).

"""


import stattools
import numpy as np
import matplotlib.pyplot as plt

def sinc_kernel(kernel_r_size: int, kernel_z_size=1) -> np.ndarray:
    """
    Generate a 2D/3D triangular kernel, resulting in :math: `sinc^2` in Fourier space.

    Args:
        kernel_r_size: The size of the kernel in the radial direction.
        kernel_z_size: The size of the kernel in the axial direction. Default is 1.

    Returns:
        A 2D/3D triangular kernel.
    """
    func_r = np.zeros(kernel_r_size)
    func_r[0:kernel_r_size // 2 + 1] = np.linspace(0, 1, (kernel_r_size + 1) // 2 + 1)[1:]
    func_r[kernel_r_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_r_size + 1) // 2 + 1)[:-1]
    func_z = np.zeros(kernel_z_size)
    func_z[0:kernel_z_size // 2 + 1] = np.linspace(0, 1, (kernel_z_size + 1) // 2 + 1)[1:]
    func_z[kernel_z_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_z_size + 1) // 2 + 1)[:-1]
    kernel = func_r[:, None, None] * func_r[None, :, None] * func_z[None, None, :]
    return kernel

def psf_kernel2d(kernel_size: int, pixel_size: tuple[float, float], dense_kernel_size=50) -> np.ndarray:
    """
    Generate a 2D kernel that has the shape of PSF in the Fourier domain (and hence the shape of OTF in the real space).

    Args:
        kernel_size: The size of the kernel.
        pixel_size: The pixel size in the real space.
        dense_kernel_size: The size of the dense kernel. Default is 50. This parameter is used for better interpolation of the PSF values on a small grid.

    Returns:
        A 2D kernel.
    """
    dx, dy = pixel_size
    dense_kernel_size = dense_kernel_size // kernel_size * kernel_size
    x_max, y_max = dx * (dense_kernel_size//2), dy * (dense_kernel_size//2)
    x = np.linspace(-x_max, x_max, dense_kernel_size)
    y = np.linspace(-y_max, y_max, dense_kernel_size)
    X, Y = np.meshgrid(x, y)
    r = (X ** 2 + Y ** 2) ** 0.5
    R = np.min((x[-1], y[-1]))
    kernel_dense = (2 / np.pi) * (np.arccos(r / R) - (r / R) * (1 - (r / R) ** 2) ** 0.5)
    kernel_dense = np.where(np.isnan(kernel_dense), 0, kernel_dense)
    kernel = stattools.downsample_circular_function(kernel_dense, (kernel_size, kernel_size))
    kernel /= np.sum(kernel) 
    return kernel

