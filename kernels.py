"""
kernels.py

This module contains functions for generating finite size real space kernels for the SSNR calculations.

Functions
    sinc_kernel: Generate a 2D/3D triangular kernel, resulting in :math: `sinc^2` in Fourier space.
    psf_kernel2d: Generate a 2D kernel that has the shape of PSF in the Fourier domain (and hence the shape of OTF in the real space).

"""


import utils
import numpy as np
import matplotlib.pyplot as plt

def sinc_kernel1d(kernel_size: int) -> np.ndarray:
    """
    Generate a 1D triangular kernel, resulting in :math: `sinc^2` in Fourier space.

    Args:
        kernel_r_size: The size of the kernel in the radial direction.

    Returns:
        A 1D triangular kernel.
    """
    func_r = np.zeros(kernel_size)
    func_r[0:kernel_size // 2 + 1] = np.linspace(0, 1, (kernel_size + 1) // 2 + 1)[1:]
    func_r[kernel_size // 2: kernel_size] = np.linspace(1, 0, (kernel_size + 1) // 2 + 1)[:-1]
    return func_r



def sinc_kernel2d(kernel_size: int) -> np.ndarray:
    """
    Generate a 2D triangular kernel, resulting in :math: `sinc^2` in Fourier space.

    Args:
        kernel_r_size: The size of the kernel in the radial direction.

    Returns:
        A 2D triangular kernel.
    """
    kernel = sinc_kernel1d(kernel_size)[:, None] * sinc_kernel1d(kernel_size)[None, :]
    return kernel

def sinc_kernel3d(kernel_r_size: int, kernel_z_size=1) -> np.ndarray:
    """
    Generate a 3D triangular kernel, resulting in :math: `sinc^2` in Fourier space.

    Args:
        kernel_r_size: The size of the kernel in the radial direction.
        kernel_z_size: The size of the kernel in the axial direction. Default is 1.

    Returns:
        A 2D/3D triangular kernel.
    """
    kernel = sinc_kernel1d(kernel_r_size)[:, None,  None] * sinc_kernel1d(kernel_r_size)[None, :, None] * sinc_kernel1d(kernel_z_size)[None, None, :]
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
    kernel = utils.downsample_circular_function(kernel_dense, (kernel_size, kernel_size))
    kernel /= np.sum(kernel) 
    return kernel


def angular_notch_kernel( 
    kernel_size_px: int, 
    cn: int, 
    f_start: float, 
    f_stop: float = 0.5, 
    beta: float = 0.3, 
    fb: float = 0.45, 
    theta0: float = 0.0, 
    design_grid: int | None = None, 
) -> np.ndarray: 
    """ 
    Design a small real-space convolution kernel whose frequency response suppresses 
    m-fold (Cn) anisotropic peaks starting at a given radius. 
     
    Parameters 
    ---------- 
    kernel_size_px : int 
        Odd spatial kernel size in pixels (e.g., 5, 7, 9, 11, 13). 
    cn : int 
        Rotational symmetry order (2, 3, 4, 6, ...). Uses cos(cn * theta). 
    f_start : float 
        Start radius in cycles/pixel where the angular notch begins (e.g., 0.20–0.30). 
        Use ~0.25 if peaks cluster near half-cutoff for Nyquist-sampled data. 
    f_stop : float, optional 
        Stop radius of the notch (default 0.5, i.e., Nyquist). The notch ramps smoothly 
        from f_start to f_stop using a raised-cosine. Must satisfy 0 < f_start < f_stop ≤ 0.5. 
    beta : float, optional 
        Notch depth (0–1 typical). Larger beta → stronger isotropization (more peak suppression). 
    fb : float, optional 
        Width of an isotropic Gaussian base exp(-(R/fb)^2) that keeps the spatial kernel compact. 
        Larger fb → tighter kernel (less base low-pass). 
    theta0 : float, optional 
        Angular offset in radians to align the notch with rotated peak constellations. 
    design_grid : int, optional 
        Odd size of the design grid in frequency domain (defaults to max(65, 8*N+1)). 
     
    Returns 
    ------- 
    K : (N, N) np.ndarray 
        Real-valued kernel normalized to sum to 1 (DC-preserving). 
    """ 
    N = int(kernel_size_px) 
    if N % 2 != 1: 
        raise ValueError("kernel_size_px must be odd.") 
    if not (0.0 < f_start < f_stop <= 0.5): 
        raise ValueError("Require 0 < f_start < f_stop ≤ 0.5 (cycles/pixel).") 
    if cn < 2: 
        raise ValueError("cn must be ≥ 2.") 
     
    # Frequency design grid (larger than spatial kernel to minimize edge effects) 
    M = design_grid if design_grid is not None else max(65, 8 * N + 1) 
    if M % 2 != 1: 
        M += 1  # enforce odd 
    u = np.fft.fftfreq(M, d=1.0)  # cycles/pixel, in [-0.5, 0.5) 
    U, V = np.meshgrid(u, u, indexing="xy") 
    U2, V2 = U**2, V**2
    R = np.sqrt(U2 + V2) 
    Theta = np.arctan2(V, U) 
     
    # Base isotropic apodization to keep the spatial kernel small 
    B = np.exp(-(R / fb) ** 2) 
     
    # Angular term with Cn symmetry 
    angular = np.cos(cn * (Theta - theta0)) 
     
    # Smooth radial mask that ramps from f_start to f_stop (raised cosine) 
    # 0           for R < f_start 
    # 0.5*(1-cos) for f_start ≤ R < f_stop 
    # 1           for R ≥ f_stop 
    t = np.clip((R - f_start) / (f_stop - f_start), 0.0, 1.0) 
    radial_mask = 0.5 * (1 - np.cos(np.pi * t)) 
     
    # Frequency response of the kernel 
    Khat = B * (1.0 - beta * angular * radial_mask) 
     
    # Transform to spatial domain and center 
    k_spatial = np.fft.ifft2(np.fft.ifftshift(Khat)).real 
    k_spatial = np.fft.fftshift(k_spatial) 
     
    # Crop to N×N and normalize DC gain to 1 
    c = M // 2 
    r0 = c - (N // 2) 
    r1 = c + (N // 2) + 1 
    K = k_spatial[r0:r1, r0:r1] 
    K /= K.sum() 
    return K 
 

if __name__=="__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=140) 
    
    K_C4_9 = angular_notch_kernel(kernel_size_px=9, cn=4, f_start=0.22, f_stop=0.5, beta=0.3, fb=0.45, theta0=0.0) 
    K_C6_7 = angular_notch_kernel(kernel_size_px=7, cn=6, f_start=0.22, f_stop=0.5, beta=0.3, fb=0.5, theta0=0.0) 

    print("9×9 kernel, C4 symmetry (start 0.22 cyc/px):\n", K_C4_9, "\n") 
    print("7×7 kernel, C6 symmetry (start 0.22 cyc/px):\n", K_C6_7)

    plt.imshow(K_C4_9)
    plt.show()
    plt.imshow(K_C6_7)
    plt.show()