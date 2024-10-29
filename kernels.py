import stattools
import numpy as np
import matplotlib.pyplot as plt

def sinc_kernel(kernel_r_size, kernel_z_size=1):
    func_r = np.zeros(kernel_r_size)
    func_r[0:kernel_r_size // 2 + 1] = np.linspace(0, 1, (kernel_r_size + 1) // 2 + 1)[1:]
    func_r[kernel_r_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_r_size + 1) // 2 + 1)[:-1]
    func_z = np.zeros(kernel_z_size)
    func_z[0:kernel_z_size // 2 + 1] = np.linspace(0, 1, (kernel_z_size + 1) // 2 + 1)[1:]
    func_z[kernel_z_size // 2: kernel_r_size] = np.linspace(1, 0, (kernel_z_size + 1) // 2 + 1)[:-1]
    kernel = func_r[:, None, None] * func_r[None, :, None] * func_z[None, None, :]
    return kernel

def psf_kernel2d(kernel_size, pixel_size, dense_kernel_size=50):
    dx, dy = pixel_size
    dense_kernel_size = dense_kernel_size // kernel_size * kernel_size
    x_max, y_max = dx * dense_kernel_size//2, dy * dense_kernel_size//2
    x = np.linspace(-x_max, x_max, dense_kernel_size)
    y = np.linspace(-y_max, y_max, dense_kernel_size)
    X, Y = np.meshgrid(x, y)
    r = (X ** 2 + Y ** 2) ** 0.5
    R = np.min((x[-1], y[-1]))
    kernel_dense = (2 / np.pi) * (np.arccos(r / R) - (r / R) * (1 - (r / R) ** 2) ** 0.5)
    kernel_dense = np.where(np.isnan(kernel_dense), 0, kernel_dense)
    kernel = np.zeros((kernel_size, kernel_size, 1))
    kernel[:, :, 0] = stattools.downsample_circular_function_vectorized(kernel_dense, (kernel_size, kernel_size))
    kernel /= np.amax(kernel)
    return kernel

