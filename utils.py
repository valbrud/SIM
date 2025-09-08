"""
utils.py

This module contains commonly used operations on arrays, required in the context of our work.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy
import wrappers

    
def off_grid_ft(array: np.ndarray, grid: np.ndarray, q_values: np.ndarray) -> np.ndarray:
    x_grid_flat = grid.reshape(-1, len(array.shape))
    q_grid_flat = q_values.reshape(-1, len(array.shape))
    phase_matrix = q_grid_flat @ x_grid_flat.T
    fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
    array_ft_values = fourier_exponents @ array.flatten()
    return array_ft_values.reshape(q_values.shape[:-1])


def find_decreasing_surface_levels3d(array: np.ndarray[tuple[int, int, int], np.float64], axes=None, direction=None) -> np.ndarray[tuple[int, int, int], np.int32]:
    """
    Assuming function is monotonically decaying around some point, finds surface levels of this function.
    No interpolation is used.

    Args:
        array (np.ndarray): 3D array to analyze.
        axes (tuple, optional): Axes for the array. Defaults to None.
        direction (int, optional): Direction to analyze. Defaults to None.

    Returns:
        np.ndarray: Mask indicating the surface levels.
    """
    if axes is None:
        ax1, ax2, ax3 = (np.arange(array.shape[0] + 1) - array.shape[0] / 2, np.arange(array.shape[1] + 1) - array.shape[1] / 2,
                         np.arange(array.shape[2] + 1) - array.shape[1] / 2)
    else:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    cx, cy, cz = np.array(array.shape) // 2
    ax1, ax2, ax3 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10], ax3[ax3 >= -1e-10]
    ax_positive = (ax1, ax2, ax3)
    if direction is None:
        max_values = (ax1[-1], ax2[-2], ax3[-3])
        ax, d = ax_positive[np.argmin(max_values)], np.argmin(max_values)
    else:
        ax, d = ax_positive[direction], direction

    mask = np.full(array.shape, -1)
    epsilon = (ax[1] - ax[0]) * 10 ** -10  # to avoid floating point errors
    for i in range(len(ax)):
        if d == 0:
            surface_value = array[cx + i, cy, cz]
        elif d == 1:
            surface_value = array[cx, cy + i, cz]
        elif d == 2:
            surface_value = array[cx, cy, cz + i]
        else:
            raise AttributeError(f"d = {direction} is an incorrect direction for a 3d array")
        mask[(mask == -1) * (array >= surface_value + epsilon)] = i
    return mask


def find_decreasing_surface_levels2d(array: np.ndarray[tuple[int, int], np.float64], axes=None, direction=None) -> np.ndarray[tuple[int, int], np.int32]:
    """
    Assuming function is monotonically decaying around some point, finds surface levels of this function.
    No interpolation is used.

    Args:
        array (np.ndarray): 2D array to analyze.
        axes (tuple, optional): Axes for the array. Defaults to None.
        direction (int, optional): Direction to analyze. Defaults to None.

    Returns:
        np.ndarray: Mask indicating the surface levels.
    """
    if axes is None:
        ax1, ax2 = np.arange(array.shape[0] + 1) - array.shape[0] / 2, np.arange(array.shape[1] + 1) - array.shape[1] / 2
    else:
        ax1, ax2 = axes[0], axes[1]
    cx, cy = np.array(array.shape) // 2
    y, x = np.meshgrid(ax2, ax1)
    ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
    ax_positive = (ax1, ax2)
    if direction is None:
        ax, d = (ax1, 0) if ax1[-1] < ax2[-1] else (ax2, 1)
    else:
        ax, d = ax_positive[direction], direction
    mask = np.full(array.shape, -1)
    epsilon = (ax[1] - ax[0]) * 10 ** -10  # to avoid floating point errors
    for i in range(len(ax)):
        surface_value = array[cx + i, cy] if not d else array[cx, cy + i]
        mask[(mask == -1) * (array >= surface_value + epsilon)] = i
    return mask


def find_decreasing_radial_surface_levels(array, axes=None):
    """Not implemented yet"""
    if axes is None:
        ax1, ax2, ax3 = (np.arange(array.shape[0] + 1) - array.shape[0] / 2, np.arange(array.shape[1] + 1) - array.shape[1] / 2,
                         np.arange(array.shape[2] + 1) - array.shape[1] / 2)
    else:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    return


def average_mask(array: np.ndarray[np.float64], mask: np.ndarray[np.int32], shape='same') -> np.ndarray[np.float64]:
    """
    Averages an array along the surface levels of the mask.

    Args:
        array (np.ndarray): Array to average.
        mask (np.ndarray[np.int32]): Mask indicating regions to average.
        shape (str, optional): Shape of the output array. Defaults to 'same'.

    Returns:
        np.ndarray: Averaged array.
    """
    if shape == 'reduced':
        averaged = np.zeros((np.amax(mask + 1)))
    else:
        averaged = np.zeros(array.shape)
    for i in range(np.amax(mask) + 1):
        elements = array[mask == i]
        average = np.sum(elements) / elements.size
        if shape == 'same':
            averaged[mask == i] = average
        elif shape == 'reduced':
            averaged[i] = average
        else:
            raise AttributeError("Unknown output shape")
    return averaged


# def average_rings2d(array, axes=None):
#     if axes is None:
#         ax1, ax2 = np.arange(array.shape[0]) - array.shape[0]/2, np.arange(array.shape[1]) - array.shape[1] / 2
#     else:
#         ax1, ax2 = axes[0], axes[1]
#
#     y, x = np.meshgrid(ax2, ax1)
#     ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
#     ax = ax1 if ax1[-1] < ax2[-1] else ax2
#     averaged = np.zeros(ax.size)
#     r = (x**2 + y**2)**0.5
#     r_old = -1
#     for i in range(len(ax)):
#         if i == 25:
#             pass
#         ring = array[(r <= ax[i] + 10**(-10)) * (r > r_old)]
#         averaged[i] = np.sum(ring)/ring.size
#         r_old = ax[i] + 10**(-10)
#     return averaged
#     # scipy.interpolate.RegularGridInterpolator((ax1, ax2), array, method='linear',
#     #                                           bounds_error=False,
#     #                                           fill_value=0.)
#     # r = ax1[ax1 >= -1e-10] if ax1[-1] < ax2[-1] else ax2[ax2 >= -1e-10]
#     # theta = np.arange(0, 2 * np.pi, 2 * np.pi / r.size)


def average_rings2d(array: np.ndarray, axes: tuple[np.ndarray] = None, num_angles=360, number_of_samples: int = None):
    """
    Averages the 2D array radially using bilinear interpolation in polar coordinates.

    Parameters:
        array: 2D numpy array to average radially.
        axes: Tuple of arrays representing the grid axes (ax1, ax2).
        num_samples: Number of radial samples (r) to take.
        num_angles: Number of angular samples (theta).

    Returns:
        radii: Radial distances at which the interpolation is performed.
        averaged: Radially averaged values.
    """
    if axes is None:
        ax1, ax2 = np.arange(array.shape[0]) - array.shape[0] / 2, np.arange(array.shape[1]) - array.shape[1] / 2
    else:
        ax1, ax2 = axes[0], axes[1]

    interpolator = RectBivariateSpline(ax1, ax2, array)
    ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
    ax = ax1 if ax1[-1] < ax2[-1] else ax2
    if number_of_samples:
        ax = np.linspace(ax[0], ax[-1], number_of_samples)
    # Create the interpolation function for the array

    averaged = np.zeros(ax.size)

    for i, r in enumerate(ax):
        # Parametric equations for points on a circle of radius r
        theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        sample_x = r * np.cos(theta)
        sample_y = r * np.sin(theta)
        allowed_values = (sample_x >= ax[0]) * (sample_y >= ax[0])
        sample_x = sample_x[allowed_values]
        sample_y = sample_y[allowed_values]
        # Perform bilinear interpolation at these points
        interpolated_values = interpolator.ev(sample_x, sample_y)
        # plt.plot(interpolated_values)
        # plt.show()
        # Radially average by taking the mean of interpolated values
        averaged[i] = np.mean(interpolated_values)

    return averaged


def average_rings3d(array: np.ndarray[tuple[int, int, int], ...], axes: tuple[np.ndarray, np.ndarray, np.ndarray] = None) -> np.ndarray[tuple[int, int], ...]:
    """
    Averages the 3D array radially by averaging each 2D slice.

    Args:
        array (np.ndarray): 3D array to average.
        axes (tuple, optional): Axes for the array. Defaults to None.

    Returns:
        np.ndarray: Radially averaged values.
    """
    if axes is None:
        axes = (np.arange(array.shape[0]) - array.shape[0] / 2, np.arange(array.shape[1]) - array.shape[1] / 2,
                np.arange(array.shape[2]) - array.shape[2] / 2)
    ring_averages = []
    for iz in range(array.shape[2]):
        ring_averages.append(average_rings2d(array[:, :, iz], (axes[0], axes[1])))
    ring_averages = np.array(ring_averages).T
    return ring_averages


def expand_ring_averages2d(averaged: np.ndarray[int, ...], axes: tuple[np.ndarray, np.ndarray] = None) -> np.ndarray[tuple[int, int], ...]:
    """
    Expands the radially averaged 2D array back to its original shape.

    Args:
        averaged (np.ndarray): Radially averaged values.
        axes (tuple, optional): Axes for the array. Defaults to None.

    Returns:
        np.ndarray: Expanded array.
    """
    if axes is None:
        ax1, ax2 = np.arange(averaged.shape[0]) - averaged.shape[0] / 2, np.arange(averaged.shape[0]) - averaged.shape[0] / 2
    else:
        ax1, ax2 = axes[0], axes[1]

    x, y = np.meshgrid(ax1, ax2)
    r = (x ** 2 + y ** 2) ** 0.5
    ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
    ax = ax1 if ax1[-1] < ax2[-1] else ax2
    shape = r.shape
    expanded = np.zeros(shape)
    r_old = -1
    for i in range(len(ax)):
        expanded[(r <= ax[i] + 10 ** -10) * (r > r_old)] = averaged[i]
        r_old = ax[i] + 10 ** -10
    return expanded


def expand_ring_averages3d(averaged: np.ndarray[tuple[int, int], ...], axes: tuple[np.ndarray, np.ndarray, np.ndarray] = None) -> np.ndarray[tuple[int, int, int], ...]:
    """
    Expands the radially averaged 3D array back to its original shape.

    Args:
        averaged (np.ndarray): Radially averaged values.
        axes (tuple, optional): Axes for the array. Defaults to None.

    Returns:
        np.ndarray: Expanded array.
    """
    if axes is None:
        axes = (np.arange(averaged.shape[0]) - averaged.shape[0] / 2, np.arange(averaged.shape[0]) - averaged.shape[0] / 2, np.arange(averaged.shape[1]) - averaged.shape[1] / 2)
    expanded = np.zeros((axes[0].size, axes[1].size, axes[2].size))
    for iz in range(axes[2].size):
        expanded[:, :, iz] = expand_ring_averages2d(averaged[:, iz], axes=(axes[0], axes[1]))
    return expanded


def estimate_localized_peaks(array, axes):
    """
    Estimates localized peaks in a 3D array.
    This particular implementation is very bad but quite universal.

    Args:
        array (np.ndarray): 3D array to analyze.
        axes (tuple): Axes for the array.

    Returns:
        tuple: Localized peaks and their amplitudes.
    """
    if len(array.shape) != 3:
        raise ValueError("Estimation is only available for 3d arrays now")
    array_abs = np.abs(array)
    max_value = np.amax(array_abs)
    array = np.where(array_abs > max_value * 10 ** -4, array, 0)
    array_abs = np.where(array_abs > max_value * 10 ** -2, array_abs, 0)
    maxima_indices = np.array(np.where((array_abs[1:-1, 1:-1, 1:-1] > array_abs[0:-2, 1:-1, 1:-1]) * (array_abs[1:-1, 1:-1, 1:-1] > array_abs[2:, 1:-1, 1:-1]) *
                                       (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 0:-2, 1:-1]) * (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 2:, 1:-1]) *
                                       (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 1:-1, 0:-2]) * (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 1:-1, 2:]))) + 1
    maxima_indices = list(zip(maxima_indices[0], maxima_indices[1], maxima_indices[2]))
    fourier_peaks, std = gaussian_maxima_fitting(array_abs, axes, maxima_indices)
    amplitudes = np.zeros(len(maxima_indices), dtype=np.complex128)
    distances2 = np.zeros((len(fourier_peaks), axes[0].size, axes[1].size, axes[2].size))
    for i in range(len(fourier_peaks)):
        xdist2 = (axes[0] - fourier_peaks[i][0]) ** 2
        ydist2 = (axes[1] - fourier_peaks[i][1]) ** 2
        zdist2 = (axes[2] - fourier_peaks[i][2]) ** 2
        Xdist2, Ydist2, Zdist2 = np.meshgrid(ydist2, xdist2, zdist2)
        Rdist2 = Xdist2 + Ydist2 + Zdist2
        a = np.argmin(Rdist2)
        distances2[i] = Rdist2
    minimal_distance_array = np.min(distances2, axis=0)
    nearest_neighbor_array = np.zeros((len(axes[0]), len(axes[1]), len(axes[2])))
    for i in range(1, len(fourier_peaks)):
        nearest_neighbor_array = np.where(distances2[i] == minimal_distance_array, i, nearest_neighbor_array)
    for i in range(0, len(fourier_peaks)):
        amplitudes[i] = np.sum(array[nearest_neighbor_array == i])
    normalization = np.amax(np.abs(amplitudes))
    amplitudes /= normalization
    return fourier_peaks, amplitudes


def gaussian_maxima_fitting(array, axes, maxima_indices, size=5):
    """
    Fits Gaussian functions to the maxima in a 3D array.

    Args:
        array (np.ndarray): 3D array to analyze.
        axes (tuple): Axes for the array.
        maxima_indices (list): Indices of the maxima.
        size (int, optional): Size of the fitting window. Defaults to 5.

    Returns:
        tuple: Fitted maxima and their standard deviations.
    """
    maxima_fitted = []
    std = []
    for index in maxima_indices:
        fitted_values = array[index[0] - size // 2:index[0] + size // 2 + 1,
                        index[1] - size // 2:index[1] + size // 2 + 1,
                        index[2] - size // 2:index[2] + size // 2 + 1]

        x, y, z = (axes[0][index[0] - size // 2:index[0] + size // 2 + 1],
                   axes[1][index[1] - size // 2:index[1] + size // 2 + 1],
                   axes[2][index[2] - size // 2:index[2] + size // 2 + 1])
        X, Y, Z = np.meshgrid(x, y, z)
        n = np.sum(fitted_values)
        x0 = np.sum(X * fitted_values) / n
        y0 = np.sum(Y * fitted_values) / n
        z0 = np.sum(Z * fitted_values) / n
        sigma = (1 / (2 * n) * np.sum(fitted_values * ((X - x0) ** 2 + (Y - y0) ** 2)) + (Z - z0) ** 2) ** 0.5
        maxima_fitted.append((x0, y0, z0))
        std.append(sigma)

    return np.array(maxima_fitted), np.array(std)


def downsample_circular_function(dense_function, small_size):
    """
    Downsample a circularly symmetric function from a large grid to a smaller grid using a vectorized approach.

    Parameters:
        dense_function: 2D NumPy array representing the function values on the large grid (e.g., 51 x 51).
        small_size: Tuple (m, n) representing the size of the small grid (e.g., (5, 5)).

    Returns:
        small_grid: 2D NumPy array representing the downsampled function on the smaller grid.
    """
    # Get the size of the large grid
    large_size = dense_function.shape

    # Compute scaling factors
    scale_x = large_size[0] // small_size[0]
    scale_y = large_size[1] // small_size[1]

    # Reshape the large grid into blocks of size (scale_x, scale_y)
    reshaped = dense_function.reshape(small_size[0], scale_x, small_size[1], scale_y)

    # Compute the mean of each block to downsample
    sparse_function = reshaped.mean(axis=(1, 3))
    # plt.figure(1)
    # plt.imshow(dense_function)
    # plt.figure(2)
    # plt.imshow(sparse_function)
    # plt.show()
    return sparse_function


def reverse_interpolation_nearest(x_axis, y_axis, points, values):
    """
    AI-generated function. 

    Interpolate values from known points to a grid, affecting only the nearest grid cells.

    Parameters:
        x_axis: 1D array representing the x-coordinates of the grid.
        y_axis: 1D array representing the y-coordinates of the grid.
        points: Array of known points' coordinates, shape (N, 2).
        values: Array of known values at the points, shape (N,).

    Returns:
        interpolated_grid: 2D array of interpolated values on the grid.
    """
    # Create an empty grid
    grid_shape = (len(x_axis), len(y_axis))
    interpolated_grid = np.zeros(grid_shape)

    # Loop through each point and its corresponding value
    for (px, py), value in zip(points, values):
        # Find the indices of the nearest grid points around the given point
        i = np.searchsorted(x_axis, px) - 1
        j = np.searchsorted(y_axis, py) - 1

        # Ensure indices are within valid bounds
        if i < 0 or j < 0 or i >= len(x_axis) - 1 or j >= len(y_axis) - 1:
            continue

        # Coordinates of the four surrounding grid points
        x1, x2 = x_axis[i], x_axis[i + 1]
        y1, y2 = y_axis[j], y_axis[j + 1]

        # Bilinear weights based on the distance to the surrounding points
        wx1 = (x2 - px) / (x2 - x1)
        wx2 = (px - x1) / (x2 - x1)
        wy1 = (y2 - py) / (y2 - y1)
        wy2 = (py - y1) / (y2 - y1)

        # Distribute the value to the four surrounding grid points
        interpolated_grid[i, j] += value * wx1 * wy1  # Top-left
        interpolated_grid[i + 1, j] += value * wx2 * wy1  # Top-right
        interpolated_grid[i, j + 1] += value * wx1 * wy2  # Bottom-left
        interpolated_grid[i + 1, j + 1] += value * wx2 * wy2  # Bottom-right

    return interpolated_grid

def expand_kernel(kernel: np.ndarray, target_shape: tuple[int]) -> np.ndarray:
    """
    Expand a kernel to match the target shape, centering it in the new shape.
    
    Args:
        kernel (np.ndarray): The kernel to expand.
        target_shape (tuple[int]): The desired shape of the kernel.
    Returns:
        np.ndarray: The expanded kernel.
    """

    shape = np.array(kernel.shape, dtype=np.int32)

    if ((shape % 2) == 0).any():
        raise ValueError("Size of the kernel must be odd!")

    if (shape > target_shape).any():
        raise ValueError("Size of the kernel is bigger than of the PSF!")

    if (shape < target_shape).any():
        kernel_expanded = np.zeros(target_shape, dtype=kernel.dtype)

        # Build slice objects for each dimension, to center `kernel_new` in `kernel_expanded`.
        slices = []
        for dim in range(len(shape)):
            center = target_shape[dim] // 2
            half_span = shape[dim] // 2
            start = center - half_span
            stop = start + shape[dim]
            slices.append(slice(start, stop))

        kernel_expanded[tuple(slices)] = kernel
        kernel = kernel_expanded
    return kernel

def upsample(image, factor: int = 2, add_shot_noize: bool = False) -> np.ndarray:
    # Compute new shape after upsampling
    original_shape = np.array(image.shape, dtype=np.int32)
    new_shape = np.round(original_shape * factor).astype(int)

    # Compute Fourier transform of the image
    image_ft = wrappers.wrapped_fftn(image)

    # Compute padding widths for each dimension
    pad_width = []
    for orig, new in zip(original_shape, new_shape):
        pad = (new - orig)//2
        pad_width.append((pad, pad))

    # Pad the Fourier coefficients with zeros
    ft_padded = np.pad(image_ft, pad_width, mode='constant')
    if add_shot_noize:
        # Add shot noise to the padded Fourier coefficients
        variance = np.sum(image)**2 / 2 
        center_slices = tuple(slice(pad, pad + orig) for (pad, _), orig in zip(pad_width, original_shape))
        noise = np.random.normal(0, variance**0.5, ft_padded.shape) + 1j * np.random.normal(0, variance**0.5, ft_padded.shape)
        noise[center_slices] = 0
        ft_padded += noise

    upsampled = wrappers.wrapped_ifftn(ft_padded) 
    return upsampled
