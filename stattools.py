import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy
def find_decreasing_surface_levels3d(array, axes=None, direction=None):
    if axes is None:
        ax1, ax2, ax3 = (np.arange(array.shape[0] + 1) - array.shape[0]/2, np.arange(array.shape[1] + 1) - array.shape[1] / 2,
                                            np.arange(array.shape[2] + 1) - array.shape[1] / 2)
    else:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    cx, cy, cz = np.array(array.shape)//2
    ax1, ax2, ax3 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10], ax3[ax3 >= -1e-10]
    ax_positive = (ax1, ax2, ax3)
    if direction is None:
        max_values = (ax1[-1], ax2[-2], ax3[-3])
        ax, d = ax_positive[np.argmin(max_values)], np.argmin(max_values)
    else:
        ax, d = ax_positive[direction], direction

    mask = np.full(array.shape, -1)
    epsilon = (ax[1] - ax[0]) * 10**-10 # to avoid floating point errors
    for i in range(len(ax)):
        if d == 0:
            surface_value = array[cx + i, cy, cz]
        elif d == 1:
            surface_value = array[cx, cy + i, cz]
        elif d == 2:
            surface_value = array[cx, cy, cz + i]
        else:
            raise AttributeError(f"d = {direction} is an incorrect direction for a 3d array")
        mask[(mask == -1) * (array >=  surface_value + epsilon)] = i
    return mask

def find_decreasing_surface_levels2d(array, axes=None, direction = None):
    if axes is None:
        ax1, ax2 = np.arange(array.shape[0] + 1) - array.shape[0]/2, np.arange(array.shape[1] + 1) - array.shape[1] / 2
    else:
        ax1, ax2 = axes[0], axes[1]
    cx, cy = np.array(array.shape)//2
    y, x = np.meshgrid(ax2, ax1)
    ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
    ax_positive = (ax1, ax2)
    if direction is None:
        ax, d = (ax1, 0) if ax1[-1] < ax2[-1] else (ax2, 1)
    else:
        ax, d = ax_positive[direction], direction
    mask = np.full(array.shape, -1)
    epsilon = (ax[1] - ax[0]) * 10**-10 # to avoid floating point errors
    for i in range(len(ax)):
        surface_value = array[cx + i, cy] if not d else array[cx, cy + i]
        mask[(mask == -1) * (array >= surface_value + epsilon)] = i
    return mask

def find_decreasing_radial_surface_levels(array, axes=None):
    if axes is None:
        ax1, ax2, ax3 = (np.arange(array.shape[0] + 1) - array.shape[0]/2, np.arange(array.shape[1] + 1) - array.shape[1] / 2,
                                            np.arange(array.shape[2] + 1) - array.shape[1] / 2)
    else:
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    return
def average_mask(array, mask, shape='same'):
    if shape == 'reduced':
        averaged = np.zeros((np.amax(mask + 1)))
    else:
        averaged = np.zeros(array.shape)
    for i in range(np.amax(mask) + 1):
        elements = array[mask == i]
        average = np.sum(elements)/elements.size
        if shape == 'same':
            averaged[mask == i] = average
        elif shape == 'reduced':
            averaged[i] = average
        else:
            raise AttributeError("Unknown output shape")
    return averaged
def average_rings2d(array, axes=None):
    if axes is None:
        ax1, ax2 = np.arange(array.shape[0]) - array.shape[0]/2, np.arange(array.shape[1]) - array.shape[1] / 2
    else:
        ax1, ax2 = axes[0], axes[1]

    y, x = np.meshgrid(ax2, ax1)
    ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
    ax = ax1 if ax1[-1] < ax2[-1] else ax2
    averaged = np.zeros(ax.size)
    r = (x**2 + y**2)**0.5
    r_old = -1
    for i in range(len(ax)):
        if i == 25:
            pass
        ring = array[(r <= ax[i] + 10**(-10)) * (r > r_old)]
        averaged[i] = np.sum(ring)/ring.size
        r_old = ax[i] + 10**(-10)
    return averaged
    # circle_sums = []
    # circle_point_numbers = []
    # unity_array = np.ones(array.shape)
    # for i in range(len(ax), 0, -1):
    #     r = ax[i - 1]
    #     mask = x ** 2 + y ** 2 > r ** 2
    #     np.putmask(array, mask, 0.)
    #     np.putmask(unity_array, mask, 0.)
    #     circle_sums.append(np.sum(array))
    #     circle_point_numbers.append(np.sum(unity_array))
    # circle_sums.append(0)
    # circle_point_numbers.append(0)
    # ring_sums = np.diff(circle_sums)
    # ring_numbers = np.diff(circle_point_numbers)
    # ring_averages = ring_sums / ring_numbers
    # return ring_averages[::-1]

def average_rings3d(array, axes=None):
    if axes is None:
        axes = (np.arange(array.shape[0]) - array.shape[0]/2, np.arange(array.shape[1]) - array.shape[1]/2,
                np.arange(array.shape[2]) - array.shape[2]/2)
    ring_averages = []
    for iz in range(array.shape[2]):
        ring_averages.append(average_rings2d(array[:, :, iz], (axes[0], axes[1])))
    ring_averages = np.array(ring_averages).T
    return ring_averages

def expand_ring_averages2d(averaged, axes=None):
    if axes is None:
        ax1, ax2 = np.arange(averaged.shape[0]) - averaged.shape[0]/ 2, np.arange(averaged.shape[0]) - averaged.shape[0] / 2
    else:
        ax1, ax2 = axes[0], axes[1]

    x, y = np.meshgrid(ax1, ax2)
    r = (x**2 + y**2)**0.5
    ax1, ax2 = ax1[ax1 >= -1e-10], ax2[ax2 >= -1e-10]
    ax = ax1 if ax1[-1] < ax2[-1] else ax2
    shape = r.shape
    expanded = np.zeros(shape)
    r_old = -1
    for i in range(len(ax)):
        expanded[(r <= ax[i]+10**-10) * (r > r_old)] = averaged[i]
        r_old = ax[i] + 10**-10
    return expanded

def expand_ring_averages3d(averaged, axes=None):
    if axes is None:
        axes = (np.arange(averaged.shape[0]) - averaged.shape[0]/2, np.arange(averaged.shape[0]) - averaged.shape[0]/2, np.arange(averaged.shape[1]) - averaged.shape[1]/2)
    expanded = np.zeros((axes[0].size, axes[1].size, axes[2].size))
    for iz in range(axes[2].size):
        expanded[:, :, iz] = expand_ring_averages2d(averaged[:, iz], axes=(axes[0], axes[1]))
    return expanded


def estimate_localized_peaks(array, axes):
    if len(array.shape) != 3:
        raise ValueError("Estimation is only available for 3d arrays now")
    array_abs = np.abs(array)
    max_value = np.amax(array_abs)
    array = np.where(array_abs > max_value * 10**-4, array, 0)
    array_abs = np.where(array_abs > max_value * 10**-2, array_abs, 0)
    maxima_indices = np.array(np.where((array_abs[1:-1, 1:-1, 1:-1] > array_abs[0:-2, 1:-1, 1:-1]) * (array_abs[1:-1, 1:-1, 1:-1] > array_abs[2:, 1:-1, 1:-1]) *
                                       (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 0:-2, 1:-1]) * (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 2:, 1:-1]) *
                                       (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 1:-1, 0:-2]) * (array_abs[1:-1, 1:-1, 1:-1] > array_abs[1:-1, 1:-1, 2:]))) + 1
    maxima_indices = list(zip(maxima_indices[0], maxima_indices[1], maxima_indices[2]))
    fourier_peaks, std = gaussian_maxima_fitting(array_abs, axes, maxima_indices)
    amplitudes = np.zeros(len(maxima_indices), dtype=np.complex128)
    distances2 = np.zeros((len(fourier_peaks), axes[0].size, axes[1].size, axes[2].size))
    for i in range(len(fourier_peaks)):
        xdist2 = (axes[0] - fourier_peaks[i][0])**2
        ydist2 = (axes[1] - fourier_peaks[i][1])**2
        zdist2 = (axes[2] - fourier_peaks[i][2])**2
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
    maxima_fitted = []
    std = []
    for index in maxima_indices:
        fitted_values = array[index[0]-size//2:index[0] + size//2 + 1,
                              index[1]-size//2:index[1] + size//2 + 1,
                              index[2]-size//2:index[2] + size//2 + 1]

        x, y, z = (axes[0][index[0]-size//2:index[0] + size//2 + 1],
                   axes[1][index[1]-size//2:index[1] + size//2 + 1],
                   axes[2][index[2]-size//2:index[2] + size//2 + 1])
        X, Y, Z = np.meshgrid(x, y, z)
        n = np.sum(fitted_values)
        x0 = np.sum(X * fitted_values) / n
        y0 = np.sum(Y * fitted_values) / n
        z0 = np.sum(Z * fitted_values) / n
        sigma = (1 / (2 * n) * np.sum(fitted_values * ((X - x0) ** 2 + (Y - y0) ** 2)) + (Z - z0)**2) ** 0.5
        maxima_fitted.append((x0, y0, z0))
        std.append(sigma)

    return np.array(maxima_fitted), np.array(std)


def downsample_circular_function_vectorized(dense_function, small_size):
    """
    Downsample a circularly symmetric function from a large grid to a smaller grid using a vectorized approach.

    Parameters:
    - dense_function: 2D NumPy array representing the function values on the large grid (e.g., 51 x 51).
    - small_size: Tuple (m, n) representing the size of the small grid (e.g., (5, 5)).

    Returns:
    - small_grid: 2D NumPy array representing the downsampled function on the smaller grid.
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
    Interpolate values from known points to a grid, affecting only the nearest grid cells.

    Parameters:
    - x_axis: 1D array representing the x-coordinates of the grid.
    - y_axis: 1D array representing the y-coordinates of the grid.
    - points: Array of known points' coordinates, shape (N, 2).
    - values: Array of known values at the points, shape (N,).

    Returns:
    - interpolated_grid: 2D array of interpolated values on the grid.
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