import numpy as np
from scipy.optimize import minimize
def average_ring(array, axes=None):
    if not axes:
        ax1, ax2 = np.arange(array.shape[0]) - array.shape[0]/ 2, np.arange(array.shape[1]) - array.shape[1] / 2
    else:
        ax1, ax2 = axes[0], axes[1]

    x, y = np.meshgrid(ax1, ax2)
    ax1, ax2 = ax1[ax1 >= 0], ax2[ax2 >= 0]
    ax = ax1 if ax1[-1] < ax2[-1] else ax2
    circle_sums = []
    circle_point_numbers = []
    unity_array = np.ones(array.shape)
    for i in range(len(ax), 0, -1):
        r = ax[i - 1]
        mask = x ** 2 + y ** 2 > r ** 2
        np.putmask(array, mask, 0.)
        np.putmask(unity_array, mask, 0.)
        circle_sums.append(np.sum(array))
        circle_point_numbers.append(np.sum(unity_array))
    circle_sums.append(0)
    circle_point_numbers.append(0)
    ring_sums = np.diff(circle_sums)
    ring_numbers = np.diff(circle_point_numbers)
    ring_averages = ring_sums / ring_numbers
    return ring_averages[::-1]

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



