import numpy as np

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