"""
ShapesGenerator.py

This module contains functions for generating various simulated images used in simulations.

Functions:
    generate_random_spheres: Generates an array with random spheres.
    generate_sphere_slices: Generates an array with slices of spheres.
    generate_random_lines: Generates an image with randomly oriented lines.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
def generate_random_spheres(psf_size, point_number, r = 0.1, N=10, I = 1000):
    np.random.seed(1234)
    indices = np.array(np.meshgrid(np.arange(point_number), np.arange(point_number),
                                   np.arange(point_number))).T.reshape(-1, 3)
    indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
        point_number, point_number, point_number, 3)
    grid = (indices / point_number - 1 / 2)
    shape = grid.shape[:3]
    array = np.zeros(shape)
    grid *= psf_size[None, None, None, :]
    sx, sy, sz = psf_size
    cx = np.random.rand(N) * sx - sx//2
    cy = np.random.rand(N) * sy - sy//2
    cz = np.random.rand(N) * sz - sz//2
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i])**2, axis=3)
        array[dist2 < r**2] += I
    return array

def generate_sphere_slices(psf_size, point_number, r = 0.1, N=10, I = 1000):
    np.random.seed(1234)
    indices = np.array(np.meshgrid(np.arange(point_number), np.arange(point_number),
                                   np.arange(point_number))).T.reshape(-1, 3)
    indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
        point_number, point_number, point_number, 3)
    grid = (indices / point_number - 1 / 2)
    shape = grid.shape[:3]
    array = np.zeros(shape)
    grid *= psf_size[None, None, None, :]
    sx, sy, sz = psf_size
    cx = np.random.rand(N) * sx - sx//2
    cy = np.random.rand(N) * sy - sy//2
    # cz = (np.random.rand(N) * sz - sz//2)/sz * r
    cz = np.zeros(N)
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i])**2, axis=3)
        array[dist2 < r**2] += I
    return array


def generate_random_lines(point_number, psf_size, line_width, num_lines, intensity):
    """
    Generate an image with randomly oriented lines.

    :param point_number: Number of points defining the size of the image grid (image will be point_number x point_number).
    :param psf_size: Tuple of (psf_x_size, psf_y_size) defining scaling in x and y directions.
    :param line_width: Width of the lines.
    :param num_lines: Number of lines to generate.
    :param intensity: Total intensity of each line.
    :return: Generated image with lines.
    """
    np.random.seed(1234)

    # Calculate the grid spacing based on psf_size and point_number
    dx = psf_size[0] / point_number
    dy = psf_size[1] / point_number

    # Create an empty image of size (point_number, point_number)
    image = np.zeros((point_number, point_number), dtype=np.float32)

    for _ in range(num_lines):
        # Randomly generate start and end points for the line within the scaled grid
        x1, y1 = np.random.uniform(0, point_number * dx), np.random.uniform(0, point_number * dy)
        x2, y2 = np.random.uniform(0, point_number * dx), np.random.uniform(0, point_number * dy)

        # Calculate line points using interpolation between start and end points
        num_points = int(max(abs(x2 - x1) / dx, abs(y2 - y1) / dy)) + 1
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        # Convert floating points to image grid indices using bilinear intensity distribution
        for x, y in zip(x_coords, y_coords):
            x_idx = x / dx
            y_idx = y / dy

            x_low = int(np.floor(x_idx))
            y_low = int(np.floor(y_idx))
            x_high = min(x_low + 1, point_number - 1)
            y_high = min(y_low + 1, point_number - 1)

            # Bilinear interpolation to distribute intensity between neighboring pixels
            image[x_low, y_low] += intensity / num_points * (x_high - x_idx) * (y_high - y_idx)
            image[x_high, y_low] += intensity / num_points * (x_idx - x_low) * (y_high - y_idx)
            image[x_low, y_high] += intensity / num_points * (x_high - x_idx) * (y_idx - y_low)
            image[x_high, y_high] += intensity / num_points * (x_idx - x_low) * (y_idx - y_low)

    # Apply Gaussian smoothing to create smooth line edges
    smoothed_image = gaussian_filter(image, sigma=(line_width, line_width))

    # Normalize intensity so that each line has the specified total intensity
    if smoothed_image.max() > 0:
        smoothed_image = (smoothed_image / smoothed_image.max()) * intensity

    return smoothed_image