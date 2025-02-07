"""
ShapesGenerator.py

This module contains functions for generating various simulated images used in simulations.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_random_spherical_particles(image_size: np.ndarray[[int, int, int], np.int64], point_number: int, r=0.1, N=10, I=1000) -> np.ndarray:
    """
    Generates an array with random spheres.

    Args:
        image_size (np.ndarray[[int, int, int], np.int64]): Size of the point spread function in each dimension.
        point_number (int): Number of points in each dimension.
        r (float, optional): Radius of the spheres. Defaults to 0.1.
        N (int, optional): Number of spheres to generate. Defaults to 10.
        I (int, optional): Intensity of the spheres. Defaults to 1000.

    Returns:
        np.ndarray: Array with random spheres.
    """
    # np.random.seed(1234)
    indices = np.array(np.meshgrid(np.arange(point_number), np.arange(point_number),
                                   np.arange(point_number))).T.reshape(-1, 3)
    indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
        point_number, point_number, point_number, 3)
    grid = (indices / point_number - 1 / 2)
    shape = grid.shape[:3]
    array = np.zeros(shape)
    grid *= image_size[None, None, None, :]
    sx, sy, sz = image_size
    cx = np.random.rand(N) * sx - sx // 2
    cy = np.random.rand(N) * sy - sy // 2
    cz = np.random.rand(N) * sz - sz // 2
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i]) ** 2, axis=3)
        array[dist2 < r ** 2] += I
    return array


def generate_sphere_slices(image_size: tuple[int, int, int], point_number: int, r=0.1, N=10, I=1000) -> np.ndarray:
    """
    Generates a thin slice with random spheres.

    Args:
        image_size (tuple[int, int, int]): Size of the point spread function in each dimension.
        point_number (int): Number of points in each dimension.
        r (float, optional): Radius of the spheres. Defaults to 0.1.
        N (int, optional): Number of spheres to generate. Defaults to 10.
        I (int, optional): Intensity of the spheres. Defaults to 1000.

    Returns:
        np.ndarray: A thin slice of random spheres.
    """
    np.random.seed(1234)
    indices = np.array(np.meshgrid(np.arange(point_number), np.arange(point_number),
                                   np.arange(point_number))).T.reshape(-1, 3)
    indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
        point_number, point_number, point_number, 3)
    grid = (indices / point_number - 1 / 2)
    shape = grid.shape[:3]
    array = np.zeros(shape)
    grid *= image_size[None, None, None, :]
    sx, sy, sz = image_size
    cx = np.random.rand(N) * sx - sx // 2
    cy = np.random.rand(N) * sy - sy // 2
    # cz = (np.random.rand(N) * sz - sz//2)/sz * r
    cz = np.zeros(N)
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i]) ** 2, axis=3)
        array[dist2 < r ** 2] += I
    return array


def generate_random_lines(image_size: tuple[int, int], point_number: int, line_width: float, num_lines: int, intensity: float) -> np.ndarray:
    """
    Generate an image with randomly oriented lines.

    :param point_number: Number of points defining the size of the image grid (image will be point_number x point_number).
    :param image_size: Tuple of (psf_x_size, psf_y_size) defining scaling in x and y directions.
    :param line_width: Width of the lines.
    :param num_lines: Number of lines to generate.
    :param intensity: Total intensity of each line.
    :return: Generated image with lines.
    """
    # np.random.seed(1234)

    # Calculate the grid spacing based on psf_size and point_number
    dx = image_size[0] / point_number
    dy = image_size[1] / point_number

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
            image[x_low, y_low] += intensity  * (x_high - x_idx) * (y_high - y_idx)
            image[x_high, y_low] += intensity  * (x_idx - x_low) * (y_high - y_idx)
            image[x_low, y_high] += intensity  * (x_high - x_idx) * (y_idx - y_low)
            image[x_high, y_high] += intensity  * (x_idx - x_low) * (y_idx - y_low)

    # Apply Gaussian smoothing to create smooth line edges
    smoothed_image = gaussian_filter(image, sigma=(line_width, line_width))

    # Normalize intensity so that each line has the specified total intensity
    # if np.amax(smoothed_image) > 0:
    #     smoothed_image = (smoothed_image / np.sum(smoothed_image)) * intensity * num_lines

    return smoothed_image

def generate_line_grid_2d(
        image_size: tuple[int, int],
        pitch: float,
        line_width: float,
        intensity: float
) -> np.ndarray:
    """
    Generate a 2D regular line grid with a given pitch, random orientation,
    and random initial position.

    Args:
        image_size (tuple[int, int]): Height and width of the 2D image.
        pitch (float): Distance between lines.
        line_width (float): Width of the lines.
        intensity (float): Intensity (value) assigned to the pixels where lines exist.

    Returns:
        np.ndarray: 2D array of shape (image_size[0], image_size[1])
                    with the generated line grid.
    """
    # Unpack size for clarity
    height, width = image_size

    # Prepare empty (zero) grid
    grid = np.zeros((height, width), dtype=np.float32)

    # Random orientation in [0, 2π)
    theta = np.random.uniform(0, 2 * np.pi)

    # Random offset in X and Y directions, each within [0, pitch)
    offset_x = np.random.uniform(0, pitch)
    offset_y = np.random.uniform(0, pitch)

    # Create coordinate arrays
    y_coords, x_coords = np.indices((height, width))

    # Shift coordinates by the random offset
    x_shifted = x_coords - offset_x
    y_shifted = y_coords - offset_y

    # Project each point onto the direction of the lines
    # Lines run perpendicular to the normal vector given by theta.
    # Here we use dot product with the direction (cos(theta), sin(theta)).
    projection = x_shifted * np.cos(theta) + y_shifted * np.sin(theta)

    # Compute distance (mod pitch) to the nearest line
    # The expression (projection % pitch) gives how far a point is
    # within one 'pitch period'.  The distance to the closest line
    # is the minimum to 0 or pitch boundaries.
    distance_to_line = projection % pitch
    distance_to_line = np.minimum(distance_to_line, pitch - distance_to_line)

    # Wherever distance_to_line < (line_width / 2), we set the intensity
    line_mask = distance_to_line < (line_width / 2)
    grid[line_mask] = intensity

    return grid
