"""
ShapesGenerator.py

This module provides functions for generating various model objects. 

Functions:
    generate_random_spherical_particles - Generate 3D arrays with randomly positioned spherical particles
    generate_sphere_slices - Generate 3D arrays with random spheres confined to a thin slice
    generate_random_lines - Generate n-D images filled with randomly oriented line segments
    generate_line_grid_2d - Generate 2D regular line grids with random orientation and position
    make_circle_grid - Generate 2D grids of circles with specified spacing and radius

All functions support customizable parameters for size, intensity, and geometric properties
to facilitate diverse simulation scenarios.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _parse_point_number(point_number, dim: int) -> tuple[int, ...]:
    if isinstance(point_number, int):
        return (point_number,) * dim
    shape = tuple(point_number)
    if len(shape) != dim:
        raise ValueError("`point_number` dimensionality does not match `image_size`.")
    return shape


def generate_random_spherical_particles(image_size: tuple[int, int, int],
                                        point_number: int | tuple[int, ...],
                                        radius: float = 0.1,
                                        num_particles: int = 10,
                                        intensity: int = 1000,
                                        generate_default: bool = False) -> np.ndarray:
    """
    Generate a 3D array containing randomly positioned spherical particles.

    Creates a 3D volume with spheres of specified radius placed at random positions.
    Each sphere contributes a constant intensity to all voxels within its volume.

    Parameters
    ----------
    image_size : tuple[int, int, int]
        Physical size of the 3D volume in each dimension (sx, sy, sz).
    point_number : int or (Nx, Ny[, Nz])
        Number of grid points along each dimension for discretization.
        • If a single int is given it is used for *every* dimension.
    r : float, optional
        Radius of each sphere. Default is 0.1.
    N : int, optional
        Number of spheres to generate. Default is 10.
    I : float, optional
        Intensity value added to voxels within each sphere. Default is 1000.

    Returns
    -------
    np.ndarray
        3D array of shape (point_number, point_number, point_number) containing
        the generated spherical particles.
    """
    if generate_default:
        np.random.seed(1234)

    dim = len(image_size)
    if dim not in (2, 3):
        raise ValueError("Invalid dimensionality. Choose either 2 or 3.")

    shape = _parse_point_number(point_number, dim)
    axes = [np.arange(n) for n in shape]
    indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # (..., dim)

    grid = (indices / np.asarray(shape) - 1 / 2)
    array = np.zeros(shape, dtype=np.float32)
    grid = grid * np.asarray(image_size)

    centers = np.column_stack([
        np.random.rand(num_particles) * float(image_size[d]) - float(image_size[d]) / 2
        for d in range(dim)
    ])

    for i in range(num_particles):
        dist2 = np.sum((grid - centers[(None,) * dim + (i,)]) ** 2, axis=dim)
        array[dist2 < radius ** 2] += intensity

    return array


def generate_sphere_slices(image_size: tuple[int, int, int],
                           point_number: int | tuple[int, int, int],
                           r: float = 0.1,
                           N: int = 10,
                           I: int = 1000,
                           generate_default: bool = False) -> np.ndarray:
    """
    Generate a 3D array with random spheres confined to a thin slice.

    Creates a 3D volume with spheres positioned randomly in the XY plane but
    constrained to a thin slice in the Z direction (centered at z=0).
    Useful for simulating 2D-like structures in 3D space.

    Parameters
    ----------
    image_size : tuple[int, int, int]
        Physical size of the 3D volume in each dimension (sx, sy, sz).
    point_number : int or (Nx, Ny[, Nz])
        Number of grid points along each dimension for discretization.
        • If a single int is given it is used for *every* dimension.
    r : float, optional
        Radius of each sphere. Default is 0.1.
    N : int, optional
        Number of spheres to generate. Default is 10.
    I : float, optional
        Intensity value added to voxels within each sphere. Default is 1000.

    Returns
    -------
    np.ndarray
        3D array of shape (point_number, point_number, point_number) containing
        the generated spherical particles in a thin slice.
    """
    if generate_default:
        np.random.seed(1234)

    dim = len(image_size)
    if dim != 3:
        raise ValueError("`generate_sphere_slices` expects a 3-D `image_size`.")

    shape = _parse_point_number(point_number, dim)
    axes = [np.arange(n) for n in shape]
    indices = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    grid = (indices / np.asarray(shape) - 1 / 2)
    array = np.zeros(shape, dtype=np.float32)
    grid = grid * np.asarray(image_size)

    sx, sy, sz = map(float, image_size)
    cx = np.random.rand(N) * sx - sx / 2
    cy = np.random.rand(N) * sy - sy / 2
    cz = np.zeros(N, dtype=np.float32)
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i]) ** 2, axis=3)
        array[dist2 < r ** 2] += I

    return array


from itertools import product
from typing import Sequence, Tuple


def generate_random_lines(
    image_size: Sequence[float],
    point_number: int | Tuple[int, ...],
    line_width: float | Sequence[float],
    num_lines: int,
    intensity: float,
    generate_default: bool = False
) -> np.ndarray:
    """
    AI-generated functions. 

    Generate an n-D image (2-D or 3-D) filled with randomly oriented line segments.

    Parameters
    ----------
    image_size : (Dx, Dy) or (Dx, Dy, Dz)
        Physical size of the PSF support along each axis.
    point_number : int or (Nx, Ny) or (Nx, Ny, Nz)
        Number of pixels/voxels along each axis of the output grid.
        • If a single int is given it is used for *every* dimension.

    Returns
    -------
    img : ndarray
        • shape (Nx, Ny)               if `image_size` has length 2  
        • shape (Nx, Ny, Nz)           if `image_size` has length 3
    """
    if generate_default:
        np.random.seed(1234)

    dim = len(image_size)
    if dim not in (2, 3):
        raise ValueError("Only 2-D and 3-D images are supported.")

    shape = _parse_point_number(point_number, dim)

    spacings = tuple(s / n for s, n in zip(image_size, shape))   # dx, dy[, dz]
    sigmas = (line_width,) * dim if np.isscalar(line_width) else tuple(line_width)

    img = np.zeros(shape, dtype=np.float32)

    # ------------- helper for n-D linear interpolation ----------------------
    def add_intensity(indices_f: np.ndarray):
        """
        Distribute intensity at fractional index position `indices_f`
        to the 2**dim neighbouring voxels/pixels.
        """
        lows = np.floor(indices_f).astype(int)
        fracs = indices_f - lows
        highs = np.minimum(lows + 1, np.array(shape) - 1)

        # iterate over all 2**dim vertex combinations: (low/high, low/high, …)
        for corner in product((0, 1), repeat=dim):
            idx = tuple(highs[d] if corner[d] else lows[d] for d in range(dim))
            w = 1.0
            for d, c in enumerate(corner):
                w *= fracs[d] if c else (1.0 - fracs[d])
            img[idx] += intensity * w

    # ------------- draw random line segments --------------------------------
    for _ in range(num_lines):
        # start & end points in physical coordinates
        p0 = np.random.uniform([0]*dim, image_size)
        p1 = np.random.uniform([0]*dim, image_size)

        # number of sampled points along the segment
        steps = int(
            max(abs((p1 - p0) / spacings).max(), 1)
        ) + 1
        t_vals = np.linspace(0.0, 1.0, steps)

        for t in t_vals:
            pos_physical = p0 + t * (p1 - p0)
            idx_frac = pos_physical / spacings
            add_intensity(idx_frac)

    # ------------- Gaussian blur to obtain finite width ---------------------
    img = gaussian_filter(img, sigma=sigmas)

    return img

def generate_line_grid_2d(
        image_size: tuple[int, int],
        pitch: float,
        line_width: float,
        intensity: float,
        generate_default: bool = False
) -> np.ndarray:
    """
    AI-generated function. 

    Generate a 2D regular line grid with random orientation and position.

    Creates a 2D image with parallel lines arranged in a regular grid pattern.
    The orientation and initial position of the grid are randomized.

    Parameters
    ----------
    image_size : tuple[int, int]
        Height and width of the 2D image in pixels.
    pitch : float
        Distance between adjacent lines in pixels.
    line_width : float
        Width of each line in pixels.
    intensity : float
        Intensity value assigned to pixels within the lines.

    Returns
    -------
    np.ndarray
        2D array of shape (image_size[0], image_size[1]) containing
        the generated line grid pattern.
    """

    if generate_default:
        np.random.seed(1234)

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

def make_circle_grid(im_size=128, spacing=16, radius=5, value=1.0):
    """
    AI-generated function.

    Generate a 2D grid of circles with specified spacing and radius.

    Creates a square image with circles arranged in a regular grid pattern.
    Each circle has the same radius and intensity value.

    Parameters
    ----------
    im_size : int, optional
        Size of the square image in pixels. Default is 128.
    spacing : int, optional
        Distance between circle centers in pixels. Default is 16.
    radius : float, optional
        Radius of each circle in pixels. Default is 5.
    value : float, optional
        Intensity value assigned to pixels within circles. Default is 1.0.

    Returns
    -------
    np.ndarray
        2D array of shape (im_size, im_size) containing the circle grid pattern.
    """
    img = np.zeros((im_size, im_size), dtype=np.float32)
    yy, xx = np.indices(img.shape)
    centres = np.arange(spacing // 2, im_size, spacing)
    for cy in centres:
        for cx in centres:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            img[mask] = value
    return img
