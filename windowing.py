"""
widnosing.py

This module provides functions to modify the image near the edges for different purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def make_mask_cosine_edge2d(shape: tuple[int, int], edge: int) -> np.ndarray:
    """
    2D Weight mask that vanishes with the cosine distance to the edge.

    Args:
        shape (tuple[int, int]): Shape of the mask.
        edge (int): Width of the edge.

    Returns:
        np.ndarray: The mask.
    """
    # no valid edge -> no masking
    if edge <= 0:
        return np.ones(shape)
    # instead of computing the mask directly, the relative distance to the nearest
    # edge within the configured width is computed. this only needs to be done
    # once for one corner and can then be mirrored accordingly.
    d = np.linspace(0.0, 1.0, num=edge)
    dx, dy = np.meshgrid(d, d)
    dxy = np.hypot(dx, dy)
    dcorner = np.where(dx < dy, dx, dy)
    dcorner = np.where(dxy < dcorner, dxy, dcorner)
    print(dcorner.shape)
    dist = np.ones(shape)
    dist[..., :edge, :] = d[:, np.newaxis]
    dist[..., -edge:, :] = d[::-1, np.newaxis]
    dist[..., :, :edge] = d
    dist[..., :, -edge:] = d[::-1]
    dist[..., :edge, :edge] = dcorner
    dist[..., -edge:, :edge] = dcorner[::-1, :]
    dist[..., :edge, -edge:] = dcorner[:, ::-1]
    dist[..., -edge:, -edge:] = dcorner[::-1, ::-1]
    # convert distance to weight
    return np.sin(0.5 * np.pi * dist)

# plt.figure().clear()
# plt.imshow(make_mask_cosine_edge2d((100, 200), 25))
# plt.xlabel("x (pixel)")
# plt.ylabel("y (pixel)")
# plt.colorbar(label="pixel weight")
# plt.tight_layout()

def make_mask_cosine_edge3d(shape: tuple[int, int, int], edge: int) -> np.ndarray:
    """
    3D Weight mask that vanishes with the cosine distance to the edges.

    Args:
        shape (tuple[int, int, int]): Shape of the mask.
        edge (int): Width of the edge.

    Returns:
        np.ndarray: The mask.
    """
    if edge <= 0:
        return np.ones(shape)

    d = np.linspace(0.0, 1.0, num=edge)
    dx, dy = np.meshgrid(d, d)
    # dxy = np.hypot(dx, dy)
    dcorner = np.where(dx < dy, dx, dy)
    # print(dcorner)
    # dcorner = np.where(dxy < dcorner, dxy, dcorner)

    dx, dy, dz = np.meshgrid(d, d, d)
    # dxyz = np.hypot(dx, dy, dz)
    d3corner = np.minimum(np.minimum(dx, dy), dz)

    dist = np.ones(shape)
    dist[:edge, :, :] = d[:, np.newaxis, np.newaxis]
    dist[-edge:, :, :] = d[::-1, np.newaxis, np.newaxis]
    dist[:, :edge, :] *= d[:, np.newaxis]
    dist[:, -edge:, :] *= d[::-1, np.newaxis]
    dist[:, :, :edge] *= d
    dist[:, :, -edge:] *= d[::-1]

    dist[:edge, :edge, :] = dcorner[..., np.newaxis]
    dist[-edge:, :edge, :] = dcorner[::-1, :, np.newaxis]
    dist[:edge, -edge:, :] = dcorner[:, ::-1, np.newaxis]
    dist[-edge:, -edge:, :] = dcorner[::-1, ::-1, np.newaxis]
    dist[:edge, :, :edge] = dcorner[:, np.newaxis, :]
    dist[-edge:, :, :edge] = dcorner[::-1, np.newaxis, :]
    dist[:edge, :, -edge:] = dcorner[:, np.newaxis, ::-1]
    dist[-edge:, :, -edge:] = dcorner[::-1, np.newaxis, ::-1]
    dist[:, :edge, :edge] = dcorner[np.newaxis, :, :]
    dist[:, -edge:, :edge] = dcorner[np.newaxis, ::-1, :]
    dist[:, :edge, -edge:] = dcorner[np.newaxis, :, ::-1]
    dist[:, -edge:, -edge:] = dcorner[np.newaxis, ::-1, ::-1]

    dist[:edge, :edge, :edge] = d3corner
    dist[-edge:, :edge, :edge] = d3corner[::-1, ...]
    dist[:edge, -edge:, :edge] = d3corner[:, ::-1, :]
    dist[:edge, :edge, -edge:] = d3corner[..., ::-1]
    dist[-edge:, -edge:, :edge] = d3corner[::-1, ::-1, :]
    dist[-edge:, :edge, -edge:] = d3corner[::-1, :, ::-1]
    dist[:edge, -edge:, -edge:] = d3corner[:, ::-1, ::-1]
    dist[-edge:, -edge:, -edge:] = d3corner[::-1, ::-1, ::-1]

    return np.sin(0.5 * np.pi * dist)

if __name__=="__main__":
    # Test the 3D mask generation
    mask_3d = make_mask_cosine_edge3d((50, 50, 50), 10)

    # Visualize a slice of the 3D mask
    fig, ax = plt.subplots()
    mp1 = ax.imshow(mask_3d[:, 0, :], cmap='viridis')
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    plt.colorbar(mp1, label="pixel weight")
    plt.tight_layout()


    def update(val):
        ax.clear()
        IM = mask_3d[:, :, int(val)]
        print(np.amax(IM))
        mp1.set_data(np.abs(IM))
        plt.draw()
        ax.set_aspect(1. / ax.get_data_ratio())


    slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
    slider_ssnr = Slider(slider_loc, 'fz', 0,49)  # slider properties
    slider_ssnr.on_changed(update)

    plt.show()

def gaussian_attenuation(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """
    2D Gaussian attenuation mask.

    Args:
        shape (tuple[int, int]): Shape of the mask.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: The mask.
    """
    x = np.arange(shape[1]) - shape[1] / 2
    y = np.arange(shape[0]) - shape[0] / 2
    xx, yy = np.meshgrid(x, y)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))