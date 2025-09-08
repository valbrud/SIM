"""
large_FOV_operations.py

This module provides image processing operations specifically designed for large field-of-view
microscopy applications. It includes functions for correcting field aberrations and applying
vignetting effects that are common in wide-field imaging systems.

Functions:
    introduce_field_aberrations - Apply field aberration correction to images
    radial_fade - Apply circular vignetting/fade effects to images
"""

import numpy as np

def introduce_field_aberrations(image: np.ndarray, mode: str = "reflect"):
    """
    AI-generated function. 

    Apply field aberration correction using adaptive kernel filtering.

    This function implements a spatially-varying kernel filter that corrects for
    field aberrations in microscopy images. It uses a 3x3 kernel with weights that
    vary based on spatial position to compensate for field-dependent distortions.

    The algorithm applies different weights to neighboring pixels based on their
    spatial coordinates, effectively implementing a position-dependent smoothing
    that corrects for field curvature and other aberrations.

    Parameters
    ----------
    image : np.ndarray
        Input image, either 2D (H, W) or 3D (H, W, C) array.
    mode : str, optional
        Padding mode for boundary handling. Options include 'reflect', 'constant',
        'nearest', 'mirror', 'wrap'. Default is 'reflect'.

    Returns
    -------
    np.ndarray
        Corrected image with the same shape and dtype as the input.

    Notes
    -----
    The correction uses sinusoidal weighting functions based on normalized
    spatial coordinates to create a spatially-adaptive kernel that compensates
    for field-dependent aberrations common in large field-of-view microscopy.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2‑D or 3‑D.")
    img = image.astype(np.float32)
    H, W = img.shape[:2]
    Ny, Nx = H, W

    # Coordinate grids (centred)
    y = np.arange(H) - H // 2
    x = np.arange(W) - W // 2
    X, Y = np.meshgrid(x, y)

    s_x = np.sin(X / Nx)
    s_y = np.sin(Y / Ny)
    w_a = s_x * s_y
    w_b = np.abs(s_y)        # absolute
    w_c = np.abs(s_x)        # absolute
    w_0 = 1.0                # scalar

    if img.ndim == 3:
        w_a = w_a[..., None]
        w_b = w_b[..., None]
        w_c = w_c[..., None]

    # Denominator (kernel sum) per pixel
    denom = 2.0 * w_b + 2.0 * w_c + w_0  # w_a terms cancel

    pad_width = ((1, 1), (1, 1)) + (() if img.ndim == 2 else ((0, 0),))
    padded = np.pad(img, pad_width, mode=mode)

    TL = padded[0:H,     0:W    ]
    TC = padded[0:H,     1:W+1  ]
    TR = padded[0:H,     2:W+2  ]
    CL = padded[1:H+1,   0:W    ]
    CC = padded[1:H+1,   1:W+1  ]
    CR = padded[1:H+1,   2:W+2  ]
    BL = padded[2:H+2,   0:W    ]
    BC = padded[2:H+2,   1:W+1  ]
    BR = padded[2:H+2,   2:W+2  ]

    numer = (
        w_a * TL + w_b * TC + (-w_a) * TR +
        w_c * CL + w_0 * CC + w_c * CR +
        (-w_a) * BL + w_b * BC + w_a * BR
    )

    out = numer / denom        # per‑pixel normalisation
    return out.astype(image.dtype)

def radial_fade(image: np.ndarray, fade_level: float = 0.75, power: float = 2.0):
    """
    AI-generated function. 
    
    Apply a circularly-symmetric fade (vignetting) so that
    the image centre keeps full brightness (×1.0) and the extreme
    corners are scaled by `fade_level`.

    Parameters
    ----------
    image : np.ndarray
        2-D (H×W) or 3-D (H×W×C) array, any numeric dtype.
    fade_level : float, optional
        Factor (0 … 1) that the *corners* will be multiplied by.
        • 0.75 ⇒ corners are 75 % as bright as the centre  
        • 0.0  ⇒ corners fade completely to black  
        • 1.0  ⇒ no fade at all
    power : float, optional
        Controls how fast the fall-off happens:  
        1 ≈ linear, 2 ≈ quadratic (classic vignette), >2 ≈ steeper.

    Returns
    -------
    np.ndarray
        Faded image, same shape & dtype as input.
    """
    if not (0.0 <= fade_level <= 1.0):
        raise ValueError("fade_level must be in [0, 1].")

    img = image.astype(np.float32)
    H, W = img.shape[:2]

    # Radial coordinates centred at the optical axis
    yy, xx = np.indices((H, W))
    yy = yy - H / 2.0
    xx = xx - W / 2.0
    r = np.hypot(xx, yy)
    r_norm = r / r.max()                # 0 at centre … 1 at furthest corner

    # Smooth mask: 1 at centre → fade_level at edge
    mask = fade_level + (1.0 - fade_level) * (1.0 - r_norm**power)
    if img.ndim == 3:                   # broadcast over colour channels
        mask = mask[..., None]

    out = img * mask
    # Clip and cast back so dtype is preserved
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        out = np.clip(out, info.min, info.max).astype(image.dtype)
    else:
        out = out.astype(image.dtype)
    return out
