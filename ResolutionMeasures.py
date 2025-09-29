
import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import numpy as np
import matplotlib.pyplot as plt
import hpc_utils

def frc(image1, image2, is_fourier=False, num_bins=100):
    """
    AI-generated function. 
    
    Compute the Fourier Ring Correlation (FRC) between two images.
    
    Parameters:
    - image1, image2: 2D numpy arrays of identical shape.
    - is_fourier: If False, the images are assumed to be in real space.
                  If True, the images are assumed to already represent 
                  Fourier space data.
    - num_bins: Number of radial bins (rings) to compute the FRC over.
    
    Returns:
    - freq: 1D array of mean radii (spatial frequency) for each bin.
    - frc_values: 1D array of FRC values for each radial bin.
    
    The FRC in each radial shell is computed as:
    
        FRC(f) = |Σ_{k in shell} F1(k) * conj(F2(k))| /
                 sqrt( Σ_{k in shell} |F1(k)|^2 * Σ_{k in shell} |F2(k)|^2 )
    """
    
    # If input images are in real space, transform them to Fourier space
    if not is_fourier:
        f1 = hpc_utils.wrapped_fftn(image1)
        f2 = hpc_utils.wrapped_fftn(image2)
    else:
        f1, f2 = image1, image2
    
    
    # Get the image dimensions and create a radial coordinate grid
    ny, nx = image1.shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = min(np.amax(x - center_x), np.amax(y - center_y))
    
    # Define radial bins spanning from 0 to the maximum radius
    bins = np.linspace(0, max_radius, num_bins + 1)
    frc_values = np.zeros(num_bins)
    freq = np.zeros(num_bins)
    
    # Calculate the FRC for each ring (bin)
    for i in range(num_bins):
        # Create a mask for pixels within the current radial bin
        mask = (r >= bins[i]) & (r < bins[i+1])
        freq[i] = (bins[i] + bins[i+1]) / 2
        if np.any(mask):
            # Compute the numerator: cross-correlation (phase sensitive)
            numerator = np.sum(f1[mask] * np.conjugate(f2[mask]))
            # Compute the denominator: product of the power sums
            denominator = np.sqrt(np.sum(np.abs(f1[mask])**2) * np.sum(np.abs(f2[mask])**2))
            frc_values[i] = np.abs(numerator) / denominator if denominator != 0 else 0.0
            # Define the spatial frequency for the bin as the mean radius
    return frc_values, freq

def correct_for_readout_noise(image, readout_noise): ...

def frc_one_image(image, num_bins=50, readout_noise = 0):
    if readout_noise > 0:
        image = correct_for_readout_noise(image, readout_noise)
    
    rng = np.random.default_rng()
    image1 = rng.binomial(image.astype(int), 0.5)
    image2 = image - image1

    return frc(image1, image2, is_fourier=False, num_bins=num_bins)
