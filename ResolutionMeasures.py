
import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import numpy as np
import matplotlib.pyplot as plt

def frc(image1, image2, is_fourier=False, num_bins=50):
    """
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
        f1 = np.fft.fft2(image1)
        f2 = np.fft.fft2(image2)
    else:
        f1, f2 = image1, image2
    
    # Shift the zero-frequency component to the center of the spectrum
    f1 = np.fft.fftshift(f1)
    f2 = np.fft.fftshift(f2)
    
    # Get the image dimensions and create a radial coordinate grid
    ny, nx = image1.shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Define radial bins spanning from 0 to the maximum radius
    max_radius = np.max(r)
    bins = np.linspace(0, max_radius, num_bins + 1)
    frc_values = np.zeros(num_bins)
    freq = np.zeros(num_bins)
    
    # Calculate the FRC for each ring (bin)
    for i in range(num_bins):
        # Create a mask for pixels within the current radial bin
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.any(mask):
            # Compute the numerator: cross-correlation (phase sensitive)
            numerator = np.sum(f1[mask] * np.conjugate(f2[mask]))
            # Compute the denominator: product of the power sums
            denominator = np.sqrt(np.sum(np.abs(f1[mask])**2) * np.sum(np.abs(f2[mask])**2))
            frc_values[i] = np.abs(numerator) / denominator if denominator != 0 else 0.0
            # Define the spatial frequency for the bin as the mean radius
            freq[i] = (bins[i] + bins[i+1]) / 2
    return freq, frc_values


def _prepare_fourier(image, is_fourier):
    """
    Helper to compute Fourier transform of an image (if in real space)
    and shift zero frequency to the center.
    """
    if not is_fourier:
        F = np.fft.fft2(image)
    else:
        F = image.copy()
    return np.fft.fftshift(F)

def high_pass_mask(shape, cutoff, smooth_width=0):
    """
    Create a high-pass filter mask in Fourier space.
    
    Parameters:
      shape : tuple of ints
          The shape (ny, nx) of the Fourier image.
      cutoff : float
          The cutoff radius (in pixel units relative to the Fourier image center).
          Frequencies with r < cutoff are suppressed.
      smooth_width: float, optional
          The width over which to apply a smooth transition.
          If zero, a hard cutoff is used.
    
    Returns:
      mask : 2D numpy array (same shape) containing values between 0 and 1.
    """
    ny, nx = shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    if smooth_width > 0:
        # Use a smooth transition: a sigmoid centered at cutoff.
        mask = 1.0 / (1 + np.exp(-(r - cutoff) / smooth_width))
    else:
        # Hard cutoff.
        mask = (r >= cutoff).astype(np.float64)
    return mask

def decorrelation_measure(image1, image2, cutoff, is_fourier=False, num_bins=50, smooth_width=0):
    """
    Compute a decorrelation measure between two images after high-pass filtering
    their Fourier transforms.
    
    The procedure is:
      1. (Optional) Transform images to Fourier space and shift the zero frequency.
      2. Apply a high-pass filter mask (with cutoff, and optional smooth transition)
         to remove low-frequency content.
      3. Bin the remaining Fourier coefficients into radial annuli.
      4. For each annulus, compute the normalized cross-correlation:
         
            C(f) = |Σ F1 * conj(F2)| / sqrt( Σ|F1|^2 * Σ|F2|^2 )
         
         and define the decorrelation in that annulus as:
         
            decor(f) = 1 - C(f)
         
         Thus, decorrelation is near zero when high-frequency content is very similar,
         and tends to 1 when they are highly uncorrelated.
    
    Parameters:
      image1, image2 : 2D numpy arrays
          The input images (must be the same shape).
      cutoff : float
          The radius (in Fourier pixel units) below which frequencies are suppressed.
      is_fourier : bool, optional
          Set to True if the inputs are already Fourier transforms.
      num_bins : int, optional
          Number of radial bins to use.
      smooth_width : float, optional
          If greater than zero, use a smooth transition for the high-pass filter.
    
    Returns:
      freq : 1D numpy array of mean radial frequencies (per bin).
      decorrelation : 1D numpy array of decorrelation values per radial bin.
    """
    # Prepare Fourier representations.
    F1 = _prepare_fourier(image1, is_fourier)
    F2 = _prepare_fourier(image2, is_fourier)

    # Create high pass mask.
    mask = high_pass_mask(image1.shape, cutoff, smooth_width=smooth_width)
    
    # Apply the mask.
    F1_hp = F1 * mask
    F2_hp = F2 * mask
    
    # Create a radial coordinate grid.
    ny, nx = image1.shape
    y_idx, x_idx = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2)
    
    # Define radial bins.
    max_radius = np.max(r)
    bins = np.linspace(0, max_radius, num_bins + 1)
    decor_values = np.zeros(num_bins)
    freq = np.zeros(num_bins)
    
    # For each radial bin, compute the normalized cross-correlation (only for coefficients above cutoff)
    for i in range(num_bins):
        # Create a mask for pixels within the current radial bin.
        bin_mask = (r >= bins[i]) & (r < bins[i+1])
        if np.any(bin_mask):
            # Only consider the coefficients in the bin that were passed by the high-pass filter.
            # (Since our high-pass filter is applied over all frequencies,
            #  bins below the cutoff will be mostly zero.)
            F1_bin = F1_hp[bin_mask]
            F2_bin = F2_hp[bin_mask]
            numerator = np.sum(F1_bin * np.conjugate(F2_bin))
            denominator = np.sqrt(np.sum(np.abs(F1_bin)**2) * np.sum(np.abs(F2_bin)**2))
            corr = np.abs(numerator) / denominator if denominator != 0 else 0.0
            decor_values[i] = 1.0 - corr
            freq[i] = (bins[i] + bins[i+1]) / 2
    return freq, decor_values

