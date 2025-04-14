import numpy as np
import scipy
import skimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from scipy.signal import fftconvolve

# Shallow copy of a skimage richardson-lucy function for convenience
richardson_lucy_skimage = skimage.restoration.richardson_lucy

def richardson_lucy_homebrew(image: np.ndarray, psf: np.ndarray, step_number: int = 10, storeIntermediate: bool = True, regularization_filter: float = 10 ** -10) -> tuple[np.ndarray, np.ndarray]:
    """
    Self-made implementations of the Richardson-Lucy deconvolution algorithm.
    Test-proven to be identical to skimage one. Present to be able to play with the algorithm if needed.

    Args:
        image (np.ndarray): The input image to be deconvolved.
        psf (np.ndarray): The point spread function.
        step_number (int, optional): Number of iterations to perform. Defaults to 10.
        storeIntermediate (bool, optional): Whether to store intermediate results. Defaults to True.
        regularization_filter (float, optional): Regularization filter to avoid division by zero. Defaults to 10**-10.

    Returns:
        np.ndarray: The deconvolved image.
        np.ndarray: History of intermediate results (if storeIntermediate is True).
    """
    f0 = np.array(image)
    size = np.array(image.shape)

    history = np.zeros((step_number, *size)) if storeIntermediate else None

    f = f0
    g = psf
    for step in range(step_number):
        if storeIntermediate:
            history[step] = f
        inner_convolution = scipy.signal.convolve(g, f, 'same')
        f = scipy.signal.convolve(image / (inner_convolution + regularization_filter), np.flip(g), 'same') * f
        f_ft = np.abs(scipy.fft.fftshift(scipy.fft.fft2(scipy.fft.fftshift(f))))

    object_estimated = f

    return object_estimated, history


def richardson_lucy_blind_homebrew(image: np.ndarray, psf0: np.ndarray, step_number: int = 10, storeIntermediate: bool = True, regularization_filter: float = 10 ** -10,
                                   steps_per_iteration: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements the blind Richardson-Lucy deconvolution algorithm.

    Args:
        image (np.ndarray): The input image to be deconvolved.
        psf0 (np.ndarray, optional): The initial point spread function. Defaults to None.
        step_number (int, optional): Number of iterations to perform. Defaults to 10.
        storeIntermediate (bool, optional): Whether to store intermediate results. Defaults to True.
        regularization_filter (float, optional): Regularization filter to avoid division by zero. Defaults to 10**-10.
        steps_per_iteration (int, optional): Number of steps per iteration for PSF and object estimation. Defaults to 10.

    Returns:
        np.ndarray: The deconvolved image.
        np.ndarray: The estimated point spread function.
        np.ndarray: History of intermediate results (if storeIntermediate is True).
    """

    def _iterate_psf(image, object, psf):
        c, f, g, = image, object, psf
        for step in range(steps_per_iteration):
            inner_convolution = scipy.signal.convolve(g, f, 'same')
            g = scipy.signal.convolve(c / (inner_convolution + regularization_filter), np.flip(f), 'same') * g

        return g

    def _iterate_object(image, object, psf):
        c, f, g, = image, object, psf
        for step in range(steps_per_iteration):
            inner_convolution = scipy.signal.convolve(g, f, 'same')
            f = scipy.signal.convolve(c / (inner_convolution + regularization_filter), np.flip(g), 'same') * f
        return f

    f0 = np.array(image)
    size = np.array(image.shape)

    history = np.zeros((2, step_number, *size)) if storeIntermediate else None

    if not psf0 is None:
        g0 = psf0
    else:
        g0 = np.zeros(size)
        g0[:, :] = 1 / g0.size

    f = f0
    g = g0
    for step in range(step_number):
        if storeIntermediate:
            history[0, step] = g
            history[1, step] = f
        g = _iterate_psf(image, f, g)
        f = _iterate_object(image, f, g)
    psf_estimated = g
    object_estimated = f

    return object_estimated, psf_estimated, history


def bayesian_gaussian_frequency_estimate(image_ft: np.ndarray, noise_power: np.ndarray, known_average: np.ndarray,
                                         known_variance: np.ndarray, otf: np.ndarray, numerical_threshold: float = 10 ** -10) -> np.ndarray:
    """
    Estimates ground truth spatial frequencies in the assumption that they, as well as the noise, are Gaussian.
    p(f|I) = p(I|f)p(f) / p(I) = ((i/g)*g^2*Sigma^2 + sigma^2*fa) / (sigma^2 + Sigma^2*g^2)

    Args:
        image_ft (np.ndarray): Noisy, otf-multiplied image spatial frequencies. (i = g * f + n)
        noise_power (np.ndarray): Noise power spectrum.
        known_average (np.ndarray): a-priory information about the average spatial frequencies of the ground truth.
        known_variance (np.ndarray): a-priory information about the variance of the spatial frequencies of the ground truth.
        otf (np.ndarray): Optical transfer function. Assumed to be normalized by its maximal value.
        numerical_threshold (float, optional): Defaults to 10**-10.
    Returns:
        np.ndarray: Estimated true spatial frequencies
    """
    i = image_ft
    sigma = noise_power
    fa = known_average
    Sigma = known_variance
    g = otf
    return np.where(g > numerical_threshold, ((i / g) * Sigma ** 2 * g ** 2 + sigma ** 2 * fa) / (sigma ** 2 + Sigma ** 2 * g ** 2), 0)


def image_of_maximal_surprise_estimate(image_ft: np.ndarray, noise_power: np.ndarray, known_average: np.ndarray,
                                                    known_variance: np.ndarray, otf: np.ndarray, max_iters: int = 100, tolerance: float = 1e-8,
                                                    numerical_threshold: float = 10 ** -10) -> np.ndarray:
    """
    Estimates ground truth spatial frequencies, finding object that cares most mutual information with the observed image.
    Mif = p(I|f) p(f) log (p(I|f) p(f) / p(I))
    It is assumed that the object and the noise are Gaussian.

    Args:
        image_ft (np.ndarray): Noisy, otf-multiplied image spatial frequencies. (i = g * f + n)
        noise_power (np.ndarray): Noise power spectrum.
        known_average (np.ndarray): a-priory information about the average spatial frequencies of the ground truth.
        known_variance (np.ndarray): a-priory information about the variance of the spatial frequencies of the ground truth.
        otf (np.ndarray): Optical transfer function. Assumed to be normalized by its maximal value.
        max_iters (int): Maximum number of iterations for iterative solution search.
        tolerance (float): Tolerance for iterative solution search.
        numerical_threshold (float, optional): Defaults to 10**-10.
    Returns:
        np.ndarray: Estimated true spatial frequencies
    """
    i = image_ft
    sigma = noise_power
    fa = known_average
    Sigma = known_variance
    g = otf

    # Auxiliary functions
    def M1(sigma, Sigma, g):
        return 1 / 2 * np.sum(np.log((sigma ** 2 + Sigma ** 2 * g ** 2) / sigma ** 2))

    def D(sigma, Sigma, g, i, f, fa):
        Dp = (i - fa * g) ** 2 / (2 * (g ** 2 * Sigma ** 2 + sigma ** 2))
        Dp = np.sum(np.where(g > numerical_threshold, Dp, 0))
        Dm = np.sum(np.where(g > numerical_threshold, (i - g * f) ** 2 / (2 * sigma ** 2), 0))
        return Dp - Dm

    def A0(M1, D, Sigma, g):
        return np.where(g > numerical_threshold, (M1 + D) / (Sigma ** 2 * g ** 2), 0)

    def A1(M1, D, sigma):
        return (1 + M1 + D) / (sigma ** 2)

    def C0(A0, fa):
        return np.where(g > numerical_threshold, A0 * fa, 0)

    def C1(A1, g, i):
        return np.where(g > numerical_threshold, A1 * i / g, 0)

    max_iters = 250
    tolerance = 1e-8

    # Bayesian estimate is typically close
    fnew = np.where(g > numerical_threshold, ((i / g) * Sigma ** 2 * g ** 2 + sigma ** 2 * fa) / (sigma ** 2 + Sigma ** 2 * g ** 2), 0)

    # Solving polynomial equation of degree 4 iteratively to avoid clumsy formulas.
    # It is typically enough 3-5 iterations to converge.
    for _ in range(max_iters):
        fold = fnew
        m1 = M1(sigma, Sigma, g)
        d = D(sigma, Sigma, g, i, fold, fa)
        a0 = A0(m1, d, Sigma, g)
        a1 = A1(m1, d, sigma)
        c0 = C0(a0, fa)
        c1 = C1(a1, g, i)
        fnew = np.where(g > numerical_threshold, (c1 + c0) / (a1 + a0), 0)
        # print(f'fold={fold[N // 2, N // 2]}, fnew={fnew[N // 2, N // 2]}')
        delta = fnew - fold
        if np.all(np.abs(delta) < tolerance):
            break

    return
