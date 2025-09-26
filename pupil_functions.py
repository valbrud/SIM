import numpy as np
from math import factorial
import scipy

def setup_phi(Nphi, float_type=np.float32):
    Nphi = int(Nphi)
    phi  = np.linspace(float_type(0.0), float_type(2.0) * np.pi, Nphi, endpoint=False, dtype=float_type)
    return phi

def setup_rho_legendre(Nrho, float_type=np.float32):
    x_gl, _ = scipy.special.roots_legendre(int(Nrho))         # returns xp arrays
    x_gl = x_gl.astype(float_type, copy=False)
    u_nodes = 0.5 * (x_gl + float_type(1.0))
    rho     = np.sqrt(u_nodes)
    return rho

def radial_zernike(n, m, r, float_type=np.float32):
    """
    Compute the radial part R_{n,|m|}(r) of the Zernike polynomial
    for each r in the 1D array `r`.
    """
    m_abs = abs(m)
    R = np.zeros_like(r, dtype=float_type)

    # The sum goes up to floor((n - m_abs)/2)
    upper_k = (n - m_abs) // 2
    for k in range(upper_k + 1):
        c = ((-1) ** k
                * factorial(n - k)
                / (factorial(k)
                * factorial((n + m_abs) // 2 - k)
                * factorial((n - m_abs) // 2 - k)))
        R += c * r ** (n - 2 * k)
    R *= np.sqrt(2 * (n + 1) / (1 + (m == 0)))
    return R

def azimuthal_zernike(m, phi):
        """
        Compute the azimuthal part for Zernike polynomial Z_n^m.
        - if m >= 0: cos(m * phi)
        - if m <  0: sin(|m| * phi)

        `phi` is a 1D array of angles.
        """
        if m >= 0:
            return np.cos(m * phi)
        else:
            return np.sin(abs(m) * phi)

def compute_pupil_plane_aberrations(zernieke_polynomials, Nrho, Nphi, float_type=np.float32):
    """
    Construct a 2D pupil-plane aberration by summing Zernike modes using HCIPy's zernike().

    Parameters
    ----------
    zernieke_polynomials : dict
        Dictionary with keys = (n, m) and values = amplitudes.
        Example: {(2, 2): 0.1, (3, 1): -0.05, (4, -2): 0.07, ...}
    rho : ndarray
        1D array of radial coordinates (0 <= rho <= 1 typically).
    phi : ndarray
        1D array of azimuthal coordinates (in radians, e.g. -π to +π or 0 to 2π).

    Returns
    -------
    aberration : ndarray
        The resulting 2D aberration (same shape as rho, phi).
    """

    phi = setup_phi(Nphi, float_type)
    rho = setup_rho_legendre(Nrho, float_type)

    RHO, PHI = np.meshgrid(rho, phi, indexing='ij')
    # grid = PolarGrid(SeparatedCoords((rho, phi)))
    aberration = np.zeros((rho.size, phi.size))

    for (n, m), amplitude in zernieke_polynomials.items():
        # aberration += amplitude * zernike(n, m, grid=grid)
        aberration += amplitude * radial_zernike(n, m, RHO, float_type) * azimuthal_zernike(m, PHI)
    return aberration


def make_vortex_pupil(
    Nrho,
    Nphi,
    m=1,
    rho_inner=0.0,
    rho_outer=1.0,
    amplitude=None,
    complex_type=np.complex64,
):
    """
    Build a true vortex pupil P(ρ, φ) = A(ρ) * exp(i m φ) within an annulus.

    Parameters
    ----------
    Nrho : int
        Number of radial quadrature nodes.
    Nphi : int
        Number of angular nodes.
    m : int
        Topological charge of the vortex (phase term exp(i*m*phi)).
    rho_inner : float
        Inner radius (0 <= rho_inner < rho_outer).
    rho_outer : float
        Outer radius (rho_inner < rho_outer <= 1).
    amplitude : None or (Nu,) array or callable
        Radial amplitude A(ρ). If None -> A(ρ)=1 on the annulus.
        If array, must have shape (Nu,).
        If callable, it will be called as amplitude(rho) and must return (Nu,).
    dtype : np.dtype
        Complex dtype of the returned array.

    Returns
    -------
    P : (Nu, Nphi) complex array
        The complex pupil over (ρ, φ).
    """
    rho = setup_rho_legendre(Nrho)
    phi = setup_phi(Nphi)

    if not (0.0 <= rho_inner < rho_outer <= 1.0):
        raise ValueError("Require 0 <= rho_inner < rho_outer <= 1.")

    # Radial amplitude profile A(ρ)
    if amplitude is None:
        A = np.ones_like(rho, dtype=float)
    elif callable(amplitude):
        A = np.asarray(amplitude(rho), dtype=float)
    else:
        A = np.asarray(amplitude, dtype=float)
        if A.shape != rho.shape:
            raise ValueError(f"amplitude must have shape {rho.shape}, got {A.shape}")

    # Annular aperture mask M(ρ)
    M = ((rho >= rho_inner) & (rho <= rho_outer)).astype(float)  # (Nu,)

    # Helical phase term e^{i m φ}
    phase_phi = np.exp(1j * m * phi)  # (Nphi,)

    # Broadcast to (Nu, Nphi)
    P = (A * M)[:, None] * phase_phi[None, :]

    return P.astype(complex_type, copy=False)

