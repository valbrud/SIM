import numpy as np
import hpc_utils


# -----------------------------------------------------------------------------
# Minimal pupil-grid helpers
# -----------------------------------------------------------------------------
def _make_xy_pupil_grid(N, xp):
    """
    Assumption: x[0]=-1, x[N//2]=0, x[N-1]=1 (same for y).
    """
    x = xp.linspace(-1.0, 1.0, int(N), dtype=xp.float32)
    X, Y = xp.meshgrid(x, x, indexing="ij")
    return X, Y


def _ensure_rho_phi(N, RHO, PHI, xp):
    """
    Returns RHO, PHI on the backend (numpy/cupy), shape (N,N).
    If None, compute from the assumed [-1,1] pupil grid.
    """
    if RHO is None or PHI is None:
        X, Y = _make_xy_pupil_grid(N, xp)
        rho = xp.sqrt(X * X + Y * Y)
        phi = xp.arctan2(Y, X)
        # map to [0, 2π)
        phi = xp.mod(phi, 2.0 * xp.pi)
        return rho, phi

    rho = xp.asarray(RHO)
    phi = xp.asarray(PHI)
    if rho.shape != (N, N) or phi.shape != (N, N):
        raise ValueError(f"RHO and PHI must have shape {(N, N)}; got {rho.shape} and {phi.shape}.")
    return rho, phi


def _as_scalar_or_grid(val, N, xp, dtype):
    """
    Accept scalar or (N,N). (Also tolerates (1,1) etc via asarray broadcasting rules.)
    """
    if val is None:
        return xp.ones((N, N), dtype=dtype)

    a = xp.asarray(val)
    if a.ndim == 0:
        return xp.full((N, N), a.astype(dtype), dtype=dtype)
    if a.shape == (N, N):
        return a.astype(dtype, copy=False)
    raise ValueError(f"Expected scalar or (N,N) array; got shape {a.shape}.")

import numpy as np
import hpc_utils

def get_fresnel_coefficients(
    ns,
    nm,
    NA,
    RHO=None,
    PHI=None,
    N=256,
    device="auto",
    use_complex64=True,
):
    """
    Fresnel transmission coefficients (amplitude) for s and p polarization,
    for an interface from medium nm (incident) -> sample ns (transmitted).

    Returns
    -------
    ts, tp : np.ndarray
        Real arrays of shape (N,N) on the pupil grid.
    """
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()
    float_type = xp.float32 if use_complex64 else xp.float64

    if RHO is None or PHI is None:
        # build rho/phi on assumed [-1,1] grid
        x = xp.linspace(-1.0, 1.0, int(N), dtype=float_type)
        X, Y = xp.meshgrid(x, x, indexing="ij")
        rho = xp.sqrt(X * X + Y * Y)
    else:
        rho = xp.asarray(RHO).astype(float_type, copy=False)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(f"RHO must be (N,N); got {rho.shape}.")
        N = int(rho.shape[0])

    inside = rho <= float_type(1.0)

    # angle in incident medium (nm): sin(theta_m) = (NA/nm)*rho
    sin_t1 = (float_type(NA) / float_type(nm)) * rho
    sin_t1 = xp.clip(sin_t1, 0.0, 1.0)

    cos_t1 = xp.sqrt(xp.clip(1.0 - sin_t1 * sin_t1, 0.0, 1.0))

    # Snell: nm*sin(t1) = ns*sin(t2)
    sin_t2 = (float_type(nm) / float_type(ns)) * sin_t1
    # beyond critical angle -> evanescent; set transmission to 0 (pragmatic choice)
    propagating = sin_t2 <= float_type(1.0)
    sin_t2 = xp.clip(sin_t2, 0.0, 1.0)
    cos_t2 = xp.sqrt(xp.clip(1.0 - sin_t2 * sin_t2, 0.0, 1.0))

    n1 = float_type(nm)
    n2 = float_type(ns)

    # amplitude transmission coefficients (field)
    # ts = 2 n1 cos1 / (n1 cos1 + n2 cos2)
    # tp = 2 n1 cos1 / (n2 cos1 + n1 cos2)
    denom_s = (n1 * cos_t1 + n2 * cos_t2)
    denom_p = (n2 * cos_t1 + n1 * cos_t2)

    ts = (2.0 * n1 * cos_t1) / xp.where(denom_s == 0, float_type(1.0), denom_s)
    tp = (2.0 * n1 * cos_t1) / xp.where(denom_p == 0, float_type(1.0), denom_p)

    mask = inside & propagating
    ts = xp.where(mask, ts, float_type(0.0))
    tp = xp.where(mask, tp, float_type(0.0))

    # return NumPy arrays
    ts_np = ts.get() if mode == "gpu" else np.asarray(ts)
    tp_np = tp.get() if mode == "gpu" else np.asarray(tp)
    return ts_np, tp_np


# -----------------------------------------------------------------------------
# Coherent PSF: now just an FFT of the Cartesian pupil
# -----------------------------------------------------------------------------
def compute_2d_psf_coherent(
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,      # expected (N,N) complex (Cartesian pupil samples)
    high_NA=False,
    device="auto",
    use_complex64=True,

):
    """
    FFT-based coherent field from a Cartesian pupil.

    Assumptions:
      - pupil_function is sampled on an N×N Cartesian pupil grid spanning [-1,1]×[-1,1].
      - outside the unit disk (rho>1) is ignored (forced to 0).
      - output is the centered FFT (via hpc_utils.wrapped_fftn), returned as NumPy.
    """
    if pupil_function is None:
        raise ValueError("FFT path requires pupil_function as an (N,N) array.")

    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type = xp.float32 if use_complex64 else xp.float64

    P = xp.asarray(pupil_function).astype(complex_type, copy=False)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"pupil_function must be a square (N,N) array; got {P.shape}.")
    N = int(P.shape[0])

    rho, _phi = _ensure_rho_phi(N, RHO, PHI, xp)
    inside = rho <= float_type(1.0)

    # High-NA apodization (same functional form you used, but on the Cartesian grid)
    if high_NA:
        salpha = float_type(NA / nmedium)
        # avoid negative due to numerical issues near edge
        arg = 1.0 - (rho * salpha) ** 2
        arg = xp.clip(arg, 0.0, 1.0)
        A = (1.0 / (arg ** 0.25)).astype(float_type, copy=False)
        P = P * A

    # enforce aperture
    P = xp.where(inside, P, complex_type(0.0))

    # scale roughly like an integral over pupil plane
    # if coordinates are linspace(-1,1,N), step is 2/(N-1)
    du = float_type(2.0 / (N - 1))
    dv = du
    scale = (du * dv).astype(float_type, copy=False)

    E = scale * hpc_utils.wrapped_fftn(P, axes=(0, 1), norm=None)
    # wrapped_fftn returns NumPy already
    return np.asarray(E, dtype=(np.complex64 if use_complex64 else np.complex128))


# -----------------------------------------------------------------------------
# Vectorial components: build 5 augmented pupils on Cartesian grid, FFT each
# -----------------------------------------------------------------------------
def compute_2d_vectorial_components_free_dipole(
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,     # expected (N,N) complex
    tp=1.0,                  # scalar or (N,N)
    ts=1.0,                  # scalar or (N,N)
    high_NA=True,
    device="auto",
    use_complex64=True,
):
    if pupil_function is None:
        raise ValueError("FFT path requires pupil_function as an (N,N) array.")

    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type = xp.float32 if use_complex64 else xp.float64

    P_base = xp.asarray(pupil_function).astype(complex_type, copy=False)
    if P_base.ndim != 2 or P_base.shape[0] != P_base.shape[1]:
        raise ValueError(f"pupil_function must be a square (N,N) array; got {P_base.shape}.")
    N = int(P_base.shape[0])

    rho, phi = _ensure_rho_phi(N, RHO, PHI, xp)
    inside = rho <= float_type(1.0)

    # Fresnel factors
    tp_g = _as_scalar_or_grid(tp, N, xp, float_type)
    ts_g = _as_scalar_or_grid(ts, N, xp, float_type)

    # Geometry
    salpha = float_type(NA / nmedium)
    sin_theta = xp.clip(salpha * rho, 0.0, 1.0).astype(float_type, copy=False)
    cos_theta = xp.sqrt(xp.clip(1.0 - sin_theta * sin_theta, 0.0, 1.0)).astype(float_type, copy=False)

    # Radial weights
    F0 = (0.5 * (ts_g + tp_g * cos_theta)).astype(float_type, copy=False)
    F1 = ((tp_g / xp.sqrt(float_type(2.0))) * sin_theta).astype(float_type, copy=False)
    F2 = (0.5 * (ts_g - tp_g * cos_theta)).astype(float_type, copy=False)

    # Angular harmonics
    c1 = xp.cos(phi).astype(float_type, copy=False)
    s1 = xp.sin(phi).astype(float_type, copy=False)
    c2 = xp.cos(2.0 * phi).astype(float_type, copy=False)
    s2 = xp.sin(2.0 * phi).astype(float_type, copy=False)

    # Optional high-NA apodization applied to base pupil (common to all components)
    if high_NA:
        arg = 1.0 - (rho * salpha) ** 2
        arg = xp.clip(arg, 0.0, 1.0)
        A = (1.0 / (arg ** 0.25)).astype(float_type, copy=False)
        P0_base = P_base * A
    else:
        P0_base = P_base

    # enforce aperture
    P0_base = xp.where(inside, P0_base, complex_type(0.0))

    # Build augmented pupils on Cartesian grid
    P0  = P0_base * F0
    P1c = P0_base * F1 * c1
    P1s = P0_base * F1 * s1
    P2c = P0_base * F2 * c2
    P2s = P0_base * F2 * s2

    # FFT each (use the same scaling as coherent)
    du = float_type(2.0 / (N - 1))
    dv = du
    scale = (du * dv).astype(float_type, copy=False)

    U0  = scale * hpc_utils.wrapped_fftn(P0,  axes=(0, 1), norm=None)
    U1c = scale * hpc_utils.wrapped_fftn(P1c, axes=(0, 1), norm=None)
    U1s = scale * hpc_utils.wrapped_fftn(P1s, axes=(0, 1), norm=None)
    U2c = scale * hpc_utils.wrapped_fftn(P2c, axes=(0, 1), norm=None)
    U2s = scale * hpc_utils.wrapped_fftn(P2s, axes=(0, 1), norm=None)

    # wrapped_fftn returns NumPy arrays
    cplx = (np.complex64 if use_complex64 else np.complex128)
    return (np.asarray(U0,  dtype=cplx),
            np.asarray(U1c, dtype=cplx),
            np.asarray(U1s, dtype=cplx),
            np.asarray(U2c, dtype=cplx),
            np.asarray(U2s, dtype=cplx))



def compute_2d_incoherent_vectorial_psf_free_dipole(
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,     # expected (N,N) complex
    tp=1.0,                  # scalar or (N,N)
    ts=1.0,                  # scalar or (N,N)
    high_NA=True,
    device="auto",
    use_complex64=True,
):
    U0, U1c, U1s, U2c, U2s = compute_2d_vectorial_components_free_dipole(
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,     # expected (N,N) complex
    tp=1.0,                  # scalar or (N,N)
    ts=1.0,                  # scalar or (N,N)
    high_NA=True,
    device="auto",
    use_complex64=True,
    )

    I = (np.abs(U0) ** 2 +
         np.abs(U1c) ** 2 +
         np.abs(U1s) ** 2 +
         np.abs(U2c) ** 2 +
         np.abs(U2s) ** 2)

    s = I.sum()
    if s != 0:
        I /= s
    return I


def compute_3d_psf_coherent(
    NA,
    z_values,
    nsample=1.0,
    nmedium=1.0,
    pupil_function=None,     # (N,N) complex pupil samples, or None -> clear pupil
    high_NA=False,
    device="auto",
    use_complex64=True,
    RHO=None,
    PHI=None,
):
    """
    3D coherent field stack using FFT on a Cartesian pupil grid.

    Returns
    -------
    E3D : np.ndarray (complex)
        Shape (N, N, Nz): coherent field slices (FFT result for each z).
    """
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type   = xp.float32 if use_complex64 else xp.float64

    # Determine N and (rho,phi) grid
    if pupil_function is not None:
        P0 = xp.asarray(pupil_function).astype(complex_type, copy=False)
        if P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
            raise ValueError(f"pupil_function must be (N,N); got {P0.shape}.")
        N = int(P0.shape[0])
    else:
        # infer N from RHO if possible, else default
        if RHO is not None:
            N = int(np.asarray(RHO).shape[0])
        else:
            N = 256
        P0 = None

    # build rho/phi if needed
    if RHO is None or PHI is None:
        x = xp.linspace(-1.0, 1.0, N, dtype=float_type)
        X, Y = xp.meshgrid(x, x, indexing="ij")
        rho = xp.sqrt(X * X + Y * Y)
    else:
        rho = xp.asarray(RHO).astype(float_type, copy=False)
        if rho.shape != (N, N):
            raise ValueError(f"RHO must be shape {(N,N)}; got {rho.shape}.")

    inside = rho <= float_type(1.0)

    # base pupil: clear aperture if None
    if P0 is None:
        P0 = xp.where(inside, complex_type(1.0), complex_type(0.0))

    # u(z) construction (kept from your earlier code)
    salpha = float_type(NA / nmedium)
    z = np.asarray(z_values, dtype=np.float64)

    if NA <= nsample:
        u_values = 4.0 * np.pi * z * (nsample - np.sqrt(nsample**2 - NA**2))
    else:
        u_values = 4.0 * np.pi * z * nsample * (1.0 - np.sqrt(1.0 - float(NA / nmedium) ** 2))

    # precompute geometry terms for high NA kernel
    if high_NA:
        cos_theta = xp.sqrt(xp.clip(1.0 - (rho * salpha) ** 2, 0.0, 1.0)).astype(float_type, copy=False)
        cos_alpha = float_type(np.sqrt(max(0.0, 1.0 - float(salpha) ** 2)))
        denom = (1.0 - cos_alpha) if (1.0 - float(cos_alpha)) != 0.0 else float_type(1.0)

    # FFT scaling: pupil grid is assumed linspace(-1,1,N)
    du = float_type(2.0 / (N - 1))
    dv = du
    scale = (du * dv).astype(float_type, copy=False)

    slices = []
    for u in u_values:
        u_xp = float_type(u)
        if not high_NA:
            K = xp.exp(1j * (u_xp / 2.0) * (rho * rho)).astype(complex_type, copy=False)
        else:
            K = xp.exp(1j * (u_xp / 2.0) * (1.0 - cos_theta) / denom).astype(complex_type, copy=False)

        Pz = xp.where(inside, P0 * K, complex_type(0.0))

        Ez = scale * hpc_utils.wrapped_fftn(Pz, axes=(0, 1), norm=None)
        slices.append(Ez)

    # wrapped_fftn returns NumPy arrays; stack on host
    return np.stack(slices, axis=2)

def compute_3d_incoherent_vectorial_psf_free_dipole(
    NA,
    z_values,
    nsample=1.0,
    nmedium=1.0,
    pupil_function=None,   # (N,N) complex or None
    tp=1.0,                # scalar or (N,N)
    ts=1.0,                # scalar or (N,N)
    high_NA=True,
    device="auto",
    use_complex64=True,
    RHO=None,
    PHI=None,
):
    """
    3D incoherent vectorial PSF stack: sum of intensities of 5 Debye-like components.
    Returns I of shape (N, N, Nz), normalized to sum=1.
    """
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type   = xp.float32 if use_complex64 else xp.float64

    # Determine N and rho grid
    if pupil_function is not None:
        P0 = xp.asarray(pupil_function).astype(complex_type, copy=False)
        if P0.ndim != 2 or P0.shape[0] != P0.shape[1]:
            raise ValueError(f"pupil_function must be (N,N); got {P0.shape}.")
        N = int(P0.shape[0])
    else:
        if RHO is not None:
            N = int(np.asarray(RHO).shape[0])
        else:
            N = 256
        P0 = None

    # build rho/phi
    if RHO is None or PHI is None:
        x = xp.linspace(-1.0, 1.0, N, dtype=float_type)
        X, Y = xp.meshgrid(x, x, indexing="ij")
        rho = xp.sqrt(X * X + Y * Y)
        phi = xp.mod(xp.arctan2(Y, X), 2.0 * xp.pi)
    else:
        rho = xp.asarray(RHO).astype(float_type, copy=False)
        phi = xp.asarray(PHI).astype(float_type, copy=False)
        if rho.shape != (N, N) or phi.shape != (N, N):
            raise ValueError(f"RHO/PHI must be shape {(N,N)}; got {rho.shape}/{phi.shape}.")

    inside = rho <= float_type(1.0)

    # base pupil: clear aperture if None
    if P0 is None:
        P0 = xp.where(inside, complex_type(1.0), complex_type(0.0))

    # u(z) construction (same as scalar code)
    salpha = float_type(NA / nmedium)
    z = np.asarray(z_values, dtype=np.float64)
    if NA <= nsample:
        u_values = 4.0 * np.pi * z * (nsample - np.sqrt(nsample**2 - NA**2))
    else:
        u_values = 4.0 * np.pi * z * nsample * (1.0 - np.sqrt(1.0 - float(NA / nmedium) ** 2))

    # precompute high-NA geometry
    if high_NA:
        cos_theta = xp.sqrt(xp.clip(1.0 - (rho * salpha) ** 2, 0.0, 1.0)).astype(float_type, copy=False)
        cos_alpha = float_type(np.sqrt(max(0.0, 1.0 - float(salpha) ** 2)))
        denom = (1.0 - cos_alpha) if (1.0 - float(cos_alpha)) != 0.0 else float_type(1.0)

    slices = []
    for u in u_values:
        u_xp = float_type(u)
        if not high_NA:
            K = xp.exp(1j * (u_xp / 2.0) * (rho * rho)).astype(complex_type, copy=False)
        else:
            K = xp.exp(1j * (u_xp / 2.0) * (1.0 - cos_theta) / denom).astype(complex_type, copy=False)

        Pz = xp.where(inside, P0 * K, complex_type(0.0))

        # Your 2D component function (FFT-based) should accept backend arrays fine
        U0, U1c, U1s, U2c, U2s = compute_2d_vectorial_components_free_dipole(
            NA=NA,
            RHO=rho,
            PHI=phi,
            nmedium=nmedium,
            pupil_function=Pz,
            tp=tp,
            ts=ts,
            high_NA=high_NA,
            device=device,
            use_complex64=use_complex64,
        )

        I_z = (np.abs(U0) ** 2 +
               np.abs(U1c) ** 2 +
               np.abs(U1s) ** 2 +
               np.abs(U2c) ** 2 +
               np.abs(U2s) ** 2)
        slices.append(I_z)
        print(f"Computed z-slice with u={float(u):.3f}")

    I = np.stack(slices, axis=2)  # (N, N, Nz)
    s = I.sum()
    if s != 0:
        I /= s
    return I
