import numpy as np
import hpc_utils

# -----------------------------------------------------------------------------
# Minimal pupil-grid helpers
# -----------------------------------------------------------------------------
def _make_xy_pupil_grid(N, xp, float_type):
    """
    Assumption: pupil coordinates span [-1,1] in each axis, inclusive endpoints.
    """
    x = xp.linspace(-1.0, 1.0, int(N), dtype=float_type)
    X, Y = xp.meshgrid(x, x, indexing="ij")
    return X, Y


def _ensure_rho_phi(N, RHO, PHI, xp, float_type):
    """
    Returns RHO, PHI on the backend (numpy/cupy), shape (N,N).
    If None, compute from the assumed [-1,1] pupil grid.
    """
    if RHO is None or PHI is None:
        X, Y = _make_xy_pupil_grid(N, xp, float_type)
        rho = xp.sqrt(X * X + Y * Y)
        phi = xp.arctan2(Y, X)
        phi = xp.mod(phi, 2.0 * xp.pi)
        return rho, phi

    rho = xp.asarray(RHO).astype(float_type, copy=False)
    phi = xp.asarray(PHI).astype(float_type, copy=False)
    if rho.shape != (N, N) or phi.shape != (N, N):
        raise ValueError(f"RHO and PHI must have shape {(N, N)}; got {rho.shape} and {phi.shape}.")
    return rho, phi


def _as_scalar_or_grid(val, N, xp, dtype):
    """
    Accept scalar or (N,N).
    """
    if val is None:
        return xp.ones((N, N), dtype=dtype)

    a = xp.asarray(val)
    if a.ndim == 0:
        return xp.full((N, N), a.astype(dtype), dtype=dtype)
    if a.shape == (N, N):
        return a.astype(dtype, copy=False)
    raise ValueError(f"Expected scalar or (N,N) array; got shape {a.shape}.")


def _ensure_psf_coords(psf_coordinates):
    """
    psf_coordinates must be a tuple/list of 1D arrays, length 2 for 2D PSFs:
      (x_coords, y_coords)
    They must be uniformly spaced for CZT.
    """
    if psf_coordinates is None:
        raise ValueError("psf_coordinates must be provided as (x_coords, y_coords).")
    if not isinstance(psf_coordinates, (tuple, list)) or len(psf_coordinates) != 2:
        raise ValueError("psf_coordinates must be a tuple/list of two 1D arrays: (x_coords, y_coords).")
    return psf_coordinates[0], psf_coordinates[1]


def _pupil_coords_1d(N, xp, float_type):
    """
    1D pupil coordinate array u in [-1,1], used along each pupil axis.
    """
    return xp.linspace(-1.0, 1.0, int(N), dtype=float_type)


def _coords_to_backend_1d(arr, xp, float_type):
    a = xp.asarray(arr).astype(float_type, copy=False)
    if a.ndim != 1:
        raise ValueError("Coordinate arrays must be 1D.")
    return a


def _scale_coordinates(q_coords, xp, float_type, use_2pi, n_over_l=1.0):
    """
    Map user PSF coordinates to the CZT 'q' parameter in exp(-i q x).

    Many Fourier optics codes use exp(-i 2π f x). In a medium, effective
    wavenumber scales with n, so the PSF shrinks ~1/n if x is in real units.
    We include an explicit multiplicative factor n_over_l (typically nmedium),
    and optionally 2π.

    Result: q = (2π if use_2pi else 1) * n_over_l * q_coords
    """
    q = _coords_to_backend_1d(q_coords, xp, float_type)
    scale = (2.0 * xp.pi) if use_2pi else float_type(1.0)
    return (scale * float_type(n_over_l)) * q


# -----------------------------------------------------------------------------
# Fresnel coefficients (unchanged output grid: pupil plane)
# -----------------------------------------------------------------------------
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
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()
    float_type = xp.float32 if use_complex64 else xp.float64

    if RHO is None or PHI is None:
        x = xp.linspace(-1.0, 1.0, int(N), dtype=float_type)
        X, Y = xp.meshgrid(x, x, indexing="ij")
        rho = xp.sqrt(X * X + Y * Y)
    else:
        rho = xp.asarray(RHO).astype(float_type, copy=False)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError(f"RHO must be (N,N); got {rho.shape}.")
        N = int(rho.shape[0])

    inside = rho <= float_type(1.0)

    sin_t1 = (float_type(NA) / float_type(nm)) * rho
    sin_t1 = xp.clip(sin_t1, 0.0, 1.0)
    cos_t1 = xp.sqrt(xp.clip(1.0 - sin_t1 * sin_t1, 0.0, 1.0))

    sin_t2 = (float_type(nm) / float_type(ns)) * sin_t1
    propagating = sin_t2 <= float_type(1.0)
    sin_t2 = xp.clip(sin_t2, 0.0, 1.0)
    cos_t2 = xp.sqrt(xp.clip(1.0 - sin_t2 * sin_t2, 0.0, 1.0))

    n1 = float_type(nm)
    n2 = float_type(ns)

    denom_s = (n1 * cos_t1 + n2 * cos_t2)
    denom_p = (n2 * cos_t1 + n1 * cos_t2)

    ts = (2.0 * n1 * cos_t1) / xp.where(denom_s == 0, float_type(1.0), denom_s)
    tp = (2.0 * n1 * cos_t1) / xp.where(denom_p == 0, float_type(1.0), denom_p)

    mask = inside & propagating
    ts = xp.where(mask, ts, float_type(0.0))
    tp = xp.where(mask, tp, float_type(0.0))

    ts_np = ts.get() if mode == "gpu" else np.asarray(ts)
    tp_np = tp.get() if mode == "gpu" else np.asarray(tp)
    return ts_np, tp_np


# -----------------------------------------------------------------------------
# Coherent PSF: CZT-based FT of Cartesian pupil onto requested PSF coordinates
# -----------------------------------------------------------------------------
def compute_2d_psf_coherent(
    psf_coordinates,
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,      # expected (..., N, N) complex (Cartesian pupil samples)
    high_NA=False,
    device="auto",
    use_complex64=True,
    use_2pi=True,
):
    """
    CZT-based coherent field from a Cartesian pupil evaluated on the user grid.

    psf_coordinates: (x_coords, y_coords) 1D arrays specifying desired PSF-plane coordinates.
    Convention: kernel exp(-i * (2π) * (u*x + v*y)) if use_2pi=True.
    """
    if pupil_function is None:
        raise ValueError("Requires pupil_function as an array (..., N, N).")

    x_coords, y_coords = _ensure_psf_coords(psf_coordinates)

    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type = xp.float32 if use_complex64 else xp.float64

    P = xp.asarray(pupil_function).astype(complex_type, copy=False)
    if P.ndim < 2 or P.shape[-1] != P.shape[-2]:
        raise ValueError(f"pupil_function must end with (N,N); got {P.shape}.")
    N = int(P.shape[-1])

    rho, _phi = _ensure_rho_phi(N, RHO, PHI, xp, float_type)
    inside = rho <= float_type(1.0)

    # High-NA apodization on Cartesian grid
    if high_NA:
        salpha = float_type(NA / nmedium)
        arg = 1.0 - (rho * salpha) ** 2
        arg = xp.clip(arg, 0.0, 1.0)
        A_apod = (1.0 / (arg ** 0.25)).astype(float_type, copy=False)
        P = P * A_apod

    # enforce aperture
    P = xp.where(inside, P, complex_type(0.0))

    # Pupil coordinates (u,v) in [-1,1]
    u = _pupil_coords_1d(N, xp, float_type)
    v = u

    # Requested PSF coordinates -> conjugate coordinates qx,qy
    qx = _scale_coordinates(x_coords, xp, float_type, use_2pi, n_over_l=nmedium)
    qy = _scale_coordinates(y_coords, xp, float_type, use_2pi, n_over_l=nmedium)

    # Compute 2D FT using separable CZT along last two axes
    E = hpc_utils.czt_nd_fourier(P, (u, v), (qx, qy), axes=(-2, -1), rtol=1e-3, atol=1e-3)

    # Return NumPy
    E_np = xp.asnumpy(E) if mode == "gpu" else np.asarray(E)
    return E_np.astype(np.complex64 if use_complex64 else np.complex128, copy=False)


# -----------------------------------------------------------------------------
# Vectorial components: build 5 augmented pupils on Cartesian grid, CZT each
# -----------------------------------------------------------------------------
def compute_2d_vectorial_components_free_dipole(
    psf_coordinates,
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,     # expected (..., N, N) complex
    tp=1.0,                  # scalar or (N,N)
    ts=1.0,                  # scalar or (N,N)
    high_NA=True,
    device="auto",
    use_complex64=True,
    use_2pi=True,
):
    if pupil_function is None:
        raise ValueError("Requires pupil_function as an array (..., N, N).")

    x_coords, y_coords = _ensure_psf_coords(psf_coordinates)

    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type = xp.float32 if use_complex64 else xp.float64

    P_base = xp.asarray(pupil_function).astype(complex_type, copy=False)
    if P_base.ndim < 2 or P_base.shape[-1] != P_base.shape[-2]:
        raise ValueError(f"pupil_function must end with (N,N); got {P_base.shape}.")
    N = int(P_base.shape[-1])

    rho, phi = _ensure_rho_phi(N, RHO, PHI, xp, float_type)
    inside = rho <= float_type(1.0)

    tp_g = _as_scalar_or_grid(tp, N, xp, float_type)
    ts_g = _as_scalar_or_grid(ts, N, xp, float_type)

    salpha = float_type(NA / nmedium)
    sin_theta = xp.clip(salpha * rho, 0.0, 1.0).astype(float_type, copy=False)
    cos_theta = xp.sqrt(xp.clip(1.0 - sin_theta * sin_theta, 0.0, 1.0)).astype(float_type, copy=False)

    F0 = (0.5 * (ts_g + tp_g * cos_theta)).astype(float_type, copy=False)
    F1 = ((tp_g / xp.sqrt(float_type(2.0))) * sin_theta).astype(float_type, copy=False)
    F2 = (0.5 * (ts_g - tp_g * cos_theta)).astype(float_type, copy=False)

    c1 = xp.cos(phi).astype(float_type, copy=False)
    s1 = xp.sin(phi).astype(float_type, copy=False)
    c2 = xp.cos(2.0 * phi).astype(float_type, copy=False)
    s2 = xp.sin(2.0 * phi).astype(float_type, copy=False)

    if high_NA:
        arg = 1.0 - (rho * salpha) ** 2
        arg = xp.clip(arg, 0.0, 1.0)
        A_apod = (1.0 / (arg ** 0.25)).astype(float_type, copy=False)
        P0_base = P_base * A_apod
    else:
        P0_base = P_base

    P0_base = xp.where(inside, P0_base, complex_type(0.0))

    P0  = P0_base * F0
    P1c = P0_base * F1 * c1
    P1s = P0_base * F1 * s1
    P2c = P0_base * F2 * c2
    P2s = P0_base * F2 * s2

    # Pupil coords
    u = _pupil_coords_1d(N, xp, float_type)
    v = u

    # Requested PSF coords -> q
    qx = _scale_coordinates(x_coords, xp, float_type, use_2pi, n_over_l=nmedium)
    qy = _scale_coordinates(y_coords, xp, float_type, use_2pi, n_over_l=nmedium)

    # CZT each component
    U0  = hpc_utils.czt_nd_fourier(P0,  (u, v), (qx, qy), axes=(-2, -1), rtol=1e-3, atol=1e-3)
    U1c = hpc_utils.czt_nd_fourier(P1c, (u, v), (qx, qy), axes=(-2, -1), rtol=1e-3, atol=1e-3)
    U1s = hpc_utils.czt_nd_fourier(P1s, (u, v), (qx, qy), axes=(-2, -1), rtol=1e-3, atol=1e-3)
    U2c = hpc_utils.czt_nd_fourier(P2c, (u, v), (qx, qy), axes=(-2, -1), rtol=1e-3, atol=1e-3)
    U2s = hpc_utils.czt_nd_fourier(P2s, (u, v), (qx, qy), axes=(-2, -1), rtol=1e-3, atol=1e-3)

    cplx = (np.complex64 if use_complex64 else np.complex128)
    if mode == "gpu":
        return (xp.asnumpy(U0).astype(cplx, copy=False),
                xp.asnumpy(U1c).astype(cplx, copy=False),
                xp.asnumpy(U1s).astype(cplx, copy=False),
                xp.asnumpy(U2c).astype(cplx, copy=False),
                xp.asnumpy(U2s).astype(cplx, copy=False))
    return (np.asarray(U0, dtype=cplx),
            np.asarray(U1c, dtype=cplx),
            np.asarray(U1s, dtype=cplx),
            np.asarray(U2c, dtype=cplx),
            np.asarray(U2s, dtype=cplx))


def compute_2d_incoherent_vectorial_psf_free_dipole(
    psf_coordinates,
    NA,
    RHO=None,
    PHI=None,
    nmedium=1.0,
    pupil_function=None,
    tp=1.0,
    ts=1.0,
    high_NA=True,
    device="auto",
    use_complex64=True,
    use_2pi=True,
):
    U0, U1c, U1s, U2c, U2s = compute_2d_vectorial_components_free_dipole(
        psf_coordinates=psf_coordinates,
        NA=NA,
        RHO=RHO,
        PHI=PHI,
        nmedium=nmedium,
        pupil_function=pupil_function,
        tp=tp,
        ts=ts,
        high_NA=high_NA,
        device=device,
        use_complex64=use_complex64,
        use_2pi=use_2pi,
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


# -----------------------------------------------------------------------------
# 3D coherent PSF: compute pupil phase per z, then 2D CZT each slice
# -----------------------------------------------------------------------------
def compute_3d_psf_coherent(
    psf_coordinates,
    NA,
    z_values,
    nsample=1.0,
    nmedium=1.0,
    pupil_function=None,     # (N,N) complex pupil samples, or None -> clear pupil
    high_NA=False,
    device="auto",
    use_complex64=True,
    use_2pi=True,
    RHO=None,
    PHI=None,
):
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type   = xp.float32 if use_complex64 else xp.float64

    x_coords, y_coords = _ensure_psf_coords(psf_coordinates)

    # Determine N and (rho,phi) grid
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

    # build rho if needed
    if RHO is None or PHI is None:
        X, Y = _make_xy_pupil_grid(N, xp, float_type)
        rho = xp.sqrt(X * X + Y * Y)
    else:
        rho = xp.asarray(RHO).astype(float_type, copy=False)
        if rho.shape != (N, N):
            raise ValueError(f"RHO must be shape {(N,N)}; got {rho.shape}.")

    inside = rho <= float_type(1.0)

    if P0 is None:
        P0 = xp.where(inside, complex_type(1.0), complex_type(0.0))

    salpha = float_type(NA / nmedium)
    z = np.asarray(z_values, dtype=np.float64)

    if NA <= nsample:
        u_values = 4.0 * np.pi * z * (nsample - np.sqrt(nsample**2 - NA**2))
    else:
        u_values = 4.0 * np.pi * z * nsample * (1.0 - np.sqrt(1.0 - float(NA / nmedium) ** 2))

    if high_NA:
        cos_theta = xp.sqrt(xp.clip(1.0 - (rho * salpha) ** 2, 0.0, 1.0)).astype(float_type, copy=False)
        cos_alpha = float_type(np.sqrt(max(0.0, 1.0 - float(salpha) ** 2)))
        denom = (1.0 - cos_alpha) if (1.0 - float(cos_alpha)) != 0.0 else float_type(1.0)

    # Pupil coords
    u = _pupil_coords_1d(N, xp, float_type)
    v = u

    # Requested PSF coords -> q
    qx = _scale_coordinates(x_coords, xp, float_type, use_2pi, n_over_l=nmedium)
    qy = _scale_coordinates(y_coords, xp, float_type, use_2pi, n_over_l=nmedium)

    slices = []
    for uval in u_values:
        u_xp = float_type(uval)
        if not high_NA:
            K = xp.exp(1j * (u_xp / 2.0) * (rho * rho)).astype(complex_type, copy=False)
        else:
            K = xp.exp(1j * (u_xp / 2.0) * (1.0 - cos_theta) / denom).astype(complex_type, copy=False)

        Pz = xp.where(inside, P0 * K, complex_type(0.0))

        Ez = hpc_utils.czt_nd_fourier(Pz, (u, v), (qx, qy), axes=(0, 1), atol=1e-4, rtol=1e-4)
        slices.append(Ez)

    E3D = xp.stack(slices, axis=2)  # (Mx, My, Nz)
    return xp.asnumpy(E3D) if mode == "gpu" else np.asarray(E3D)


def compute_3d_incoherent_vectorial_psf_free_dipole(
    psf_coordinates,
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
    use_2pi=True,
    RHO=None,
    PHI=None,
):
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()

    complex_type = xp.complex64 if use_complex64 else xp.complex128
    float_type   = xp.float32 if use_complex64 else xp.float64

    x_coords, y_coords = _ensure_psf_coords(psf_coordinates)

    # Determine N and rho/phi
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

    if RHO is None or PHI is None:
        X, Y = _make_xy_pupil_grid(N, xp, float_type)
        rho = xp.sqrt(X * X + Y * Y)
        phi = xp.mod(xp.arctan2(Y, X), 2.0 * xp.pi)
    else:
        rho = xp.asarray(RHO).astype(float_type, copy=False)
        phi = xp.asarray(PHI).astype(float_type, copy=False)
        if rho.shape != (N, N) or phi.shape != (N, N):
            raise ValueError(f"RHO/PHI must be shape {(N,N)}; got {rho.shape}/{phi.shape}.")

    inside = rho <= float_type(1.0)

    if P0 is None:
        P0 = xp.where(inside, complex_type(1.0), complex_type(0.0))

    salpha = float_type(NA / nmedium)
    z = np.asarray(z_values, dtype=np.float64)

    if NA <= nsample:
        u_values = 4.0 * np.pi * z * (nsample - np.sqrt(nsample**2 - NA**2))
    else:
        u_values = 4.0 * np.pi * z * nsample * (1.0 - np.sqrt(1.0 - float(NA / nmedium) ** 2))

    if high_NA:
        cos_theta = xp.sqrt(xp.clip(1.0 - (rho * salpha) ** 2, 0.0, 1.0)).astype(float_type, copy=False)
        cos_alpha = float_type(np.sqrt(max(0.0, 1.0 - float(salpha) ** 2)))
        denom = (1.0 - cos_alpha) if (1.0 - float(cos_alpha)) != 0.0 else float_type(1.0)

    # Pupil coords and requested PSF coords -> q
    u = _pupil_coords_1d(N, xp, float_type)
    v = u
    qx = _scale_coordinates(x_coords, xp, float_type, use_2pi, n_over_l=nmedium)
    qy = _scale_coordinates(y_coords, xp, float_type, use_2pi, n_over_l=nmedium)

    slices = []
    for uval in u_values:
        u_xp = float_type(uval)
        if not high_NA:
            K = xp.exp(1j * (u_xp / 2.0) * (rho * rho)).astype(complex_type, copy=False)
        else:
            K = xp.exp(1j * (u_xp / 2.0) * (1.0 - cos_theta) / denom).astype(complex_type, copy=False)

        Pz = xp.where(inside, P0 * K, complex_type(0.0))

        U0, U1c, U1s, U2c, U2s = compute_2d_vectorial_components_free_dipole(
            psf_coordinates=psf_coordinates,
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
            use_2pi=use_2pi,
        )

        I_z = (np.abs(U0) ** 2 +
               np.abs(U1c) ** 2 +
               np.abs(U1s) ** 2 +
               np.abs(U2c) ** 2 +
               np.abs(U2s) ** 2)
        slices.append(I_z)

    I = np.stack(slices, axis=2)  # (Mx, My, Nz)
    s = I.sum()
    if s != 0:
        I /= s
    return I
