import numpy as np

def _detect_gpu():
    """Return ('gpu', cupy_module) if GPU is available, otherwise ('cpu', None)."""
    try:
        import cupy as cp
        _ = cp.asarray([0], dtype=cp.float32)  # smoke test
        return 'gpu', cp
    except Exception:
        return 'cpu', None

def _setup_phi(Nphi, xp, float_type):
    Nphi = int(Nphi)
    phi  = xp.linspace(float_type(0.0), float_type(2.0) * xp.pi, Nphi, endpoint=False, dtype=float_type)
    wphi = (float_type(2.0) * xp.pi) / float_type(Nphi)
    return phi, wphi

def _setup_rho_legendre(Nu, legendre_roots_fn, xp, float_type):
    x_gl, w_gl = legendre_roots_fn(int(Nu))         # returns xp arrays
    x_gl = x_gl.astype(float_type, copy=False)
    w_gl = w_gl.astype(float_type, copy=False)
    u_nodes = 0.5 * (x_gl + float_type(1.0))
    w_u     = 0.5 * w_gl
    rho     = xp.sqrt(u_nodes)
    w_r     = 0.5 * w_u
    return rho, w_r

def _cpu_roots_legendre(N):
    x, w = np.polynomial.legendre.leggauss(N)
    return np.asarray(x), np.asarray(w)

def _gpu_roots_legendre(N):
    # prefer GPU native; fallback to CPU then copy
    import cupy as cp
    try:
        from cupyx.scipy.special import roots_legendre as _roots
        return _roots(N)
    except Exception:
        x, w = np.polynomial.legendre.leggauss(N)
        return cp.asarray(x), cp.asarray(w)

def j0_bessel_cpu(x):
    try:
        from scipy.special import j0 as _j0
        return _j0(x)
    except Exception as e:
        j0 = getattr(np.special, "j0", None)
        if j0 is None:
            raise RuntimeError("CPU path requires scipy.special.j0 (or NumPy>=2.0 np.special.j0).") from e
        return j0(x)

def j0_bessel_gpu(x):
    from cupyx.scipy.special import j0 as _j0
    return _j0(x)

def _cpu_mem_budget(safety=0.70):
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
        return free, int(free * safety)
    except Exception:
        return None, None

def _gpu_mem_budget(safety=0.70):
    try:
        import cupy as cp
        free, _ = cp.cuda.Device().mem_info
        return int(free), int(free * safety)
    except Exception:
        return None, None

def _dtype_bytes(use_complex64):
    return (4, 8) if use_complex64 else (8, 16)  # (float_bytes, complex_bytes)


def _normalize_pupil_array(pupil_array, rho, phi, xp):
    """
    explicit pupil:
      None          -> (Nu,) ones, radial
      (Nu,)         -> radial
      (Nu, Nphi)    -> general
    """
    if pupil_array is None:
        return xp.ones_like(rho), True
    P = xp.asarray(pupil_array)
    if P.ndim == 1:
        if P.shape[0] != rho.shape[0]:
            raise ValueError(f"pupil (Nu,) expected length {rho.size}, got {P.shape[0]}")
        return P, True
    if P.ndim == 2:
        if P.shape != (rho.size, phi.size):
            raise ValueError(f"pupil (Nu,Nphi) expected {(rho.size, phi.size)}, got {P.shape}")
        return P, False
    raise ValueError("pupil must be None, (Nu,) or (Nu, Nphi).")


def _get_apodization(rho, xp, salpha):
    """
    Apodization over ρ:
      None        -> (Nu,) ones
      (Nu,) array -> used as-is
    """
    A = xp.asarray(1 / (1 - (rho * salpha) ** 2) ** 0.25)
    return A

def _integrate_radial_common(
    vx, vy, rho, w_r, P, use_complex64, j0_fn, xp,
    mem_budget_bytes=None,     # None → no tiling
):
    float_type  = xp.float32 if use_complex64 else xp.float64
    complex_type= xp.complex64 if use_complex64 else xp.complex128
    float_bytes, complex_bytes = _dtype_bytes(use_complex64)

    vmag = xp.sqrt(vx * vx + vy * vy).astype(float_type, copy=False)  # (Nx,Ny)

    # Estimate peak for full (Nx,Ny,Nu) J0 buffer (rough)
    if mem_budget_bytes is not None:
        NxNy = vmag.size
        per_rho_bytes = NxNy * (complex_bytes + 2 * float_bytes)
        total_need = per_rho_bytes * rho.size
        if total_need > mem_budget_bytes:
            # ρ-tiling
            rho_block = max(8, int(mem_budget_bytes // max(1, per_rho_bytes)))
            rho_block = int(min(rho.size, rho_block))
            if rho_block < 8:
                rho_block = 8
            E_acc = xp.zeros_like(vmag, dtype=complex_type)
            s = 0
            while s < rho.size:
                e = min(rho.size, s + rho_block)
                J0 = j0_fn(vmag[..., None] * rho[None, None, s:e])
                I_rho = (float_type(2.0) * xp.pi) * J0 * P[None, None, s:e]
                E_acc = E_acc + xp.tensordot(I_rho, w_r[s:e], axes=([2], [0]))
                s = e
            return E_acc.astype(complex_type, copy=False)

    # Single pass
    J0 = j0_fn(vmag[..., None] * rho[None, None, :])                 # (Nx,Ny,Nu)
    I_rho = (float_type(2.0) * xp.pi) * J0 * P[None, None, :]        # (Nx,Ny,Nu)
    return xp.tensordot(I_rho, w_r, axes=([2], [0])).astype(complex_type, copy=False)

def _integrate_phi_common(
    vx, vy, rho, phi, w_r, wphi, P, use_complex64, xp,
    mem_budget_bytes=None,     # None → no tiling
):
    """
    Accumulates Iphi (Nx,Ny,Nu) by tiling over φ to respect memory.
    """
    float_type  = xp.float32 if use_complex64 else xp.float64
    complex_type= xp.complex64 if use_complex64 else xp.complex128
    float_bytes, complex_bytes = _dtype_bytes(use_complex64)

    Nx, Ny = vx.shape
    Iphi_acc = xp.zeros((Nx, Ny, rho.size), dtype=complex_type)

    # Choose φ tile
    if mem_budget_bytes is not None:
        per_phi_bytes = (vx.size * rho.size) * (3 * float_bytes + 2 * complex_bytes)
        acc_bytes = (vx.size * rho.size) * complex_bytes
        max_block = max(1, int((mem_budget_bytes - acc_bytes) // max(1, per_phi_bytes)))
        phi_block = int(min(phi.size, max(8, max_block)))
    else:
        phi_block = int(phi.size)

    p = 0
    while p < phi.size:
        q = min(phi.size, p + phi_block)
        phi_tile = phi[p:q]                           # (P,)
        cphi = xp.cos(phi_tile); sphi = xp.sin(phi_tile)
        kx = vx[..., None, None] * rho[None, None, :, None] * cphi[None, None, None, :]
        ky = vy[..., None, None] * rho[None, None, :, None] * sphi[None, None, None, :]
        phase = -1j * (kx + ky)
        kernel = xp.exp(phase)
        P_tile = P[:, p:q]                            # (Nu,P)
        integrand = kernel * P_tile[None, None, :, :] # (Nx,Ny,Nu,P)
        Iphi_acc += wphi * xp.sum(integrand, axis=-1) # (Nx,Ny,Nu)
        p = q

    return xp.tensordot(Iphi_acc, w_r, axes=([2], [0])).astype(complex_type, copy=False)

# Compute PSF and OTF under the assumption of Abbe's Sine condition
def compute_2d_psf_coherent(
    grid,
    NA,
    nmedium = 1.0, 
    pupil_function=None,   # None | (Nu,) | (Nu, Nphi)
    Nphi=256,
    Nu=129,
    high_NA=False,         # True for high NA, False for paraxial
    device='auto',         # 'auto' | 'cpu' | 'gpu'
    use_complex64=True,
    safety=0.70,
):
    """
    Manager + compute in one: selects CPU/GPU mode explicitly via device_mode,
    then runs the shared integrators. Always returns a NumPy array.
    """
    # ---- pick device mode & xp ----
    if device == 'cpu':
        device_mode, xp = 'cpu', np
    elif device == 'gpu':
        device_mode, cp_mod = _detect_gpu()
        if device_mode != 'gpu':
            raise RuntimeError("CuPy/GPU requested but not available.")
        xp = cp_mod
    else:  # 'auto'
        device_mode, cp_mod = _detect_gpu()
        xp = (cp_mod if device_mode == 'gpu' else np)

    # dtypes
    float_type   = xp.float32 if use_complex64 else xp.float64
    complex_type = xp.complex64 if use_complex64 else xp.complex128

    # memory budget & specialized funcs
    if device_mode == 'gpu':
        _, mem_budget_bytes = _gpu_mem_budget(safety)
        roots_legendre_fn   = _gpu_roots_legendre
        j0_fn               = j0_bessel_gpu
        to_xp               = xp.asarray
        to_host             = (lambda a: a.get())
    else:
        _, mem_budget_bytes = _cpu_mem_budget(safety)
        roots_legendre_fn   = _cpu_roots_legendre
        j0_fn               = j0_bessel_cpu
        to_xp               = xp.asarray  # np.asarray
        to_host             = (lambda a: a)  # already NumPy

    # ---- inputs on chosen device ----
    grid_xp = to_xp(grid)
    if grid_xp.shape[-1] != 2:
        raise ValueError("grid must have shape (..., 2) with (vx, vy).")
    vx = (2.0 * NA * xp.pi * grid_xp[..., 0]).astype(float_type)
    vy = (2.0 * NA * xp.pi * grid_xp[..., 1]).astype(float_type)

    # ---- quadrature on device ----
    phi,  wphi  = _setup_phi(Nphi, xp, float_type)
    rho,  w_r   = _setup_rho_legendre(Nu, roots_legendre_fn, xp, float_type)

    # ---- pupil normalization on device ----
    P, is_radial = _normalize_pupil_array(pupil_function, rho, phi, xp)

    if high_NA:
        salpha = NA/nmedium
        A = _get_apodization(rho, xp, salpha)
        P = (P * A) if is_radial else (P * A[:, None])
    # ---- compute using shared integrators (with budget for tiling) ----
    if is_radial:
        E = _integrate_radial_common(
            vx, vy, rho, w_r, P, use_complex64, j0_fn, xp,
            mem_budget_bytes=mem_budget_bytes
        )
    else:
        E = _integrate_phi_common(
            vx, vy, rho, phi, w_r, wphi, P, use_complex64, xp,
            mem_budget_bytes=mem_budget_bytes
        )

    # ---- return host ndarray (NumPy) ----
    E = E.astype(complex_type, copy=False)
    return to_host(E)


def compute_3d_psf_coherent(
    grid2d,
    NA,
    z_values,
    nsample = 1.0,
    nmedium = 1.0,
    pupil_function=None,         # None | (Nu,) | (Nu, Nphi)
    high_NA=False,               # True for high NA, False for paraxial
    Nphi=256,
    Nu=129,
    device='auto',               # 'auto' | 'cpu' | 'gpu'
    use_complex64=True,
    safety=0.70,
):
    """
    Compute a 3D coherent PSF by stacking 2D slices at defocus values z_values.

    For each z, we pass to compute_2d_psf_coherent() a pupil equal to:
        P(ρ, φ) = P(ρ, φ) * K(ρ; z)

    Defocus kernels K_defocus(ρ; z):
      - Paraxial:
          K(ρ; z) = exp( i * (2π * z * NA^2 / n_sample) * ρ^2 )
        (equivalently exp(i u^2/2) with u = sqrt(2 * 2π * z * NA^2 / n_sample) * ρ)

      - High NA (Debye-like, same medium):
          K(ρ; z) = exp( i * (2π * n_sample * z) * sqrt(1 - (NA/n_sample)^2 * ρ^2) )

    Returns
    -------
    E3D : np.ndarray (complex)
        Shape (Nz, *grid.shape[:-1]). Each slice is the complex coherent field.
    """

    rho, _ = _setup_rho_legendre(Nu, _cpu_roots_legendre, np, np.float32)  # (Nu,)
    E = []
    salpha = NA/nmedium
    if NA <= nsample:
        u_values = 4 * np.pi * z_values * (nsample - np.sqrt(nsample ** 2 - NA ** 2))
    else:
        u_values = 4 * np.pi * z_values * nsample * (1 - np.sqrt(1 - salpha ** 2))

    for u in u_values:

        if not high_NA:
            defocus_kernel = np.exp(1j * u/2 * (rho ** 2))
        else:
            cos_theta = np.sqrt(1.0 - (rho * salpha) ** 2)
            cos_alpha = np.sqrt(1.0 - salpha ** 2)
            defocus_kernel = np.exp(1j * u/2 * (1 - cos_theta) / (1 - cos_alpha))

        if pupil_function is None:
            P_z = defocus_kernel
        else:
            if pupil_function.ndim == 1:
                P_z = pupil_function * defocus_kernel                     
            else:
                P_z = pupil_function * defocus_kernel[:, None]

        Ez = compute_2d_psf_coherent(
            grid=grid2d,
            NA=NA,
            nmedium=nmedium,
            pupil_function=P_z,
            Nphi=Nphi,
            Nu=Nu,
            high_NA=high_NA,
            device=device,
            use_complex64=use_complex64,
            safety=safety,
        )

        E.append(Ez)
    return np.stack(E, axis=2)
