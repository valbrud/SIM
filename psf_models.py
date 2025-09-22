import numpy as np
import time

def _cpu_free_and_budget(safety=0.70):
    """Return (free_bytes, budget_bytes) for system RAM, or (None, None) if psutil missing."""
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
        return free, int(free * safety)
    except Exception:
        return None, None

def _cpu_suggest_workers(budget_bytes, per_tile_bytes, reserve_cores=1):
    """
    Suggest a thread count that respects both cores and memory.
    If budget or per_tile_bytes is unknown, fallback to (cores - reserve).
    """
    total = max(1, os.cpu_count() or 1)
    by_cores = max(1, total - reserve_cores)
    if not budget_bytes or not per_tile_bytes:
        return by_cores
    by_mem = max(1, int(budget_bytes // max(1, per_tile_bytes)))
    return max(1, min(by_cores, by_mem))

def _gpu_free_and_budget(cp_mod, safety=0.70):
    """Return (free_bytes, budget_bytes) for the active CUDA device."""
    try:
        free_bytes, _ = cp_mod.cuda.Device().mem_info
        return int(free_bytes), int(free_bytes * safety)
    except Exception:
        return None, None

def _dtype_byte_sizes(use_complex64):
    """(float_bytes, complex_bytes) for current precision."""
    if use_complex64:
        return 4, 8    # float32, complex64
    else:
        return 8, 16   # float64, complex128
    
def _pick_backend(device='auto'):
    """
    Returns (xp, cp) where xp is numpy or cupy. cp is cupy module or None.
    device='auto' -> GPU if CuPy available and a device exists, else CPU.
    device='cpu'  -> force NumPy
    device='gpu'  -> force CuPy (raises if unavailable)
    """
    if device == 'cpu':
        return np, None
    try:
        import cupy as cp
        if device in ('auto', 'gpu'):
            # If we can allocate a tiny array, assume GPU is usable.
            cp.asarray([0], dtype=cp.float32)
            return cp, cp
    except Exception:
        pass
    if device == 'gpu':
        raise RuntimeError("CuPy/GPU requested but not available.")
    return np, None


def _roots_legendre(N, cp_mod):
    """
    Legendre nodes/weights on [-1,1] for the chosen backend.
    - On CPU: numpy.polynomial.legendre.leggauss
    - On GPU: cupyx.scipy.special.roots_legendre if available; else compute on CPU and copy.
    Returns (x, w) on xp with dtype float64.
    """
    if cp_mod is None:
        x, w = np.polynomial.legendre.leggauss(N)
        return x, w
    # GPU path
    try:
        from cupyx.scipy.special import roots_legendre as _roots
        x, w = _roots(N)
        return x, w
    except Exception:
        # Fallback: compute on CPU then send to GPU
        x, w = np.polynomial.legendre.leggauss(N)
        return cp_mod.asarray(x), cp_mod.asarray(w)


def _ensure_xp_array(a, xp):
    """Move a NumPy/CuPy array to the chosen backend/dtype."""
    # If already that backend, cheap cast; else copy/convert.
    return xp.asarray(a)


def _normalize_pupil_array(pupil_array, rho, phi, xp):
    """
    Normalize/validate the explicit pupil array.

    Inputs
        pupil_array: explicit pupil array or None.
        - None                -> unit radial pupil
        - shape (Nu,)         -> radial-only pupil P(ρ) (uses 1D J0 integration)
        - shape (Nu, Nphi)    -> general pupil P(ρ, φ) (uses 2D φ integration)
    Returns
        P        : xp.ndarray, shape (Nu,) for radial-only, or (Nu, Nphi) for φ-dependent
        is_radial: bool  (True if shape is (Nu,))
    """
    if pupil_array is None:
        # default: unit radial pupil
        return xp.ones_like(rho), True

    P = xp.asarray(pupil_array)
    if P.ndim == 1:
        if P.shape[0] != rho.shape[0]:
            raise ValueError(f"pupil (Nu,) expected length {rho.size}, got {P.shape[0]}")
        return P, True

    if P.ndim == 2:
        Nu_expected, Nphi_expected = rho.shape[0], phi.shape[0]
        if P.shape != (Nu_expected, Nphi_expected):
            raise ValueError(
                f"pupil (Nu,Nphi) expected {(Nu_expected, Nphi_expected)}, got {P.shape}"
            )
        return P, False


def j0_bessel(x, xp, cp_mod):
    """
    Backend-aware J0(x) evaluator.
    - GPU: cupyx.scipy.special.j0
    - CPU: scipy.special.j0  (raises a clear error if SciPy is missing)
    """
    if cp_mod is None:
        try:
            from scipy.special import j0 as _j0cpu
        except Exception as e:
            raise RuntimeError(
                "CPU path requires scipy.special.j0; please install SciPy."
            ) from e
        return _j0cpu(x)
    else:
        from cupyx.scipy.special import j0 as _j0gpu
        return _j0gpu(x)
    

def compute_2d_psf_coherent_no_aberrations(
    grid,
    NA, 
    pupil_function=None,
    Nphi=256,
    Nu=129,
    device='auto',
    use_complex64=True,
):
    """
    AI-generated function. Architecture is developed by a human.

    Scalar coherent PSF (no aberrations), low-NA approximation.
    Integrand: exp(-i * rho * (vx*cos(phi) + vy*sin(phi))) * P(rho)
    Quadrature:
      - φ: periodic trapezoid, Nphi points on [0, 2π)
      - ρ: u-substitution (u = ρ^2) with Gauss–Legendre on u ∈ [0,1]
            ρ = √u,   dρ = du / (2√u)  =>  ∫ f(ρ) ρ dρ = 1/2 ∫ f(√u) du
    Normalization: on-axis field E(0) = 1 (using the *same* quadrature).
    Inputs:
      grid: (..., 2) array with spatial freqs (cycles) → internally scaled by 2π.
      pupil_function: callable P(rho) over [0,1], or None for unit pupil.
      Nphi: number of φ samples (endpoint excluded). Power-of-two is nice.
      Nu:   number of Gauss–Legendre nodes in u ∈ [0,1].
      device: 'auto' | 'cpu' | 'gpu'
      use_complex64: True -> complex64 (fast, light), False -> complex128
    Returns:
      E: complex field with shape grid[...,0].shape (same x–y shape as grid)
    """
    
    xp, cp_mod = _pick_backend(device)
    float_type = xp.float32 if use_complex64 else xp.float64
    complex_type = xp.complex64 if use_complex64 else xp.complex128

    float_bytes, complex_bytes = _dtype_byte_sizes(use_complex64)
    gpu_free, gpu_budget = _gpu_free_and_budget(cp_mod, safety=0.70)

    # --- parse inputs and basic arrays ---
    grid = _ensure_xp_array(grid, xp)
    if grid.shape[-1] != 2:
        raise ValueError("grid must have shape (..., 2) with (vx, vy).")
    vx = (2.0 * NA *  xp.pi * grid[..., 0]).astype(float_type)  # angular spatial frequency
    vy = (2.0 * NA *  xp.pi * grid[..., 1]).astype(float_type)

    # φ-quad (periodic trapezoid)
    Nphi = int(Nphi)
    phi  = xp.linspace(float_type(0.0), float_type(2.0)*xp.pi, Nphi, endpoint=False, dtype=float_type)  # [Nphi]
    wphi = (float_type(2.0) * xp.pi) / float_type(Nphi)

    # u-quad (Gauss–Legendre on [0,1]) via nodes/weights on [-1,1]
    x_gl, w_gl = _roots_legendre(int(Nu), cp_mod)  # float64 by default
    x_gl = x_gl.astype(float_type, copy=False)
    w_gl = w_gl.astype(float_type, copy=False)
    # Map to u ∈ [0,1]
    u_nodes = 0.5 * (x_gl + float_type(1.0))     # [Nu]
    w_u     = 0.5 * w_gl                 # [Nu]
    rho     = xp.sqrt(u_nodes)           # ρ = √u
    w_r     = 0.5 * w_u                  # because ∫ f(ρ) ρ dρ = 1/2 ∫ f(√u) du

    P, is_radial = _normalize_pupil_array(pupil_function, rho, phi, xp)

    if is_radial:
        vmag = xp.sqrt(vx * vx + vy * vy).astype(float_type, copy=False)  # (Nx,Ny)

        # Estimate memory for full (Nx,Ny,Nu) J0 path on GPU
        need_bytes = None
        if cp_mod is not None and gpu_budget is not None:
            NxNy = vmag.size
            Nu_local = rho.size
            # crude: temp J0 (complex) + a couple of float temps
            need_bytes = NxNy * Nu_local * (complex_bytes + 2 * float_bytes)

        if (cp_mod is not None) and (gpu_budget is not None) and (need_bytes > gpu_budget):
            # Tile over ρ so the working set fits VRAM
            # choose a safe rho_block based on budget
            per_rho_bytes = vmag.size * (complex_bytes + 2 * float_bytes)
            rho_block = max(8, int(gpu_budget // max(1, per_rho_bytes)))
            rho_block = int(min(rho.size, rho_block))
            if rho_block < 8:
                rho_block = 8  # tiny safety floor

            E_acc = xp.zeros_like(vmag, dtype=complex_type)
            start = 0
            while start < rho.size:
                stop = min(rho.size, start + rho_block)
                J0 = j0_bessel(vmag[..., None] * rho[None, None, start:stop], xp, cp_mod)     # (Nx,Ny,nb)
                I_rho = (float_type(2.0) * xp.pi) * J0 * P[None, None, start:stop]           # (Nx,Ny,nb)
                E_acc = E_acc + xp.tensordot(I_rho, w_r[start:stop], axes=([2], [0]))
                start = stop
            E = E_acc.astype(complex_type, copy=False)

        else:
            # Full, single-pass evaluation fits in VRAM or we are on CPU
            J0 = j0_bessel(vmag[..., None] * rho[None, None, :], xp, cp_mod)                 # (Nx,Ny,Nu)
            I_rho = (float_type(2.0) * xp.pi) * J0 * P[None, None, :]                        # (Nx,Ny,Nu)
            E = xp.tensordot(I_rho, w_r, axes=([2], [0])).astype(complex_type, copy=False)   # (Nx,Ny)
    else:
        # Accumulate (Nx,Ny,Nu) without allocating the full (Nx,Ny,Nu,Nphi)
        Nx, Ny = vx.shape
        Iphi_acc = xp.zeros((Nx, Ny, rho.size), dtype=complex_type)

        # Pick a phi_block that fits GPU VRAM (or just use full Nphi on CPU)
        if cp_mod is not None and gpu_budget is not None:
            # rough per-φ-sample cost for (Nx,Ny,Nu,P): kx, ky, phase (floats) + kernel, integrand (complex)
            per_phi_bytes = (vx.size * rho.size) * (3 * float_bytes + 2 * complex_bytes)
            # plus the accumulator once
            acc_bytes = (vx.size * rho.size) * complex_bytes
            # pick the largest block that fits budget
            max_block = max(1, int((gpu_budget - acc_bytes) // max(1, per_phi_bytes)))
            phi_block = int(min(Nphi, max(8, max_block)))
        else:
            phi_block = Nphi  # CPU or unknown budget → single pass

        p0 = 0
        while p0 < Nphi:
            ps = slice(p0, min(p0 + phi_block, Nphi))
            phi_tile = phi[ps]                         # (P,)
            cphi = xp.cos(phi_tile)                    # (P,)
            sphi = xp.sin(phi_tile)                    # (P,)

            kx = vx[..., None, None] * rho[None, None, :, None] * cphi[None, None, None, :]
            ky = vy[..., None, None] * rho[None, None, :, None] * sphi[None, None, None, :]
            phase = -1j * (kx + ky)                    # (Nx,Ny,Nu,P)
            kernel = xp.exp(phase)                     # (Nx,Ny,Nu,P)

            P_tile = P[:, ps]                          # (Nu,P)
            integrand = kernel * P_tile[None, None, :, :]   # (Nx,Ny,Nu,P)
            Iphi_acc += wphi * xp.sum(integrand, axis=-1)   # (Nx,Ny,Nu)

            p0 = ps.stop

        E = xp.tensordot(Iphi_acc, w_r, axes=([2], [0])).astype(complex_type, copy=False)  # (Nx,Ny)
    return E if xp == np else E.get()