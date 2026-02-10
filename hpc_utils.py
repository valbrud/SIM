# hpc_utils.py
# Supplementary utilities for HPC (High-Performance Computing) tasks.
# Provides backend selection and array manipulation functions to support both CPU and GPU computations.
# Designed to automatically select GPU if possible by default and not produce overhead when using CPU.

import numpy as _np

# --- Global backend state ----------------------------------------------------
_MODE      = 'cpu'     # 'cpu' | 'gpu'
_xp        = _np       # numpy or cupy
_fft_mod   = _np.fft   # numpy.fft or cupyx.scipy.fft
_sig_mod   = None      # scipy.signal or cupyx.scipy.signal


def _detect_gpu():
    try:
        import cupy as cp
        _ = cp.asarray([0], dtype=cp.float32)  # smoke test
        return 'gpu', cp
    except Exception:
        return 'cpu', None


def pick_backend(device='auto'):
    """
    Configure the global backend once.
    device: 'auto' | 'cpu' | 'gpu'
    """
    global _MODE, _xp, _fft_mod, _sig_mod

    device = (device or 'auto').lower()
    if device not in ('auto', 'cpu', 'gpu'):
        raise ValueError("pick_backend(device) expects 'auto', 'cpu', or 'gpu'")

    if device == 'cpu':
        _MODE, _xp, _fft_mod = 'cpu', _np, _np.fft
        try:
            import scipy.signal as _scipy_signal
            _sig_mod = _scipy_signal
        except Exception:
            _sig_mod = None
        return _MODE

    if device == 'gpu':
        mode, cp = _detect_gpu()
        if mode != 'gpu':
            raise RuntimeError("GPU requested but CuPy is not available.")
        import cupyx.scipy.fft as cpx_fft
        import cupyx.scipy.signal as cpx_signal
        _MODE, _xp, _fft_mod, _sig_mod = 'gpu', cp, cpx_fft, cpx_signal
        return _MODE

    # auto
    mode, cp = _detect_gpu()
    if mode == 'gpu':
        import cupyx.scipy.fft as cpx_fft
        import cupyx.scipy.signal as cpx_signal
        _MODE, _xp, _fft_mod, _sig_mod = 'gpu', cp, cpx_fft, cpx_signal
    else:
        _MODE, _xp, _fft_mod = 'cpu', _np, _np.fft
        try:
            import scipy.signal as _scipy_signal
            _sig_mod = _scipy_signal
        except Exception:
            _sig_mod = None
    return _MODE


def get_backend():
    """Return ('cpu'|'gpu', xp_module)."""
    return _MODE, _xp


# --- Zero-copy helpers -------------------------------------------------------
def _as_xp(a):
    """
    Return 'a' as an array on the current backend with zero-copy when possible:
      - CPU backend: NumPy array input returns the same object
      - GPU backend: CuPy array input returns the same object; NumPy→CuPy copies (necessary)
    """
    if _xp is _np:
        return a if isinstance(a, _np.ndarray) else _np.asarray(a)
    # GPU path
    try:
        import cupy as _cp
        if isinstance(a, _cp.ndarray):
            return a
    except Exception:
        pass
    return _xp.asarray(a)

def _to_numpy(a):
    """
    Return a NumPy ndarray with zero-copy when already NumPy; move device→host only if needed.
    """
    if isinstance(a, _np.ndarray):
        return a
    if _xp is _np:
        # Not a NumPy ndarray (python scalar/list); asarray will create once.
        return _np.asarray(a)
    # GPU -> host copy
    try:
        return _xp.asnumpy(a)
    except Exception:
        return _np.asarray(a)


# --- Shift-safe FFT wrappers (match your wrappers.wrapper_ft semantics) ------
def wrapped_fft(a, n=None, axis=-1, norm=None, device='auto'):
    x = _as_xp(a)
    y = _xp.fft.fftshift(_fft_mod.fft(_xp.fft.ifftshift(x), n=n, axis=axis, norm=norm))
    return _to_numpy(y)

def wrapped_ifft(a, n=None, axis=-1, norm=None, device='auto'):
    x = _as_xp(a)
    y = _xp.fft.fftshift(_fft_mod.ifft(_xp.fft.ifftshift(x), n=n, axis=axis, norm=norm))
    return _to_numpy(y)

def wrapped_fftn(a, s=None, axes=None, norm=None, device='auto'):
    x = _as_xp(a)
    y = _xp.fft.fftshift(_fft_mod.fftn(_xp.fft.ifftshift(x), s=s, axes=axes, norm=norm))
    return _to_numpy(y)

def wrapped_ifftn(a, s=None, axes=None, norm=None, device='auto'):
    x = _as_xp(a)
    y = _xp.fft.fftshift(_fft_mod.ifftn(_xp.fft.ifftshift(x), s=s, axes=axes, norm=norm))
    return _to_numpy(y)


# --- Convolution wrappers (drop-in SciPy/CuPy signal) -----------------------
def convolve2d(in2d, kernel, mode='same', boundary='fill', fillvalue=0, device='auto'):
    if _sig_mod is None:
        raise RuntimeError("Signal module not available on this backend.")

    a = _as_xp(in2d)
    k = _as_xp(kernel)
    y = _sig_mod.convolve2d(a, k, mode=mode, boundary=boundary, fillvalue=fillvalue)
    return _to_numpy(y)

def convolve(in1, in2, mode='full', method='auto', device='auto'):
    if _sig_mod is None:
        raise RuntimeError("Signal module not available on this backend.")

    a = _as_xp(in1)
    b = _as_xp(in2)
    y = _sig_mod.convolve(a, b, mode=mode, method=method)
    return _to_numpy(y)


# --- Chirp-Z transform (Bluestein) for grid-defined FT -----------------------
# Computes along any axis, supports CPU/GPU via your global backend.
# Requires uniformly spaced coordinate arrays (dx constant, dq constant)

def _is_uniform_grid_1d(x, rtol=1e-4, atol=1e-4):
    """
    Check if 1D coordinate array is uniformly spaced.
    Returns (is_uniform, x0, dx).
    """
    if x.ndim != 1:
        raise ValueError("Coordinate arrays must be 1D.")
    n = int(x.size)
    if n < 2:
        return True, float(x[0]) if n == 1 else 0.0, 1.0

    # Use NumPy for diagnostics (host-side) to avoid device sync explosions.
    xn = _to_numpy(x)
    dxs = _np.diff(xn.astype(_np.float64))
    # print(f"_is_uniform_grid_1d: dxs = {dxs}")
    dx0 = dxs[0]
    if _np.allclose(dxs, dx0, rtol=rtol, atol=atol):
        return True, float(xn[0]), float(dx0)
    return False, float(xn[0]), float(dx0)


def precompute_czt_1d_from_coords(x_coords, q_coords, rtol=1e-6, atol=1e-10):
    """
    Precompute chirp-factors and chirp-kernel FFT for a 1D CZT that evaluates
    the Fourier sum on a uniformly spaced q-grid given uniformly spaced x-grid.

    We compute (per axis):
        F[q_m] = dx * sum_{k=0}^{N-1} f[x_k] * exp(-i q_m x_k)
    with x_k = x0 + k dx, q_m = q0 + m dq.

    Returns:
        A_chirp (N,), B_chirp (M,), D_kernel_fft (L,)
        and metadata dict with dx, dq, x0, q0, N, M, L
    """
    mode, xp = get_backend()

    x = _as_xp(x_coords)
    q = _as_xp(q_coords)

    okx, x0, dx = _is_uniform_grid_1d(x, rtol=rtol, atol=atol)
    okq, q0, dq = _is_uniform_grid_1d(q, rtol=rtol, atol=atol)
    if not okx:
        raise ValueError("x_coords must be uniformly spaced for CZT (constant dx). "
                         "Nonuniform x requires a NUFFT.")
    if not okq:
        raise ValueError("q_coords must be uniformly spaced for CZT (constant dq). "
                         "Nonuniform q requires a NUFFT or chunking/interpolation.")

    N = int(x.size)
    M = int(q.size)
    if N <= 0 or M <= 0:
        raise ValueError("Empty coordinate arrays are not allowed.")

    L = N + M - 1

    # Fourier kernel convention: exp(-i q x)
    # Define W = exp(-i dx*dq), sqW = exp(-i dx*dq/2)
    dx_dq = dx * dq
    sqW = xp.exp(-1j * dx_dq / 2.0)
    W = sqW * sqW  # exp(-1j * dx*dq)

    # A_scalar chosen so that A_scalar^{-k} = exp(-i q0 * (k dx))
    # => A_scalar = exp(+i q0 dx)
    A_scalar = xp.exp(1j * q0 * dx)

    # Input chirp-factor:
    # g[k] = f[k] * A_scalar^{-k} * W^{k^2/2}
    # with W^{k^2/2} = (sqW)^k * W^{k(k-1)/2}
    k = xp.arange(N, dtype=xp.float64)
    A_chirp = (sqW / A_scalar) ** k * (W ** (k * (k - 1.0) / 2.0))

    # Output chirp-factor:
    # F[m] = dx * exp(-i q_m x0) * W^{m^2/2} * (g*h)[m]
    # exp(-i q_m x0) = exp(-i q0 x0) * exp(-i m dq x0)
    # W^{m^2/2} = (sqW)^m * W^{m(m-1)/2}
    m = xp.arange(M, dtype=xp.float64)
    phase0 = xp.exp(-1j * q0 * x0)     # constant in m
    phase_step = xp.exp(-1j * dq * x0) # geometric in m
    B_chirp = (dx * phase0) * (sqW * phase_step) ** m * (W ** (m * (m - 1.0) / 2.0))

    # Chirp-kernel: h[n] = W^{-n^2/2} for n = -(N-1) .. (M-1)
    # Build via Vtmp[n] = W^{n^2/2} = (sqW)^n * W^{n(n-1)/2}
    # Then h[n] = conj(Vtmp[n]) because |W|=1 here.
    idx = xp.arange(max(N, M), dtype=xp.float64)  # need up to max(N-1, M-1)
    Vtmp = (sqW ** idx) * (W ** (idx * (idx - 1.0) / 2.0))

    d = xp.zeros(L, dtype=xp.complex128)
    # n = 0..M-1 goes to d[0..M-1]
    d[:M] = xp.conj(Vtmp[:M])
    # n = -(N-1)..-1 goes to d[L-(N-1)..L-1] with n = -i mapping to i
    if N > 1:
        i = xp.arange(1, N, dtype=xp.int64)
        d[L - i] = xp.conj(Vtmp[i])

    D_kernel_fft = _fft_mod.fft(d)

    meta = dict(dx=dx, dq=dq, x0=x0, q0=q0, N=N, M=M, L=L)
    return A_chirp, B_chirp, D_kernel_fft, meta


def czt_1d_apply(datain, A_chirp, B_chirp, D_kernel_fft, axis=-1):
    """
    Apply 1D CZT (Bluestein) along a given axis of an arbitrary-dimensional array.

    Args:
        datain: array-like, any shape
        A_chirp: (N,) complex chirp-factor (input chirp)
        B_chirp: (M,) complex chirp-factor (output chirp)
        D_kernel_fft: (L,) complex FFT(chirp-kernel)
        axis: axis to transform along (default last)

    Returns:
        dataout: same shape as datain but axis-length becomes M
                 (stays on current backend; use _to_numpy if you want host)
    """
    mode, xp = get_backend()
    x = _as_xp(datain)

    axis = int(axis)
    if axis < 0:
        axis += x.ndim

    N = int(A_chirp.size)
    M = int(B_chirp.size)
    L = int(D_kernel_fft.size)

    if x.shape[axis] != N:
        raise ValueError(f"Axis length mismatch: datain.shape[axis]={x.shape[axis]} "
                         f"but A_chirp has N={N}.")

    # Move target axis to last, flatten batch dims
    x_mv = xp.moveaxis(x, axis, -1)
    batch_shape = x_mv.shape[:-1]
    K = int(_np.prod(batch_shape)) if batch_shape else 1
    x2 = x_mv.reshape((K, N))

    # Pre-chirp and pad to length L
    cztin = xp.zeros((K, L), dtype=xp.complex128)
    cztin[:, :N] = x2 * A_chirp[None, :]

    # Convolution via FFT: ifft( fft(g)*FFT(h) )
    F = _fft_mod.fft(cztin, axis=1)
    F *= D_kernel_fft[None, :]
    conv = _fft_mod.ifft(F, axis=1)

    # Post-chirp, take first M
    y2 = conv[:, :M] * B_chirp[None, :]

    # Reshape back, move axis to original position
    y_mv = y2.reshape(batch_shape + (M,))
    y = xp.moveaxis(y_mv, -1, axis)
    return y


def czt_nd_fourier(datain, x_coords, q_coords, axes=None, rtol=1e-6, atol=1e-12):
    """
    Multi-dimensional FT via separable per-axis CZT.

    Args:
        datain: array-like with ndim = D' >= D
        x_coords: tuple/list of 1D coordinate arrays (x, y, z, ...)
                 length D. Each must be uniformly spaced.
        q_coords: tuple/list of 1D frequency-coordinate arrays (qx, qy, qz, ...)
                 length D. Each must be uniformly spaced.
        axes: which axes of datain correspond to these coordinates.
              If None, uses the last D axes.
        rtol, atol: grid uniformity checks

    Returns:
        dataout: transformed array, same ndim as datain,
                 with each transformed axis length replaced by len(q_coords[d]).
    """
    mode, xp = get_backend()
    x = _as_xp(datain)

    if not isinstance(x_coords, (tuple, list)) or not isinstance(q_coords, (tuple, list)):
        raise ValueError("x_coords and q_coords must be tuples/lists of 1D arrays.")
    D = len(x_coords)
    if D != len(q_coords):
        raise ValueError("x_coords and q_coords must have the same number of dimensions.")

    if axes is None:
        axes = list(range(x.ndim - D, x.ndim))
    else:
        axes = list(axes)
        if len(axes) != D:
            raise ValueError("axes must have the same length as x_coords/q_coords.")

    # Apply 1D CZT sequentially per axis (separable kernel exp(-i q·x))
    y = x
    for d in range(D):
        A_chirp, B_chirp, D_fft, _meta = precompute_czt_1d_from_coords(
            x_coords[d], q_coords[d], rtol=rtol, atol=atol
        )
        y = czt_1d_apply(y, A_chirp, B_chirp, D_fft, axis=axes[d])

    return y
