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