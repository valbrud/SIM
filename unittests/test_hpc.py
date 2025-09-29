# test_hpc.py
import os
import gc
import time
import math
import numpy as np
import pytest

import hpc_utils


# ----------------------------
# Helpers
# ----------------------------

def gpu_available() -> bool:
    """Return True if we can pick the GPU backend, False otherwise."""
    try:
        hpc_utils.pick_backend('gpu')
        mode, _ = hpc_utils.get_backend()
        # restore CPU to avoid accidental leakage into other tests
        hpc_utils.pick_backend('cpu')
        return (mode == 'gpu')
    except Exception:
        try:
            hpc_utils.pick_backend('cpu')
        except Exception:
            pass
        return False


def time_many(func, repeats=5, warmup=2):
    """
    Run func() warmup times, then repeats times; return (median, all_samples).
    Uses gc + perf_counter for less jitter.
    """
    samples = []
    for _ in range(warmup):
        func()
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    samples.sort()
    median = samples[len(samples)//2]
    return median, samples


def gaussian_kernel(k: int, sigma: float) -> np.ndarray:
    """Simple normalized 2D Gaussian kernel of odd size k."""
    assert k % 2 == 1, "kernel size must be odd"
    ax = np.arange(-(k//2), k//2 + 1)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    G = np.exp(-(X*X + Y*Y) / (2.0 * sigma * sigma))
    G /= G.sum()
    return G


# Allow quick tuning from environment
REPEATS = int(os.getenv("HPC_TEST_REPEATS", "5"))
WARMUP  = int(os.getenv("HPC_TEST_WARMUP",  "2"))

# Problem sizes chosen to be reasonably quick in CI but still illustrative
FFT_SHAPE = tuple(int(x) for x in os.getenv("HPC_TEST_FFT_SHAPE", "1024,1024").split(","))
IMG_SHAPE = tuple(int(x) for x in os.getenv("HPC_TEST_IMG_SHAPE", "1024,1024").split(","))
K_SIZE    = int(os.getenv("HPC_TEST_KERNEL_SIZE", "17"))
K_SIGMA   = float(os.getenv("HPC_TEST_KERNEL_SIGMA", "3.0"))


# ----------------------------
# Correctness sanity checks
# ----------------------------

@pytest.mark.parametrize("shape", [FFT_SHAPE])
def test_wrapped_fftn_ifftn_correctness(shape):
    rng = np.random.default_rng(123)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # CPU reference
    hpc_utils.pick_backend('cpu')
    X_cpu = hpc_utils.wrapped_fftn(x)
    x_cpu = hpc_utils.wrapped_ifftn(X_cpu)

    # GPU (if available)
    if gpu_available():
        hpc_utils.pick_backend('gpu')
        X_gpu = hpc_utils.wrapped_fftn(x)
        x_gpu = hpc_utils.wrapped_ifftn(X_gpu)

        # Compare CPU vs GPU within reasonable tolerance
        assert np.allclose(X_cpu, X_gpu, rtol=1e-5, atol=1e-7)
        assert np.allclose(x_cpu, x_gpu, rtol=1e-5, atol=1e-7)


def test_convolve2d_correctness():
    rng = np.random.default_rng(42)
    img = rng.standard_normal(IMG_SHAPE)
    ker = gaussian_kernel(K_SIZE, K_SIGMA)

    hpc_utils.pick_backend('cpu')
    y_cpu = hpc_utils.convolve2d(img, ker, mode='same', boundary='wrap')

    if gpu_available():
        hpc_utils.pick_backend('gpu')
        y_gpu = hpc_utils.convolve2d(img, ker, mode='same', boundary='wrap')
        # Allow a slightly looser tolerance for convolution (FFT/alg differences under the hood)
        assert np.allclose(y_cpu, y_gpu, rtol=1e-5, atol=1e-6)


# ----------------------------
# Performance comparisons (printed; no hard asserts)
# ----------------------------

@pytest.mark.parametrize("shape", [FFT_SHAPE])
def test_perf_wrapped_fftn_cpu_vs_gpu(shape, capsys):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape)

    # CPU timing
    hpc_utils.pick_backend('cpu')
    cpu_median, cpu_samples = time_many(lambda: hpc_utils.wrapped_fftn(x), repeats=REPEATS, warmup=WARMUP)

    msg = [f"[FFT2D] shape={shape}"]
    msg.append(f"CPU  median: {cpu_median:.6f}s  samples={['%.6f'%v for v in cpu_samples]}")

    # GPU timing (if available)
    if gpu_available():
        hpc_utils.pick_backend('gpu')
        # one extra warmup to trigger plan creation reliably
        _ = hpc_utils.wrapped_fftn(x)
        gpu_median, gpu_samples = time_many(lambda: hpc_utils.wrapped_fftn(x), repeats=REPEATS, warmup=WARMUP)
        msg.append(f"GPU  median: {gpu_median:.6f}s  samples={['%.6f'%v for v in gpu_samples]}")
        if gpu_median > 0:
            msg.append(f"Speedup (CPU/GPU): {cpu_median/gpu_median:.2f}×")
    else:
        msg.append("GPU not available → skipped")

    print("\n".join(msg))
    # ensure output is shown in pytest logs
    captured = capsys.readouterr()
    assert "[FFT2D]" in captured.out


def test_perf_wrapped_ifftn_cpu_vs_gpu(capsys):
    rng = np.random.default_rng(9)
    x = rng.standard_normal(FFT_SHAPE) + 1j * rng.standard_normal(FFT_SHAPE)

    # CPU timing
    hpc_utils.pick_backend('cpu')
    cpu_median, cpu_samples = time_many(lambda: hpc_utils.wrapped_ifftn(x), repeats=REPEATS, warmup=WARMUP)

    msg = [f"[IFFT2D] shape={FFT_SHAPE}"]
    msg.append(f"CPU  median: {cpu_median:.6f}s  samples={['%.6f'%v for v in cpu_samples]}")

    # GPU timing (if available)
    if gpu_available():
        hpc_utils.pick_backend('gpu')
        _ = hpc_utils.wrapped_ifftn(x)
        gpu_median, gpu_samples = time_many(lambda: hpc_utils.wrapped_ifftn(x), repeats=REPEATS, warmup=WARMUP)
        msg.append(f"GPU  median: {gpu_median:.6f}s  samples={['%.6f'%v for v in gpu_samples]}")
        if gpu_median > 0:
            msg.append(f"Speedup (CPU/GPU): {cpu_median/gpu_median:.2f}×")
    else:
        msg.append("GPU not available → skipped")

    print("\n".join(msg))
    captured = capsys.readouterr()
    assert "[IFFT2D]" in captured.out


def test_perf_convolve2d_cpu_vs_gpu(capsys):
    rng = np.random.default_rng(21)
    img = rng.standard_normal(IMG_SHAPE)
    ker = gaussian_kernel(K_SIZE, K_SIGMA)

    # CPU timing
    hpc_utils.pick_backend('cpu')
    cpu_median, cpu_samples = time_many(lambda: hpc_utils.convolve2d(img, ker, mode='same', boundary='wrap'),
                                        repeats=REPEATS, warmup=WARMUP)

    msg = [f"[CONV2D] img={IMG_SHAPE} kernel={K_SIZE}x{K_SIZE} boundary=wrap"]
    msg.append(f"CPU  median: {cpu_median:.6f}s  samples={['%.6f'%v for v in cpu_samples]}")

    # GPU timing (if available)
    if gpu_available():
        hpc_utils.pick_backend('gpu')
        _ = hpc_utils.convolve2d(img, ker, mode='same', boundary='wrap')
        gpu_median, gpu_samples = time_many(lambda: hpc_utils.convolve2d(img, ker, mode='same', boundary='wrap'),
                                            repeats=REPEATS, warmup=WARMUP)
        msg.append(f"GPU  median: {gpu_median:.6f}s  samples={['%.6f'%v for v in gpu_samples]}")
        if gpu_median > 0:
            msg.append(f"Speedup (CPU/GPU): {cpu_median/gpu_median:.2f}×")
    else:
        msg.append("GPU not available → skipped")

    print("\n".join(msg))
    captured = capsys.readouterr()
    assert "[CONV2D]" in captured.out
