import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import unittest
import time
from psf_models import compute_2d_psf_coherent_no_aberrations
import wrappers

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

alpha = 2 * np.pi / 5
n = 1.518
NA = n * np.sin(alpha)
dx = 1 / (4 * NA)  
dy = dx
dz = 1 / (2 * (1 - np.cos(alpha)))

Nl = 255
Nz = 51 

max_r = Nl // 2 * dx
max_z = Nz // 2 * dz

psf_size2d = 2 * np.array((max_r, max_r))
psf_size3d = 2 * np.array((max_r, max_r, max_z))

x = np.linspace(-max_r, max_r, Nl)
y = np.copy(x)
z = np.linspace(-max_r, max_r, Nz)

x_grid2d = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1) 
x_grid3d = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)

fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nl)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), Nl)
fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), Nz)

q_grid2d = np.stack(np.meshgrid(fx, fy, indexing='ij'), axis=-1)
q_grid3d = np.stack(np.meshgrid(fx, fy, fz, indexing='ij'), axis=-1)

fxn = fx / (2 * NA)
fyn = fy / (2 * NA)
fzn = fz / (1 - np.cos(alpha))

def compute_incoherent_psf_and_otf(E_field):
    """
    Given complex field E(x,y), compute incoherent PSF = |E|^2 and its OTF
    (Fourier transform of the intensity). Returns (PSF, OTF) with OTF normalized
    so that OTF[0,0] is ~1.
    """
    I = np.abs(E_field)**2
    I /= np.sum(I)
    # OTF is the FFT of intensity (no shift needed for origin at [0,0])
    OTF = wrappers.wrapped_fftn(I).real
    return I, OTF

def plot_psf_otf_plot(I, OTF, title):
    fig = plt.figure(figsize=(10, 4.2), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(I, origin='lower')
    ax1.set_title(f"PSF intensity\n{title}")
    # fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(np.abs(OTF), origin='lower')  # show magnitude
    ax2.set_title(f"|OTF| (FFT of PSF)\n{title}")
    # fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x[Nl//2:], I[Nl//2, Nl//2:], label='fx cut (fy=0)')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(fyn[Nl//2:], OTF[Nl//2:, Nl//2], label='fy cut (fx=0)')
    fig.tight_layout()
    plt.show()

    # fig.savefig(outpath, dpi=140)
    # plt.close(fig)

def _gpu_available():
    try:
        import cupy as cp
        # attempt a tiny alloc
        _ = cp.asarray([0], dtype=cp.float32)
        return True
    except Exception:
        return False



def make_vortex_pupil(
    rho,
    phi,
    m=1,
    rho_inner=0.0,
    rho_outer=1.0,
    amplitude=None,
    dtype=np.complex128,
):
    """
    Build a true vortex pupil P(ρ, φ) = A(ρ) * exp(i m φ) within an annulus.

    Parameters
    ----------
    rho : (Nu,) array
        Radial quadrature nodes in [0,1].
    phi : (Nphi,) array
        Angular nodes in [0, 2π).
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
    rho = np.asarray(rho)
    phi = np.asarray(phi)

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

    return P.astype(dtype, copy=False)

# ---- tests -------------------------------------------------------------------

class TestPSFOTF(unittest.TestCase):

    def test_cpu_basic(self):
        """CPU: sanity run, normalization, finite values, and plots."""
        
        # --- TIME MARKER START ---
        t_start = time.time()
        E = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA=n * np.sin(alpha), pupil_function=None, Nphi=256, Nu=129, device='cpu'
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---
        
        self.assertEqual(E.shape, x_grid2d[...,0].shape)
        self.assertTrue(np.isfinite(E).all())
        # On-axis normalization: origin is [0,0] in our grid construction
        # self.assertAlmostEqual(abs(E[0,0]), 1.0, places=2)

        I, OTF = compute_incoherent_psf_and_otf(E)
        plot_psf_otf_plot(I, OTF, "CPU, flat pupil")

    @unittest.skipUnless(_gpu_available(), "CuPy/GPU not available")
    def test_gpu_basic(self):
        """GPU: sanity run mirrors CPU pipeline; plots saved."""
        
        # --- TIME MARKER START ---
        t_start = time.time()
        E = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA=n * np.sin(alpha), pupil_function=None, Nphi=256, Nu=129, device='gpu'
        )
        t_end = time.time()
        print(f"test_gpu_basic: compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---
        
        self.assertEqual(E.shape, x_grid2d[..., 0].shape)
        # bring back to host if it's a CuPy array
        try:
            import cupy as cp
            if isinstance(E, cp.ndarray):
                E = cp.asnumpy(E)
        except Exception:
            pass
        self.assertTrue(np.isfinite(E).all())
        # self.assertAlmostEqual(abs(E[0,0]), 1.0, places=2)

        I, OTF = compute_incoherent_psf_and_otf(E)
        plot_psf_otf_plot(I, OTF, "GPU, flat pupil")

    def test_nontrivial_pupil_vortex(self):
        """Non-trivial pupil (vortex-like radial apodization proxy)."""
        Nphi = 255
        Nu = 129
        float_type = np.float64
        xp = np
        cp_mod = None
        from psf_models import _roots_legendre
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

        P_vortex = make_vortex_pupil(rho, phi, m=1, rho_inner=0.0, rho_outer=1.0)

        # --- TIME MARKER START ---
        t_start = time.time()
        E = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=P_vortex, Nphi=Nphi, Nu=Nu, device='gpu'
        )
        t_end = time.time()
        print(f"test_nontrivial_pupil_vortex_like: compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")

        t_start = time.time()
        E = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=P_vortex, Nphi=Nphi, Nu=Nu, device='gpu'
        )
        t_end = time.time()
        print(f"test_nontrivial_pupil_vortex_like: compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")

        
        t_start = time.time()
        E = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=P_vortex, Nphi=Nphi, Nu=Nu, device='gpu'
        )
        t_end = time.time()
        print(f"test_nontrivial_pupil_vortex_like: compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---

        # self.assertEqual(E.shape, x_grid2d[...,0].shape)
        # self.assertTrue(np.isfinite(E).all())
        # self.assertAlmostEqual(abs(E[0,0]), 1.0, places=2)  # normalized

        I, OTF = compute_incoherent_psf_and_otf(E)
        plot_psf_otf_plot(I, OTF, "CPU, vortex-like pupil (radial proxy)")

    def test_low_vs_high_quadrature(self):
        """CPU: compare low vs high integration settings (robustness + plots)."""

        # Low-accuracy (fast)
        # --- TIME MARKER START ---
        t_start = time.time()
        E_low = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=None, Nphi=64, Nu=33, device='cpu'
        )
        t_end = time.time()
        print(f"test_low_vs_high_quadrature (low): compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---
        
        # High-accuracy (reference-ish)
        # --- TIME MARKER START ---
        t_start = time.time()
        E_high = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=None, Nphi=129, Nu=129, device='cpu'
        )
        t_end = time.time()
        print(f"test_low_vs_high_quadrature (high): compute_2d_psf_coherent_no_aberrations took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---

        # Basic checks
        self.assertEqual(E_low.shape, x_grid2d[...,0].shape)
        self.assertEqual(E_high.shape, x_grid2d[...,0].shape)
        self.assertTrue(np.isfinite(E_low).all())
        self.assertTrue(np.isfinite(E_high).all())
        # Both are normalized on-axis (origin at [0,0])
        # self.assertAlmostEqual(abs(E_low[0,0]), 1.0, places=2)
        # self.assertAlmostEqual(abs(E_high[0,0]), 1.0, places=3)

        # Save comparison plots
        I_low, OTF_low = compute_incoherent_psf_and_otf(E_low)
        I_high, OTF_high = compute_incoherent_psf_and_otf(E_high)

        plot_psf_otf_plot(I_low, OTF_low, "CPU, low quadrature (Nphi=64, Nu=33)")
        plot_psf_otf_plot(I_high, OTF_high, "CPU, high quadrature (Nphi=256, Nu=129)")


# Optional: a second GPU test to pair with low-vs-high, but it’s skipped if no GPU
class TestPSFOTF_GPU_Extras(unittest.TestCase):
    @unittest.skipUnless(_gpu_available(), "CuPy/GPU not available")
    def test_gpu_low_vs_high_quadrature(self):
        E_low = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=None, Nphi=64, Nu=33, device='gpu'
        )
        E_high = compute_2d_psf_coherent_no_aberrations(
            x_grid2d, NA, pupil_function=None, Nphi=129, Nu=129, device='gpu'
        )
        # Bring to host if needed
        try:
            import cupy as cp
            if isinstance(E_low, cp.ndarray):  E_low  = cp.asnumpy(E_low)
            if isinstance(E_high, cp.ndarray): E_high = cp.asnumpy(E_high)
        except Exception:
            pass

        self.assertTrue(np.isfinite(E_low).all())
        self.assertTrue(np.isfinite(E_high).all())

        I_low, OTF_low = compute_incoherent_psf_and_otf(E_low)
        I_high, OTF_high = compute_incoherent_psf_and_otf(E_high)
        plot_psf_otf_plot(I_low, OTF_low, "GPU, low quadrature")
        plot_psf_otf_plot(I_high, OTF_high, "GPU, high quadrature")


if __name__ == "__main__":
    unittest.main()
    plt.show()