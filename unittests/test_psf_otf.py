import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import unittest
import time
from psf_models import *
from pupil_functions import make_vortex_pupil, compute_pupil_plane_aberrations
import hpc_utils
import pupil_functions
from otf_fast import compute_3d_otf_fast
# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

alpha = 2 * np.pi / 5
n = 1
NA = n * np.sin(alpha)
dx = 1 / (8 * NA)  
dy = dx
dz = 1 / n / (4 * (1 - np.cos(alpha)))

Nl = 71
Nz = 21 

max_r = Nl // 2 * dx
max_z = Nz // 2 * dz

psf_size2d = 2 * np.array((max_r, max_r))
psf_size3d = 2 * np.array((max_r, max_r, max_z))

x = np.linspace(-max_r, max_r, Nl)
y = np.copy(x)
z = np.linspace(-max_z, max_z, Nz)

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
    OTF = hpc_utils.wrapped_fftn(I).real
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


# ---- tests -------------------------------------------------------------------

class TestPSFOTF2D(unittest.TestCase):

    def test_cpu_basic(self):
        """CPU: sanity run, normalization, finite values, and plots."""
        
        # --- TIME MARKER START ---
        t_start = time.time()
        E = compute_2d_psf_coherent(
            x_grid2d, NA=n * np.sin(alpha), pupil_function=None, Nphi=256, Nu=129, device='cpu'
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")
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
        E = compute_2d_psf_coherent(
            x_grid2d, NA=n * np.sin(alpha), pupil_function=None, Nphi=256, Nu=129, device='gpu'
        )
        t_end = time.time()
        print(f"test_gpu_basic: compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")
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
        Nphi = 455
        Nrho = 125
        P_vortex = make_vortex_pupil(Nrho, Nphi, m=1, rho_inner=0.0, rho_outer=1.0)

        # --- TIME MARKER START ---
        t_start = time.time()
        E = compute_2d_psf_coherent(
            x_grid2d, NA, pupil_function=P_vortex, Nphi=Nphi, Nu=Nrho, device='gpu'
        )
        t_end = time.time()
        print(f"test_nontrivial_pupil_vortex_like: compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")

        t_start = time.time()
        E = compute_2d_psf_coherent(
            x_grid2d, NA, pupil_function=P_vortex, Nphi=Nphi, Nu=Nrho, device='gpu'
        )
        t_end = time.time()
        print(f"test_nontrivial_pupil_vortex_like: compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")

        
        t_start = time.time()
        E = compute_2d_psf_coherent(
            x_grid2d, NA, pupil_function=P_vortex, Nphi=Nphi, Nu=Nrho, device='gpu'
        )
        t_end = time.time()
        print(f"test_nontrivial_pupil_vortex_like: compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")
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
        E_low = compute_2d_psf_coherent(
            x_grid2d, NA, pupil_function=None, Nphi=64, Nu=33, device='gpu'
        )
        t_end = time.time()
        print(f"test_low_vs_high_quadrature (low): compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---
        
        # High-accuracy (reference-ish)
        # --- TIME MARKER START ---
        t_start = time.time()
        E_high = compute_2d_psf_coherent(
            x_grid2d, NA, pupil_function=None, Nphi=129, Nu=129, device='gpu'
        )
        t_end = time.time()
        print(f"test_low_vs_high_quadrature (high): compute_2d_psf_coherent took {t_end - t_start:.4f} seconds")
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

    def test_vectorial_vs_scalar(self):
        """Test scalar vs vectorial."""
        t_start = time.time()
        E_scalar = compute_2d_psf_coherent(
            grid=x_grid2d,
            NA=n * np.sin(alpha),
            nmedium=n,
            high_NA=True,
            pupil_function=None,
            Nphi=101,
            Nu=101,
            device='gpu', 
            safety=0.9
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_3d_scalar_psf took {t_end - t_start:.4f} seconds")

        Is, OTFs = compute_incoherent_psf_and_otf(E_scalar)

        t_start = time.time()
        Iv = compute_2d_incoherent_vectorial_psf_free_dipole(
            grid=x_grid2d,
            NA=n * np.sin(alpha),
            nmedium=n,
            high_NA=True,
            pupil_function=None,
            Nphi=101,
            Nu=101,
            device='gpu', 
            safety=0.9
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_3d_vectorial_psf took {t_end - t_start:.4f} seconds")

        OTFv = hpc_utils.wrapped_fftn(Iv).real

        plot_psf_otf_plot(np.log1p(10**4 * Is[:,:]), np.log1p(10**4 * OTFs[:,:]), "PSF/OTF central z slice low_NA_model")
        plot_psf_otf_plot(np.log1p(10 ** 4 * Iv[:,:]), np.log1p(10 ** 4 * OTFv[:,:]), "PSF/OTF central z slice high_NA_model")

        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        axes[0].set_title("PSF x cut (y=0, z=center)")
        axes[1].set_title("PSF z cut (x=0, y=center)")
        axes[0].plot(np.log1p(10**4 * Is[:, Nl//2]), label="Scalar model")
        axes[0].plot(np.log1p(10**4 * Iv[Nl//2, :]), label="Vectorial model")
        axes[0].legend()

        axes[1].plot(OTFs[Nl//2, :], label="Scalar model")
        axes[1].plot(OTFv[Nl//2, :], label = "Vectorial model")
        axes[1].legend()


        plt.show()

    def test_vectorial_zernieke(self):
        """Test high NA vs low NA."""
        zernieke = {
            (2, -2): 0.072,  # Astigmatism Oblique
        }
        aberration_phase = pupil_functions.compute_pupil_plane_aberrations(zernieke, Nrho=129, Nphi=256)
        aberration_function = np.exp(1j * 2 * np.pi * aberration_phase)

        t_start = time.time()
        psf = compute_2d_incoherent_vectorial_psf_free_dipole(
            grid=x_grid2d,
            NA=n * np.sin(alpha),
            nmedium=n,
            high_NA=True,
            pupil_function=aberration_function,
            device='gpu', 
            safety=0.9,
        )
        t_end = time.time()
        print(f"test_gpu_basic: compute_2d_vectorial_psf with aberrations took {t_end - t_start:.4f} seconds")
        OTF = hpc_utils.wrapped_fftn(psf).real
        plot_psf_otf_plot(np.log1p(10 ** 4 * psf[:,:]), np.log1p(10 ** 4 * OTF[:,:]), "PSF/OTF central z slice high_NA_model with Zernike aberrations")

class TestPSFOTF3D(unittest.TestCase):

    def test_high_NA_no_aberrations(self):
        
        """Test high NA vs low NA."""
        t_start = time.time()
        E_high = compute_3d_psf_coherent(
            grid2d=x_grid2d,
            z_values=z,
            NA=n * np.sin(alpha),
            nmedium=n,
            nsample=n,
            high_NA=True,
            pupil_function=None,
            Nphi=201,
            Nrho=101,
            device='gpu'
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_3d_psf_coherent took {t_end - t_start:.4f} seconds")
        # --- TIME MARKER END ---
        
        self.assertEqual(E_high.shape, x_grid3d[...,0].shape)
        self.assertTrue(np.isfinite(E_high).all())

        I_high, OTF_high = compute_incoherent_psf_and_otf(E_high)

        E_low = compute_3d_psf_coherent(
            grid2d=x_grid2d,
            z_values=z,
            NA=n * np.sin(alpha),
            nmedium=n,
            nsample=n,
            high_NA=False,
            pupil_function=None,
            Nphi=201,
            Nu=101,
            device='gpu'
        )

        I_low, OTF_low = compute_incoherent_psf_and_otf(E_low)

        plot_psf_otf_plot(np.log1p(10**4 * I_low[:,:,Nz//2]), np.log1p(10**4 *OTF_low[:,:,Nz//2]), "PSF/OTF central z slice low_NA_model")
        plot_psf_otf_plot(np.log1p(10 ** 4 * I_high[:,:,Nz//2]), np.log1p(10 ** 4 * OTF_high[:,:,Nz//2]), "PSF/OTF central z slice high_NA_model")

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].set_title("PSF x cut (y=0, z=center)")
        axes[0, 1].set_title("PSF z cut (x=0, y=center)")
        axes[1, 0].set_title("OTF fx cut (fy=0, fz=center)")
        axes[1, 1].set_title("OTF fz cut (fx=0, fy=center)")
        axes[0, 0].plot(np.log1p(10**4 * I_low[:, Nl//2, Nz//2]), label="Paraxial approximation")
        axes[0, 0].plot(np.log1p(10**4 * I_high[Nl//2, :, Nz//2]), label="high_NA_model")
        axes[0, 0].legend()
        axes[0, 1].plot(I_low[Nl//2, Nl//2, :], label="Paraxial approximation")
        axes[0, 1].plot(I_high[Nl//2, Nl//2, :], label="high_NA_model")
        axes[0, 1].legend()
        axes[1, 0].plot(OTF_low[Nl//2, :, Nz//2], label="Paraxial approximation")
        axes[1, 0].plot(OTF_high[Nl//2, :, Nz//2], label = "high_NA_model")
        axes[1, 0].legend()
        axes[1, 1].plot(OTF_low[Nl//2, Nl//4, :], label = "Paraxial approximation")
        axes[1, 1].plot(OTF_high[Nl//2, Nl//4, :], label = "high_NA_model")
        axes[1, 1].legend()

        plt.show()

    def test_vectorial_vs_scalar(self):
        """Test high NA vs low NA."""
        t_start = time.time()
        E_scalar = compute_3d_psf_coherent(
            grid2d=x_grid2d,
            z_values=z,
            NA=n * np.sin(alpha),
            nmedium=n,
            nsample=n,
            high_NA=True,
            pupil_function=None,
            Nphi=101,
            Nu=101,
            device='gpu', 
            safety=0.9
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_3d_scalar_psf took {t_end - t_start:.4f} seconds")

        Is, OTFs = compute_incoherent_psf_and_otf(E_scalar)

        t_start = time.time()
        Iv = compute_3d_incoherent_vectorial_psf_free_dipole(
            grid2d=x_grid2d,
            z_values=z,
            NA=n * np.sin(alpha),
            nmedium=n,
            nsample=n,
            high_NA=True,
            pupil_function=None,
            Nphi=101,
            Nu=101,
            device='gpu', 
            safety=0.9
        )
        t_end = time.time()
        print(f"test_cpu_basic: compute_3d_vectorial_psf took {t_end - t_start:.4f} seconds")

        OTFv = hpc_utils.wrapped_fftn(Iv).real

        plot_psf_otf_plot(np.log1p(10 ** 4 * Is[:,:,Nz//2]), np.log1p(10 ** 4 * OTFs[:,:,Nz//2]), "PSF/OTF central z slice low_NA_model")
        plot_psf_otf_plot(np.log1p(10 ** 4 * Iv[:,:,Nz//2]), np.log1p(10 ** 4 * OTFv[:,:,Nz//2]), "PSF/OTF central z slice high_NA_model")

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].set_title("PSF x cut (y=0, z=center)")
        axes[0, 1].set_title("PSF z cut (x=0, y=center)")
        axes[1, 0].set_title("OTF fx cut (fy=0, fz=center)")
        axes[1, 1].set_title("OTF fz cut (fx=0, fy=center)")
        axes[0, 0].plot(np.log1p(10**4 * Is[:, Nl//2, Nz//2]), label="Scalar model")
        axes[0, 0].plot(np.log1p(10**4 * Iv[Nl//2, :, Nz//2]), label="Vectorial model")
        axes[0, 0].legend()
        axes[0, 1].plot(Is[Nl//2, Nl//2, :], label="Scalar model")
        axes[0, 1].plot(Iv[Nl//2, Nl//2, :], label="Vectorial model")
        axes[0, 1].legend()
        axes[1, 0].plot(OTFs[Nl//2, :, Nz//2], label="Scalar model")
        axes[1, 0].plot(OTFv[Nl//2, :, Nz//2], label = "Vectorial model")
        axes[1, 0].legend()
        axes[1, 1].plot(OTFs[Nl//2, Nl//4, :], label = "Scalar model")
        axes[1, 1].plot(OTFv[Nl//2, Nl//4, :], label = "Vectorial model")
        axes[1, 1].legend()

        plt.show()

class TestFastOTF3D(unittest.TestCase):
    def test_fast_otf(self):
        """Test high NA vs low NA."""
        zernieke = {
            (2, -2): 0.072,  # Astigmatism Oblique
        }
        aberration_phase = pupil_functions.compute_pupil_plane_aberrations(zernieke, Nrho=129, Nphi=256)
        aberration_function = np.exp(1j * 2 * np.pi * aberration_phase)
    
        t_start = time.time()
        OTF_fast = compute_3d_otf_fast(
            q_grid=q_grid3d,
            NA=n * np.sin(alpha),
            z_values=z,
            nsample=n,
            nmedium=n,
            pupil_function=aberration_function,         
            high_NA=True,
            Nphi=101,
            Nrho=101,
            device='gpu',
            use_complex64=True,
            minimum_grid_point_number=512,
        )
        t_end = time.time()
        print(f"fast OTF route: compute_3d_otf_fast took {t_end - t_start:.4f} s")

        # Reconstruct PSF from fast OTF for a rough comparison
        I_fast = np.fft.ifftn(OTF_fast).real
        plt.imshow(np.log1p(10**4 * I_fast[:,:,Nz//2]), origin='lower')

    def test_scalar_fast_otf_vs_psf_route(self):
        """Compare 3D OTF from fast CTF route vs PSF-based route."""

        # --- scalar PSF + OTF via direct 3D PSF computation ---
        t_start = time.time()
        E_scalar = compute_3d_psf_coherent(
            grid2d=x_grid2d,
            z_values=z,
            NA=n * np.sin(alpha),
            nmedium=n,
            nsample=n,
            high_NA=True,
            pupil_function=None,
            Nphi=101,
            Nrho=101,
            device='gpu',
            safety=0.9,
        )
        t_end = time.time()
        print(f"scalar PSF route: compute_3d_psf_coherent took {t_end - t_start:.4f} s")

        I_psf, OTF_psf = compute_incoherent_psf_and_otf(E_scalar)

        # --- fast 3D OTF via CTF autocorrelation on Ewald sphere ---
        t_start = time.time()
        OTF_fast = compute_3d_otf_fast(
            q_grid=q_grid3d,
            NA=n * np.sin(alpha),
            z_values=z,
            nsample=n,
            nmedium=n,
            pupil_function=None,          # or your Zernike-based callable
            high_NA=True,
            Nphi=101,
            Nrho=101,
            device='gpu',
            use_complex64=True,
            minimum_grid_point_number=32,
        )
        t_end = time.time()
        print(f"fast OTF route: compute_3d_otf_fast took {t_end - t_start:.4f} s")

        # Reconstruct PSF from fast OTF for a rough comparison
        I_fast = np.fft.ifftn(OTF_fast).real

        # --- slices ---
        zc  = Nz // 2          # central z index
        kzc = Nz // 2          # central kz index
        yc  = Nl // 2          # central y index
        kyc = Nl // 2          # central ky index

        # central PSF slice (x–y at z center)
        psf_slice_std  = I_psf[:, :, zc]
        psf_slice_fast = I_fast[:, :, zc]

        # OTF slices: kx–ky at kz center
        otf_xy_std  = OTF_psf[:, :, kzc].real
        otf_xy_fast = OTF_fast[:, :, kzc].real

        # OTF slices: kx–kz at ky center
        otf_xz_std  = OTF_psf[:, kyc, :].real
        otf_xz_fast = OTF_fast[:, kyc, :].real

        # --- 2D PSF/OTF field plots ---
        plot_psf_otf_plot(
            np.log1p(10**4 * psf_slice_std),
            np.log1p(10**4 * otf_xy_std),
            "PSF/OTF (PSF route, central slices)",
        )
        plot_psf_otf_plot(
            np.log1p(10**4 * psf_slice_fast),
            np.log1p(10**4 * otf_xy_fast),
            "PSF/OTF (fast OTF route, central slices)",
        )

        # --- 1D line-cut comparisons ---
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].set_title("PSF x cut (y=0, z=center)")
        axes[0, 0].plot(
            np.log1p(10**4 * psf_slice_std[:, yc]),
            label="PSF route",
        )
        axes[0, 0].plot(
            np.log1p(10**4 * psf_slice_fast[:, yc]),
            label="fast OTF route",
        )
        axes[0, 0].legend()

        axes[0, 1].set_title("PSF z cut (x=0, y=center)")
        axes[0, 1].plot(
            I_psf[Nl // 2, Nl // 2, :],
            label="PSF route",
        )
        axes[0, 1].plot(
            I_fast[Nl // 2, Nl // 2, :],
            label="fast OTF route",
        )
        axes[0, 1].legend()

        axes[1, 0].set_title("OTF fx cut (fy=0, fz=center)")
        axes[1, 0].plot(
            OTF_psf[:, yc, kzc].real,
            label="PSF route",
        )
        axes[1, 0].plot(
            OTF_fast[:, yc, kzc].real,
            label="fast OTF route",
        )
        axes[1, 0].legend()

        axes[1, 1].set_title("OTF fz cut (fx=0, fy=center)")
        axes[1, 1].plot(
            OTF_psf[Nl // 2, yc, :].real,
            label="PSF route",
        )
        axes[1, 1].plot(
            OTF_fast[Nl // 2, yc, :].real,
            label="fast OTF route",
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
    plt.show()
