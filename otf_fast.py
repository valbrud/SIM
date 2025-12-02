import numpy as np
import hpc_utils


# ---------- backend helpers ----------

def _get_backend(device):
    """
    Pick CPU/GPU backend using your existing hpc_utils.
    Returns (mode, xp, to_xp, to_host).
    """
    hpc_utils.pick_backend(device)
    mode, xp = hpc_utils.get_backend()  # mode in {'cpu','gpu'}

    if mode == "gpu":
        to_xp   = xp.asarray
        to_host = lambda a: a.get()
    else:
        to_xp   = xp.asarray
        to_host = lambda a: a

    return mode, xp, to_xp, to_host


# ---------- grid densification (CPU) ----------

def _compute_upsample_factors(shape, minimum_grid_point_number):
    """
    For each spatial dimension, if N < minimum_grid_point_number,
    choose the smallest integer factor f such that N*f >= minimum_grid_point_number.
    Otherwise f = 1.
    """
    factors = []
    for N in shape:
        if N >= minimum_grid_point_number:
            factors.append(1)
        else:
            f = int(np.ceil(float(minimum_grid_point_number) / float(N)))
            factors.append(max(1, f))
    return tuple(factors)


def _make_dense_q_grid_cpu(q_grid, factors, float_type):
    """
    Given q_grid of shape (Nx,Ny,Nz,3) (NumPy, CPU) and integer factors (fx,fy,fz),
    build q_grid_dense (NumPy, CPU) spanning the same min/max in each q-direction.
    """
    if q_grid.ndim != 4 or q_grid.shape[-1] != 3:
        raise ValueError("q_grid must have shape (Nx, Ny, Nz, 3).")

    Nx, Ny, Nz, _ = q_grid.shape
    fx, fy, fz = factors

    if fx == fy == fz == 1:
        return q_grid.astype(float_type, copy=False)

    qx_1d = q_grid[:, 0, 0, 0].astype(float_type, copy=False)
    qy_1d = q_grid[0, :, 0, 1].astype(float_type, copy=False)
    qz_1d = q_grid[0, 0, :, 2].astype(float_type, copy=False)

    Nx_dense = Nx * fx
    Ny_dense = Ny * fy
    Nz_dense = Nz * fz

    qx_dense = np.linspace(qx_1d[0], qx_1d[-1], Nx_dense, dtype=float_type)
    qy_dense = np.linspace(qy_1d[0], qy_1d[-1], Ny_dense, dtype=float_type)
    qz_dense = np.linspace(qz_1d[0], qz_1d[-1], Nz_dense, dtype=float_type)

    QX, QY, QZ = np.meshgrid(qx_dense, qy_dense, qz_dense, indexing="ij")
    q_grid_dense = np.stack((QX, QY, QZ), axis=-1).astype(float_type, copy=False)
    return q_grid_dense


# ---------- sphere mask on CPU ----------

def _binary_sphere_shell_mask_cpu(q_grid_dense):
    """
    Build a binary shell mask on CPU:
      - inside = |q| <= 1
      - eroded = binary_erosion(inside)
      - shell  = inside & ~eroded
    Uses scipy.ndimage on CPU. If not available, falls back to 'inside' only.
    """
    qx = q_grid_dense[..., 0]
    qy = q_grid_dense[..., 1]
    qz = q_grid_dense[..., 2]

    r = np.sqrt(qx * qx + qy * qy + qz * qz)
    inside = (r <= 1.0)

    try:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(inside)
        shell = inside & ~eroded
    except Exception:
        shell = inside

    return shell


# ---------- mapping from sphere → pupil coordinates ----------

def _sphere_points_to_pupil(
    qx, qy, qz, NA, nsample, xp, float_type, eps=1e-7
):
    """
    Map non-zero direction vectors (qx,qy,qz) in the OBJECT medium
    to pupil-plane polar coordinates (rho, phi) using Abbe's sine condition:

        rho = sin(theta_medium) / sin(alpha),

    where theta_medium is the polar angle in the object medium and
          sin(alpha) = NA / nsample  (semi-opening angle of the lens in that medium).

    q = (qx,qy,qz) is assumed proportional to the direction cosines in the object medium.
    """
    # Normalize direction to unit vector
    norm = xp.sqrt(qx * qx + qy * qy + qz * qz) + float_type(eps)
    nx = qx / norm
    ny = qy / norm
    nz = qz / norm

    # cos(theta) = nz; sin(theta) from 1 - cos^2
    nz_clipped = xp.clip(nz, float_type(-1.0), float_type(1.0))
    cos_theta = nz_clipped
    sin_theta = xp.sqrt(xp.clip(float_type(1.0) - cos_theta * cos_theta,
                                float_type(0.0), float_type(1.0)))

    # Abbe mapping: rho = sin(theta) / sin(alpha)
    sin_alpha = float_type(NA) / float_type(nsample)
    # If NA > nsample, clip to avoid nonsense
    if sin_alpha > 1.0:
        sin_alpha = float_type(1.0)

    rho = sin_theta / sin_alpha

    # azimuth
    phi = xp.arctan2(ny, nx)

    return rho.astype(float_type, copy=False), phi.astype(float_type, copy=False)


# ---------- pupil evaluation at scattered points ----------

def _evaluate_pupil_at_points(pupil_function, rho, phi, xp, complex_type):
    """
    Evaluate pupil/aberration at scattered pupil points (rho, phi).

    - If pupil_function is None: return ones (no aberration).
    - If pupil_function is callable: assumed to accept (rho, phi) on the
      current backend (xp) and return a complex array of the same shape.
    """
    if pupil_function is None:
        return xp.ones_like(rho, dtype=complex_type)

    if callable(pupil_function):
        P = pupil_function(rho, phi)
        P = xp.asarray(P)
        if P.shape != rho.shape:
            raise ValueError(
                "Callable pupil_function must return array of same shape as rho."
            )
        return P.astype(complex_type, copy=False)

    raise ValueError(
        "In compute_3d_otf_fast, pupil_function must be None or callable "
        "(for Zernike-based pupils etc.)."
    )


# ---------- main function: 3D OTF (scalar) ----------

def compute_3d_otf_fast(
    q_grid,
    NA,
    z_values,
    nsample=1.0,
    nmedium=1.0,
    pupil_function=None,         # None | callable(rho,phi)
    high_NA=False,               # kept for API symmetry; not used in scalar CTF
    Nphi=256,                    # unused here
    Nrho=129,                    # unused here
    device="auto",               # 'auto' | 'cpu' | 'gpu'
    use_complex64=True,
    safety=0.70,                 # unused here
    minimum_grid_point_number=32,
):
    """
    Fast 3D scalar OTF computation.

    Parameters
    ----------
    q_grid : np.ndarray, shape (Nx,Ny,Nz,3)
        3D frequency grid (object medium), last axis are (qx,qy,qz).
        The OTF is returned on this grid (after internal upsampling + block-sum).
    NA, nsample :
        Used to define the pupil mapping rho = sin(theta_medium)/sin(alpha),
        where sin(alpha) = NA / nsample.
    pupil_function : None or callable
        If callable, must be P(rho,phi) and return complex pupil field
        (e.g. from Zernike expansion) at those polar coordinates.
        If None, a unity pupil (no aberration) is assumed.
    minimum_grid_point_number : int
        If an axis has fewer points than this, it is upsampled by an integer
        factor so that N * factor >= minimum_grid_point_number.

    Returns
    -------
    OTF : np.ndarray, complex
        3D OTF on the original q_grid shape (Nx,Ny,Nz).
    """
    # --- prepare CPU arrays for grid + mask ---
    float_type_np = np.float32 if use_complex64 else np.float64
    q_grid_np = np.asarray(q_grid, dtype=float_type_np)

    if q_grid_np.ndim != 4 or q_grid_np.shape[-1] != 3:
        raise ValueError("q_grid must have shape (Nx, Ny, Nz, 3).")

    Nx, Ny, Nz, _ = q_grid_np.shape

    # 0) upsample grid if needed (CPU)
    factors = _compute_upsample_factors((Nx, Ny, Nz), minimum_grid_point_number)
    fx, fy, fz = factors

    q_grid_dense_cpu = _make_dense_q_grid_cpu(q_grid_np, factors, float_type_np)
    Nx_dense, Ny_dense, Nz_dense, _ = q_grid_dense_cpu.shape

    # 1) sphere shell mask on CPU
    shell_mask_cpu = _binary_sphere_shell_mask_cpu(q_grid_dense_cpu)

    # --- now move to selected backend for heavy computations ---
    mode, xp, to_xp, to_host = _get_backend(device)

    float_type   = xp.float32 if use_complex64 else xp.float64
    complex_type = xp.complex64 if use_complex64 else xp.complex128

    q_grid_dense = to_xp(q_grid_dense_cpu).astype(float_type, copy=False)
    shell_mask   = to_xp(shell_mask_cpu)

    # indices of non-zero shell points
    idx = xp.where(shell_mask)
    if len(idx[0]) == 0:
        raise RuntimeError("Sphere shell mask is empty on the given q_grid.")

    qx_shell = q_grid_dense[..., 0][idx]
    qy_shell = q_grid_dense[..., 1][idx]
    qz_shell = q_grid_dense[..., 2][idx]

    # 2) map shell points to pupil-plane (rho,phi)
    rho_shell, phi_shell = _sphere_points_to_pupil(
        qx_shell, qy_shell, qz_shell, NA, nsample, xp, float_type
    )

    # limit to inside the pupil
    valid = rho_shell <= float_type(1.0)
    if not xp.all(valid):
        rho_shell = rho_shell[valid]
        phi_shell = phi_shell[valid]

        shell_mask_valid = xp.zeros_like(shell_mask, dtype=bool)
        shell_mask_valid_idx = tuple(i[valid] for i in idx)
        shell_mask_valid[shell_mask_valid_idx] = True
        shell_mask = shell_mask_valid
        idx = xp.where(shell_mask)

    # 3) evaluate pupil (Zernike etc.) at those pupil-plane points
    P_shell = _evaluate_pupil_at_points(
        pupil_function, rho_shell, phi_shell, xp, complex_type
    )

    # build CTF on dense grid
    CTF_dense = xp.zeros((Nx_dense, Ny_dense, Nz_dense), dtype=complex_type)
    CTF_dense[idx] = P_shell

    # 4) OTF as convolution (autocorrelation) of CTF:
    #    OTF = IFFT( |FFT(CTF)|^2 )
    Fk = xp.fft.fftn(CTF_dense)
    power_spectrum = xp.abs(Fk) ** 2
    OTF_dense = xp.fft.ifftn(power_spectrum)
    OTF_dense = OTF_dense.astype(complex_type, copy=False)

    # 5) downsample back to original grid by summing blocks of size (fx,fy,fz)
    otf = OTF_dense
    Nx_d, Ny_d, Nz_d = Nx_dense, Ny_dense, Nz_dense

    if fx > 1:
        otf = otf.reshape(Nx, fx, Ny_d, Nz_d).sum(axis=1)
        Nx_d = Nx
    if fy > 1:
        otf = otf.reshape(Nx_d, Ny, fy, Nz_d).sum(axis=2)
        Ny_d = Ny
    if fz > 1:
        otf = otf.reshape(Nx_d, Ny_d, Nz, fz).sum(axis=3)
        Nz_d = Nz

    assert otf.shape == (Nx, Ny, Nz)

    return to_host(otf)
