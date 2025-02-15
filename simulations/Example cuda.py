import math, cmath
import numpy as np
from numba import cuda, complex128, float64, int32


# Device helper: compute complex exponential exp(i * arg)
@cuda.jit(device=True)
def cexp(arg):
    # return exp(i*arg) as a complex number
    return complex(math.cos(arg), math.sin(arg))


# CUDA kernel that computes h[x,y,i] by performing 2D Simpson integration over (rho, phi)
# Each thread handles one (x, y, i) coordinate.
@cuda.jit
def compute_h_kernel(u, v, phase_change, psy,
                     pupil, rho, phi,
                     alpha, dphi, drho,
                     h):
    # Get the thread indices corresponding to spatial x, y and integration index i.
    ix = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    iy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    iz = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    nx = u.shape[0]
    ny = u.shape[1]
    ni = u.shape[2]
    nrho = rho.shape[0]
    nphi = phi.shape[0]

    if ix < nx and iy < ny and iz < ni:
        # local values for current (x,y,i)
        u_val = u[ix, iy, iz]
        v_val = v[ix, iy, iz]
        psy_val = psy[ix, iy]

        s_alpha = math.sin(alpha)
        c_alpha = math.cos(alpha)

        # We'll store the φ-integrated value at each rho sample in a local array.
        # (We assume that nrho is not larger than 100.)
        f_r = cuda.local.array(100, complex128)

        # Loop over rho (outer integration)
        for ir in range(nrho):
            rho_val = rho[ir]
            # Precompute the apodization factor:
            denom = 1.0 - rho_val * rho_val * s_alpha * s_alpha
            # avoid division by zero
            if denom < 1e-12:
                denom = 1e-12
            A = pupil[ir] * rho_val / (denom ** 0.25)

            # u-dependent part: note that the factor ((1 - sqrt(denom))/(1-c_alpha))
            # should be safe if alpha != 0.
            if (1.0 - c_alpha) != 0:
                arg1 = (u_val * 0.5) * ((1.0 - math.sqrt(denom)) / (1.0 - c_alpha))
            else:
                arg1 = 0.0
            u_term = cexp(arg1)  # exp(i*arg1)

            # Now integrate over phi using Simpson’s rule.
            sum_phi = 0.0 + 0.0j
            for ip in range(nphi):
                phi_val = phi[ip]
                # v-dependent part: exp(-i * v * rho * cos(phi - psy))
                arg2 = v_val * rho_val * math.cos(phi_val - psy_val)
                v_term = cexp(-arg2)
                # Multiply by precomputed phase change at (rho,phi)
                pc = phase_change[ir, ip]
                # Compute integrand at (ir, ip)
                integrand = A * u_term * v_term * pc

                # Simpson weight for phi:
                if ip == 0 or ip == nphi - 1:
                    weight = 1.0
                elif ip % 2 == 1:
                    weight = 4.0
                else:
                    weight = 2.0
                sum_phi += integrand * weight

            # Simpson integration over phi: factor dphi/3.
            f_r[ir] = (dphi / 3.0) * sum_phi

        # Now integrate over rho using Simpson’s rule.
        sum_r = 0.0 + 0.0j
        for ir in range(nrho):
            if ir == 0 or ir == nrho - 1:
                weight = 1.0
            elif ir % 2 == 1:
                weight = 4.0
            else:
                weight = 2.0
            sum_r += f_r[ir] * weight
        result = (drho / 3.0) * sum_r

        h[ix, iy, iz] = result


# Host function to set up data and launch the CUDA kernel.
def compute_h_cuda(u, v, phase_change, psy, pupil, rho, phi, alpha):
    # Convert inputs to float64/complex128 as needed.
    # Here u, v, phase_change, psy, pupil, rho, phi are assumed to be numpy arrays.
    # Also compute integration steps.
    dphi = 2 * np.pi / (phi.size)
    drho = (rho[-1] - rho[0]) / (rho.size - 1)

    # Transfer arrays to the device.
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    d_phase_change = cuda.to_device(phase_change)
    d_psy = cuda.to_device(psy)
    d_pupil = cuda.to_device(pupil)
    d_rho = cuda.to_device(rho)
    d_phi = cuda.to_device(phi)

    # Allocate output array on the device.
    h_shape = u.shape  # (nx, ny, n_i)
    d_h = cuda.device_array(h_shape, dtype=np.complex128)

    # Configure grid and block dimensions.
    nx, ny, ni = u.shape
    threadsperblock = (8, 8, 1)
    blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (ni + threadsperblock[2] - 1) // threadsperblock[2]
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Launch the kernel.
    compute_h_kernel[blockspergrid, threadsperblock](d_u, d_v, d_phase_change, d_psy,
                                                     d_pupil, d_rho, d_phi,
                                                     alpha, dphi, drho,
                                                     d_h)
    # Copy the result back to the host.
    h = d_h.copy_to_host()
    return h


# =============================================================================
# Example usage:
# (You must define or load your own arrays for u, v, c_vectors, etc.)
# =============================================================================
if __name__ == '__main__':
    alpha = 2 * np.pi / 5
    nmedium = 1.5
    nobject = 1.5
    NA = nmedium * np.sin(alpha)
    theta = np.asin(0.9 * np.sin(alpha))
    fz_max_diff = nmedium * (1 - np.cos(alpha))
    dx = 1 / (8 * NA)
    dy = dx
    dz = 1 / (4 * fz_max_diff)
    N = 101
    max_r = N // 2 * dx
    max_z = N // 2 * dz
    psf_size = 2 * np.array((max_r, max_r, max_z))
    dV = dx * dy * dz
    x = np.linspace(-max_r, max_r, N)
    y = np.copy(x)
    z = np.linspace(-max_z, max_z, N)

    # Sample input shapes (adjust to your actual shapes)
    nx, ny, ni = 128, 128, 10
    # u and v: for each pixel and each mode; here we use dummy data.
    u = np.random.rand(nx, ny, ni).astype(np.float64)
    v = np.random.rand(nx, ny, ni).astype(np.float64)

    # c_vectors: shape (nx, ny, something, 2); we only use [:,:,:,0] and [:,:,:,1]
    # For demonstration, create dummy c_vectors.
    c_vectors = np.random.rand(nx, ny, 1, 2).astype(np.float64)
    vx = 2 * np.pi * c_vectors[:, :, :, 0]
    vy = 2 * np.pi * c_vectors[:, :, :, 1]
    psy = np.arctan2(vy, vx)[:, :, 0]  # shape (nx, ny)

    # rho and phi arrays for integration:
    nrho = 70
    nphi = 70
    # make sure rho does not hit 1 exactly:
    rho = np.linspace(0, 1 - 1e-9, nrho).astype(np.float64)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False).astype(np.float64)

    # Compute the pupil-plane aberrations and phase_change.
    # (Assume OpticalSystem.compute_pupil_plane_abberations returns an array of shape (nrho, nphi))
    # For demonstration, we use dummy aberration function:
    aberration_function = np.random.rand(nrho, nphi).astype(np.float64)
    nm = 1.0  # some constant
    phase_change = np.exp(1j * 2 * np.pi * nm * aberration_function).astype(np.complex128)

    # Assume pupil_function is given.
    # For demonstration, let pupil_function(rho) be ones.
    pupil = np.ones_like(rho).astype(np.float64)

    # Assume alpha is a scalar (in radians)
    alpha = 0.5

    # Compute h on the GPU.
    h = compute_h_cuda(u, v, phase_change, psy, pupil, rho, phi, alpha)

    # Finally, compute intensity I = |h|^2 (on the host)
    I = np.abs(h * np.conjugate(h)).real

    # (For debugging/visualization you could display I, e.g., plt.imshow(I[:,:,0]))
    print("Computed intensity shape:", I.shape)