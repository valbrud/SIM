import numpy
import numpy as np
import scipy as sp
import scipy.interpolate
import wrappers


class Lens:
    def __init__(self, alpha=0.3, regularization_parameter=0.01):
        self.alpha = alpha
        self.e = regularization_parameter / (4 * np.sin(self.alpha / 2) ** 2)

        self.psf = np.zeros((1, 1, 1))
        self.otf = np.zeros((1, 1, 1))
        self.otf_frequencies = None
        self.psf_coordinates = None
        self.shifted_otfs = {}
        self.wvdiff_otfs = {}

    def PSF(self, c_vectors):
        r = (c_vectors[:, :, :, 0] ** 2 + c_vectors[:, :, :, 1] ** 2) ** 0.5
        z = c_vectors[:, :, :, 2]
        v = 2 * np.pi * r * np.sin(self.alpha)
        u = 2 * np.pi * z * np.sin(self.alpha / 2) ** 2

        def integrand(rho):
            return np.exp(- 1j * (u[:, :, :, None] / 2 * rho ** 2)) * sp.special.j0(
                rho * v[:, :, :, None]) * 2 * np.pi * rho

        rho = np.linspace(0, 1, 100)
        integrands = integrand(rho)
        h = sp.integrate.simpson(integrands, rho)
        I = (h * h.conjugate()).real
        return I

    # Could not get good numbers yet
    def regularized_analytic_OTF(self, f_vector):
        f_r = (f_vector[0] ** 2 + f_vector[1] ** 2) ** 0.5
        f_z = f_vector[2]
        l = f_r / np.sin(self.alpha)
        s = f_z / (4 * np.sin(self.alpha / 2) ** 2)
        if l > 2:
            return 0

        def p_max(theta):
            D = 4 - l ** 2 * (1 - np.cos(theta) ** 2)
            return (-l * np.cos(theta) + D ** 0.5) / 2

        def integrand(p, theta):
            denum = self.e ** 2 + (abs(s) - p * l * np.cos(theta)) ** 2
            return 8 * self.e * p / denum

        otf, _ = sp.integrate.dblquad(integrand, 0, np.pi / 2, lambda x: 0, p_max)
        return (otf)

    def compute_PSF_and_OTF(self, psf_size, N):

        dx = psf_size[0] / N
        dy = psf_size[1] / N
        dz = psf_size[2] / N

        x = np.arange(-psf_size[0] / 2, psf_size[0] / 2, dx)
        y = np.arange(-psf_size[1] / 2, psf_size[1] / 2, dy)
        z = np.arange(-psf_size[2] / 2, psf_size[2] / 2, dz)

        self.psf_coordinates = np.array((x, y, z))

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / psf_size[0], N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / psf_size[1], N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / psf_size[2], N)

        self.otf_frequencies = np.array((fx, fy, fz))

        psf = np.zeros((N, N, N))
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        grid = c_vectors_sorted.reshape((len(x), len(y), len(z), 3))
        psf = self.PSF(grid)
        self.psf = psf / np.sum(psf[:, :, int(N / 2)])
        self.otf = np.abs(wrappers.wrapped_ifftn(self.psf))

    def compute_wvdiff_otfs(self, wv_group1, wv_group2=None):
        if not wv_group2:
            wv_group2 = np.copy(wv_group1)

        x, y, z = self.psf_coordinates[0], self.psf_coordinates[1], self.psf_coordinates[2]
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        for wv1 in wv_group1:
            for wv2 in wv_group2:
                wvdiff = wv2 - wv1
                if not tuple(wvdiff) in self.wvdiff_otfs.keys():
                    phases = np.dot(c_vectors_sorted, wvdiff)
                    phases = phases.reshape((len(x), len(y), len(z)))
                    psf_phase_shifted = self.psf * np.exp(1j * phases)
                    self.wvdiff_otfs[tuple(wvdiff)] = np.sum(psf_phase_shifted) / (len(x) * len(y) * len(z))
        return self.wvdiff_otfs

    def compute_shifted_otf(self, wavevectors):
        x, y, z = self.psf_coordinates[0], self.psf_coordinates[1], self.psf_coordinates[2]
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        for wavevector in wavevectors:
            if np.sum(np.abs(wavevector)) == 0:
                self.shifted_otfs[tuple(wavevector)] = self.otf
            else:
                phases = np.dot(c_vectors_sorted, wavevector)
                phases = phases.reshape((len(x), len(y), len(z)))
                psf_phase_shifted = self.psf * np.exp(1j * phases)
                self.shifted_otfs[tuple(wavevector)] = np.abs(wrappers.wrapped_ifftn(psf_phase_shifted))
        return self.shifted_otfs

    def interpolate_otf(self, k_shift):
        axes = 2 * np.pi * self.otf_frequencies
        otf = self.otf
        interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method="linear", bounds_error=False,
                                                                 fill_value=0.)
        qx = (2 * np.pi * self.otf_frequencies[0] - k_shift[0])
        qy = (2 * np.pi * self.otf_frequencies[1] - k_shift[1])
        qz = (2 * np.pi * self.otf_frequencies[2] - k_shift[2])
        interpolation_points = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        interpolation_points = interpolation_points[np.lexsort((interpolation_points[:, 2], interpolation_points[:, 1],
                                                                interpolation_points[:, 0]))]
        otf_interpolated = interpolator(interpolation_points)
        otf_interpolated = otf_interpolated.reshape(self.otf_frequencies[0].size, self.otf_frequencies[1].size,
                                                    self.otf_frequencies[2].size)
        return otf_interpolated

    def interpolate_otf_at_one_point(self, q):
        axes = 2 * np.pi * self.otf_frequencies
        otf = self.otf
        interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method="linear", bounds_error=False,
                                                                 fill_value=0.)
        otf_interpolated = interpolator(q)
        return otf_interpolated


class Illumination:
    def __init__(self, intensity_plane_waves, spacial_shifts):
        self.spacial_shifts = spacial_shifts
        self.M_t = len(self.spacial_shifts)
        self.waves = intensity_plane_waves


class NoiseEstimator:
    def __init__(self, illumination, optical_system):
        self.illumination = illumination
        self.optical_system = optical_system

    def Dj(self, q_grid, method="Fourier"):
        d_j = 0
        for m in range(len(self.illumination.waves)):
            a_m = self.illumination.waves[m].amplitude
            k_m = self.illumination.waves[m].wavevector
            if method == "Fourier":
                d_j += np.abs(a_m) ** 2 * np.abs(self.optical_system.shifted_otfs[tuple(k_m)]) ** 2
            elif method == "Scipy":
                d_j += np.abs(a_m) ** 2 * np.abs(self.optical_system.interpolate_otf(k_m)) ** 2

        d_j *= self.illumination.M_t
        return d_j

    def Vj(self, q_grid, method="Fourier"):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m1 in range(len(self.illumination.waves)):
            for m2 in range(len(self.illumination.waves)):
                a_m1 = self.illumination.waves[m1].amplitude
                a_m2 = self.illumination.waves[m2].amplitude
                a_m12 = self.illumination.waves[m1 - m2].amplitude
                k_m1 = self.illumination.waves[m1].wavevector
                k_m2 = self.illumination.waves[m2].wavevector
                if method == "Fourier":
                    otf1 = self.optical_system.shifted_otfs[tuple(k_m1)]
                    otf2 = self.optical_system.shifted_otfs[tuple(k_m2)]
                    otf3 = self.optical_system.wvdiff_otfs[tuple(k_m2 - k_m1)]
                elif method == "Scipy":
                    otf1 = self.optical_system.interpolate_otf(k_m1)
                    otf2 = self.optical_system.interpolate_otf(k_m2)
                    otf3 = self.optical_system.interpolate_otf_at_one_point(k_m2 - k_m1)
                else:
                    raise ValueError("The interpolation method is unknown")

                v_j += a_m1 * a_m2.conjugate() * a_m12 * otf1.conjugate() * otf2 * otf3
        v_j *= self.illumination.M_t
        return v_j

    def SSNR(self, q_axes, method="Fourier"):
        if method == "Fourier":
            if len(self.optical_system.shifted_otfs) == 0 or len(self.optical_system.wvdiff_otfs) == 0:
                raise AttributeError("Shifted OTFs required for this method are not calculated. \
                    Use OpticalSystem.compute_shifted_otfs and OpticalSystem.compute_wvdiff_otfs to compute them")

        qx, qy, qz = np.array(q_axes[0]), np.array(q_axes[1]), np.array(q_axes[2])
        q_vectors = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -2], q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, qz.size, 3)
        dj = self.Dj(q_grid, method)
        ssnr = np.zeros(dj.shape, dtype=np.complex128)
        print(ssnr.shape)
        vj = self.Vj(q_grid, method)
        mask = (vj != 0) * (dj != 0)
        numpy.putmask(ssnr, mask, np.abs(dj) ** 2 / vj)
        return ssnr
