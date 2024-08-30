import numpy as np
import scipy as sp
import scipy.interpolate
import wrappers
import matplotlib.pyplot as plt
import Windowing
from abc import abstractmethod
from VectorOperations import VectorOperations
class OpticalSystem:
    # Other scipy interpolation methods are not directly supported, because they require too much computation time.
    # Nevertheless, just add them to the list if needed
    supported_interpolation_methods = ["linear", "Fourier"]
    def __init__(self, interpolation_method):
        self.psf = None
        self.otf = None
        self.interpolator = None
        self._otf_frequencies = None
        self._psf_coordinates = None
        self._interpolation_method = None
        self.interpolation_method = interpolation_method
        self._shifted_otfs = {}
        self._wvdiff_otfs = {}

    def compute_psf_and_otf_cordinates(self, psf_size, N):
        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N)
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N)
        z = np.linspace(-psf_size[2] / 2, psf_size[2] / 2, N)

        self.psf_coordinates = np.array((x, y, z))

    @property
    def psf_coordinates(self):
        return self._psf_coordinates

    @psf_coordinates.setter
    def psf_coordinates(self, new_coordinates):
        self._psf_coordinates = new_coordinates
        x, y, z, = new_coordinates
        Nx, Ny, Nz = x.size, y.size, z.size
        Lx, Ly, Lz = 2 * new_coordinates[:, -1]

        fx = np.linspace(-Nx/(2 * Lx), Nx/(2 * Lx) , Nx)
        fy = np.linspace(-Ny/(2 * Ly), Ny/(2 * Ly) , Ny)
        fz = np.linspace(-Nz/(2 * Lz), Nz/(2 * Lz) , Nz)
        self._otf_frequencies = np.array((fx, fy, fz))

    @property
    def otf_frequencies(self):
        return self._otf_frequencies

    @property
    def interpolation_method(self):
        return self._interpolation_method

    @interpolation_method.setter
    def interpolation_method(self, new_method):
        if new_method not in OpticalSystem.supported_interpolation_methods:
            raise AttributeError("Interpolation method ", new_method, " is not supported")
        self._interpolation_method = new_method

    @abstractmethod
    def compute_psf_and_otf(self): ...

    def compute_q_grid(self):
        if self.otf_frequencies is None:
            raise AttributeError("Major bug, OTF frequencies are missing")
        q_axes = 2 * np.pi * self.otf_frequencies
        qx, qy, qz = np.array(q_axes[0]), np.array(q_axes[1]), np.array(q_axes[2])
        q_vectors = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -2], q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, qz.size, 3)
        return q_grid

    def compute_effective_otfs_projective_3dSIM(self, illumination):
        waves = illumination.waves
        effective_psfs={}
        effective_otfs={}
        X, Y, Z = np.meshgrid(*self.psf_coordinates)
        for r in range(illumination.Mr):
            angle = illumination.angles[r]
            indices = illumination.rearranged_indices
            for xy_indices in illumination.indices2d:
                effective_psf = 0
                for z_index in indices[xy_indices]:
                    wavevector = VectorOperations.rotate_vector3d(
                        waves[(*xy_indices, z_index)].wavevector, np.array((0, 0, 1)), angle)
                    amplitude = waves[(*xy_indices, z_index)].amplitude
                    phase_shifted = np.exp(1j * np.einsum('ijkl,i ->jkl', np.array((X, Y, Z)), wavevector)) * self.psf
                    effective_psf += amplitude * phase_shifted
                effective_psfs[(r, xy_indices)] = effective_psf
                effective_otfs[(r, xy_indices)] = wrappers.wrapped_fftn(effective_psf)
        return effective_otfs

    def compute_effective_otfs_true_3dSIM(self, illumination):
        waves = illumination.waves
        effective_psfs={}
        effective_otfs={}
        X, Y, Z = np.meshgrid(*self.psf_coordinates)
        for r in range(illumination.Mr):
            wavevectors, indices = illumination.get_wavevectors(r)
            for w in range(len(wavevectors)):
                amplitude = waves[indices[w]].amplitude
                phase_shifted = np.exp(1j * np.einsum('ijkl,i ->jkl', np.array((X, Y, Z)), wavevectors[w])) * self.psf
                effective_psf = amplitude * phase_shifted
                effective_psfs[(r, w)] = effective_psf
                effective_otfs[(r, w)] = wrappers.wrapped_fftn(effective_psf)

        return effective_psfs, effective_otfs


    def _prepare_interpolator(self):
        if self.otf_frequencies is None or self.otf is None:
            raise AttributeError("OTF or axes are not computed yet. This method can not be called at this stage")
        axes = 2 * np.pi * self.otf_frequencies
        otf = self.otf
        self.interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method=self._interpolation_method,
                                                                 bounds_error=False,
                                                                 fill_value=0.)

    def interpolate_otf(self, k_shift):
        if self.interpolation_method == "Fourier":
            raise AttributeError("Due to the major code refactoring, Fourier interpolation is temporarily not available")

        else:
            qx = (2 * np.pi * self.otf_frequencies[0] - k_shift[0])
            qy = (2 * np.pi * self.otf_frequencies[1] - k_shift[1])
            qz = (2 * np.pi * self.otf_frequencies[2] - k_shift[2])
            interpolation_points = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
            interpolation_points = interpolation_points[
                np.lexsort((interpolation_points[:, 2], interpolation_points[:, 1],
                            interpolation_points[:, 0]))]
            if self.interpolator is None:
                raise AttributeError("Interpolator does not exist. Compute OTF to prepare the interpolator")

            otf_interpolated = self.interpolator(interpolation_points)
            otf_interpolated = otf_interpolated.reshape(self.otf_frequencies[0].size, self.otf_frequencies[1].size,
                                                        self.otf_frequencies[2].size)
            return otf_interpolated


class Lens(OpticalSystem):
    def __init__(self, alpha=np.pi/4, refractive_index=1,  regularization_parameter=0.01, interpolation_method="linear"):
        super().__init__(interpolation_method)
        self.n = refractive_index
        self.alpha = alpha
        self.e = regularization_parameter / (4 * np.sin(self.alpha / 2) ** 2)

    def PSF(self, c_vectors, mask = None):
        r = (c_vectors[:, :, :, 0] ** 2 + c_vectors[:, :, :, 1] ** 2) ** 0.5
        z = c_vectors[:, :, :, 2]
        v = 2 * np.pi * r * self.n * np.sin(self.alpha)
        u = 8 * np.pi * z * self.n * np.sin(self.alpha / 2) ** 2

        def integrand(rho):
            return np.exp(- 1j * (u[:, :, :, None] / 2 * rho ** 2)) * sp.special.j0(
                rho * v[:, :, :, None]) * 2 * np.pi * rho

        rho = np.linspace(0, 1, 100)
        integrands = integrand(rho)
        h = sp.integrate.simpson(integrands, x=rho)
        I = (h * h.conjugate()).real
        if mask:
            shape = np.array(c_vectors.shape[:-1])
            m = mask(shape, np.amin(shape)//5)
            I *= m
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

    def _mask_OTF(self):
        ...
    def compute_psf_and_otf(self, parameters=None, apodization_filter=None):
        if not self.psf_coordinates and not parameters:
            raise AttributeError("Compute psf first or provide psf parameters")
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)
        self._wvdiff_otfs = {}
        self._shifted_otfs = {}
        x, y, z = self.psf_coordinates
        N = x.size
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        grid = c_vectors_sorted.reshape((len(x), len(y), len(z), 3))
        psf = self.PSF(grid, apodization_filter)
        self.psf = psf / np.sum(psf)
        self.otf = np.abs(wrappers.wrapped_ifftn(self.psf)).astype(complex)
        self.otf /= np.amax(self.otf)
        self._prepare_interpolator()

