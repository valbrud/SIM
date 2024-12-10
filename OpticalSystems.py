"""
OpticalSystems.py

This module contains classes for simulating and analyzing optical systems.

Note: More reasonable interface for accessing and calculating of the PSF and OTF is expected in the future.
For this reason the detailed documentation on the computations is not provided yet.
"""

import numpy as np
import scipy as sp
import scipy.interpolate
from numpy import ndarray

import wrappers
import matplotlib.pyplot as plt
import Windowing
from abc import abstractmethod

from Illumination import Illumination
from VectorOperations import VectorOperations


class OpticalSystem:
    """
    Base class for optical systems, providing common functionality.

    Attributes:
        supported_interpolation_methods (list): List of supported interpolation methods.
        psf (np.ndarray): Point Spread Function.
        otf (np.ndarray): Optical Transfer Function.
        interpolator (scipy.interpolate.RegularGridInterpolator): Interpolator for OTF.
        _otf_frequencies (np.ndarray): Frequencies for OTF.
        _psf_coordinates (np.ndarray): Coordinates for PSF.
        _interpolation_method (str): Interpolation method.
    """

    #: A list of currently supported interpolation methods.
    #: Other scipy interpolation methods are not directly supported due to high memory usage.
    #: Add them to the list if needed.
    #: Currently, meaningless, but changes are expected.
    #: Fourier is interpolation is used for SIM by default.
    #: Linear interpolation is available with the "interpolate_OTF" method if needed.
    supported_interpolation_methods = ["linear", "Fourier"]

    def __init__(self, interpolation_method: str):
        self.psf = None
        self.otf = None
        self.interpolator = None
        self._otf_frequencies = None
        self._psf_coordinates = None
        self._interpolation_method = None
        self.interpolation_method = interpolation_method

    @property
    def interpolation_method(self):
        return self._interpolation_method

    @interpolation_method.setter
    def interpolation_method(self, new_method: str):
        if new_method not in OpticalSystem.supported_interpolation_methods:
            raise AttributeError("Interpolation method ", new_method, " is not supported")
        self._interpolation_method = new_method

    @property
    def psf_coordinates(self):
        return self._psf_coordinates

    @psf_coordinates.setter
    @abstractmethod
    def psf_coordinates(self, new_coordinates):
        ...

    @property
    def otf_frequencies(self):
        return self._otf_frequencies

    @abstractmethod
    def compute_psf_and_otf_cordinates(self, psf_size: tuple[int], N: int):
        """
        Compute the PSF and OTF coordinate axes.

        Args:
            psf_size (tuple): Size of the PSF.
            N (int): Number of points.
        """
        pass

    @abstractmethod
    def compute_psf_and_otf(self) -> tuple[ndarray[tuple[int, int, int], np.float64],
                                           ndarray[tuple[int, int, int], np.float64]]:
        """
        Compute the PSF and OTF.
        """
        pass

    @abstractmethod
    def compute_q_grid(self) -> ndarray[tuple[int, int, int, 3], np.float64]:
        """
        Compute the q-grid for the OTF.

        Returns:
            np.ndarray: Computed q-grid.
        """
        pass

    @abstractmethod
    def compute_x_grid(self) -> ndarray[tuple[int, int, int, 3], np.float64]:
        """
        Compute the x-grid for the PSF.

        Returns:
            np.ndarray: Computed x-grid.
        """
        pass

    def _prepare_interpolator(self):
        """
        Prepare the interpolator based on OTF values and axes.

        Raises:
            AttributeError: If OTF or axes are not computed yet.
        """
        if self.otf_frequencies is None or self.otf is None:
            raise AttributeError("OTF or axes are not computed yet. This method can not be called at this stage")
        axes = 2 * np.pi * self.otf_frequencies
        otf = self.otf
        self.interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method=self._interpolation_method,
                                                                      bounds_error=False,
                                                                      fill_value=0.)

    def interpolate_otf(self, k_shift: ndarray[3, np.float64]) -> ndarray[tuple[int, int, int], np.float64]:
        """
        Interpolate the OTF with a given shift.

        Args:
            k_shift (np.ndarray): Shift vector for interpolation.

        Returns:
            np.ndarray: Interpolated OTF.
        """
        pass


class OpticalSystem2D(OpticalSystem):

    def __init__(self, interpolation_method):
        super().__init__(interpolation_method)

    def compute_psf_and_otf_cordinates(self, psf_size: tuple[float], N: int):
        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N)
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N)

        self.psf_coordinates = np.array((x, y))

    @OpticalSystem.psf_coordinates.setter
    def psf_coordinates(self, new_coordinates):
        self._psf_coordinates = new_coordinates
        x, y = new_coordinates
        Nx, Ny = x.size, y.size
        Lx, Ly = 2 * new_coordinates[:, -1]

        fx = np.linspace(-Nx / (2 * Lx), Nx / (2 * Lx), Nx)
        fy = np.linspace(-Ny / (2 * Ly), Ny / (2 * Ly), Ny)
        self._otf_frequencies = np.array((fx, fy))

    def compute_q_grid(self) -> ndarray[tuple[int, int, 2], np.float64]:
        if self.otf_frequencies is None:
            raise AttributeError("Major bug, OTF frequencies are missing")
        q_axes = 2 * np.pi * self.otf_frequencies
        qx, qy = np.array(q_axes[0]), np.array(q_axes[1])
        q_vectors = np.array(np.meshgrid(qx, qy)).T.reshape(-1, 2)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, 2)
        return q_grid

    def compute_x_grid(self) -> ndarray[tuple[int, int, 2], np.float64]:
        if self.psf_coordinates is None:
            raise AttributeError("PSF coordinates are missing")
        x_axes = self.psf_coordinates
        x, y = np.array(x_axes[0]), np.array(x_axes[1])
        x_vectors = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        x_sorted = x_vectors[np.lexsort((x_vectors[:, -1], x_vectors[:, 0]))]
        x_grid = x_sorted.reshape(x.size, y.size, 2)
        return x_grid

    def compute_effective_otfs_2dSIM(self, illumination: Illumination) -> dict[tuple[int, tuple[int]], np.float64]:
        """
        Compute the effective OTFs for 2D SIM illumination
        (in the case of 2D SIM they are just shifted).

        Args:
            illumination: Illumination object containing wave information.

        Returns:
            dict: Effective OTFs.
        """
        waves = illumination.waves
        effective_psfs = {}
        effective_otfs = {}
        if len(illumination.indices2d) != len(illumination.indices3d):
            raise AttributeError("2D optical system assums 2D illumination!")
        X, Y = np.meshgrid(*self.psf_coordinates)
        for r in range(illumination.Mr):
            wavevectors, indices = illumination.get_wavevectors_projected(r)
            for wavevector, index in zip(wavevectors, indices):
                effective_psf = 0
                amplitude = waves[(*index, 0)].amplitude
                phase_shifted = np.exp(1j * np.einsum('ijk,i ->jk', np.array((X, Y)), wavevector)) * self.psf
                effective_psf += amplitude * phase_shifted
                effective_psfs[(r, index)] = effective_psf
                effective_otfs[(r, index)] = wrappers.wrapped_fftn(effective_psf)
                # effective_otfs[(r, xy_indices)] = np.transpose(wrappers.wrapped_fftn(effective_psf), axes=(1, 0, 2))
                # plt.imshow(np.abs((effective_otfs[(r, xy_indices)][:, :, 25])))
                # plt.show()
        return effective_otfs

    def interpolate_otf(self, k_shift: ndarray[3, np.float64]) -> ndarray[tuple[int, int, int], np.float64]:
        if self.interpolation_method == "Fourier":
            raise AttributeError("Due to the major code refactoring, Fourier interpolation is temporarily not available")

        else:
            qx = (2 * np.pi * self.otf_frequencies[0] - k_shift[0])
            qy = (2 * np.pi * self.otf_frequencies[1] - k_shift[1])
            interpolation_points = np.array(np.meshgrid(qx, qy)).T.reshape(-1, 2)
            interpolation_points = interpolation_points[
                np.lexsort((interpolation_points[:, 1],
                            interpolation_points[:, 0]))]

            if self.interpolator is None:
                raise AttributeError("Interpolator does not exist. Compute OTF to prepare the interpolator")

            otf_interpolated = self.interpolator(interpolation_points)
            otf_interpolated = otf_interpolated.reshape(self.otf_frequencies[0].size, self.otf_frequencies[1].size)
            return otf_interpolated

    def compute_psf_and_otf(self) -> tuple[np.float64, np.float64]:

        ...


class OpticalSystem3D(OpticalSystem):

    def __init__(self, interpolation_method):
        super().__init__(interpolation_method)

    def compute_psf_and_otf_cordinates(self, psf_size, N):
        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N)
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N)
        z = np.linspace(-psf_size[2] / 2, psf_size[2] / 2, N)

        self.psf_coordinates = np.array((x, y, z))

    @OpticalSystem.psf_coordinates.setter
    def psf_coordinates(self, new_coordinates):
        self._psf_coordinates = new_coordinates
        x, y, z, = new_coordinates
        Nx, Ny, Nz = x.size, y.size, z.size
        Lx, Ly, Lz = 2 * new_coordinates[:, -1]

        fx = np.linspace(-Nx / (2 * Lx), Nx / (2 * Lx), Nx)
        fy = np.linspace(-Ny / (2 * Ly), Ny / (2 * Ly), Ny)
        fz = np.linspace(-Nz / (2 * Lz), Nz / (2 * Lz), Nz)
        self._otf_frequencies = np.array((fx, fy, fz))

    def compute_q_grid(self) -> ndarray[tuple[int, int, int, int], np.float64]:
        if self.otf_frequencies is None:
            raise AttributeError("Major bug, OTF frequencies are missing")
        q_axes = 2 * np.pi * self.otf_frequencies
        qx, qy, qz = np.array(q_axes[0]), np.array(q_axes[1]), np.array(q_axes[2])
        q_vectors = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -2], q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, qz.size, 3)
        return q_grid

    def compute_x_grid(self) -> ndarray[tuple[int, int, int, int], np.float64]:
        if self.psf_coordinates is None:
            raise AttributeError("PSF coordinates are missing")
        x_axes = self.psf_coordinates
        x, y, z = np.array(x_axes[0]), np.array(x_axes[1]), np.array(x_axes[2])
        x_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        x_sorted = x_vectors[np.lexsort((x_vectors[:, -2], x_vectors[:, -1], x_vectors[:, 0]))]
        x_grid = x_sorted.reshape(x.size, y.size, z.size, 3)
        return x_grid

    def compute_effective_otfs_projective_3dSIM(self, illumination: Illumination) -> dict[tuple[int, tuple[int]], np.float64]:
        """
        Compute the effective OTFs for projective 3D SIM illumination.

        Args:
            illumination: Illumination object containing wave information.

        Returns:
            dict: Effective OTFs.
        """
        waves = illumination.waves
        effective_psfs = {}
        effective_otfs = {}
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
                    phase_shifted = np.transpose(np.exp(1j * np.einsum('ijkl,i ->jkl', np.array((X, Y, Z)), wavevector)), axes=(1, 0, 2)) * self.psf
                    effective_psf += amplitude * phase_shifted
                effective_psfs[(r, xy_indices)] = effective_psf
                effective_otfs[(r, xy_indices)] = wrappers.wrapped_fftn(effective_psf)
                # plt.gca().set_title(f'{xy_indices}')
                # plt.imshow(np.abs(np.log(1 + 10**8 * effective_otfs[(r, xy_indices)][:, :, 50])))
                # plt.show()
                # effective_otfs[(r, xy_indices)] = np.transpose(wrappers.wrapped_fftn(effective_psf), axes=(1, 0, 2))
                # plt.imshow(np.abs((effective_otfs[(r, xy_indices)][:, :, 25])))
                # plt.show()
        return effective_otfs

    def compute_effective_otfs_true_3dSIM(self, illumination: Illumination) -> dict[tuple[int, tuple[int]], np.float64]:
        """
        Compute the effective OTFs for true 3D SIM
        (in the case of true 3D SIM, they are just shifted).

        Args:
            illumination: Illumination object containing wave information.

        Returns:
            tuple: Effective PSFs and OTFs.
        """
        waves = illumination.waves
        effective_psfs = {}
        effective_otfs = {}
        X, Y, Z = np.meshgrid(*self.psf_coordinates)
        for r in range(illumination.Mr):
            wavevectors, indices = illumination.get_wavevectors(r)
            for index in indices:
                amplitude = waves[indices].amplitude
                phase_shifted = np.exp(1j * np.transpose(np.einsum('ijkl,i ->jkl', np.array((X, Y, Z)), wavevectors[index])), axes=(1, 0, 2)) * self.psf
                effective_psf = amplitude * phase_shifted
                effective_psfs[(r, index)] = effective_psf
                effective_otfs[(r, index)] = wrappers.wrapped_fftn(effective_psf)

        return effective_otfs

    def interpolate_otf(self, k_shift: ndarray[3, np.float64]) -> np.float64:
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

    def compute_psf_and_otf(self) -> tuple[np.float64, np.float64]:

        ...


#
class System4f2D(OpticalSystem2D):
    def __init__(self, alpha=np.pi / 4, refractive_index=1, interpolation_method="linear"):
        super().__init__(interpolation_method)
        self.n = refractive_index
        self.alpha = alpha
        self.NA = self.n * np.sin(self.alpha)

    def _PSF(self, c_vectors, mask=None):
        r = (c_vectors[:, :, 0] ** 2 + c_vectors[:, :, 1] ** 2) ** 0.5
        v = 2 * np.pi * r * self.NA
        I = (2 * scipy.special.j1(v) / v) ** 2
        cx, cy = np.array(I.shape) // 2
        I[cx, cy] = 1
        I /= np.sum(I)
        return I

    def _PSF_from_pupil_function(self, pupil_function, mask=None):
        E = wrappers.wrapped_fftn(pupil_function)
        I = np.abs(E) ** 2
        I /= np.sum(I)
        return I

    def _mask_OTF(self):
        ...

    def compute_psf_and_otf(self, parameters=None, pupil_function=None, mask=None):
        if self.psf_coordinates is None and parameters is None and pupil_function is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        elif parameters is not None:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)
        x, y = self.psf_coordinates
        N = x.size
        c_vectors = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 1], c_vectors[:, 0]))]
        grid = c_vectors_sorted.reshape((len(x), len(y), 2))
        if pupil_function is None:
            psf = self._PSF(grid, mask=None)
        else:
            psf = self._PSF_from_pupil_function(pupil_function)
        self.psf = psf / np.sum(psf)
        self.otf = np.abs(wrappers.wrapped_ifftn(self.psf)).astype(complex)
        self.otf /= np.amax(self.otf)
        self._prepare_interpolator()
        return self.psf, self.otf


class System4f3D(OpticalSystem3D):
    def __init__(self, alpha=np.pi / 4, refractive_index_sample=1, refractive_index_medium=1, regularization_parameter=0.01, interpolation_method="linear"):
        super().__init__(interpolation_method)
        self.ns = refractive_index_sample
        self.nm = refractive_index_medium
        self.alpha = alpha
        self.e = regularization_parameter / (4 * np.sin(self.alpha / 2) ** 2)
        self.NA = self.nm * np.sin(self.alpha)

    def _PSF(self, c_vectors, high_NA=False, apodization_function="Sine", pupil_function=lambda rho: 1, mask=None):
        r = (c_vectors[:, :, :, 0] ** 2 + c_vectors[:, :, :, 1] ** 2) ** 0.5
        z = c_vectors[:, :, :, 2]
        v = 2 * np.pi * r * self.NA
        if self.NA <= self.ns:
            u = 4 * np.pi * z * (self.ns - np.sqrt(self.ns ** 2 - self.NA ** 2))
        else:
            u = 4 * np.pi * z * self.ns * (1 - np.cos(self.alpha))

        if not high_NA:
            def integrand(rho):
                return pupil_function(rho) * np.exp(- 1j * (u[:, :, :, None] / 2 * rho ** 2)) * sp.special.j0(
                    rho * v[:, :, :, None]) * 2 * np.pi * rho

            rho = np.linspace(0, 1, 100)
            integrands = integrand(rho)
            h = sp.integrate.simpson(integrands, x=rho)
            I = (h * h.conjugate()).real

        else:
            def integrand(theta):
                return apodization_function(theta) * np.sin(theta) * np.exp(1j * u[:, :, :, None] *
                                                                            np.sin(theta / 2) ** 2 / (2 * np.sin(self.alpha / 2) ** 2)
                                                                            ) * sp.special.j0(v[:, :, :, None] * np.sin(theta) / np.sin(self.alpha))

            if apodization_function == "Sine":
                def apodization_function(theta):
                    test = pupil_function(np.sin(theta))
                    return pupil_function(np.sin(theta)) * (np.cos(theta)) ** 0.5

            theta = np.linspace(0, self.alpha, 100)
            integrands = integrand(theta)
            h = sp.integrate.simpson(integrands, x=theta)
            I = (h * h.conjugate()).real

        if mask:
            shape = np.array(c_vectors.shape[:-1])
            m = mask(shape, np.amin(shape) // 5)
            I *= m

        return I

    # Could not get good numbers yet
    def _mask_OTF(self):
        ...

    def compute_psf_and_otf(self, parameters=None, high_NA=False, apodization_function="Sine", pupil_function=lambda rho: 1, mask=None):
        if self.psf_coordinates is None and parameters is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)
        x, y, z = self.psf_coordinates
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        grid = c_vectors_sorted.reshape((len(x), len(y), len(z), 3))
        psf = self._PSF(grid, high_NA, apodization_function="Sine", pupil_function=pupil_function, mask=mask)
        self.psf = psf / np.sum(psf)
        self.otf = np.abs(wrappers.wrapped_ifftn(self.psf)).astype(complex)
        self.otf /= np.amax(self.otf)
        self._prepare_interpolator()
        return self.psf, self.otf
