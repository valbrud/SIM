"""
OpticalSystems.py

This module contains classes for simulating and analyzing optical systems.

Note: More reasonable interface for accessing and calculating of the PSF and OTF is expected in the future.
For this reason the detailed documentation on the computations is not provided yet.
"""
from joblib import Parallel, delayed
import numpy as np
import scipy as sp
import scipy.interpolate
from numpy import ndarray
from hcipy import zernike, SeparatedCoords, PolarGrid
from math import factorial

import wrappers
import matplotlib.pyplot as plt
import windowing
from abc import abstractmethod

from Illumination import Illumination
from VectorOperations import VectorOperations
from Dimensions import DimensionMetaAbstract


class OpticalSystem(metaclass=DimensionMetaAbstract):
    """
    Base class for optical systems, providing common functionality.
    The base class implements all the functionality but cannot be implemented.
    Use dimensional children classes instead.

    Attributes:
        supported_interpolation_methods (list): List of supported interpolation methods.
        psf (np.ndarray): Point Spread Function.
        otf (np.ndarray): Optical Transfer Function.
        interpolator (scipy.interpolate.RegularGridInterpolator): Interpolator for OTF.
        _otf_frequencies (np.ndarray): Frequencies for OTF.
        _psf_coordinates (np.ndarray): Coordinates for PSF.
        _interpolation_method (str): Interpolation method.
    """

    dimensionality = None 

    #: A list of currently supported interpolation methods.
    #: Other scipy interpolation methods are not directly supported due to high memory usage.
    #: Add them to the list if needed.
    #: Currently, meaningless, but changes are expected.
    #: Fourier is interpolation is used for SIM by default.
    #: Linear interpolation is available with the "interpolate_OTF" method if needed.
    supported_interpolation_methods = ["linear", "Fourier"]

    def __init__(self, interpolation_method: str, normalize_otf = 'True'):
        self.psf = None
        self.otf = None
        self._x_grid = None
        self._q_grid = None
        self.interpolator = None
        self._otf_frequencies = None
        self._psf_coordinates = None
        self._interpolation_method = None
        self.interpolation_method = interpolation_method
        self.normalize_otf = True


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

    # To avoid unnecessary memory usage, the grid is stored only if needed.
    @property
    def x_grid(self):
        if self._x_grid is None:
            self._compute_x_grid()
        return self._x_grid

    @property
    def q_grid(self):
        if self._q_grid is None:
            self._compute_q_grid()
        return self._q_grid

    @abstractmethod
    def compute_psf_and_otf_cordinates(self, psf_size: tuple[int], N: int, account_for_pixel_size: bool = False) -> None:
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

    def _compute_q_grid(self) -> ndarray[tuple[int], np.float64]:
        """
        Compute the q-grid for the OTF.

        Returns:
            np.ndarray: Computed q-grid.
        """
        self._q_grid= np.stack(np.meshgrid(*self.otf_frequencies, indexing='ij'), axis=-1)

    def _compute_x_grid(self) -> ndarray[tuple[int], np.float64]:
        """
        Compute the x-grid for the PSF.

        Returns:
            np.ndarray: Computed x-grid.
        """
        self._x_grid=np.stack(np.meshgrid(*self.psf_coordinates, indexing='ij'), axis=-1)
    
    def _account_for_pixel_size(self):
        x_grid = self.x_grid
        dx = x_grid[*([1] * self.dimensionality)] - x_grid[*([0]*self.dimensionality)]
        q_grid = self.q_grid
        q_grid_flat = q_grid.reshape(-1, self.dimensionality)
        otf_pixel = np.prod(np.sinc(q_grid_flat * dx[None, :]), axis=1)
        self.otf = self.otf * otf_pixel.reshape(*self.otf.shape)

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

    @staticmethod
    def radial_zernike(n, m, r):
        """
        Compute the radial part R_{n,|m|}(r) of the Zernike polynomial
        for each r in the 1D array `r`.
        """
        m_abs = abs(m)
        R = np.zeros_like(r, dtype=float)

        # The sum goes up to floor((n - m_abs)/2)
        upper_k = (n - m_abs) // 2
        for k in range(upper_k + 1):
            c = ((-1) ** k
                 * factorial(n - k)
                 / (factorial(k)
                    * factorial((n + m_abs) // 2 - k)
                    * factorial((n - m_abs) // 2 - k)))
            R += c * r ** (n - 2 * k)
        R *= np.sqrt(2 * (n + 1) / (1 + (m == 0)))
        return R

    @staticmethod
    def azimuthal_zernike(m, phi):
        """
        Compute the azimuthal part for Zernike polynomial Z_n^m.
        - if m >= 0: cos(m * phi)
        - if m <  0: sin(|m| * phi)

        `phi` is a 1D array of angles.
        """
        if m >= 0:
            return np.cos(m * phi)
        else:
            return np.sin(abs(m) * phi)

    @staticmethod
    def compute_pupil_plane_abberations(zernieke_polynomials, rho, phi):
        """
        Construct a 2D pupil-plane aberration by summing Zernike modes using HCIPy's zernike().

        Parameters
        ----------
        zernieke_polynomials : dict
            Dictionary with keys = (n, m) and values = amplitudes.
            Example: {(2, 2): 0.1, (3, 1): -0.05, (4, -2): 0.07, ...}
        rho : ndarray
            1D array of radial coordinates (0 <= rho <= 1 typically).
        phi : ndarray
            1D array of azimuthal coordinates (in radians, e.g. -π to +π or 0 to 2π).

        Returns
        -------
        aberration : ndarray
            The resulting 2D aberration (same shape as rho, phi).
        """
        RHO, PHI = np.meshgrid(rho, phi, indexing='ij')
        # grid = PolarGrid(SeparatedCoords((rho, phi)))
        aberration = np.zeros((rho.size, phi.size))

        for (n, m), amplitude in zernieke_polynomials.items():
            # aberration += amplitude * zernike(n, m, grid=grid)
            aberration += amplitude * OpticalSystem.radial_zernike(n, m, RHO) * OpticalSystem.azimuthal_zernike(m, PHI)
        return aberration


class OpticalSystem2D(OpticalSystem):

    dimensionality = 2

    def __init__(self, interpolation_method, normalize_otf=True):
        super().__init__(interpolation_method, normalize_otf)

    def compute_psf_and_otf_cordinates(self, psf_size: tuple[float], N: int):
        if type(N) is int:
            N = (N, N)
        elif type(N) is tuple:
            if len(N) != 2:
                raise AttributeError("N should be integer or a tuple of two integers")
            
        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N[0])
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N[1])

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

    def compute_psf_and_otf(self, account_for_pixel_size: bool = False) -> tuple[np.float64, np.float64]:

        ...


class OpticalSystem3D(OpticalSystem):

    dimensionality = 3

    def __init__(self, interpolation_method, normalize_otf = True):
        super().__init__(interpolation_method, normalize_otf)

    def compute_psf_and_otf_cordinates(self, psf_size, N):
        if type(N) is int:
            N = (N, N, N)
        elif type(N) is tuple:
            if len(N) != 3:
                raise AttributeError("N should be integer or a tuple of three integers")

        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N[0])
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N[1])
        z = np.linspace(-psf_size[2] / 2, psf_size[2] / 2, N[2])

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

    def interpolate_otf(self, k_shift: ndarray[3, np.float64]) -> np.ndarray[tuple[int, int, int], np.float64]:
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

    def compute_psf_and_otf(self, account_for_pixel_size: bool = False) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                                           np.ndarray[tuple[int, int, int], np.float64]]: ...


class System4f2DCoherent(OpticalSystem2D): 
    def __init__(self,
                 alpha=np.pi / 4,
                 refractive_index=1,
                 interpolation_method="linear", 
                 normalize_otf = False):
        
        super().__init__(interpolation_method, normalize_otf)
        self.n = refractive_index
        self.alpha = alpha
        self.NA = self.n * np.sin(self.alpha)

    def _PSF(self, grid):
        r = (grid[:, :, 0] ** 2 + grid[:, :, 1] ** 2) ** 0.5
        v = 2 * np.pi * r * self.NA
        E = 2 * scipy.special.j1(v) / v
        cx, cy = grid.shape[0] // 2, grid.shape[1] // 2
        E[cx, cy] = 1
        return E

    def _PSF_from_pupil_function(self, pupil_function):
        E = wrappers.wrapped_ifftn(pupil_function)
        return E

    def compute_psf_and_otf(self, parameters=None, pupil_function =None, account_for_pixel_size: bool = False)\
            -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                     np.ndarray[tuple[int, int, int], np.float64]]:
        if self.psf_coordinates is None and parameters is None and pupil_function is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        elif parameters is not None:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)

        grid = np.stack(np.meshgrid(*self.psf_coordinates), axis=-1)

        if pupil_function is None:
            psf = self._PSF(grid)
        else:
            psf = self._PSF_from_pupil_function(pupil_function)

        self.otf = np.abs(wrappers.wrapped_fftn(self.psf)).astype(complex)
        self.otf = self.otf / np.amax(self.otf) if self.normalize_otf else self.otf
        self._prepare_interpolator()
        return self.psf, self.otf


class System4f3DCoherent(OpticalSystem3D):
    def __init__(self,
                 alpha=np.pi / 4, 
                 refractive_index_sample=1, 
                 refractive_index_medium=1, 
                 interpolation_method="linear", 
                 normalize_otf = False):
        
        super().__init__(interpolation_method, normalize_otf)
        self.ns = refractive_index_sample
        self.nm = refractive_index_medium
        self.alpha = alpha
        self.NA = self.nm * np.sin(self.alpha)

    def _integrand_no_aberrations(self, rho, phi, i, u, v, psy, pupil_function):
        """
        Compute the integrand for slice index i.
        Note: here u, v, and psy are arrays such that:
          - u and v have shape (nx, ny, nz)
          - psy has shape (nx, ny) (computed from grid)
        """
        # Apodization part (rho is 1D, so broadcasting adds new axes)
        apodization_part = pupil_function(rho) * rho / ((1 - rho ** 2 * np.sin(self.alpha) ** 2) ** 0.25)
        # u-dependent part: use u[:,:,i] (resulting shape (nx, ny)) and add integration axis
        u_dependent_part = np.exp(1j * (u[:, :, i, None] / 2 *
                                        ((1 - np.sqrt(1 - rho ** 2 * np.sin(self.alpha) ** 2)) / (1 - np.cos(self.alpha)))))
        # v-dependent part: use v[:,:,i] and psy (both with shape (nx, ny))
        v_dependent_part = np.exp(-1j * v[:, :, i, None, None] * rho[None, None, :, None] *
                                  np.cos(phi[None, None, None, :] - psy[:, :, None, None]))
        # The integrand shape becomes (nx, ny, len(rho), len(phi))
        return apodization_part[None, None, :, None] * u_dependent_part[:, :, :, None] * v_dependent_part

    def _process_slice(self, i, u, v, rho, phi, phase_change, psy, pupil_function):
        """
        Process a single slice (index i) of u, integrating first over phi and then over rho.
        """
        integrands = self._integrand_no_aberrations(rho, phi, i, u, v, psy, pupil_function)
        # Multiply by the aberration phase change (which has shape (len(rho), len(phi)))
        integrands_aberrated = integrands * phase_change[None, None, :, :]
        # Integrate over phi (axis=3)
        integrated_phi = scipy.integrate.simpson(integrands_aberrated, x=phi, axis=3)
        # Then integrate over rho (axis=2)
        integrated_rho = scipy.integrate.simpson(integrated_phi, x=rho, axis=2)
        return integrated_rho  # shape: (nx, ny)

    def _process_chunk(self, indices, u, v, rho, phi, phase_change, psy, pupil_function):
        # Process a small chunk (list) of slices.
        results = [self._process_slice(i, u, v, rho, phi, phase_change, psy, pupil_function)
                   for i in indices]
        # Stack the results along the third axis.
        return np.stack(results, axis=2)

    def _compute_h_parallel(self, u, rho, phi, phase_change, psy, pupil_function, n_jobs=3, chunksize=5):
        """
        Parallel version: split the integration over the z-dimension into chunks.
        u, v, and psy are computed in _PSF, and u.shape is (nx, ny, nz).
        """
        nz = u.shape[2]
        indices = np.arange(nz)
        chunks = [indices[i:i + chunksize] for i in range(0, nz, chunksize)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_chunk)(chunk, u, self.v, rho, phi, phase_change, psy, pupil_function)
            for chunk in chunks)
        # Concatenate results along the z-dimension.
        return np.concatenate(results, axis=2)

    # --- End parallelization helpers ---
    def _PSF(self, grid,
             high_NA=False,
             pupil_function=lambda rho: 1,
             integrate_rho=True,
             zernieke={},
             **kwargs):
        r = (grid[:, :, :, 0] ** 2 + grid[:, :, :, 1] ** 2) ** 0.5
        z = grid[:, :, :, 2]
        v = 2 * np.pi * r * self.NA
        self.v = v  # store for use in the parallel helper
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
            E = sp.integrate.simpson(integrands, x=rho)

        else:
            # Here Abbe sine condition is assumed to be satisfied, and the P(theta) apodization function is assumed to be sqrt(cos(theta))
            # Integration in theta is much more stable numerically at the values of alpha approaching 90 degrees. However, it's not
            # as convenient to work with aberrations, that are given in the pupil plane and are more precisely computed in terms of rho.
            # For these reasons, both implementations are provided.
            if not integrate_rho:
                def integrand(theta):
                    return (pupil_function(np.sin(theta) / np.sin(self.alpha)) * (np.cos(theta)) ** 0.5 * np.sin(theta)
                            * np.exp(1j * u[:, :, :, None] / 2 * (np.sin(theta / 2) ** 2 / np.sin(self.alpha / 2) ** 2))
                            * sp.special.j0(v[:, :, :, None] * np.sin(theta) / np.sin(self.alpha)))

                theta = np.linspace(0, self.alpha - 1e-9, 100)
                integrands = integrand(theta)
                E = sp.integrate.simpson(integrands, x=theta)

            else:
                # Taking final value slightly less than 1 to avoid division by zero in the integrand
                if not zernieke:
                    rho = np.linspace(0, 1 - 1e-9, 100)
                    def integrand(rho):
                        return (pupil_function(rho) * rho / (1 - rho**2 * np.sin(self.alpha)**2)**0.25
                            * np.exp(1j * (u[:, :, :, None] / 2 * ((1 - np.sqrt(1 - rho ** 2 * np.sin(self.alpha) ** 2)) / (1 - np.cos(self.alpha)))))
                            * sp.special.j0(v[:, :, :, None] * rho)
                            )
                    integrands = integrand(rho)
                    E = sp.integrate.simpson(integrands, x=rho)

                else:
                    rho = np.linspace(0, 1 - 1e-9, 100)
                    vx, vy = 2 * np.pi * grid[:, :, :, 0], 2 * np.pi * grid[:, :, :, 1]
                    psy = np.arctan2(vy, vx)[:, :, 0]
                    dphi = 2 * np.pi / 100
                    phi = np.arange(0, 2 * np.pi, dphi)
                    aberration_function = OpticalSystem.compute_pupil_plane_abberations(zernieke, rho, phi)
                    # plt.plot(aberration_function[50, :])
                    # plt.show()
                    phase_change = np.exp(1j * 2 * np.pi * self.nm * aberration_function)
                    E = np.zeros(u.shape, dtype=np.complex128)
                    def integrand_no_aberrations(rho, phi, i):
                        apodization_part = pupil_function(rho) * rho / (1 - rho ** 2 * np.sin(self.alpha) ** 2) ** 0.25
                        u_dependent_part = np.exp(1j * (u[:, :, i, None] / 2 * ((1 - np.sqrt(1 - rho ** 2 * np.sin(self.alpha) ** 2)) / (1 - np.cos(self.alpha)))))
                        v_dependent_part = np.exp(-1j * v[:, :, i, None, None] * rho[None, None, :, None] * np.cos(phi[None, None, None, :] - psy[:, :, None, None]))
                        return apodization_part[None, None, :, None] * u_dependent_part[:, :, :, None] * v_dependent_part

                    # for i in range(u.shape[2]):
                    #     integrands = integrand_no_aberrations(rho, phi, i)
                    #     integrands_aberrated = integrands * phase_change[None, None, :, :]
                    #     integrated_phi = sp.integrate.simpson(integrands_aberrated, axis=3, x=phi)
                    #     integrated_rho = sp.integrate.simpson(integrated_phi, axis=2, x=rho)
                    #     E[:, :, i] = integrated_rho

                    # Replace your for-loop with a parallelized version:
                    E = np.stack(Parallel(n_jobs=2)(
                        delayed(lambda i: sp.integrate.simpson(
                            np.sum(integrand_no_aberrations(rho, phi, i) * phase_change[None, None, :, :], axis=3) * dphi,
                        x = rho, axis=2))(i) for i in range(u.shape[2])
                    ), axis=2)

        return E

    def compute_psf_and_otf(self, parameters=None,
                            high_NA=False,
                            pupil_function=lambda rho: 1,
                            account_for_pixel_size: bool = False,
                            integrate_rho=False,
                            zernieke={}) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
    np.ndarray[tuple[int, int, int], np.float64]]:
        if self.psf_coordinates is None and parameters is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)

        grid = np.stack(np.meshgrid(*self.psf_coordinates), axis=-1)
        psf = self._PSF(grid, high_NA, 
                        pupil_function=pupil_function, 
                        integrate_rho=integrate_rho,
                        zernieke=zernieke)
        
        self.otf = wrappers.wrapped_fftn(self.psf)
        if self.normalize_otf:
            self.otf /= np.amax(self.otf)
            self.psf = psf / np.sum(psf)

        self._prepare_interpolator()
        return self.psf, self.otf


class System4f2D(System4f2DCoherent):
    def __init__(self,
                 alpha=np.pi / 4,
                 refractive_index=1,
                 interpolation_method="linear", 
                 normalize_otf=True):
        
        super().__init__(alpha,
                 refractive_index,
                 interpolation_method, 
                 normalize_otf)

    def _PSF(self, grid, save_pupil_function=False):
        E = super()._PSF(grid)
        if save_pupil_function:
            pupil_function = np.where(self.q_grid[:, :, 0] ** 2 + self.q_grid[:, :, 1] ** 2 < self.NA ** 2, 1, 0)
            self.__dict__['pupil_function'] = pupil_function
        I = np.abs(E) ** 2
        I /= np.sum(I)
        return I

    def _PSF_from_pupil_function(self, pupil_function, save_pupil_function=False):
        if save_pupil_function:
            self.__dict__['pupil_function'] = pupil_function
    
        E = wrappers.wrapped_fftn(pupil_function)
        I = np.abs(E) ** 2
        I /= np.sum(I)
        return I

    def compute_psf_and_otf(self,
                            parameters=None,
                            pupil_function =None, 
                            save_pupil_function=False, 
                            account_for_pixel_size: bool = False)\
            -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                     np.ndarray[tuple[int, int, int], np.float64]]:
        
        if self.psf_coordinates is None and parameters is None and pupil_function is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        elif parameters is not None:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)

        grid = np.stack(np.meshgrid(*self.psf_coordinates), axis=-1)

        if pupil_function is None:
            psf = self._PSF(grid, save_pupil_function=save_pupil_function)

        else:
            psf = self._PSF_from_pupil_function(pupil_function, save_pupil_function=save_pupil_function)
                    
        self.psf = psf / np.sum(psf)
        self.otf = np.abs(wrappers.wrapped_fftn(self.psf)).astype(complex)
        self.otf /= np.amax(self.otf)
        if account_for_pixel_size:
            self._account_for_pixel_size()
        self._prepare_interpolator()

        return self.psf, self.otf


class System4f3D(System4f3DCoherent):
    def __init__(self,
                 alpha=np.pi / 4, 
                 refractive_index_sample=1, 
                 refractive_index_medium=1, 
                 interpolation_method="linear", 
                 normalize_otf = True):
        
        super().__init__(alpha, 
                 refractive_index_sample, 
                 refractive_index_medium, 
                 interpolation_method, 
                 normalize_otf)



    def _PSF(self, grid,
             high_NA=False,
             pupil_function=lambda rho: 1,
             integrate_rho=True,
             zernieke={},
             save_pupil_function=False,
             **kwargs):
        
        E = super()._PSF(grid,
                        high_NA,
                        pupil_function,
                        integrate_rho,
                        zernieke,
                        **kwargs)
        
        if save_pupil_function:
            self.__dict__['pupil_function'] = wrappers.wrapped_fftn(E)
        
        I = (E * E.conjugate()).real
        return I

    def compute_psf_and_otf(self, parameters=None,
                            high_NA=False,
                            pupil_function=lambda rho: 1,
                            account_for_pixel_size: bool = False,
                            integrate_rho=False,
                            zernieke={}, 
                            save_pupil_function=False,
                            ) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                                       np.ndarray[tuple[int, int, int], np.float64]]:
        if self.psf_coordinates is None and parameters is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)

        grid = np.stack(np.meshgrid(*self.psf_coordinates), axis=-1)
        psf = self._PSF(grid, high_NA, 
                        pupil_function=pupil_function, 
                        integrate_rho=integrate_rho,
                        zernieke=zernieke,
                        save_pupil_function=save_pupil_function)
        self.psf = psf / np.sum(psf)
        self.otf = wrappers.wrapped_fftn(self.psf)
        self.otf /= np.amax(self.otf)
        if account_for_pixel_size:
            self._account_for_pixel_size()
        self._prepare_interpolator()
        return self.psf, self.otf


class PointScanningImagingSystem(OpticalSystem):
    """
    Base class for a point-scanning microscopy.
    """

    def __init__(self,
                 optical_system_excitation,
                 optical_system_detection, 
                 aperture = None, 
                 interpolation_method = "linear",
                 normalize_otf = True):
        
        if not isinstance(optical_system_excitation, OpticalSystem):
            raise TypeError("optical_system_excitation must be an instance of OpticalSystem")
        if not isinstance(optical_system_detection, OpticalSystem):
            raise TypeError("optical_system_detection must be an instance of OpticalSystem")
        if not optical_system_excitation.dimensionality == optical_system_detection.dimensionality:
            raise ValueError("Excitation and detection systems must have the same dimensionality")
        
        super().__init__(interpolation_method, normalize_otf)

        #The case here must be handled if the systems have different psf coordinates.
        #For now it is assmed that they are the same.
        if not np.array_equal(optical_system_excitation.psf_coordinates, optical_system_detection.psf_coordinates):
            raise ValueError("Excitation and detection systems must have the same psf coordinates")
        
        self.optical_system_excitation = optical_system_excitation
        self.optical_system_detection = optical_system_detection



class Confocal(PointScanningImagingSystem):
    """
    Class for simulating a confocal microscope system.
    """

    def compute_psf_and_otf(self):
        """
        Compute the PSF and OTF for the confocal system.
        """
        if not self.optical_system_excitation.psf:
            psf_excitation, otf_excitation = self.optical_system_excitation.compute_psf_and_otf()
        if not self.optical_system_detection.psf:
            psf_detection, otf_detection = self.optical_system_detection.compute_psf_and_otf()

        
        # Compute final PSF and OTF including aperture
        self.psf = psf_excitation * psf_detection
        self.otf = wrappers.wrapped_fftn(self.psf)
        if self.normalize_otf:
            self.otf /= np.amax(self.otf)
        self.psf /= np.sum(self.psf)


class RCM(PointScanningImagingSystem):
    ...

class ISM(PointScanningImagingSystem):
    ...


class Confocal2D(Confocal): ...
class Confocal3D(Confocal): ... 

class RCM2D(RCM): ...
class RCM3D(RCM): ...

class ISM2D(ISM): ...
class ISM3D(ISM): ...