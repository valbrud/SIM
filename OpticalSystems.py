"""
OpticalSystems.py

This module contains classes for simulating and analyzing optical systems.

Note: More reasonable interface for accessing and calculating of the PSF and OTF is expected in the future.
For this reason the detailed documentation on the computations is not provided yet.
"""
import utils
from joblib import Parallel, delayed
import numpy as np
import scipy as sp
import scipy.interpolate
from numpy import ndarray
from math import factorial

import hpc_utils
import matplotlib.pyplot as plt
import windowing
from abc import abstractmethod

from Illumination import Illumination
from VectorOperations import VectorOperations
from Dimensions import DimensionMetaAbstract

import pupil_functions
import psf_models

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

    # Dictionary of known pupil elements and their corresponding functions. Only calls the default versions of 
    # the functions. To use full functionality, compute the pupil function manually and provide as an array.  
    known_pupil_elements = {'vortex' : pupil_functions.make_vortex_pupil} 

    def __init__(self, interpolation_method: str, normalize_otf = 'True', computed_size: int = 0):
        self._psf = None
        self._otf = None
        self._x_grid = None
        self._q_grid = None
        self.interpolator = None
        self._otf_frequencies = None
        self._psf_coordinates = None
        self._interpolation_method = None
        self.interpolation_method = interpolation_method
        self.normalize_otf = True
        self._otf_pixel = None

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self, new_psf):
        self._psf = new_psf
        self._psf /= np.sum(self._psf)
        self._otf = hpc_utils.wrapped_fftn(new_psf)

    @property
    def otf(self):
        return self._otf

    @otf.setter
    def otf(self, new_otf):
        self._otf = new_otf
        self._psf = hpc_utils.wrapped_ifftn(new_otf)

    @property 
    def otf_pixel(self):
        return self._otf_pixel 

    @property
    def otf_with_pixel_size_correction(self):
        if self.otf_pixel is None:
            self.compute_pixel_correction()
        return self._otf * self._otf_pixel
    
    @property
    def psf_with_pixel_size_correction(self):
        corrected_psf = hpc_utils.wrapped_iffn(self.pixel_size_corrected_otf)
        return corrected_psf / np.sum(corrected_psf)
    
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
    def compute_psf_and_otf_coordinates(self, psf_size: tuple[int], N: int) -> None:
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
    
    def compute_pixel_correction(self):
        x_grid = self.x_grid
        dx = x_grid[*([1] * self.dimensionality)] - x_grid[*([0]*self.dimensionality)]
        q_grid = self.q_grid
        q_grid_flat = q_grid.reshape(-1, self.dimensionality)
        otf_pixel = np.prod(np.sinc(q_grid_flat * dx[None, :]), axis=1)
        otf_pixel = otf_pixel.reshape(np.array(q_grid.shape)[:-1])
        self._otf_pixel = otf_pixel
        return otf_pixel 
    
    def _prepare_interpolator(self):
        """
        Prepare the interpolator based on OTF values and axes.

        Raises:
            AttributeError: If OTF or axes are not computed yet.
        """
        if self.otf_frequencies is None or self.otf is None:
            raise AttributeError("OTF or axes are not computed yet. This method can not be called at this stage")
        axes = [2 * np.pi * np.array(self.otf_frequencies[ax]) for ax in range(self.dimensionality)]
        otf = self.otf
        self.interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method=self._interpolation_method,
                                                                      bounds_error=False,
                                                                      fill_value=0.)

    def upsample(self, factor=1, include_coordinates=True):
        """
        Upsample the PSF by zero-padding the OTF.
        Coordinates are changed accordingly.
        """
        if factor == 1:
            return
        if include_coordinates:
            self.psf_coordinates = tuple(np.linspace(coord[0], coord[-1], coord.size * factor - coord.size % 2) for coord in self.psf_coordinates)
        self.psf = utils.upsample(self.psf, factor)
        self._x_grid = None
        self._q_grid = None

    def interpolate_otf(self, k_shift: ndarray[3, np.float64]) -> ndarray[tuple[int, int, int], np.float64]:
        """
        Interpolate the OTF with a given shift.

        Args:
            k_shift (np.ndarray): Shift vector for interpolation.

        Returns:
            np.ndarray: Interpolated OTF.
        """
        pass
    
    def _normalize_psf__and_otf(self):
        if self.normalize_otf:
            self._psf /= np.sum(self.psf)
            self._otf /= np.max(np.abs(self.otf))

    def _get_pupil_function(self, Nrho: int, Nphi: int, pupil_element, pupil_function, zernieke={}) -> ndarray[tuple[int, int], np.complex128]:
        
        if pupil_element and pupil_function is not None:
                raise AttributeError("In the case of a custom pupil_function, pupil element is not accepted!")
            
        if pupil_element:
            pupil_function = self._get_pupil_function_of_an_element(pupil_element)
            
        if pupil_function is not None:
            if not pupil_function.shape == (Nrho, Nphi):
                raise AttributeError("Dimensions of the pupil_function don't match integration dimensions! ")

        if zernieke:
            aberration_phase = pupil_functions.compute_pupil_plane_aberrations(
                zernieke_polynomials=zernieke, 
                Nrho=Nrho, 
                Nphi=Nphi, 
            )
            aberration_function = np.exp(1j * 2 * np.pi * aberration_phase)
            if not pupil_function is None:
                if pupil_function.shape == (Nrho):
                    pupil_function[:, None] *= aberration_function
                elif pupil_function.shape == (Nrho, Nphi):
                    pupil_function *= aberration_function
            else:
                pupil_function = aberration_function

        return pupil_function

    def _get_pupil_function_of_an_element(self, Nrho: int, Nphi: int, pupil_element: str = "") -> ndarray[tuple[int, int], np.complex128]:
        """
        Get the pupil function of a given optical element.

        Args:
            pupil_element: An instance of an optical element class.

        Returns:
            np.ndarray: Pupil function of the element.
        """
        if not pupil_element: 
            return None
        
        if not pupil_element in OpticalSystem.known_pupil_elements:
            raise AttributeError("Pupil element ", pupil_element, " is not known to the OpticalSystem class. Compute" \
            " pupil function manually instead or add the element to the known_pupil_elements.")

        return OpticalSystem.known_pupil_elements[pupil_element](Nrho, Nphi)
    
    @abstractmethod
    def get_uniform_pupil(self):
        """ 
        Returns a uniformly filled with ones pupil function. Beware that the coordinates of this one 
        are not equal to the rho and phi used elsewhere. This function computes the pupil function on a
        uniform frequency grid and used for apodization.
        """
        pass

class OpticalSystem2D(OpticalSystem):

    dimensionality = 2

    def __init__(self, interpolation_method, normalize_otf=True, computed_size: int = 0):
        super().__init__(interpolation_method, normalize_otf)
        self.computed_size = computed_size

    def compute_psf_and_otf_coordinates(self, psf_size: tuple[float], N: int):
        if type(N) is int:
            N = (N, N)
        elif type(N) is tuple:
            if len(N) != 2:
                raise AttributeError("N should be integer or a tuple of two integers")
            
        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N[0])
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N[1])

        self.psf_coordinates = (x, y)

    @OpticalSystem.psf_coordinates.setter
    def psf_coordinates(self, new_coordinates):
        self._psf_coordinates = new_coordinates
        x, y = new_coordinates
        Nx, Ny = x.size, y.size
        Lx, Ly = 2 * new_coordinates[0][-1], 2 * new_coordinates[1][-1]

        fx = np.linspace(-Nx / (2 * Lx), Nx / (2 * Lx), Nx)
        fy = np.linspace(-Ny / (2 * Ly), Ny / (2 * Ly), Ny)
        self._otf_frequencies = (fx, fy)
        self._x_grid = None
        self._q_grid = None


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

    #Required by the abstract base class. Implementation is provided in the subclasses.
    def compute_psf_and_otf(self) -> tuple[np.float64, np.float64]:
        pass
    
    def get_uniform_pupil(self):
        rho = np.sqrt(self.q_grid[..., 0]**2 + self.q_grid[..., 1]**2)
        pupil_function = np.where(rho <= self.NA, 1., 0.)
        return pupil_function

    @property
    def computational_grid(self):
        if not self.computed_size:
            return super().x_grid
        else:
            centerx, centery = self.psf_coordinates[0].size // 2, self.psf_coordinates[1].size // 2
            x, y = self.psf_coordinates[0][centerx - self.computed_size // 2:centerx + self.computed_size // 2 + 1], self.psf_coordinates[1][centery - self.computed_size // 2:centery + self.computed_size // 2 + 1]
            grid = np.meshgrid(x, y, indexing='ij')
            return np.stack(grid, axis=-1)
        
class OpticalSystem3D(OpticalSystem):

    dimensionality = 3

    def __init__(self, interpolation_method, normalize_otf = True):
        super().__init__(interpolation_method, normalize_otf)

    def compute_psf_and_otf_coordinates(self, psf_size, N):
        if type(N) is int:
            N = (N, N, N)
        elif type(N) is tuple:
            if len(N) != 3:
                raise AttributeError("N should be integer or a tuple of three integers")

        x = np.linspace(-psf_size[0] / 2, psf_size[0] / 2, N[0])
        y = np.linspace(-psf_size[1] / 2, psf_size[1] / 2, N[1])
        z = np.linspace(-psf_size[2] / 2, psf_size[2] / 2, N[2])

        self.psf_coordinates = (x, y, z)

    @OpticalSystem.psf_coordinates.setter
    def psf_coordinates(self, new_coordinates):
        self._psf_coordinates = new_coordinates
        x, y, z, = new_coordinates
        Nx, Ny, Nz = x.size, y.size, z.size
        Lx, Ly, Lz = 2 * new_coordinates[0][-1], 2 * new_coordinates[1][-1], 2 * new_coordinates[2][-1]

        fx = np.linspace(-Nx / (2 * Lx), Nx / (2 * Lx), Nx)
        fy = np.linspace(-Ny / (2 * Ly), Ny / (2 * Ly), Ny)
        fz = np.linspace(-Nz / (2 * Lz), Nz / (2 * Lz), Nz)
        self._otf_frequencies = (fx, fy, fz)

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

    #Required by the abstract base class. Implementation is provided in the subclasses.
    def compute_psf_and_otf(self) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                                           np.ndarray[tuple[int, int, int], np.float64]]: 
        pass

    def get_uniform_pupil(self):
        rho = np.sqrt(self.q_grid[..., 0, 0]**2 + self.q_grid[..., 0, 1]**2)
        pupil_function = np.where(rho <= self.NA, 1., 0.)
        return pupil_function

class System4f2DCoherent(OpticalSystem2D): 
    def __init__(self,
                 alpha=np.pi / 4,
                 refractive_index=1,
                 interpolation_method="linear", 
                 normalize_otf = False, 
                 computed_size:int = 0):
        
        super().__init__(interpolation_method, normalize_otf, computed_size)
        self.nm = refractive_index
        self.alpha = alpha
        self.NA = self.nm * np.sin(self.alpha)

    def compute_psf_and_otf(self, parameters=None,
                            high_NA=False,
                            pupil_element="",
                            pupil_function=None, 
                            zernieke={}, 
                            Nrho = 129, 
                            Nphi = 129)\
            -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                     np.ndarray[tuple[int, int, int], np.float64]]:
        
        if self.psf_coordinates is None and parameters is None and pupil_function is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        
        elif parameters is not None:
            psf_size, N = parameters
            self.compute_psf_and_otf_coordinates(psf_size, N)

        pupil_function = self._get_pupil_function(Nrho, Nphi, pupil_element, pupil_function, zernieke)

        psf = psf_models.compute_2d_psf_coherent(
            grid=self.computational_grid, 
            NA=self.NA, 
            nmedium=self.nm, 
            pupil_function=pupil_function, 
            high_NA=high_NA, 
            Nrho=Nrho, 
            Nphi=Nphi, 
        )

        if self.computed_size:
            psf = utils.expand_kernel(psf, (self.psf_coordinates[0].size, self.psf_coordinates[1].size))

        self.psf = psf 
        self.otf = hpc_utils.wrapped_fftn(self.psf)
        
        self._normalize_psf__and_otf()

        # self._prepare_interpolator()
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


    def compute_psf_and_otf(self, parameters=None,
                            high_NA=False,
                            pupil_element=None,
                            pupil_function=None, 
                            zernieke={}, 
                            Nrho = 129, 
                            Nphi = 129,) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                                                np.ndarray[tuple[int, int, int], np.float64]]:
        if self.psf_coordinates is None and parameters is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_coordinates(psf_size, N)

        grid2d = np.stack(np.meshgrid(self.psf_coordinates[0], self.psf_coordinates[1], indexing='ij'), axis=-1)
        z_values = self.psf_coordinates[2]

        pupil_function = self._get_pupil_function(Nrho, Nphi, pupil_element, pupil_function, zernieke)

        psf = psf_models.compute_3d_psf_coherent(
            grid2d=grid2d, 
            NA=self.NA,
            z_values=z_values, 
            nsample=self.ns, 
            nmedium=self.nm, 
            pupil_function=pupil_function, 
            high_NA=high_NA, 
            Nrho=Nrho, 
            Nphi=Nphi, 
        )

        self.psf = psf 
        self.otf = hpc_utils.wrapped_fftn(self.psf)
        
        self._normalize_psf__and_otf()

        # self._prepare_interpolator()
        return self.psf, self.otf


class System4f2D(System4f2DCoherent):
    def __init__(self,
                 alpha=np.pi / 4,
                 refractive_index=1,
                 interpolation_method="linear", 
                 normalize_otf=True, 
                 computed_size:int = 0):
        
        super().__init__(alpha,
                 refractive_index,
                 interpolation_method, 
                 normalize_otf, 
                 computed_size)
    
    def compute_psf_and_otf(self, parameters=None,
                            high_NA=False,
                            vectorial=False,
                            pupil_element=None,
                            pupil_function=None, 
                            zernieke={}, 
                            Nrho = 129, 
                            Nphi = 129, 
                            ) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                                                np.ndarray[tuple[int, int, int], np.float64]]:
        
        if self.psf_coordinates is None and parameters is None and pupil_function is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        
        elif parameters is not None:
            psf_size, N = parameters
            self.compute_psf_and_otf_coordinates(psf_size, N)
        
        if not vectorial:
            csf, _ = super().compute_psf_and_otf(parameters, high_NA, pupil_element, pupil_function, zernieke, Nrho, Nphi)
            psf = np.abs(csf) ** 2

        else:
            pupil_function = self._get_pupil_function(Nrho, Nphi, pupil_element, pupil_function, zernieke)
            
            psf = psf_models.compute_2d_incoherent_vectorial_psf_free_dipole(
                grid=self.computational_grid,
                NA=self.NA,
                nmedium=self.nm, 
                pupil_function=pupil_function, 
                high_NA=high_NA, 
                Nrho=Nrho, 
                Nphi=Nphi, 
            )

            if self.computed_size:
                psf = utils.expand_kernel(psf, (self.psf_coordinates[0].size, self.psf_coordinates[1].size))

        self._otf = hpc_utils.wrapped_fftn(psf).real
        self._psf = psf.real
        
        self._normalize_psf__and_otf()

        # self._prepare_interpolator()
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

    def compute_psf_and_otf(self, parameters=None,
                            high_NA=False,
                            vectorial=False,
                            pupil_element=None,
                            pupil_function=None, 
                            zernieke={}, 
                            Nrho = 129, 
                            Nphi = 129, 
                            ) -> tuple[np.ndarray[tuple[int, int, int], np.float64],
                                                np.ndarray[tuple[int, int, int], np.float64]]:
        if self.psf_coordinates is None and parameters is None:
            raise AttributeError("Compute psf first or provide psf parameters")
        
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_coordinates(psf_size, N)

        if not vectorial:
            csf, _ = super().compute_psf_and_otf(None, high_NA, pupil_element, pupil_function, zernieke, Nrho, Nphi)
            psf = np.abs(csf) ** 2

        else:
            grid2d = np.stack(np.meshgrid(self.psf_coordinates[0], self.psf_coordinates[1], indexing='ij'), axis=-1)
            z_values = self.psf_coordinates[2]

            pupil_function = self._get_pupil_function(Nrho, Nphi, pupil_element, pupil_function, zernieke)

            psf = psf_models.compute_3d_incoherent_vectorial_psf_free_dipole(
                grid2d=grid2d, 
                NA=self.NA,
                z_values=z_values, 
                nsample=self.ns, 
                nmedium=self.nm, 
                pupil_function=pupil_function, 
                high_NA=high_NA, 
                Nrho=Nrho, 
                Nphi=Nphi, 
            )

        self._otf = hpc_utils.wrapped_fftn(psf).real
        self._psf = psf.real

        self._normalize_psf__and_otf()

        # self._prepare_interpolator()
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
        self.otf = hpc_utils.wrapped_fftn(self.psf)
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