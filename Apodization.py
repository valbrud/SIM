"""
Apodization.py

This module provides apodization methods for band-limited imaging systems.
Apodization functions suppress artifacts near the frequency support boundary, improving
image quality in widefield, confocal, point-scanning, and SIM modalities.

Classes:
    TriangularApodization - Apodization based on triangular (linear roll-off) weighting of the transfer function.
    TriangularApodizationSIM - Triangular apodization specialized for SIM effective transfer functions.
    AutoconvolutionApodization - Abstract base for autoconvolution-based optimal apodization (Stallinga et al., 2022).
    AutoconvolutionApodizationWidefield - Autoconvolution apodization for widefield microscopy.
    AutoconvolutionApodizationPointScanning - Autoconvolution apodization for point-scanning modalities.
    AutoconvolutionApodizationSIM - Autoconvolution apodization base for SIM (dimension-specific).
    AutoconvolutionApodizationSIM2D - 2D autoconvolution apodization for SIM.
    AutoconvolutionApodizationSIM3D - 3D autoconvolution apodization for SIM using Ewald sphere construction.

Functions:
    _build_edwald_sphere_mask - Build a spherical mask in frequency domain representing the Ewald sphere.
"""

import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import Illumination
from OpticalSystems import OpticalSystem2D, OpticalSystem3D, OpticalSystem
from abc import ABC, abstractmethod
import hpc_utils, utils
import numpy as np
import scipy 
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
from Dimensions import DimensionMetaAbstract
from OpticalSystems import System4f3DCoherent
from SSNRCalculator import SSNRSIM2D, SSNRSIM3D

class ApodizationSIM(metaclass=DimensionMetaAbstract):
    def __init__(self, optical_system: OpticalSystem, illumination: Illumination.PlaneWavesSIM, plane_wave_wavevectors: list[np.ndarray], Ndense: int = 201):
        self.optical_system = optical_system
        self.illumination = illumination
        if plane_wave_wavevectors: 
            self.plane_wave_wavevectors = plane_wave_wavevectors
        else: 
            self.plane_wave_wavevectors = []
            if not illumination.source_electromagnetic_plane_waves and not plane_wave_wavevectors:
                raise ValueError("Plane wave wavevectors required.")
            for plane_wave in illumination.source_electromagnetic_plane_waves:
                self.plane_wave_wavevectors.append(plane_wave.wavevector)
            
        self.Ndense = Ndense
        self.grid_coordinates = None
        self._apodization_function = None   

    @property
    def apodization_function(self):
        return self._apodization_function

    @abstractmethod
    def _get_dense_grid(self):
        pass 
    
    @abstractmethod
    def _compute_ideal_ctf_support(self):
        pass 
    
    def _interpolate_to_system_grid(self, transfer_function_dense):
        interpolator = scipy.interpolate.RegularGridInterpolator(self.grid_coordinates, transfer_function_dense, method='linear',
                                                                bounds_error=True,
                                                                fill_value=0.)
        
        return interpolator(self.optical_system.q_grid.flatten()).reshape(self.optical_system.otf.shape)
    
class ApodizationSIM2D(ApodizationSIM):
    dimensionality = 2
    def __init__(self, optical_system: OpticalSystem2D, illumination: Illumination.IlluminationPlaneWaves2D, plane_wave_wavevectors: list[np.ndarray] = None, Ndense: int = 201):
            if self.dimensionality != optical_system.dimensionality:
                raise ValueError("The optical system and apodization dimensions do not match.")
            super().__init__(optical_system, illumination, plane_wave_wavevectors, Ndense=Ndense)

    def _get_dense_grid(self):
        qx, qy = self.optical_system.otf_frequencies
        qx_min, qx_max = qx.min(), qx.max()
        qy_min, qy_max = qy.min(), qy.max()
        Nx, Ny = max(self.Ndense, len(qx)), max(self.Ndense, len(qy))
        qx_dense = np.linspace(qx_min, qx_max, Nx)
        qy_dense = np.linspace(qy_min, qy_max, Ny)
        self.grid_coordinates = (qx_dense, qy_dense)
        return np.stack(np.meshgrid(qx_dense, qy_dense, indexing='ij'), axis=-1)

    def _compute_ideal_ctf_support(self):
        ideal_ctf_support = np.zeros(self._get_dense_grid().shape[:-1], dtype=np.float64)
        grid = self._get_dense_grid()

        for Mr in range(self.illumination.Mr):
            for plane_wave_wavevector in self.plane_wave_wavevectors:
                wavevector = np.copy(plane_wave_wavevector)[:2]
                wavevector = VectorOperations.rotate_vector2d(np.copy(wavevector), self.illumination.angles[Mr])
                circular_mask = utils.build_circular_mask(grid, wavevector / (2 * np.pi), self.optical_system.NA / np.sin(self.optical_system.alpha))
                ideal_ctf_support[circular_mask] = 1

        return ideal_ctf_support
    
class ApodizationSIM3D(ApodizationSIM):
    dimensionality = 3
    def __init__(self, optical_system: OpticalSystem3D, illumination: Illumination.IlluminationPlaneWaves3D, plane_wave_wavevectors: list[np.ndarray] = None, Ndense: int = 201):
        if self.dimensionality != optical_system.dimensionality:
            raise ValueError("The optical system and apodization dimensions do not match.")
        super().__init__(optical_system, illumination, plane_wave_wavevectors, Ndense=Ndense)

    def _get_dense_grid(self):
        qx, qy, qz = self.optical_system.otf_frequencies
        qx_min, qx_max = qx.min(), qx.max()
        qy_min, qy_max = qy.min(), qy.max()
        qz_min, qz_max = qz.min(), qz.max()
        Nx, Ny, Nz = max(self.Ndense, len(qx)), max(self.Ndense, len(qy)), max(self.Ndense, len(qz))
        qx_dense = np.linspace(qx_min, qx_max, Nx)
        qy_dense = np.linspace(qy_min, qy_max, Ny)
        qz_dense = np.linspace(qz_min, qz_max, Nz)
        self.grid_coordinates = (qx_dense, qy_dense, qz_dense)
        return np.stack(np.meshgrid(qx_dense, qy_dense, qz_dense, indexing='ij'), axis=-1)
    
    def _compute_ideal_ctf_support(self):
        q_grid_dense = self._get_dense_grid()
        ideal_ctf_support = np.zeros(q_grid_dense.shape[:-1], dtype=np.float64)
        kz_shift = np.zeros(3)
        widefield_ctf_support = None
        for wavevector in self.plane_wave_wavevectors:
            if np.isclose(wavevector[0], 0, atol=10**-6) and np.isclose(wavevector[1], 0, atol=10**-6):
                kz_shift = wavevector[2]
                # widefield_ctf_support = self.optical_system.get_transfer_function_support()
                break
        if kz_shift.sum() == 0:
            kz_shift = max([wavevector[2] for wavevector in self.plane_wave_wavevectors])

        self.plane_wave_wavevectors = [wavevector - np.array([0, 0, kz_shift]) for wavevector in self.plane_wave_wavevectors]
        for Mr in range(self.illumination.Mr):
            for wavevector in self.plane_wave_wavevectors:
                wavevector_rotated = np.copy(wavevector)
                wavevector_rotated[:2] = VectorOperations.rotate_vector2d(np.copy(wavevector[:2]), self.illumination.angles[Mr])
                q_center = wavevector_rotated / (2 * np.pi)
                angle_cutoff = self.optical_system.alpha
                q_radius = self.optical_system.NA / np.sin(angle_cutoff)
                # if np.isclose(np.sum(wavevector[:2]), 0, atol=10**-6):
                #     ideal_ctf_support = np.where(widefield_ctf_support, 1., ideal_ctf_support)
                #     continue
                sphere_mask = utils.build_edwald_sphere_mask(q_grid_dense, q_center, q_radius, angle_cutoff)
                ideal_ctf_support = np.where(sphere_mask, 1., ideal_ctf_support)
        _, _, slider = utils.imshow3D(ideal_ctf_support, cmap='gray', mode='real', vmin=0, vmax=1)
        plt.show()
        return ideal_ctf_support
    
class TriangularApodization(metaclass=DimensionMetaAbstract):
    """
    Apodization based on triangular (linear roll-off) weighting of a band-limited transfer function.
    The apodization function equals (1 - r/r_edge)^power, where r is the radial ratio
    to the transfer function support boundary.

    Attributes:
        transfer_function (np.ndarray): The band-limited transfer function to apodize.
        apodization_function (np.ndarray): Computed apodization weights.
        power (float): Exponent controlling the roll-off steepness.
    """
    def __init__(self, transfer_function, power=1):
        self._transfer_function = transfer_function
        self._apodization_function = None
        self._power = power

        self._compute_apodization_function()

    @property
    def transfer_function(self):
        return self._transfer_function
    
    @property
    def apodization_function(self):
        return self._apodization_function
    
    @property
    def power(self):
        return self._power
    
    @power.setter
    def power(self, new_power):
        self._power = new_power
        self._compute_apodization_function()
    
    def _compute_apodization_function(self, **kwargs):
        """
        Computes the apodization function for the given band-limited transfer function.
        """
        transfer_function_mask = np.where(np.abs(self.transfer_function) > 0, 1, 0)
        interior_mask = scipy.ndimage.binary_dilation(transfer_function_mask, iterations=1)
        surface_mask = ~transfer_function_mask & interior_mask
        triangular_function = utils.radial_ratio(self.transfer_function, surface_mask)
        triangular_function = np.nan_to_num(triangular_function)
        self._apodization_function = np.where(triangular_function<=1+1e-10, 1-triangular_function, 0) ** self._power
        _, _, slider=utils.imshow3D(self._apodization_function, cmap='gray', vmin=0, vmax=1)
        plt.show()
        # plt.imshow(self._apodization_function)
        # plt.show()  


class TriangularApodizationSIM(ApodizationSIM, TriangularApodization):
    """
    Triangular apodization specialized for SIM: computes the effective transfer function 
    from a sum of effective OTFs, then applies the triangular roll-off.
    """
    def __init__(self, optical_system: OpticalSystem, illumination: Illumination.PlaneWavesSIM, power=1, noise_cutoff=10**-4, Ndense=201):
        super().__init__(optical_system, illumination, plane_wave_wavevectors=None, Ndense=Ndense)
        ctf_support = self._compute_ideal_ctf_support()
        effective_otf = scipy.signal.convolve(ctf_support, np.flip(ctf_support).conjugate(), mode='same')
        effective_otf /= np.amax(effective_otf)
        effective_otf = np.where(effective_otf > noise_cutoff * np.amax(effective_otf), effective_otf, 0)
        TriangularApodization.__init__(self, effective_otf, power=power)

class TriangularApodizationSIM2D(TriangularApodizationSIM, ApodizationSIM2D):
    dimensionality = 2
    def __init__(self, optical_system: OpticalSystem2D, illumination: Illumination.IlluminationPlaneWaves2D, power=1, noise_cutoff=10**-4, Ndense=201):
        super().__init__(optical_system, illumination, power=power, noise_cutoff=noise_cutoff, Ndense=Ndense)
        
class TriangularApodizationSIM3D(TriangularApodizationSIM, ApodizationSIM3D):
    dimensionality = 3
    def __init__(self, optical_system: OpticalSystem3D, illumination: Illumination.IlluminationPlaneWaves3D, power=1, noise_cutoff=10**-4, Ndense=201):
        super().__init__(optical_system, illumination, power=power, noise_cutoff=noise_cutoff, Ndense=Ndense)

class AutoconvolutionApodization(ABC):
    """
    This class implements apodization based on the autoconvolution method presented
    in the paper 'Optimal Transfer Functions for bandlimiting imaging' by Stallinga at al. (2022) 
    Object oriented design it for easiness of the extrapolation to different microscopy modalities.
    """
    def __init__(self):
        self._ideal_ctf = None
        self._ideal_psf = None
        self._ideal_otf = None
        self._compute_ideal_transfer_functions()

    @property
    def ideal_ctf(self):
        return self._ideal_ctf
    
    @property 
    def ideal_psf(self):
        return self._ideal_psf
    
    @property
    def ideal_otf(self):
        return self._ideal_otf

    @property
    def apodization_function(self):
        return self._ideal_otf
    
    @abstractmethod
    def _compute_ideal_transfer_functions(self, **kwargs):
        """
        Computes the ideal transfer functions for the given pupil function.
        """

    
class AutoconvolutionApodizationWidefield(AutoconvolutionApodization):
    def __init__(self, optical_system: OpticalSystem):
        self._optical_system = optical_system
        super().__init__()
        
    def _compute_ideal_transfer_functions(self, noise_level=10**-6):
        self._ideal_ctf = np.float64(self._optical_system.get_transfer_function_support())
        self._ideal_otf = scipy.signal.convolve(self._ideal_ctf, np.flip(self._ideal_ctf).conjugate(), mode='same')
        self._ideal_otf /= np.amax(self._ideal_otf)
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf)
        self._ideal_psf /= np.sum(self._ideal_psf)


class AutoconvolutionApodizationPointScanning(AutoconvolutionApodizationWidefield):
    def __init__(self, ctf_support_exitation, ctf_support_detection):
        self._ctf_support_exitation = ctf_support_exitation
        super().__init__(ctf_support_detection)

    @property
    def ctf_support_exitation(self):
        return self._ctf_support_exitation
    
    @ctf_support_exitation.setter
    def ctf_support_exitation(self, new_ctf_support_exitation):
        self._ctf_support_exitation = new_ctf_support_exitation
        self._compute_ideal_transfer_functions()

    @property
    def ctf_support_detection(self):
        return self.ctf_support

    @ctf_support_detection.setter
    def ctf_support_detection(self, new_ctf_support_detection):
        self.ctf_support = new_ctf_support_detection

    def _compute_ideal_transfer_functions(self, **kwargs):
        ideal_excitatiton = np.where(np.abs(self.ctf_support_exitation) > 10**-12 * np.amax(self.ctf_support_exitation), 1, 0)
        ideal_detection = np.where(np.abs(self.ctf_support) > 10**-12 * np.amax(self.ctf_support), 1, 0)
        ctf_effective = scipy.signal.convolve(ideal_excitatiton, ideal_detection, mode='same')
        self._ideal_ctf = np.where(np.abs(ctf_effective) > 10**-12 * np.amax(ctf_effective), 1, 0)
        self._ideal_otf = scipy.signal.convolve(self._ideal_ctf, np.flip(self._ideal_ctf).conjugate(), mode='same')
        self._ideal_otf /= np.amax(self._ideal_otf)
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf)
        self._ideal_psf /= np.sum(self._ideal_psf)


AutoconvolutionApodizationConfocal = AutoconvolutionApodizationPointScanning
AutoconvolutionApodizationISM = AutoconvolutionApodizationPointScanning
AutoconvolutionApodizationRCM = AutoconvolutionApodizationPointScanning

class AutoconvolutionApodizationSIM(ApodizationSIM, AutoconvolutionApodization):
    def __init__(self, optical_system: OpticalSystem, illumination: Illumination.PlaneWavesSIM, plane_wave_wavevectors: list[np.ndarray] = None, Ndense: int = 201):
        super().__init__(optical_system, illumination, plane_wave_wavevectors, Ndense=Ndense)
        AutoconvolutionApodization.__init__(self)

    @property
    def apodization_function(self):
        return self._ideal_otf
    
    def _compute_ideal_transfer_functions(self):
        dense_ctf = self._compute_ideal_ctf_support()
        dense_otf = scipy.signal.convolve(dense_ctf, np.flip(dense_ctf).conjugate(), mode='same')
        if (np.array(dense_otf.shape) < self.Ndense).any():
            self.ideal_otf = self._interpolate_to_system_grid(dense_otf)
            self.ideal_ctf = self._interpolate_to_system_grid(dense_ctf)
        else: 
            self._ideal_otf = dense_otf
            self._ideal_ctf = dense_ctf

        self._ideal_otf /= np.amax(self._ideal_otf)
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf).real
        self._ideal_psf /= np.sum(self._ideal_psf)


class AutoconvolutionApodizationSIM2D(AutoconvolutionApodizationSIM, ApodizationSIM2D):
    dimensionality = 2
    def __init__(self, optical_system: OpticalSystem2D, illumination: Illumination.IlluminationPlaneWaves2D, plane_wave_wavevectors: list[np.ndarray] = None, Ndense: int = 201):
        super().__init__(optical_system, illumination, plane_wave_wavevectors, Ndense=Ndense)


class AutoconvolutionApodizationSIM3D(AutoconvolutionApodizationSIM, ApodizationSIM3D):
    dimensionality = 3
    def __init__(self, optical_system: OpticalSystem3D, illumination: Illumination.IlluminationPlaneWaves3D, plane_wave_wavevectors: list[np.ndarray] = None, Ndense: int = 201):
        super().__init__(optical_system, illumination, plane_wave_wavevectors, Ndense=Ndense)

        

