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

import utils
import scipy.signal
import Illumination
from OpticalSystems import OpticalSystem2D, OpticalSystem3D, OpticalSystem
from abc import ABC, abstractmethod
import hpc_utils
import numpy as np
import scipy 
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
from Dimensions import DimensionMetaAbstract
from OpticalSystems import System4f3DCoherent
from SSNRCalculator import SSNRSIM2D, SSNRSIM3D

class TriangularApodization():
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
    
    @abstractmethod
    def _compute_apodization_function(self, **kwargs):
        """
        Computes the apodization function for the given band-limited transfer function.
        """
        transfer_function_mask = np.where(np.abs(self.transfer_function) > 0, 1, 0)
        interior_mask = scipy.ndimage.binary_erosion(transfer_function_mask, iterations=1)
        surface_mask = transfer_function_mask & ~interior_mask
        # plt.imshow(surface_mask)
        # plt.show()
        triangular_function = utils.radial_ratio(self.transfer_function, surface_mask)
        triangular_function = np.nan_to_num(triangular_function)
        self._apodization_function = np.where(triangular_function<=1+1e-10, 1-triangular_function, 0) ** self._power
        # plt.imshow(self._apodization_function)
        # plt.show()  

    def _compute_apodization_function_2d(self):
        pass

    def _compute_apodization_function_3d(self):
        pass


class TriangularApodizationSIM(TriangularApodization):
    """
    Triangular apodization specialized for SIM: computes the effective transfer function 
    from a sum of effective OTFs, then applies the triangular roll-off.
    """
    def __init__(self, optical_system: OpticalSystem, illumination: Illumination.PlaneWavesSIM, power=1, numeric_noise=10**-4):
        _, effective_otfs = illumination.compute_effective_kernels(optical_system.psf, optical_system.psf_coordinates)
        effective_transfer_function = np.zeros_like(optical_system.otf, dtype=np.complex128)
        for otf_key in effective_otfs.keys():
            effective_transfer_function += effective_otfs[otf_key]
        effective_transfer_function /= np.amax(effective_transfer_function)
        effective_transfer_function = np.where(np.abs(effective_transfer_function) > numeric_noise, np.abs(effective_transfer_function), 0)
        # plt.imshow(effective_transfer_function)
        # plt.show()
        super().__init__(effective_transfer_function, power=power)




class AutoconvolutionApodization(ABC):
    """
    This class implements apodization based on the autoconvolution method presented
    in the paper 'Optimal Transfer Functions for bandlimiting imaging' by Stallinga at al. (20222) 
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
        ... 

    
class AutoconvolutionApodizationWidefield(AutoconvolutionApodization):
    def __init__(self, optical_system: OpticalSystem):
        self._pupil_function = optical_system.get_uniform_pupil()
        super().__init__()
        
    
    @property
    def pupil_function(self):
        return self._pupil_function
    
    @pupil_function.setter
    def pupil_function(self, new_pupil_function):
        self._pupil_function = new_pupil_function
        self._compute_ideal_transfer_functions()

    def _compute_ideal_transfer_functions(self, **kwargs):
        self._ideal_ctf = np.where(np.abs(self.pupil_function) > 10**-12 * np.amax(self.pupil_function), 1, 0)
        self._ideal_otf = scipy.signal.convolve(self._ideal_ctf, np.flip(self._ideal_ctf).conjugate(), mode='same')
        self._ideal_otf /= np.amax(self._ideal_otf)
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf)
        self._ideal_psf /= np.sum(self._ideal_psf)


class AutoconvolutionApodizationPointScanning(AutoconvolutionApodizationWidefield):
    def __init__(self, pupil_function_exitation, pupil_function_detection):
        self._pupil_function_exitation = pupil_function_exitation
        super().__init__(pupil_function_detection)

    @property
    def pupil_function_exitation(self):
        return self._pupil_function_exitation
    
    @pupil_function_exitation.setter
    def pupil_function_exitation(self, new_pupil_function_exitation):
        self._pupil_function_exitation = new_pupil_function_exitation
        self._compute_ideal_transfer_functions()

    @property
    def pupil_function_detection(self):
        return self.pupil_function

    @pupil_function_detection.setter
    def pupil_function_detection(self, new_pupil_function_detection):
        self.pupil_function = new_pupil_function_detection

    def _compute_ideal_transfer_functions(self, **kwargs):
        ideal_excitatiton = np.where(np.abs(self.pupil_function_exitation) > 10**-12 * np.amax(self.pupil_function_exitation), 1, 0)
        ideal_detection = np.where(np.abs(self.pupil_function) > 10**-12 * np.amax(self.pupil_function), 1, 0)
        ctf_effective = scipy.signal.convolve(ideal_excitatiton, ideal_detection, mode='same')
        self._ideal_ctf = np.where(np.abs(ctf_effective) > 10**-12 * np.amax(ctf_effective), 1, 0)
        self._ideal_otf = scipy.signal.convolve(self._ideal_ctf, np.flip(self._ideal_ctf).conjugate(), mode='same')
        self._ideal_otf /= np.amax(self._ideal_otf)
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf)
        self._ideal_psf /= np.sum(self._ideal_psf)


AutoconvolutionApodizationConfocal = AutoconvolutionApodizationPointScanning
AutoconvolutionApodizationISM = AutoconvolutionApodizationPointScanning
AutoconvolutionApodizationRCM = AutoconvolutionApodizationPointScanning

class AutoconvolutionApodizationSIM(AutoconvolutionApodizationWidefield, metaclass=DimensionMetaAbstract):
    """
    Autoconvolution apodization base for SIM. Extends the widefield variant by accounting
    for illumination plane-wave wavevectors when constructing the ideal CTF support.
    Dimension-specific subclasses must implement _compute_ideal_transfer_functions.
    """
    def __init__(self, optical_system: OpticalSystem, illumination: Illumination.PlaneWavesSIM, plane_wave_wavevectors: list[np.ndarray]):
        if not illumination.dimensionality == optical_system.dimensionality:
            raise ValueError("The illumination and pupil function dimensions do not match.")
        
        if plane_wave_wavevectors: 
            self.plane_wave_wavevectors = plane_wave_wavevectors
        else: 
            self.plane_wave_wavevectors = []
            if not illumination.source_electromagnetic_plane_waves and not plane_wave_wavevectors:
                raise ValueError("Plane wave wavevectors required.")
            for plane_wave in illumination.source_electromagnetic_plane_waves:
                self.plane_wave_wavevectors.append(plane_wave.wavevector)
        
        self._optical_system = optical_system
        self._illumination = illumination

        super().__init__(optical_system)

    def _compute_ideal_transfer_functions(self, **kwargs):
        raise NotImplementedError("The autoconvolution method of apodization is dimension-specific due to numeric issues.")
        # var1 = scipy.signal.convolve(self.pupil_function, np.flip(self.pupil_function).conjugate(), mode='same')
        # var2 = scipy.signal.convolve(self.pupil_function, self.pupil_function.conjugate(), mode='same')
        # var3 = np.log(1 + 10**8 * scipy.signal.convolve((self.pupil_function), np.flip(self.pupil_function).conjugate(), mode='same'))
        # plt.imshow(var1[:, :, 50].real, cmap='gray')
        # plt.show()
        # plt.imshow((var3[50, :, :].transpose()).real, cmap='gray')
        # plt.imshow((self.pupil_function[50, :, :].transpose()).real, cmap='gray')
        # plt.show()

  

class AutoconvolutionApodizationSIM2D(AutoconvolutionApodizationSIM):
    dimensionality = 2
    def __init__(self, optical_system: OpticalSystem2D, illumination: Illumination.PlaneWavesSIM, plane_wave_wavevectors: list[np.ndarray] = None):
        super().__init__(optical_system, illumination, plane_wave_wavevectors)

    def _compute_ideal_transfer_functions(self, **kwargs):
        ideal_pupil_function = np.where(np.abs(self.pupil_function) > 10**-1 * np.amax(self.pupil_function), 1., 0)
        # ideal_psf = self._optical_system.compute_psf_and_otf()
        ideal_pupil_function_ift = hpc_utils.wrapped_ifftn(ideal_pupil_function)
        
        grid = self._optical_system.x_grid
        for Mr in range(self._illumination.Mr):
            for plane_wave_wavevector in self.plane_wave_wavevectors:
                wavevector = np.copy(plane_wave_wavevector)
                wavevector[:2] = VectorOperations.rotate_vector2d(np.copy(wavevector[:2]), self._illumination.angles[Mr])
                
                wavevector = np.array([wavevector[0], wavevector[1]])
                phase_modulated = ideal_pupil_function_ift * np.exp(1j * np.einsum('ijl,l ->ij', grid, wavevector))            
                phase_shifted = hpc_utils.wrapped_fftn(phase_modulated).real
                phase_shifted /= np.amax(phase_shifted)
                # phase_shifted = np.where(phase_shifted > 10**-1, 1, 0)
                ideal_pupil_function[phase_shifted >  0.1 * np.amax(self.pupil_function)] = 1

        self._ideal_ctf = ideal_pupil_function
        self._ideal_otf = scipy.signal.convolve(self._ideal_ctf, np.flip(self._ideal_ctf).conjugate(), mode='same')
        self._ideal_otf /= np.amax(self._ideal_otf)
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf).real
        self._ideal_psf /= np.sum(self._ideal_psf)
    

def _build_edwald_sphere_mask(q_grid:np.ndarray, q_center: np.ndarray, q_radius: float, angle_cutoff: float):
    """
    Builds a spherical mask in the frequency domain representing the Ewald sphere.
    """
    Qx, Qy, Qz = q_grid[..., 0], q_grid[..., 1], q_grid[..., 2]
    qx_center, qy_center, qz_center = q_center

    distance_from_center = np.sqrt((Qx - qx_center)**2 + (Qy - qy_center)**2 + (Qz - qz_center)**2)
    angle = np.arccos((Qz - qz_center) / (distance_from_center + 1e-12))

    sphere_mask = distance_from_center <= q_radius
    interior_mask = scipy.ndimage.binary_erosion(sphere_mask, iterations=1)
    sphere_mask = sphere_mask & ~interior_mask
    sphere_mask = sphere_mask & (angle <= angle_cutoff)
    # _, _, sldier =utils.imshow3D(sphere_mask, cmap='gray', mode='abs', axis='y')
    # plt.title('Ewald Sphere Mask')
    # plt.show()
    return sphere_mask

class AutoconvolutionApodizationSIM3D(AutoconvolutionApodizationSIM):
    dimensionality = 3
    def __init__(self, optical_system: OpticalSystem3D, illumination: Illumination.PlaneWavesSIM, plane_wave_wavevectors: list[np.ndarray] = None, N_dense: int = 201):
        self.N_dense = N_dense
        super().__init__(optical_system, illumination, plane_wave_wavevectors)

    def _compute_ideal_transfer_functions(self, **kwargs):
        N_dense = self.N_dense
        # widefield_coherent_psf, _ =  System4f3DCoherent.compute_psf_and_otf(self._optical_system)
        qx, qy, qz = self._optical_system.otf_frequencies
        qx_min, qx_max = qx.min(), qx.max()
        qy_min, qy_max = qy.min(), qy.max()
        qz_min, qz_max = qz.min(), qz.max()
        qx_dense = np.linspace(qx_min, qx_max, N_dense)
        qy_dense = np.linspace(qy_min, qy_max, N_dense)
        qz_dense = np.linspace(qz_min, qz_max, N_dense)

        q_grid_dense = np.stack(np.meshgrid(qx_dense, qy_dense, qz_dense, indexing='ij'), axis=-1)
        # ideal_otf_dense = np.zeros((N_dense, N_dense, N_dense), dtype=np.float64)
        ideal_pupil_function = np.zeros((N_dense, N_dense, N_dense), dtype=np.float64)
        kz_shift = max([wavevector[2] for wavevector in self.plane_wave_wavevectors])
        for Mr in range(self._illumination.Mr):
            # print(self._illumination.angles[Mr])
            for wavevector in self.plane_wave_wavevectors:
                wavevector_rotated = np.copy(wavevector)
                wavevector_rotated[:2] = VectorOperations.rotate_vector2d(np.copy(wavevector[:2]), self._illumination.angles[Mr])
                wavevector_rotated[2] -= kz_shift
                # print(wavevector)
                q_center = wavevector_rotated / (2 * np.pi)
                angle_cutoff = self._optical_system.alpha
                q_radius = self._optical_system.NA / np.sin(angle_cutoff)
                sphere_mask = _build_edwald_sphere_mask(q_grid_dense, q_center, q_radius, angle_cutoff)
                # print(sphere_mask.any())
                ideal_pupil_function = np.where(sphere_mask, 1., ideal_pupil_function)

                # _, _, slider=utils.imshow3D(ideal_pupil_function, cmap='gray', mode='abs', scaling=10**3, axis='y')
                # plt.title('Ideal CTF Dense')
                # plt.show()

        ideal_otf_dense = np.flip(scipy.signal.convolve(ideal_pupil_function, np.flip(ideal_pupil_function).conjugate(), mode='same'))
        # plt.imshow(np.flip(ideal_otf_dense)[:, N_dense//2, :], cmap='gray')
        # plt.show()
        # _, _, slider = utils.imshow3D(np.log1p(ideal_pupil_function), cmap='gray')
        # plt.title('Ideal CTF Dense')
        # plt.show()
        # plt.imshow(np.where(np.abs(ideal_otf_dense[:, N_dense//2, :]) > 10**-6, 1, 0), cmap='gray')

        interpolator_otf = scipy.interpolate.RegularGridInterpolator((qx_dense, qy_dense, qz_dense), ideal_otf_dense, method='linear',
                                                                      bounds_error=False,
                                                                      fill_value=0.)
        
        self._ideal_otf = interpolator_otf(self._optical_system.q_grid.flatten()).reshape(self._optical_system.otf.shape)

        self._ideal_otf /= np.amax(self._ideal_otf)
        
        self._ideal_psf = hpc_utils.wrapped_ifftn(self._ideal_otf).real
        self._ideal_psf /= np.sum(self._ideal_psf)

        self._ideal_ctf = hpc_utils.wrapped_fftn(self._ideal_psf**0.5).real

        

