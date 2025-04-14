import os.path
import sys

import scipy.signal
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import Illumination
from OpticalSystems import OpticalSystem
from abc import ABC, abstractmethod
import wrappers
import numpy as np
import scipy 
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt

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
    
    @abstractmethod
    def _compute_ideal_transfer_functions(self, **kwargs):
        """
        Computes the ideal transfer functions for the given pupil function.
        """
        ... 

    
class AutoconvolutionApodizationWidefield(AutoconvolutionApodization):
    def __init__(self, pupil_function):
        self._pupil_function = pupil_function
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
        self._ideal_psf = wrappers.wrapped_ifftn(self._ideal_otf)

        
    
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
        self._ideal_psf = wrappers.wrapped_ifftn(self._ideal_otf)


AutoconvolutionApodizationConfocal = AutoconvolutionApodizationPointScanning
AutoconvolutionApodizationISM = AutoconvolutionApodizationPointScanning
AutoconvolutionApodizationRCM = AutoconvolutionApodizationPointScanning

class AutoconvolutionApodizationSIM(AutoconvolutionApodizationWidefield):
    def __init__(self, optical_system: OpticalSystem, illumination: Illumination.PlaneWavesSIM):
        if not illumination.dimensionality == optical_system.dimensionality:
            raise ValueError("The illumination and pupil function dimensions do not match.")
        
        if not 'pupil_function'in optical_system.__dict__:
            raise ValueError("The optical system does not have a stored pupil function \
                             required for the autoconvoluiton method of apodization.")

        if not illumination.electric_field_plane_waves:
            raise ValueError("The illumination does not have plane waves, whose wavevectors are\
                             required for the autoconvoluiton method of apodization.")
        
        self._optical_system = optical_system
        self._illumination = illumination

        super().__init__(self._optical_system.pupil_function)

    def _compute_ideal_transfer_functions(self, **kwargs):
        # var1 = scipy.signal.convolve(self.pupil_function, np.flip(self.pupil_function).conjugate(), mode='same')
        var2 = scipy.signal.convolve(self.pupil_function, self.pupil_function.conjugate(), mode='same')
        # plt.imshow(var1[:, :, 50].real, cmap='gray')
        # plt.show()
        plt.imshow((var2[50, :, :].transpose()).real, cmap='gray')
        plt.show()

        ideal_pupil_function = np.where(np.abs(self.pupil_function) > 10**-1 * np.amax(self.pupil_function), 1, 0)
        ideal_pupil_function_ft = wrappers.wrapped_fftn(ideal_pupil_function)

        grid = self._optical_system.x_grid
        for Mr in range(self._illumination.Mr):
            for plane_wave in self._illumination.electric_field_plane_waves:
                wavevector = np.copy(plane_wave.wavevector)
                wavevector[:2] = VectorOperations.rotate_vector2d(np.copy(wavevector[:2]), self._illumination.angles[Mr])
                
                if len(self._illumination.dimensions) == 2:
                    wavevector = np.array([wavevector[0], wavevector[1]])
                    phase_modulated = ideal_pupil_function_ft * np.exp(1j * np.einsum('ijl,l ->ij', grid, wavevector))            
                    phase_shifted = wrappers.wrapped_fftn(phase_modulated).real
                    phase_shifted /= np.amax(phase_shifted)
                    # phase_shifted = np.where(phase_shifted > 10**-1, 1, 0)

                elif len(self._illumination.dimensions) == 3:
                    phase_modulated = np.transpose(np.exp(1j * np.einsum('ijkl,l ->ijk', grid, wavevector)), axes=(1, 0, 2)).real
                    phase_shifted = wrappers.wrapped_fftn(phase_modulated).real
                    phase_shifted /= np.amax(phase_shifted)
                    phase_shifted = np.where(phase_shifted > 10**-1, 1, 0)


                ideal_pupil_function[phase_shifted >  10**-1 * np.amax(self.pupil_function)] = 1
        self._ideal_ctf = ideal_pupil_function
        self._ideal_otf = scipy.signal.convolve(self._ideal_ctf, np.flip(self._ideal_ctf).conjugate(), mode='same')
        self._ideal_psf = wrappers.wrapped_ifftn(self._ideal_otf).real
    
