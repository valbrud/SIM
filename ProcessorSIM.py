import numpy as np
import scipy
import matplotlib.pyplot as plt
import OpticalSystems
import Illumination
from VectorOperations import VectorOperations
from abc import abstractmethod
import wrappers
class ProcessorSIM:
    def __init__(self, illumination, optical_system):
        self.optical_system = optical_system
        self.illumination = illumination

    @staticmethod
    @abstractmethod
    def compute_effective_psfs_and_otfs(illumination, optical_system): ...
    def compute_sim_support(self): ...
    def compute_apodization_filter_lukosz(self): ...

    def compute_apodization_filter_autoconvolution(self): ...

class ProcessorProjective3dSIM(ProcessorSIM):
    def __init__(self, illumination, optical_system):
        super().__init__(illumination, optical_system)
    @staticmethod
    @abstractmethod
    def compute_effective_psfs_and_otfs(illumination, optical_system):
        waves = illumination.waves
        # plt.show()
        for r in range(illumination.Mr):
            angle = illumination.angles[r]
            indices = illumination.rearranged_indices
            for w in range(len(illumination.wavevectors2d)):
                effective_psf = 0
                xy_indices = illumination.indices2d[w]
                for z_index in indices[xy_indices]:
                    wavevector = VectorOperations.rotate_vector3d(
                        waves[(*xy_indices, z_index)].wavevector, np.array((0, 0, 1)), angle)
                    wavevector = np.array((0, 0, wavevector[2]))
                    amplitude = waves[(*xy_indices, z_index)].amplitude
                    phase_shifted = np.exp(1j * np.einsum('ijkl,l ->ijk', self.grid, wavevector)) * optical_system.psf
                    effective_psf += amplitude * phase_shifted
                    # print(wavevector)
                effective_psfs[(r, w)] = effective_psf
                effective_otfs[(r, w)] = wrappers.wrapped_fftn(effective_psf)

class ProcessorTrue3dSIM(ProcessorSIM): ...