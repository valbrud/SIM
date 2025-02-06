"""
ProcessorSIM.py

When implemented, this class will be a top-level class, responsible for SIM reconstructions.

Classes:
    ProcessorSIM: Base class for SIM processors.
    ProcessorProjective3dSIM: Class for processing projective 3D SIM data.
    ProcessorTrue3dSIM: Class for processing true 3D SIM data.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import OpticalSystems
import Illumination
from VectorOperations import VectorOperations
from abc import abstractmethod
import wrappers
import SIMulator
import Reconstructor
import SSNRCalculator
class ProcessorSIM:
    def __init__(self, illumination, optical_system, dim=3, projected=True, local_kernel=None, apodization_filter=None,
                 estimate_from_data=False, camera=default, mpi_optimization=False, cuda_optimization=False, prioritize_memory=False):
        self.optical_system = optical_system
        self.illumination = illumination
        if dim == 3:
            self.simulator = SIMulator.SIMulator3D(self.optical_system, self.illumination, readout_noise=readout_noise,)
            self.reconstructor = Reconstructor.Reconstructor3D(self.optical_system, self.illumination)
            if projected:
                self.ssnri= SSNRCalculator.SSNRSIM3DUniversal(self.optical_system, self.illumination, readout_noise)
        else:
            self.simulator


    @staticmethod
    @abstractmethod
    def compute_effective_psfs_and_otfs(illumination, optical_system): ...
    def compute_sim_support(self): ...
    def compute_apodization_filter_lukosz(self): ...

    def compute_apodization_filter_autoconvolution(self): ...

class ProcessorProjective3dSIM(ProcessorSIM):
    def __init__(self, illumination, optical_system):
        super().__init__(illumination, optical_system)
class ProcessorTrue3dSIM(ProcessorSIM): ...