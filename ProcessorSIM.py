"""
ProcessorSIM.py

A top-level class, combining the whole SIM functionality. 

Classes:
    ProcessorSIM: Base class for SIM processors.
"""
import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

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
    modes =('general', 'simulation', 'reconstruction')
    regularizatoinion_filters = ('TrueWiener', 'Flat', 'Constant') 
    apodization_filter = ('Lukosz', 'Autoconvolution')
    deconvolution_scheme = ('Wiener', 'Richardson-Lucy', 'Bayesian', 'MutualInformation') 
    estimation_scheme = ('CrossCorrelation')
    
    def __init__(self,
                 illumination,
                 optical_system,
                 dim=3,
                 kernel=None,
                 estimate_ssnr=False,
                 deconvolution_scheme=None,
                 regularization_filter=None,
                 apodization_filter=None,
                 estimate_patterns_from_data=False,
                 camera=None,
                 mpi_optimization=False,
                 cuda_optimization=False,
                 prioritize_memory=False
                 ):
        self.optical_system = optical_system
        self.illumination = illumination
        
        match dim: 
            case 2:
                SIMulator = SIMulator.SIMulator2D
                Reconstructor = Reconstructor.Reconstructor2D
                SSNRCalculator = SSNRCalculator.SSNRSIM2D
            case 3:
                SIMulator = SIMulator.SIMulator3D
                Reconstructor = Reconstructor.Reconstructor3D
                SSNRCalculator = SSNRCalculator.SSNRSIM3D
        

        if estimate_ssnr:
            self.ssnr_calculator = SSNRCalculator(self.optical_system, self.illumination)
                

    @staticmethod
    @abstractmethod
    def compute_effective_psfs_and_otfs(illumination, optical_system): ...
    def compute_sim_support(self): ...
    def compute_apodization_filter_lukosz(self): ...

    def compute_apodization_filter_autoconvolution(self): ...

