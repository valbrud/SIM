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
import Camera
import stattools
from deconvolution import richardson_lucy_skimage, bayesian_gaussian_frequency_estimate, image_of_maximal_surprise_estimate
import WienerFiltering
from Apodization import AutoconvolutionApodizationSIM

class ProcessorSIM:
    """
    ProcessorSIM class for simulating and reconstructing super-resolution images using structured illumination microscopy (SIM).
    
    Attributes:
        illumination: Illumination object, the illumination configuration for the SIM experiment.
        optical_system: OpticalSystem object, the optical system used in the experiment.
        kernel: numpy.ndarray, optional, the kernel used for deconvolution.
        simulator: SIMulator object, for generating simulated images.
        reconstructor: Reconstructor object, for reconstructing super-resolution images.
        ssnr_calculator: SSNRCalculator object, for computing SSNR values.
        dim: int, the number of dimensions (2 or 3).
        spatial_domain: bool, whether to use spatial domain reconstruction.
        estimate_ssnr: bool, whether to estimate SSNR.
        deconvolution_method: str, the method used for deconvolution.
        regularization_method: str, the method used for regularization.
        apodization_method: str, the method used for apodization.
        estimate_patterns_from_data: bool, whether to estimate patterns from data.
        pattern_estimation_method: str, the method used for pattern estimation.
        camera: Camera object, the camera used to capture images.
    
    Methods:
        simulate_sim_images(true_object: np.ndarray, noisy: bool=True) -> np.ndarray:
            Simulate SIM images using the provided illumination and optical system.
        generate_super_resolution_image(sim_images: np.ndarray) -> np.ndarray:
            Generate super-resolution image using the provided SIM images.
        estimate_total_ssnr_from_data(image: np.ndarray) -> np.ndarray:
            Estimate the total SSNR from the image by a method of the image splitting.
        compute_true_ssnr(true_object: np.ndarray) -> np.ndarray:
            Compute the true SSNR given the object.
        deconvolve(image: np.ndarray) -> np.ndarray:
            Deconvolve the image using the specified method.
        apodize(image: np.ndarray) -> np.ndarray:
            Apodize the image using the specified method.
    """

    regularizatoinion_methods = ('TrueWiener', 'Flat', 'Constant') 
    apodization_methods = ('Lukosz', 'Autoconvolution')
    deconvolution_methods = ('Wiener', 'Richardson-Lucy', 'Bayesian', 'MutualInformation') 
    pattern_estimation_methods = ('CrossCorrelation')
    
    def __init__(self,
                 illumination: Illumination.IlluminationPlaneWaves,
                 optical_system: OpticalSystems.OpticalSystem,
                 kernel: np.ndarray=None,
                 spatial_domain: bool=False,
                 estimate_ssnr:bool=False,
                 deconvolution_method:str=None,
                 regularization_method:str=None,
                 apodization_method:str=None,
                 estimate_patterns_from_data:bool=False,
                 pattern_estimation_method:str=None,
                 camera: Camera.Camera=None,
                 mpi_optimization: bool=False,
                 cuda_optimization: bool=False,
                 prioritize_memory: bool=False, 
                 wiener_constant: float = 1e-6, 
                 ):
        
        if illumination.dimensionality != optical_system.dimensionality:
            raise ValueError(f"Illumination and optical system dimensionality do not match: {illumination.dimensionality} vs {optical_system.dimensionality}.")
        
        if not pattern_estimation_method is None and pattern_estimation_method not in self.pattern_estimation_methods:
            raise ValueError(f"Pattern estimation method {self.pattern_estimation_method} is not supported. \
                                Supported methods are: {self.pattern_estimation_methods}.")
        
        if not deconvolution_method is None and deconvolution_method not in self.deconvolution_methods:
            raise ValueError(f"Deconvolution method {self.deconvolution_method} is not supported. \
                                Supported methods are: {self.deconvolution_methods}.")
        
        if not regularization_method is None and regularization_method not in self.regularizatoinion_methods:
            raise ValueError(f"Regularization method {self.regularization_method} is not supported. \
                                Supported methods are: {self.regularizatoinion_methods}.")
        
        if not apodization_method is None and apodization_method not in self.apodization_methods:
            raise ValueError(f"Apodization method {self.apodization_method} is not supported. \
                                Supported methods are: {self.apodization_methods}.")
        

        self.optical_system = optical_system
        self.illumination = illumination
        
        if not apodization_method is None:
            self.apodization = AutoconvolutionApodizationSIM(optical_system=optical_system, illumination=illumination)
        self.kernel = self.optical_system.psf if kernel is None else kernel
        effective_kernels, effective_kernels_ft = self.illumination.compute_effective_kernels(self.kernel, self.optical_system.psf_coordinates)
        effective_psfs, effective_otfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates) \
            if self.kernel else (effective_kernels, effective_kernels_ft)

        self.sim_dimension = len(self.illumination.dim)
        self.spatial_domain = spatial_domain
        self.wiener_constant = wiener_constant

        match self.sim_dimension: 
            case 2:
                simulator = SIMulator.SIMulator2D
                reconstructor = (Reconstructor.ReconstructorSpatialDomain2D if spatial_domain 
                                else Reconstructor.ReconstructorFourierDomain2D)
                ssnr_calculator = SSNRCalculator.SSNRSIM2D

            case 3:
                simulator = SIMulator.SIMulator3D
                reconstructor = (Reconstructor.ReconstructorSpatialDomain3D if spatial_domain 
                                else Reconstructor.ReconstructorFourierDomain3D)
                ssnr_calculator = SSNRCalculator.SSNRSIM3D

            case _:
                raise ValueError(f"Invalid dimensionality {len(self.illumination.dim)}. Supported dimensions are 2D and 3D.")
            
        if not estimate_patterns_from_data is None: 
            if pattern_estimation_method not in self.pattern_estimation_methods:
                raise ValueError(f"Pattern estimation method {pattern_estimation_method} is not supported. Supported methods are: {self.deconvolution_methods}.")
            
            match pattern_estimation_method:
                case 'CrossCorrelation':
                    pass 

        if estimate_ssnr:
            self.ssnr_calculator = ssnr_calculator(
                                                   optical_system=self.optical_system,
                                                   illumination=self.illumination, 
                                                   kernel = self.kernel,
                                                   effective_otfs = effective_otfs,
                                                   effective_kernels = effective_kernels_ft,
                                                   )

        self.simulator = simulator(
                 illumination=self.illumination,
                 optical_system=self.optical_system,
                 camera=camera,
                 effective_psfs=effective_psfs,
                 )
        
        self.reconstructor = reconstructor(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=self.kernel,
            effective_kernels=effective_kernels_ft, 
            return_ft= not spatial_domain,
        )

    def full_reconstruction(self, sim_images):
        """
        Full reconstruction of the super-resolution image using the provided SIM images.
        """
        sr_image = self.reconstructor.reconstruct(sim_images)

        return final_image
    
    def simulate_sim_images(self, true_object: np.ndarray, noisy:bool=True) -> np.ndarray:
        """
        Simulate SIM images using the provided illumination and optical system.
        """
        sim_images = self.simulator.generate_sim_images(true_object)
        if noisy:
            self.simulator.generate_noisy_images(images=sim_images)
        return sim_images
    
    def generate_super_resolution_image(self, sim_images: np.ndarray) -> np.ndarray:
        """
        Generate super-resolution image using the provided SIM images.
        """
        return self.reconstructor.reconstruct(sim_images)
    
    def estimate_total_ssnr_from_data(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate the total SSNR from the image by a method of the image splitting.
        """
        pass

    def compute_true_ssnr(self, true_object: np.ndarray) -> np.ndarray:
        """
        Compute the true SSNR given the object.
        """
        return self.ssnr_calculator.compute_ssnr(true_object)
    
    def deconvolve(self, image: np.ndarray) -> np.ndarray:
        """
        Deconvolve the image using the specified method.
        """
        deconvolved_image = image
        match self.deconvolution_method:
            case 'Wiener':
                image_ft = wrappers.wrapped_ifftn(image) if self.spatial_domain else image
                match self.regularization_method:
                    case 'TrueWiener':
                        deconvolved_image, w = WienerFiltering.WienerFilterTrue(
                            image_ft=image_ft, 
                            otf=self.ssnr_calculator.otf_sim, 
                            ssnr_calculator=self.ssnr_calculator,
                            average='rings',
                            numeric_noise=1e-12,
                        )
                    case 'Flat':
                        deconvolved_image, w = WienerFiltering.WienerFilterFlat(
                            image_ft=image_ft, 
                            otf=self.ssnr_calculator.otf_sim, 
                            ssnr_calculator=self.ssnr_calculator,
                        )
                    case 'Constant':
                        deconvolved_image, w = WienerFiltering.WienerFilterConstant(
                            image_ft=image_ft, 
                            otf=self.ssnr_calculator.otf_sim, 
                            w=self.wiener_constant,
                        )
                    case None:
                        raise ValueError(f"Regularization method is not defined. \
                                        Supported methods are: {self.regularizatoinion_methods}.")

            case 'Richardson-Lucy':     
                image = image if self.spatial_domain else wrappers.wrapped_ifftn(image)
                deconvolved_image = richardson_lucy_skimage(image, self.optical_system.psf)

            case 'Bayesian':
                image_ft = wrappers.wrapped_ifftn(image) if self.spatial_domain else image
                average_rings = stattools.average_rings2d if self.sim_dimension == 2 else stattools.average_ring_averages3d
                expand_rings = stattools.expand_ring_averages2d if self.sim_dimension == 3 else stattools.expand_ring_averages3d
                averages = average_rings(image_ft, self.optical_system.otf_frequencies)
                averages_expanded = expand_rings(averages, self.optical_system.otf_frequencies, image_ft.shape)
                deviations = image_ft - averages_expanded
                noise_power = average_rings(deviations ** 2)
                deconvolved_image = expand_rings(image_ft, noise_power, averages_expanded, self.optical_system.otf_frequencies)
                deconvolved_image = bayesian_gaussian_frequency_estimate(image_ft, noise_power, averages_expanded, self.ssnr_calculator.otf_sim)

            case 'MutualInformation':
                image_ft = wrappers.wrapped_ifftn(image) if self.spatial_domain else np.copy(image)
                average_rings = stattools.average_rings2d if self.sim_dimension == 2 else stattools.average_ring_averages3d
                expand_rings = stattools.expand_ring_averages2d if self.sim_dimension == 3 else stattools.expand_ring_averages3d
                averages = average_rings(image_ft, self.optical_system.otf_frequencies)
                averages_expanded = expand_rings(averages, self.optical_system.otf_frequencies, image_ft.shape)
                deviations = image_ft - averages_expanded
                noise_power = average_rings(deviations ** 2)
                deconvolved_image = expand_rings(image_ft, noise_power, averages_expanded, self.optical_system.otf_frequencies)
                deconvolved_image = image_of_maximal_surprise_estimate(image_ft, noise_power, averages_expanded, self.ssnr_calculator.otf_sim) 

            case None:
                raise ValueError(f"Deconvolution method is not defined. \
                                Supported methods are: {self.deconvolution_methods}.")

        return deconvolved_image
    
    def apodize(self, image: np.ndarray) -> np.ndarray:
        """
        Apodize the image using the specified method.
        """
        match self.apodization_method:
            case 'Lukosz':
                pass
            case 'Autoconvolution':
                pass
            case None:
                raise ValueError(f"Apodization method is not defined. \
                                Supported methods are: {self.apodization_methods}.")

        apodized_image = image
        return apodized_image

class ProcessorSIMPatchBased(ProcessorSIM):
    pass