"""
SIMulator.py

This module provides classes for simulating Structured Illumination Microscopy (SIM) images.
It includes functionality for generating raw SIM images from ground truth data,
adding noise to simulate realistic imaging conditions, and computing widefield images.

Classes:
    SIMulator - Base abstract class for SIM simulation
    SIMulator2D - 2D SIM image simulation
    SIMulator3D - 3D SIM image simulation
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from OpticalSystems import OpticalSystem
from abc import abstractmethod
import Sources
import hpc_utils
from Illumination import PlaneWavesSIM
from Camera import Camera
from VectorOperations import VectorOperations
from Dimensions import DimensionMeta, DimensionMetaAbstract

class SIMulator(metaclass=DimensionMetaAbstract):
    """
    Base class for simulating Structured Illumination Microscopy (SIM) images.
    
    This abstract base class provides the core functionality for SIM image simulation.
    It cannot be instantiated directly - use the dimensional subclasses (SIMulator2D, SIMulator3D) instead.
    
    The simulator generates raw SIM images from ground truth data by convolving with
    effective PSFs and applying phase modulation patterns corresponding to different
    illumination angles and phases.
    
    Attributes
    ----------
    illumination : PlaneWavesSIM
        The illumination configuration for the SIM experiment.
    optical_system : OpticalSystem
        The optical system used in the experiment.
    camera : Camera, optional
        The camera used to capture images. If None, noise is added using readout_noise_variance.
    readout_noise_variance : float
        The variance of readout noise to add to simulated images.
    effective_psfs : numpy.ndarray
        Precomputed effective PSFs for the simulation.
    phase_modulation_patterns : numpy.ndarray
        Precomputed phase modulation patterns for the reconstruction.
    """

    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0,
                 effective_psfs = None,
                 ):
        """
        Initialize the SIM simulator.

        Parameters
        ----------
        illumination : PlaneWavesSIM
            The illumination configuration for the SIM experiment.
        optical_system : OpticalSystem
            The optical system used in the experiment.
        camera : Camera, optional
            The camera used to capture images. If provided, noise will be added
            based on camera properties.
        readout_noise_variance : float, optional
            The variance of readout noise to add if no camera is provided.
            Default is 0 (no noise).
        effective_psfs : numpy.ndarray, optional
            Precomputed effective PSFs. If None, they will be computed from
            the optical system.
        """
        self.optical_system = optical_system
        self.illumination = illumination
        self.readout_noise_variance = readout_noise_variance
        self.camera = camera
        self.readout_noise_variance = 0
        if not effective_psfs:
            self.effective_psfs, _ = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        else:
            self.effective_psfs = effective_psfs
        self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

    def generate_noiseless_sim_images(self, ground_truth):
        """
        Generate noiseless SIM images from ground truth data.

        This method simulates the raw SIM images that would be captured by a microscope
        without adding any noise. The simulation convolves the ground truth with effective
        PSFs and applies phase modulation patterns for different illumination conditions.

        Parameters
        ----------
        ground_truth : numpy.ndarray
            The ground truth image to simulate SIM imaging from.

        Returns
        -------
        numpy.ndarray
            Array of shape (Mr, Mt, *psf_shape) containing the simulated noiseless SIM images,
            where Mr is the number of rotations and Mt is the number of phases per rotation.
        """
        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        # sim_images_ft = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        for sim_index in self.illumination.rearranged_indices:
            for n in range(self.illumination.Mt):
                total_phase_modulation = self.phase_modulation_patterns[sim_index] * self.illumination.phase_matrix[(sim_index[0], n, sim_index[1])]
                sim_images[sim_index[0], n] += scipy.signal.convolve(total_phase_modulation * ground_truth, self.phase_modulation_patterns[sim_index].conjugate() * self.effective_psfs[sim_index], mode='same')
                    
        sim_images = np.real(sim_images) + 10**-10
        return sim_images

    def add_noise(self, images):
        """
        Add noise to images to simulate realistic imaging conditions.

        If a camera object is provided, uses the camera's noise model.
        Otherwise, adds Poisson noise and Gaussian readout noise.

        Parameters
        ----------
        images : numpy.ndarray
            The input images to add noise to.

        Returns
        -------
        numpy.ndarray
            Images with added noise.
        """
        if self.camera:
            for one_rotation in range(self.illumination.Mr):
                for image in one_rotation:
                    image[one_rotation] = self.camera.get_image(image[one_rotation], self.readout_noise_variance)
        else:
            images = (np.random.poisson(images) +
                                  np.sqrt(self.readout_noise_variance))
            sigma = np.sqrt(self.readout_noise_variance)
            images += np.random.normal(loc=0.0, scale=sigma, size=images.shape)
        return images

    def generate_noisy_sim_images(self, ground_truth):
        """
        Generate noisy SIM images from ground truth data.

        This is a convenience method that first generates noiseless SIM images
        and then adds noise to simulate realistic imaging conditions.

        Parameters
        ----------
        ground_truth : numpy.ndarray
            The ground truth image to simulate SIM imaging from.

        Returns
        -------
        numpy.ndarray
            Array of shape (Mr, Mt, *psf_shape) containing the simulated noisy SIM images.
        """
        noiseless_images = self.generate_noiseless_sim_images(ground_truth)
        noisy_images = self.add_noise(noiseless_images)
        return noisy_images
    
    def generate_widefield(self, image):
        """
        Generate a widefield image by convolving with the optical system's PSF.

        This simulates what would be observed in conventional widefield microscopy
        without structured illumination.

        Parameters
        ----------
        image : numpy.ndarray
            The input image to convolve with the PSF.

        Returns
        -------
        numpy.ndarray
            The widefield image after convolution with the optical system's PSF.
        """
        widefield_image = scipy.signal.convolve(image, self.optical_system.psf, mode='same')
        return widefield_image


class SIMulator2D(SIMulator):
    """
    2D SIM image simulator.
    
    This class provides SIM simulation specifically for 2D images.
    It enforces that the optical system is 2D and inherits all simulation
    functionality from the base SIMulator class.
    """
    dimensionality = 2
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0,
                 effective_psfs = None,
                 ):
        if not optical_system.dimensionality == 2:
            raise ValueError("The PSF must be 2D for 2D SIM simulations.")
        super().__init__(illumination, optical_system, camera, readout_noise_variance, effective_psfs)


class SIMulator3D(SIMulator):
    """
    3D SIM image simulator.
    
    This class provides SIM simulation specifically for 3D images.
    It enforces that the optical system is 3D and inherits all simulation
    functionality from the base SIMulator class.
    """
    dimensionality = 3
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0,
                 effective_psfs = None,
                 ):
        if not optical_system.dimensionality == 3:
            raise ValueError("The PSF must be 3D for 3D SIM simulations.")
        super().__init__(illumination, optical_system, camera, readout_noise_variance, effective_psfs)

