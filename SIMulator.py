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
import copy
import utils

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
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            optical_system=(optical_system, OpticalSystem),
            readout_noise_variance=(readout_noise_variance, (int, float, np.integer, np.floating)),
        )
        if camera is not None and not isinstance(camera, Camera):
            raise TypeError(f"camera must be of type Camera when provided, got {type(camera).__name__}.")
        if effective_psfs is not None and not isinstance(effective_psfs, (dict, np.ndarray)):
            raise TypeError(
                f"effective_psfs must be of type dict or ndarray when provided, got {type(effective_psfs).__name__}."
            )
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

    def generate_noiseless_sim_images(self, ground_truth, debug=False):
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
                if debug:
                    print(sim_index, n)
                total_phase_modulation = self.phase_modulation_patterns[sim_index] * self.illumination.phase_matrix[(sim_index[0], n, sim_index[1])]
                sum = np.sum(self.effective_psfs[sim_index])
                sim_images[sim_index[0], n] += scipy.signal.convolve(total_phase_modulation * ground_truth, self.phase_modulation_patterns[sim_index] * self.effective_psfs[sim_index], mode='same')
                    
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
    
    @abstractmethod
    def generate_noiseless_thick_sample_sim_images(self, ground_truth, z_values, zernike = {}):
        """
        Abstract method to generate noiseless SIM images for a thick sample.

        This method should be implemented by subclasses to simulate SIM imaging
        of thick samples, where the ground truth is 3D and the PSF varies with depth.

        Parameters
        ----------
        ground_truth : numpy.ndarray
            The 3D ground truth image to simulate SIM imaging from.
        z_values : numpy.ndarray
            The z-values corresponding to each slice of the ground truth and PSF stack.
        zernike : dict, optional
            A dictionary of Zernike coefficients to apply to the PSF for each slice.

        Returns
        -------
        numpy.ndarray
            Array of shape (Mr, Mt, *psf_shape) containing the simulated noiseless SIM images for the thick sample.
        """
        pass

    def generate_noisy_thick_sample_sim_images(self, ground_truth, z_values, zernike = {}):
        """
        Generate noisy SIM images for a thick sample.

        This is a convenience method that first generates noiseless SIM images for a thick sample
        and then adds noise to simulate realistic imaging conditions.

        Parameters
        ----------
        ground_truth : numpy.ndarray
            The 3D ground truth image to simulate SIM imaging from.

        Returns
        -------
        numpy.ndarray
            Array of shape (Mr, Mt, *psf_shape) containing the simulated noisy SIM images for the thick sample.
        """
        noiseless_images = self.generate_noiseless_thick_sample_sim_images(ground_truth, z_values, zernike=zernike)
        noisy_images = self.add_noise(noiseless_images)
        return noisy_images

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


    def generate_noiseless_thick_sample_sim_images(self, ground_truth, psf_stack): 
        sim_images_stack = np.zeros((ground_truth.shape[0], self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        effective_psfs_default = self.effective_psfs
        for i, psf in enumerate(psf_stack):
            self.effective_psfs, _ = self.illumination.compute_effective_kernels(psf_stack[i], self.optical_system.psf_coordinates)
            sim_images_stack[i] = self.generate_noiseless_sim_images(ground_truth[..., i])
        
        sim_images = np.sum(sim_images_stack, axis=0)
        sim_images = np.real(sim_images) + 10**-10

        self.effective_psfs = effective_psfs_default

        return sim_images
        

    
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


    def generate_noiseless_thick_sample_sim_images(self, image3D, psf_stack, z_values): 
        x, y = self.optical_system.psf_coordinates
        grid3d = np.stack(np.meshgrid(x, y, np.array(z_values)), axis=-1)        
        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                illumination_density = self.illumination.get_illumination_density(grid3d, r=r, n=n)
                image_total = np.zeros((image3D.shape[0], image3D.shape[1]))
                for i, psf_slice in enumerate(psf_stack):
                    image_slice = scipy.signal.convolve(illumination_density[i] * image3D[..., i], psf_slice)
                    image_total += image_slice
                sim_images[r, n] = image_total
                    
        sim_images = np.real(sim_images) + 10**-10
        return sim_images