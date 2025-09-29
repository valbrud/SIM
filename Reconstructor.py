"""
Reconstructor.py

This module provides classes for reconstructing images from Structured Illumination Microscopy (SIM) data.
It implements various reconstruction algorithms in both Fourier and spatial domains for 2D and 3D SIM imaging.

Classes:
    ReconstructorSIM - Base abstract class for SIM reconstruction
    ReconstructorFourierDomain - Fourier domain reconstruction implementation
    ReconstructorSpatialDomain - Spatial domain reconstruction implementation
    ReconstructorFourierDomain2D - 2D Fourier domain reconstruction
    ReconstructorSpatialDomain2D - 2D spatial domain reconstruction
    ReconstructorFourierDomain3D - 3D Fourier domain reconstruction
    ReconstructorSpatialDomain3D - 3D spatial domain reconstruction
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from abc import abstractmethod
from OpticalSystems import OpticalSystem, OpticalSystem2D, OpticalSystem3D
import Sources
import hpc_utils
import hpc_utils
from Box import BoxSIM, Field
from Illumination import PlaneWavesSIM, IlluminationPlaneWaves2D, IlluminationPlaneWaves3D
from VectorOperations import VectorOperations
from mpl_toolkits.mplot3d import axes3d
from Dimensions import DimensionMetaAbstract
import utils
from windowing import make_mask_cosine_edge2d

class ReconstructorSIM(metaclass=DimensionMetaAbstract):
    """
    Base class for reconstructing images from structured illumination microscopy (SIM) data.
    The base class implements all the functionality but cannot be implemented.
    Use dimensional children classes instead.

    Attributes:
        illumination: PlaneWavesSIM object, the illumination configuration for the SIM experiment.
        optical_system: OpticalSystem object, the optical system used in the experiment.
        kernel: numpy.ndarray, optional, the kernel used for convolution.
        phase_modulation_patterns: numpy.ndarray, optional, precomputed phase modulation patterns for the reconstruction.
        effective_kernels: numpy.ndarray, optional, precomputed effective kernels for the reconstruction.
    
    methods:
        reconstruct(sim_images): Reconstructs the image from the simulated images using the specified reconstruction method.
        get_widefield(sim_images): Computes the widefield image from the simulated images.
    """

    
    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 **kwargs
                 ):
        """
        Initialize the SIM reconstructor.

        Parameters
        ----------
        illumination : PlaneWavesSIM
            The illumination configuration for the SIM experiment.
        optical_system : OpticalSystem, optional
            The optical system used in the experiment.
        kernel : numpy.ndarray, optional
            The kernel used for convolution in spatial domain reconstruction.
        phase_modulation_patterns : numpy.ndarray, optional
            Precomputed phase modulation patterns for the reconstruction.
        **kwargs
            Additional keyword arguments.
        """
        self.illumination = illumination
        self._optical_system = optical_system

        self._kernel = kernel
        self.phase_modulation_patterns = phase_modulation_patterns

        if phase_modulation_patterns:
            self.phase_modulation_patterns = phase_modulation_patterns
        else:
            if self.optical_system is None:
                raise AttributeError("If phase modulation patterns are not provided, optical system must be provided to compute them.")
            self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

    @property
    def optical_system(self):
        return self._optical_system
    
    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        if self.kernel is None:
            _, self.effective_kernels = self.illumination.compute_effective_kernels(self.kernel, self.optical_system.psf_coordinates)
        else:
            _, self.effective_kernels = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
    @property
    def kernel(self):
        return self._kernel

    @abstractmethod
    def reconstruct(self, sim_images, upsample_factor=1):
        """
        Generate a row reconstructed image from SIM data.

        .. math::
           I^{SR} = h_n \\left( \\sum_n K \\ast I_n \\right)

        The image is super-resolved, but not deconvolved, 
        so that the contrast is expected to be low. 

        Parameters
        ----------
        sim_images : numpy.ndarray
            The SIM images to reconstruct from.
        upsample_factor: int
            The factor by which to upsample the reconstructed image.
        Returns
        -------
        numpy.ndarray
            The reconstructed image.
        """
        ...
    
    def upsample(self, sim_images, upsample_factor=1):
        """
        Upsample the SIM images by the given factor.

        Parameters
        ----------
        sim_images : numpy.ndarray
            The SIM images to upsample.
        factor : int, optional
            The upsampling factor. Default is 2.

        Returns
        -------
        numpy.ndarray
            The upsampled SIM images.
        """
        upsampled = np.array([np.array([utils.upsample(image, factor=upsample_factor, add_shot_noize=True) for image in  one_rotation]) for one_rotation in sim_images], dtype=np.float32)
        return upsampled
    
    def get_widefield(self, sim_images):
        """
        Compute the widefield image from SIM data by summing all images.

        Parameters
        ----------
        sim_images : numpy.ndarray
            The SIM images.

        Returns
        -------
        numpy.ndarray
            The widefield image.
        """
        widefield_image = np.zeros(sim_images.shape[2:])
        for rotation in sim_images:
            test = np.zeros(sim_images.shape[2:])
            for image in rotation:
                widefield_image += image
                test += image
                # plt.imshow(image)
                # plt.show()
        return widefield_image


class ReconstructorFourierDomain(ReconstructorSIM):
    """
    Fourier domain reconstruction for SIM images.
    
    This class implements SIM reconstruction in the Fourier domain using
    effective kernels and phase modulation patterns.
    """

    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 effective_kernels=None,
                 return_ft=False,
                 **kwargs
                 ):
        """
        Initialization is the same as of the base class + extra parameter that may be needed
        for numeric efficiency. 

        Parameters
        ----------
        effective_kernels : numpy.ndarray, optional
            Precomputed effective kernels for reconstruction.
        return_ft : bool, optional
            Whether to return the Fourier transform instead of the spatial image.

        """
        
        super().__init__(illumination,
                         optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns)
        
        if kernel is not None: 
            self.kernel = utils.expand_kernel(kernel, self.optical_system.otf.shape)

        if effective_kernels:
            self.effective_kernels = effective_kernels
        else:
            if self.optical_system is None:
                raise AttributeError("If effective kernels are not provided, optical system must be provided to compute them.")
            if kernel is None:
                _, self.effective_kernels = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
            else:
                _, self.effective_kernels = self.illumination.compute_effective_kernels(self.kernel, self.optical_system.psf_coordinates)
        self.return_ft = return_ft

    @ReconstructorSIM.kernel.setter
    def kernel(self, new_kernel):
        self._kernel = utils.expand_kernel(new_kernel, self.optical_system.otf.shape)
        _, self.effective_kernels = self.illumination.compute_effective_kernels(self._kernel, self.optical_system.psf_coordinates)


    def _compute_shifted_image_ft(self, image, r, m):
        """
        Compute the Fourier transform of a phase-shifted image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image.
        r : int
            Rotation index.
        m : int
            Modulation index.

        Returns
        -------
        numpy.ndarray
            The Fourier transform of the phase-shifted image.
        """
        phase_shifted = image * self.phase_modulation_patterns[r, m]
        shifted_image_ft = hpc_utils.wrapped_fftn(phase_shifted)
        return shifted_image_ft

    def reconstruct(self, sim_images, upsample_factor=1):
        """
        Explicitely performs SIM reconstruction in the Fourier domain.
        """
        if upsample_factor > 1:
            sim_images = self.upsample(sim_images, upsample_factor)
        # Notations are as in C. Smith et al., "Structured illumination microscopy with noise-controlled image reconstructions", 2021
        reconstructed_image_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
        for r in range(sim_images.shape[0]):
            image1rotation_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            for _, sim_index in zip(*self.illumination.get_wavevectors_projected(r)):
                m = sim_index[1]
                # Jrm = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                sum_shifts = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                for n in range(sim_images.shape[1]):
                    image_shifted_ft = self._compute_shifted_image_ft(sim_images[r, n], r, m)
                    sum_shifts += self.illumination.phase_matrix[(r, n, m)] * image_shifted_ft
                # plt.imshow(np.log(1 + 10**8 * np.abs(sum_shifts)))
                # plt.show()
                # plt.imshow(np.log(1 + 10**8 * np.abs(self.effective_kernels[(r, m)])))
                # plt.show()
                image1rotation_ft += sum_shifts * self.effective_kernels[(r, m)].conjugate()
            reconstructed_image_ft += image1rotation_ft
        # plt.imshow(np.log(1 + 10**8 * np.abs(reconstructed_image_ft)))
        # plt.show()
        if self.return_ft:
            return reconstructed_image_ft
        reconstructed_image = np.abs(hpc_utils.wrapped_ifftn(reconstructed_image_ft))
        return reconstructed_image


class ReconstructorSpatialDomain(ReconstructorSIM):
    """
    Spatial domain reconstruction for SIM images.
    
    This class implements SIM reconstruction in the spatial domain using
    convolution with illumination patterns.
    """

    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 **kwargs
                 ):
        
        super().__init__(illumination,
                         optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns
                         )

        if kernel is None:
            otf_shape = np.array(self.optical_system.otf.shape, dtype=np.int32)
            kernel = np.zeros(otf_shape, dtype=np.float64)
            kernel[otf_shape[0] // 2 + 1, otf_shape[1] // 2 + 1] = 1

        self.kernel = kernel
        if phase_modulation_patterns is None:
            self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

        self.illumination_patterns = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        for n in range(self.illumination.Mt):
            for harmonic in self.illumination.harmonics:
                r = harmonic[0]
                m = tuple([harmonic[1][dimension] for dimension in range(len(self.illumination.dimensions)) if self.illumination.dimensions[dimension]])
                self.illumination_patterns[r, n] += (self.illumination.harmonics[harmonic].amplitude * self.illumination.phase_matrix[(r, n, m)] * self.phase_modulation_patterns[harmonic])

        self.illumination_patterns = np.array(self.illumination_patterns, dtype=np.float64)

    @ReconstructorSIM.kernel.setter
    def kernel(self, new_kernel):
        self._kernel = new_kernel

    def reconstruct(self, sim_images, upsample_factor=1):
        """
        Explicitely performs SIM reconstruction in the spatial domain.
        """
        if upsample_factor > 1:
            sim_images = self.upsample(sim_images, upsample_factor)
        reconstructed_image = np.zeros(sim_images.shape[2:], dtype=np.float64)
        for r in range(sim_images.shape[0]):
            image1rotation = np.zeros(sim_images.shape[2:], dtype=np.float64)
            for n in range(sim_images.shape[1]):

                image_convolved = hpc_utils.convolve2d(
                    sim_images[r, n], self.kernel, mode='same', boundary='wrap'
                )
                image1rotation += self.illumination_patterns[r, n] * image_convolved

            # plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(image1rotation))))
            # plt.show() 
            reconstructed_image += image1rotation
        # mask = make_mask_cosine_edge2d(image1rotation.shape, 50)
        # reconstructed_image *= mask
        return reconstructed_image

class ReconstructorFourierDomain2D(ReconstructorFourierDomain):

    dimensionality = 2

    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system: OpticalSystem2D = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 effective_kernels=None,
                 return_ft=False,
                 **kwargs
                 ):

        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
        
        if not isinstance(optical_system, OpticalSystem2D):
            raise TypeError("optical_system must be an instance of OpticalSystem2D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns, 
                         effective_kernels=effective_kernels,
                         return_ft=return_ft,
        )

class ReconstructorSpatialDomain2D(ReconstructorSpatialDomain):

    dimensionality = 2
    
    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system: OpticalSystem2D = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 **kwargs
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
        
        if not isinstance(optical_system, OpticalSystem2D):
            raise TypeError("optical_system must be an instance of OpticalSystem2D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                        )
        
class ReconstructorFourierDomain3D(ReconstructorFourierDomain):

    dimensionality = 3
    
    def __init__(self,
                 illumination: IlluminationPlaneWaves3D,
                 optical_system: OpticalSystem3D = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 effective_kernels=None,
                 return_ft=False,
                 **kwargs
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves3D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves3D")
        
        if not isinstance(optical_system, OpticalSystem3D):
            raise TypeError("optical_system must be an instance of OpticalSystem3D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                         effective_kernels=effective_kernels,
                         return_ft=return_ft,
                         )

class ReconstructorSpatialDomain3D(ReconstructorSpatialDomain):

    dimensionality = 3
    
    def __init__(self,
                 illumination: IlluminationPlaneWaves3D,
                 optical_system: OpticalSystem3D = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 **kwargs
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves3D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves3D")
        
        if not isinstance(optical_system, OpticalSystem3D):
            raise TypeError("optical_system must be an instance of OpticalSystem3D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                        )