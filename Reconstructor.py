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
import scipy, cupy
from abc import abstractmethod
from OpticalSystems import OpticalSystem, OpticalSystem2D, OpticalSystem3D
import Sources
import hpc_utils
import hpc_utils
from Box import BoxSIM, Field
from Illumination import PlaneWavesSIM, IlluminationPlaneWaves2D, IlluminationPlaneWaves3D, IlluminationNPhotonSIM2D
from VectorOperations import VectorOperations
from mpl_toolkits.mplot3d import axes3d
from Dimensions import DimensionMetaAbstract
import utils
from windowing import make_mask_cosine_edge2d
import multiprocessing as mp
import psutil

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
                 unitary=False,
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
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            unitary=(unitary, bool),
        )
        if optical_system is not None and not isinstance(optical_system, OpticalSystem):
            raise TypeError(f"optical_system must be of type OpticalSystem when provided, got {type(optical_system).__name__}.")
        if kernel is not None and not isinstance(kernel, np.ndarray):
            raise TypeError(f"kernel must be of type ndarray when provided, got {type(kernel).__name__}.")
        if phase_modulation_patterns is not None and not isinstance(phase_modulation_patterns, (dict, np.ndarray)):
            raise TypeError(
                "phase_modulation_patterns must be of type dict or ndarray when provided, "
                f"got {type(phase_modulation_patterns).__name__}."
            )
        self.illumination = illumination
        self._optical_system = optical_system

        self._kernel = kernel
        self.phase_modulation_patterns = phase_modulation_patterns
        self.unitary = unitary

        if phase_modulation_patterns:
            self.phase_modulation_patterns = phase_modulation_patterns
        else:
            if self.optical_system is None:
                raise AttributeError("If phase modulation patterns are not provided, optical system must be provided to compute them.")
            self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

    def _compute_effective_kernels(self):
        """
        Compute the effective kernels for the reconstruction.

        Parameters
        ----------
        kernel : numpy.ndarray
            The kernel used for convolution in spatial domain reconstruction.

        Returns
        -------
        dict
            A dictionary of effective kernels for each rotation and modulation index.
        """
        if self.optical_system is None:
            raise AttributeError("Optical system must be provided to compute effective kernels.")
        if self.kernel is None:
            _, self.effective_kernels = self.illumination.compute_effective_kernels(self.kernel, self.optical_system.psf_coordinates)
        else:
            _, self.effective_kernels = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)

    @property
    def optical_system(self):
        return self._optical_system
    
    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self._compute_effective_kernels()

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel):
        self._kernel = new_kernel
        self._compute_effective_kernels()

    @abstractmethod
    def reconstruct(self, sim_images):
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
        Returns
        -------
        numpy.ndarray
            The reconstructed image.
        """
        ...
    
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
                 unitary=False,
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
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            return_ft=(return_ft, bool),
            unitary=(unitary, bool),
        )
        if optical_system is not None and not isinstance(optical_system, OpticalSystem):
            raise TypeError(f"optical_system must be of type OpticalSystem when provided, got {type(optical_system).__name__}.")
        if kernel is not None and not isinstance(kernel, np.ndarray):
            raise TypeError(f"kernel must be of type ndarray when provided, got {type(kernel).__name__}.")
        if phase_modulation_patterns is not None and not isinstance(phase_modulation_patterns, (dict, np.ndarray)):
            raise TypeError(
                "phase_modulation_patterns must be of type dict or ndarray when provided, "
                f"got {type(phase_modulation_patterns).__name__}."
            )
        if effective_kernels is not None and not isinstance(effective_kernels, (dict, np.ndarray)):
            raise TypeError(
                f"effective_kernels must be of type dict or ndarray when provided, got {type(effective_kernels).__name__}."
            )
        
        super().__init__(illumination,
                         optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns, 
                         unitary=unitary)
        
        if kernel is not None: 
            self.kernel = utils.expand_kernel(kernel, self.optical_system.otf.shape)

        if effective_kernels:
            self.effective_kernels = effective_kernels
            # for sim_index in self.effective_kernels:
                # plt.imshow(np.log1p(10**3 * np.abs(self.effective_kernels[sim_index])).T, cmap='gray', origin='lower')
                # plt.title(f"Effective kernel in the reconstructor {sim_index}")
                # plt.show()
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
        # plt.imshow(new_kernel.imag)
        # plt.show()
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
        phase_shifted = image * self.phase_modulation_patterns[r, m].conjugate()
        shifted_image_ft = hpc_utils.wrapped_fftn(phase_shifted)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(np.log1p(10**8 * np.abs(shifted_image_ft.T)), cmap='gray', origin='lower')
        # ax[0].set_title(f'FT of image r={r}, m={m}')
        # ax[1].imshow(np.log1p(10**8 * np.abs(self.effective_kernels[r, m].T)), cmap='gray', origin='lower')
        # ax[1].set_title(f'Effective kernel r={r}, m={m}')
        # plt.show()
        return shifted_image_ft

    def reconstruct(self, sim_images):
        """
        Explicitely performs SIM reconstruction in the Fourier domain.
        """
        # Notations are as in C. Smith et al., "Structured illumination microscopy with noise-controlled image reconstructions", 2021
        reconstructed_image_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
        for r in range(sim_images.shape[0]):
            image1rotation_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            for _, sim_index in zip(*self.illumination.get_wavevectors_projected(r)):
                m = sim_index[1]
                # Jrm = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                sum_shifts = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                    # plt.imshow(np.log1p(10**8 * np.abs(image_shifted_ft)), cmap='gray')
                    # plt.title(f'FT of image r={r}, m={m}')
                    # plt.show()
                    #replace with inverse
                for n in range(sim_images.shape[1]):
                    if self.unitary:
                        sum_shifts += self.illumination.phase_matrix[(r, n, m)].conjugate() * sim_images[r, n]
                    else:
                        sum_shifts += self.illumination.Mt * self.illumination.phase_matrix_inverse[(r, n, m)] * sim_images[r, n]
                sum_shifts_ft = self._compute_shifted_image_ft(sum_shifts, r, m)

                        # fig, ax = plt.subplots(1, 2)
                        # ax[0].imshow(np.log1p(10**8 * np.abs(image_shifted_ft.T)), cmap='gray', origin='lower')
                        # ax[0].set_title(f'FT of image r={r}, m={m}')
                        # ax[1].imshow(np.log1p(10**8 * np.abs(self.effective_kernels[r, m].T)), cmap='gray', origin='lower')
                        # ax[1].set_title(f'Effective kernel r={r}, m={m}')
                        # plt.show()
                # m_inv = tuple([-mi for mi in m])
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.log(1 + 10**8 * np.abs(sum_shifts_ft[..., sum_shifts_ft.shape[2]//2]).T), cmap='gray', origin='lower')
                ax[0].set_title(f'R={r}, m={m}')
                ax[1].imshow(np.log(1 + 10**8 * np.abs(self.effective_kernels[(r, m)][..., sum_shifts_ft.shape[2]//2].T)), cmap='gray', origin='lower')
                ax[1].set_title(f'kernel in the rec R={r}, m={m}')
                plt.show()
                image1rotation_ft += sum_shifts_ft * self.effective_kernels[(r, m)].conjugate()

            reconstructed_image_ft += image1rotation_ft
        # plt.imshow(np.log(1 + 10**8 * np.abs(reconstructed_image_ft[..., 10].T)), cmap='gray', origin='lower')
        # plt.show()
        if self.return_ft:
            return reconstructed_image_ft
        reconstructed_image = np.abs(hpc_utils.wrapped_ifftn(reconstructed_image_ft))
        return reconstructed_image

    def compute_spatial_frequency_orders(self, sim_images):
        """
        Compute the spatial frequency orders present in the SIM images.
        Parameters        ----------
        sim_images : numpy.ndarray
        Returns
        -------
        dict
            A dictionary of estimated unmodulated spatial frequency orders.
        """
        spatial_frequency_orders = {}
        for r in range(sim_images.shape[0]):
            for _, sim_index in zip(*self.illumination.get_wavevectors_projected(r)):
                m = sim_index[1]
                sum_shifts = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                for n in range(sim_images.shape[1]):
                    if self.unitary:
                        sum_shifts += self.illumination.phase_matrix[(r, n, m)].conjugate() * sim_images[r, n]
                    else:
                        sum_shifts += self.illumination.Mt * self.illumination.phase_matrix_inverse[(r, n, m)] * sim_images[r, n]
                sum_shifts_ft = self._compute_shifted_image_ft(sum_shifts, r, m)
                spatial_frequency_orders[(r, m)] = sum_shifts_ft
        return spatial_frequency_orders

    
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
                 unitary=False,
                 **kwargs
                 ):
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            unitary=(unitary, bool),
        )
        if optical_system is not None and not isinstance(optical_system, OpticalSystem):
            raise TypeError(f"optical_system must be of type OpticalSystem when provided, got {type(optical_system).__name__}.")
        if kernel is not None and not isinstance(kernel, np.ndarray):
            raise TypeError(f"kernel must be of type ndarray when provided, got {type(kernel).__name__}.")
        if phase_modulation_patterns is not None and not isinstance(phase_modulation_patterns, (dict, np.ndarray)):
            raise TypeError(
                "phase_modulation_patterns must be of type dict or ndarray when provided, "
                f"got {type(phase_modulation_patterns).__name__}."
            )
        
        super().__init__(illumination,
                         optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                         unitary=unitary
                         )

        if kernel is None:
            otf_shape = np.array(self.optical_system.otf.shape, dtype=np.int32)
            kernel = np.zeros(otf_shape, dtype=np.float64)
            kernel[otf_shape[0] // 2 + 1, otf_shape[1] // 2 + 1] = 1

        self.kernel = kernel
        if phase_modulation_patterns is None:
            self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

        self.modulation_patterns = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        for n in range(self.illumination.Mt):
            for harmonic in self.illumination.harmonics:
                r = harmonic[0]
                # self.modulation_patterns[r, n] = self.illumination.get_illumination_density(self.optical_system.x_grid, r=r, n=n)
                m = tuple([harmonic[1][dimension] for dimension in range(len(self.illumination.dimensions)) if self.illumination.dimensions[dimension]])

                if self.unitary:
                    self.modulation_patterns[r, n] += (self.illumination.harmonics[harmonic].amplitude.conjugate() * self.illumination.phase_matrix[(r, n, m)].conjugate() * self.phase_modulation_patterns[harmonic].conjugate())
                else:
                    self.modulation_patterns[r, n] += (self.illumination.harmonics[harmonic].amplitude.conjugate() * self.illumination.phase_matrix_inverse[(r, n, m)] * self.phase_modulation_patterns[harmonic].conjugate()) * self.illumination.Mt
                
        self.modulation_patterns = np.array(self.modulation_patterns, dtype=np.float64)
        # plt.imshow(self.modulation_patterns[0, 0])
        # plt.show()
    @ReconstructorSIM.kernel.setter
    def kernel(self, new_kernel):
        self._kernel = new_kernel

    def reconstruct(self, sim_images):
        """
        Explicitely performs SIM reconstruction in the spatial domain.
        """
        reconstructed_image = np.zeros(sim_images.shape[2:], dtype=np.float64)
        for r in range(sim_images.shape[0]):
            image1rotation = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            for n in range(sim_images.shape[1]):

                image_convolved = hpc_utils.convolve2d(
                    sim_images[r, n],  self.kernel.conjugate(), mode='same', boundary='wrap'
                )
                image1rotation += self.modulation_patterns[r, n] * image_convolved
                # plt.imshow(sim_images[r, n], cmap='gray')
                # plt.show()
                # plt.imshow(image1rotation.real, cmap='gray')
                # plt.show()
                del image_convolved

            # plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(image1rotation.to_numpy()))))
            # plt.show() 
            reconstructed_image += image1rotation.real
            del image1rotation

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
                 unitary=False
                 ):

        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem2D):
            raise TypeError("optical_system must be an instance of OpticalSystem2D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns, 
                         effective_kernels=effective_kernels,
                         return_ft=return_ft,
                         unitary=unitary
        )


class ReconstructorFourierDomain2D4F(ReconstructorFourierDomain2D):
    """
    A specialized 2D Fourier domain reconstructor for a specialized scheme of SIM reconstructions 
    with the reduced number of images "...". It is assumed that the first image in a stack is widefield, 
    and other images are images obtained with structured illumination (1 per rotation). 
    It is also assumed that the illumination is a conventional 2-beams illumination. 
    """

    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system: OpticalSystem2D = None,
                 phase_modulation_patterns=None,
                 return_ft=False,
                 unitary=False
                 ):

        illumination_expanded = IlluminationNPhotonSIM2D.init_from_linear_illumination(illumination, nphoton=2)
        for harmonic in illumination_expanded.harmonics:
            illumination_expanded.harmonics[harmonic].amplitude = 1
        effective_kernels = illumination_expanded.compute_effective_kernels(optical_system.psf, optical_system.psf_coordinates)[1]
        for kernel in effective_kernels:
            print(kernel, np.amax(np.abs(effective_kernels[kernel])))
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=None,
                         effective_kernels=effective_kernels,
                         phase_modulation_patterns=phase_modulation_patterns, 
                         return_ft=return_ft,
                         unitary=unitary
        )

    def reconstruct(self, sim_images, numeric_noise = 1e-3):
        widefield_image = sim_images[0]
        widefield_image_ft = hpc_utils.wrapped_fftn(widefield_image)
        otf_zero = self.effective_kernels[(0, (0, 0))] 
        widefield_image_ft = np.where(otf_zero > numeric_noise, widefield_image_ft, 0)
        print('widefield_image_ft', np.amax(np.abs(widefield_image_ft)))
        object_zero_ft = widefield_image_ft / (otf_zero + numeric_noise) * self.illumination.Mr * self.illumination.Mt
        print('object_zero_ft', np.amax(np.abs(object_zero_ft)))
        reconstructed_ft = np.copy(object_zero_ft) * otf_zero**2
        object_zero = hpc_utils.wrapped_ifftn(object_zero_ft)
        for r in range(self.illumination.Mr):
            object_minus_ft = self._compute_shifted_image_ft(object_zero, r, (-2, 0)) 
            # print(np.amax(np.abs(object_minus_ft)))
            sim_image = sim_images[r + 1]
            sim_image_ft = hpc_utils.wrapped_fftn(sim_image)
            sim_image_ft = np.where(np.abs(otf_zero) > numeric_noise, sim_image_ft, 0)
            # plt.plot((sim_image_ft / (otf_zero + numeric_noise))[:, sim_image_ft.shape[1] // 2])
            # plt.show()
            otf_p = self.effective_kernels[(r, (2, 0))]
            otf_m = self.effective_kernels[(r, (-2, 0))]
            otf_pp = self.effective_kernels[(r, (4, 0))]
            otf_mm = self.effective_kernels[(r, (-4, 0))]
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.log1p(10**8 * np.abs(otf_m)).T, cmap='gray', origin='lower')
            # ax[0].set_title(f'OTF minus r={r}, m=(-2, 0)')
            # ax[1].imshow(np.log1p(10**8 * np.abs(otf_mm)).T, cmap='gray', origin='lower')
            # ax[1].set_title(f'OTF minus minus r={r}, m=(-4, 0)')
            # plt.show()
            otf_mm = self.effective_kernels[(r, (-4, 0))]
            a0 = self.illumination.harmonics[(r, (0, 0))].amplitude
            a_m = self.illumination.harmonics[(r, (-2, 0))].amplitude
            a_p = self.illumination.harmonics[(r, (2, 0))].amplitude
            area_of_interest_mask = np.where((np.abs(otf_zero) > numeric_noise) & (np.abs(otf_m) > numeric_noise), 1, 0)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.log1p(10**8 * np.abs(object_minus_ft.T)), cmap='gray', origin='lower')
            # ax[0].set_title(f'Shifted widefield image r={r}, m=(-2, 0)')
            # ax[1].imshow(area_of_interest_mask.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            # ax[1].set_title(f'Area of interest mask r={r}')
            # plt.show()
            structured_part = (sim_image_ft - widefield_image_ft)
            print('sim_image', np.amax(np.abs(sim_image_ft / (otf_zero + numeric_noise))))
            print('structured_part', np.amax(np.abs(structured_part / (otf_zero + numeric_noise))))
            print('object_minus_ft rescaled', np.amax(np.abs(object_minus_ft * a_m)))
            object_plus_shifted_ft = (structured_part * area_of_interest_mask / (otf_zero + numeric_noise) - object_minus_ft * a_m * area_of_interest_mask) / a_p
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(np.log1p(10**8 * np.abs(structured_part.T)), cmap='gray', origin='lower')
            # ax[0].set_title(f'Structured part r={r}')
            # ax[1].imshow(np.log1p(10**8 * np.abs(object_plus_shifted_ft.T)), cmap='gray', origin='lower')
            # ax[1].set_title(f'Object plus shifted before rescaling r={r}')
            # ax[2].imshow(np.log1p(10**8 * np.abs(object_minus_ft.T)), cmap='gray', origin='lower')
            # ax[2].set_title(f'Object minus shifted r={r}')
            # plt.show()
            object_plus = hpc_utils.wrapped_ifftn(object_plus_shifted_ft)
            # plt.imshow(np.log1p(10**8 * np.abs(object_plus_shifted_ft.T)), cmap='gray', origin='lower')
            # plt.title(f'Object plus r={r}, m=(2, 0)')
            # plt.show()
            object_plus_ft = self._compute_shifted_image_ft(object_plus, r, (-2, 0))
            # object_plus_ft = np.where((np.abs(otf_minus) > numeric_noise) & (np.abs(otf_zero) < numeric_noise), object_plus_ft, 0)
            kernel_plus =  np.abs(otf_mm)**2 * np.abs(otf_m)**2 *  np.abs(a_p / a0)**2 / (2 * np.abs(otf_mm)**2 + np.abs(otf_m)**2 * np.abs(a_m / a0)**2 + numeric_noise) 
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.log1p(10**8 * np.abs(kernel_plus)).T, cmap='gray', origin='lower')
            # ax[0].set_title(f'Kernel plus r={r}, m=(2, 0)')
            # ax[1].imshow(np.log1p(10**8 * np.abs(object_plus_ft)).T, cmap='gray', origin='lower')
            # ax[1].set_title(f'Object plus r={r}, m=(2, 0)')
            # plt.show()
            reconstructed_ft += object_plus_ft * kernel_plus
            reconstructed_ft += np.flip(object_plus_ft, axis=(0, 1)).conjugate() * np.flip(kernel_plus, axis=(0, 1)).conjugate()  # Add the conjugate symmetric part
            # plt.imshow(np.log1p(10**8 * np.abs(reconstructed_ft.T)), cmap='gray', origin='lower')
            # plt.show()

        if self.return_ft:
            return reconstructed_ft * self.illumination.Mr * self.illumination.Mt
        reconstructed_image = np.abs(hpc_utils.wrapped_ifftn(reconstructed_ft))
        return reconstructed_image



class ReconstructorFourierDomain2D6F(ReconstructorFourierDomain2D):
    """
    A specialized 2D Fourier domain reconstructor for a specialized scheme of SIM reconstructions 
    with the reduced number of images "...". It is assumed that there are 2 images per rotation
    obtained with structured illumination with opposite phases, meaning that their sum gives a widefield image. 
    It is also assumed that the illumination is a conventional 2-beams illumination. 
    """

    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system: OpticalSystem2D = None,
                 phase_modulation_patterns=None,
                 return_ft=False,
                 unitary=False
                 ):

        illumination_expanded = IlluminationNPhotonSIM2D.init_from_linear_illumination(illumination, nphoton=2)
        for harmonic in illumination_expanded.harmonics:
            illumination_expanded.harmonics[harmonic].amplitude = 1
        effective_kernels = illumination_expanded.compute_effective_kernels(optical_system.psf, optical_system.psf_coordinates)[1]
        for kernel in effective_kernels:
            print(kernel, np.amax(np.abs(effective_kernels[kernel])))
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=None,
                         effective_kernels=effective_kernels,
                         phase_modulation_patterns=phase_modulation_patterns, 
                         return_ft=return_ft,
                         unitary=unitary
        )

    def reconstruct(self, sim_images, numeric_noise = 1e-3):
        otf_zero = self.effective_kernels[(0, (0, 0))]  
        reconstructed_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
        for r in range(self.illumination.Mr):
            otf_p = self.effective_kernels[(r, (2, 0))]
            otf_m = self.effective_kernels[(r, (-2, 0))]
            otf_pp = self.effective_kernels[(r, (4, 0))]
            otf_mm = self.effective_kernels[(r, (-4, 0))]
            widefield_image = np.sum(sim_images[r], axis=0)
            # plt.imshow(widefield_image, cmap='gray')
            # plt.show()
            widefield_image_ft = hpc_utils.wrapped_fftn(widefield_image)
            widefield_image_ft = np.where(otf_zero > numeric_noise, widefield_image_ft, 0)
            print('widefield_image_ft', np.amax(np.abs(widefield_image_ft)))
            object_zero_ft = widefield_image_ft / (otf_zero + numeric_noise) * self.illumination.Mr * self.illumination.Mt
            print('object_zero_ft', np.amax(np.abs(object_zero_ft)))
            reconstructed_ft += np.copy(object_zero_ft) * otf_zero**2
            object_zero = hpc_utils.wrapped_ifftn(object_zero_ft)
            object_minus_ft = self._compute_shifted_image_ft(object_zero, r, (-2, 0)) 
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.log1p(10**8 * np.abs(otf_m)).T, cmap='gray', origin='lower')
            # ax[0].set_title(f'OTF minus r={r}, m=(-2, 0)')
            # ax[1].imshow(np.log1p(10**8 * np.abs(otf_mm)).T, cmap='gray', origin='lower')
            # ax[1].set_title(f'OTF minus minus r={r}, m=(-4, 0)')
            # plt.show()
            otf_mm = self.effective_kernels[(r, (-4, 0))]
            a0 = self.illumination.harmonics[(r, (0, 0))].amplitude
            a_m = self.illumination.harmonics[(r, (-2, 0))].amplitude
            a_p = self.illumination.harmonics[(r, (2, 0))].amplitude
            area_of_interest_mask = np.where((np.abs(otf_zero) > numeric_noise) & (np.abs(otf_m) > numeric_noise), 1, 0)
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.log1p(10**8 * np.abs(object_minus_ft.T)), cmap='gray', origin='lower')
            # ax[0].set_title(f'Shifted widefield image r={r}, m=(-2, 0)')
            # ax[1].imshow(area_of_interest_mask.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            # ax[1].set_title(f'Area of interest mask r={r}')
            # plt.show()
            structured_part = hpc_utils.wrapped_fftn(sim_images[r][0] - sim_images[r][1])
            print('structured_part', np.amax(np.abs(structured_part / (otf_zero + numeric_noise))))
            print('object_minus_ft rescaled', np.amax(np.abs(object_minus_ft * a_m)))
            object_plus_shifted_ft = (structured_part * area_of_interest_mask / (otf_zero + numeric_noise) - object_minus_ft * a_m * area_of_interest_mask) / a_p
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(np.log1p(10**8 * np.abs(structured_part.T)), cmap='gray', origin='lower')
            # ax[0].set_title(f'Structured part r={r}')
            # ax[1].imshow(np.log1p(10**8 * np.abs(object_plus_shifted_ft.T)), cmap='gray', origin='lower')
            # ax[1].set_title(f'Object plus shifted before rescaling r={r}')
            # ax[2].imshow(np.log1p(10**8 * np.abs(object_minus_ft.T)), cmap='gray', origin='lower')
            # ax[2].set_title(f'Object minus shifted r={r}')
            # plt.show()
            object_plus = hpc_utils.wrapped_ifftn(object_plus_shifted_ft)
            # plt.imshow(np.log1p(10**8 * np.abs(object_plus_shifted_ft.T)), cmap='gray', origin='lower')
            # plt.title(f'Object plus r={r}, m=(2, 0)')
            # plt.show()
            object_plus_ft = self._compute_shifted_image_ft(object_plus, r, (-2, 0))
            # object_plus_ft = np.where((np.abs(otf_minus) > numeric_noise) & (np.abs(otf_zero) < numeric_noise), object_plus_ft, 0)
            kernel_plus =  np.abs(otf_mm)**2 * np.abs(otf_m)**2 *  np.abs(a_p / a0)**2 / (2 * np.abs(otf_mm)**2 + np.abs(otf_m)**2 * np.abs(a_m / a0)**2 + numeric_noise) 
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.log1p(10**8 * np.abs(kernel_plus)).T, cmap='gray', origin='lower')
            # ax[0].set_title(f'Kernel plus r={r}, m=(2, 0)')
            # ax[1].imshow(np.log1p(10**8 * np.abs(object_plus_ft)).T, cmap='gray', origin='lower')
            # ax[1].set_title(f'Object plus r={r}, m=(2, 0)')
            # plt.show()
            reconstructed_ft += object_plus_ft * kernel_plus
            reconstructed_ft += np.flip(object_plus_ft, axis=(0, 1)).conjugate() * np.flip(kernel_plus, axis=(0, 1)).conjugate()  # Add the conjugate symmetric part
            # plt.imshow(np.log1p(10**8 * np.abs(reconstructed_ft.T)), cmap='gray', origin='lower')
            # plt.show()

        if self.return_ft:
            return reconstructed_ft * self.illumination.Mr * self.illumination.Mt
        reconstructed_image = np.abs(hpc_utils.wrapped_ifftn(reconstructed_ft))
        return reconstructed_image

# Kept for reference, but the  idea didn't work. 
# class ReconstructorFourierDomain9FS(ReconstructorFourierDomain2D6F):
#     def reconstruct(self, sim_images, numeric_noise = 1e-3):
#         otf_zero = self.effective_kernels[(0, (0, 0))]  
#         reconstructed_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
#         a = self.illumination.harmonics[(0, (2, 0))].amplitude / self.illumination.harmonics[(0, (0, 0))].amplitude
#         for r in range(self.illumination.Mr):
#             otf_p = self.effective_kernels[(r, (2, 0))]
#             otf_m = self.effective_kernels[(r, (-2, 0))]
#             otf_pp = self.effective_kernels[(r, (4, 0))]
#             otf_mm = self.effective_kernels[(r, (-4, 0))]
#             I0 = hpc_utils.wrapped_fftn(sim_images[r][0])
#             I1 = hpc_utils.wrapped_fftn(sim_images[r][1])
#             I2 = hpc_utils.wrapped_fftn(sim_images[r][2])
#             Is = hpc_utils.wrapped_fftn(sim_images[r][0] * self.phase_modulation_patterns[(r, (-2, 0))])
#             mask = np.where((np.abs(otf_zero) > numeric_noise) & (np.abs(otf_p) > numeric_noise), 1, 0)
#             ephi = self.illumination.phase_matrix[(r, 1, (2, 0))]
#             emphi = self.illumination.phase_matrix[(r, 1, (-2, 0))]
#             a_s = 1 / a / (np.abs(otf_p) + numeric_noise)
#             a_s0 = (-a**2 - 1) / (3 * a**2) / (np.abs(otf_zero) + numeric_noise)
#             a_sphi = (1 - a**2 + ephi) / (3 * a**2) / (np.abs(otf_zero) + numeric_noise)
#             a_smphi = (1 - a**2 + emphi) / (3 * a**2) / (np.abs(otf_zero) + numeric_noise)
#             # fig, ax = plt.subplots(1, 2)
#             # im0 = ax[0].imshow(np.log1p(10**8 * np.abs(a_s.T *Is.T)), origin='lower')
#             # ax[0].set_title(f'Is r={r}')
#             # im1 = ax[1].imshow(np.log1p(10**8 * np.abs(a_s * Is + a_s0 * I0 + a_smphi * I1 + a_sphi * I2).T), origin='lower')
#             # ax[1].set_title(f'Composite r={r}')
#             # plt.colorbar(im1, ax=ax[1])
#             # plt.colorbar(im0, ax=ax[0])
#             # plt.show()

#             fmm = mask * (a_s * Is + a_s0 * I0 + a_smphi * I1 + a_sphi * I2)
#             fm = 1 / (3 * a) * (I0 + emphi * I1 + ephi * I2)
#             f0 = 1 / (3 ) * (I0 + I1 + I2)
#             fp = 1 / (3 * a) * (I0 + ephi * I1 + emphi * I2)

#             # fig, ax = plt.subplots(1, 4)
#             # ax[0].imshow(np.log1p(10**3 * np.abs(fmm.T)) * mask.T, cmap='gray', origin='lower')
#             # ax[0].set_title(f'Fmm r={r}')
#             # ax[1].imshow(np.log1p(10**8 * np.abs(fm .T)) , cmap='gray', origin='lower')
#             # ax[1].set_title(f'Fm r={r}')
#             # ax[2].imshow(np.log1p(10**8 * np.abs(f0 .T)), cmap='gray', origin='lower')
#             # ax[2].set_title(f'F0 r={r}')
#             # ax[3].imshow(np.log1p(10**8 * np.abs(fp .T)), cmap='gray', origin='lower')
#             # ax[3].set_title(f'Fp r={r}')
#             # plt.show()

#             fmm_shifted = hpc_utils.wrapped_fftn(self.phase_modulation_patterns[(r, (2, 0))]**2 * hpc_utils.wrapped_ifftn(fmm))
#             fm_shifted = hpc_utils.wrapped_fftn(self.phase_modulation_patterns[(r, (-2, 0))] * hpc_utils.wrapped_ifftn(fm))
#             f0_shifted = f0
#             fp_shifted = hpc_utils.wrapped_fftn(self.phase_modulation_patterns[(r, (2, 0))] * hpc_utils.wrapped_ifftn(fp))

#             # fig, ax = plt.subplots(1, 4)
#             # ax[0].imshow(np.log1p(10**3 * np.abs(fmm_shifted.T)), cmap='gray', origin='lower')
#             # ax[0].set_title(f'Fmm r={r}')
#             # ax[1].imshow(np.log1p(10**8 * np.abs(fm_shifted.T)) , cmap='gray', origin='lower')
#             # ax[1].set_title(f'Fm r={r}')
#             # ax[2].imshow(np.log1p(10**8 * np.abs(f0_shifted.T)), cmap='gray', origin='lower')
#             # ax[2].set_title(f'F0 r={r}')
#             # ax[3].imshow(np.log1p(10**8 * np.abs(fp_shifted.T)), cmap='gray', origin='lower')
#             # ax[3].set_title(f'Fp r={r}')
#             # plt.show()

#             K2 = a**4 * np.abs(otf_m)**2 * np.abs(otf_mm)**2 / ((a**4 + 1) *np.abs(otf_m)**2 + 3 * a**2 * otf_mm**2 + numeric_noise)
#             K1 = a**2 * np.abs(otf_m)**2 
#             Km1 = a**2 * np.abs(otf_p)**2 
#             K0 = a**2 * np.abs(otf_zero)**2 

#             # fig, ax = plt.subplots(1, 4)
#             # ax[0].imshow(np.log1p(10**8 * np.abs(K2.T)), cmap='gray', origin='lower')
#             # ax[0].set_title(f'K2 r={r}')
#             # ax[1].imshow(np.log1p(10**8 * np.abs(K1.T)), cmap='gray', origin='lower')
#             # ax[1].set_title(f'K1 r={r}')
#             # ax[2].imshow(np.log1p(10**8 * np.abs(K0.T)), cmap='gray', origin='lower')
#             # ax[2].set_title(f'K0 r={r}')
#             # ax[3].imshow(np.log1p(10**8 * np.abs(Km1.T)), cmap='gray', origin='lower')
#             # ax[3].set_title(f'Km1 r={r}')
#             # # plt.show()

#             # fig, ax = plt.subplots(1, 4)
#             # ax[0].imshow(np.log1p(10**3 * np.abs(K2.T * fmm_shifted.T)), cmap='gray', origin='lower')
#             # ax[0].set_title(f'Fmm r={r}')
#             # ax[1].imshow(np.log1p(10**8 * np.abs(Km1.T * fm_shifted.T)) , cmap='gray', origin='lower')
#             # ax[1].set_title(f'Fm r={r}')
#             # ax[2].imshow(np.log1p(10**8 * np.abs(K0.T * f0_shifted.T)), cmap='gray', origin='lower')
#             # ax[2].set_title(f'F0 r={r}')
#             # ax[3].imshow(np.log1p(10**8 * np.abs(K1.T * fp_shifted.T)), cmap='gray', origin='lower')
#             # ax[3].set_title(f'Fp r={r}')
#             # plt.show()

#             reconstructed_ft += Km1 * fm_shifted + K0 * f0_shifted + K1 * fp_shifted
#             reconstructed_ft += np.flip(K2 * fmm_shifted, axis=(0, 1)).conjugate() + K2 * fmm_shifted
#             Nx = sim_images.shape[2]
#             # reconstructed_ft[Nx//2+1:, :] = np.flip(reconstructed_ft[:Nx//2, :], axis=(0, 1)).conjugate()

#         plt.imshow(np.log1p(10**8 * np.abs(reconstructed_ft.T)), cmap='gray', origin='lower')
#         plt.show()

#         if self.return_ft:
#             return reconstructed_ft * self.illumination.Mr * self.illumination.Mt
#         reconstructed_image = np.abs(hpc_utils.wrapped_ifftn(reconstructed_ft))
#         return reconstructed_image

class ReconstructorSpatialDomain2D(ReconstructorSpatialDomain):

    dimensionality = 2
    
    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system: OpticalSystem2D = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 unitary=True
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem2D):
            raise TypeError("optical_system must be an instance of OpticalSystem2D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                         unitary=unitary
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
                 unitary=False
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves3D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves3D")
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem3D):
            raise TypeError("optical_system must be an instance of OpticalSystem3D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                         effective_kernels=effective_kernels,
                         return_ft=return_ft,
                         unitary=unitary
                         )

class ReconstructorSpatialDomain3D(ReconstructorSpatialDomain):

    dimensionality = 3
    
    def __init__(self,
                 illumination: IlluminationPlaneWaves3D,
                 optical_system: OpticalSystem3D = None,
                 kernel=None,
                 phase_modulation_patterns=None,
                 unitary=False, 
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves3D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves3D")
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem3D):
            raise TypeError("optical_system must be an instance of OpticalSystem3D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                         unitary=unitary
                        )
        
class ReconstructorSpatialDomain3DSliced(ReconstructorSpatialDomain):
    """ Slice by slice 3D reconstructor in spatial domain. """

    dimensionality = 3
    def __init__(self,
                illumination: IlluminationPlaneWaves2D,
                optical_system: OpticalSystem3D = None,
                kernel=None,
                phase_modulation_patterns=None,
                unitary=False
                    ):
        
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            if not isinstance(illumination, IlluminationPlaneWaves3D):
                raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
            else:
                illumination = IlluminationPlaneWaves2D.init_from_3D(illumination, force=True)
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem2D):
            if not isinstance(optical_system, OpticalSystem3D):
                raise TypeError("optical_system must be an instance of OpticalSystem2D")
            else: 
                optical_system = OpticalSystem2D.init_from_3D(optical_system)
        
        super().__init__(illumination=illumination,
                            optical_system=optical_system,
                            kernel=kernel,
                            phase_modulation_patterns=phase_modulation_patterns,
                            unitary=unitary
                        )
        
    def _reconstruct_slice(self, sim_images_slice):
        """ Reconstruct a separate slice of a 3D image"""
        sliced_reconstructed_image = np.zeros(sim_images_slice.shape[2:], dtype=np.float64)
        for r in range(sim_images_slice.shape[0]):
            image1rotation = np.zeros(sim_images_slice.shape[2:], dtype=np.float64)
            for n in range(sim_images_slice.shape[1]):
                image_convolved = hpc_utils.convolve2d(
                    sim_images_slice[r, n], self.kernel.conjugate(), mode='same', boundary='wrap'
                )
                temp =  self.modulation_patterns[r, n].conjugate() * image_convolved
                image1rotation += temp

                del image_convolved, temp  # free intermediates immediately
            sliced_reconstructed_image += image1rotation
            del image1rotation
        return sliced_reconstructed_image

    def reconstruct(self, sim_images, backend='gpu'):
        """
        This version of the function reconstructs a 3D image from a stack of SIM images. 
        Depending on the hardware, it is may be preferable to use gpu or cpu parallelization, 
        which are both implemented, albeit the CPU version is not well-tested yet. 
        """
        reconstructed_image = np.zeros(sim_images.shape[2:], dtype=np.float64)

        hpc_utils.pick_backend('cpu' if backend == 'cpu' else 'gpu')

        if backend != 'cpu':
            for z in range(sim_images.shape[-1]):
                reconstructed_image[:, :, z] = self._reconstruct_slice(sim_images[:, :, :, :, z])
                # plt.imshow(reconstructed_image[:, :, z], cmap='gray', origin='lower')
                # plt.show()
                # cupy.get_default_memory_pool().free_all_blocks()
            return reconstructed_image

        # ---- CPU multiprocessing path ----
        dtype_size = sim_images.dtype.itemsize
        memory_per_slice = (
            sim_images.shape[0] * (sim_images.shape[1] + 1) *
            sim_images[0, 0].size * dtype_size * 1.5
        )
        available_memory = psutil.virtual_memory().available
        num_cores = mp.cpu_count()

        max_workers_memory = max(1, int(available_memory * 0.7 / memory_per_slice))
        num_workers = min(num_cores, max_workers_memory, sim_images.shape[-1])

        if num_workers <= 1:
            for z in range(sim_images.shape[-1]):
                reconstructed_image[:, :, z] = self._reconstruct_slice(sim_images[:, :, :, :, z])
            return reconstructed_image

        # Use context manager; stream slices via imap to avoid materializing all at once
        with mp.Pool(processes=num_workers) as pool:
            slices = (sim_images[:, :, :, :, zi] for zi in range(sim_images.shape[-1]))
            for zi, result in enumerate(pool.imap(self._reconstruct_slice, slices, chunksize=1)):
                reconstructed_image[:, :, zi] = result

        return reconstructed_image
        
