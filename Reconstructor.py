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
from Illumination import PlaneWavesSIM, IlluminationPlaneWaves2D, IlluminationPlaneWaves3D
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
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(np.log(1 + 10**8 * np.abs(sum_shifts_ft).T), cmap='gray', origin='lower')
                # ax[0].set_title(f'R={r}, m={m}')
                # ax[1].imshow(np.log(1 + 10**8 * np.abs(self.effective_kernels[(r, m)].T)), cmap='gray', origin='lower')
                # ax[1].set_title(f'kernel in the rec R={r}, m={m}')
                # plt.show()
                image1rotation_ft += sum_shifts_ft * self.effective_kernels[(r, m)].conjugate()

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
                 unitary=False,
                 **kwargs
                 ):
        
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
                    self.modulation_patterns[r, n] += (self.illumination.harmonics[harmonic]* self.illumination.phase_matrix[(r, n, m)].conjugate() * self.phase_modulation_patterns[harmonic].conjugate())
                else:
                    self.modulation_patterns[r, n] += (self.illumination.Mt * self.illumination.harmonics[harmonic].amplitude.conjugate() * self.illumination.phase_matrix_inverse[(r, n, m)] * self.phase_modulation_patterns[harmonic].conjugate())
                
        self.modulation_patterns = np.array(self.modulation_patterns, dtype=np.float64)

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
                    sim_images[r, n], self.kernel.conjugate(), mode='same', boundary='wrap'
                )
                image1rotation += self.modulation_patterns[r, n] * image_convolved
                del image_convolved

            # plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(image1rotation))))
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
                 **kwargs
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
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem2D):
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
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem3D):
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
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem3D):
            raise TypeError("optical_system must be an instance of OpticalSystem3D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         phase_modulation_patterns=phase_modulation_patterns,
                        )
        
class ReconstructorSpatialDomain3DSliced(ReconstructorSpatialDomain):
    #Slice by slice 3D reconstructor in spatial domain.

    dimensionality = 3
    def __init__(self,
                illumination: IlluminationPlaneWaves2D,
                optical_system: OpticalSystem3D = None,
                kernel=None,
                phase_modulation_patterns=None,
                **kwargs
                    ):
        
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            if not isinstance(illumination, IlluminationPlaneWaves3D):
                raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
            else:
                illumination = IlluminationPlaneWaves2D.init_from_3D(illumination)
        
        if not optical_system is None and not isinstance(optical_system, OpticalSystem2D):
            if not isinstance(optical_system, OpticalSystem3D):
                raise TypeError("optical_system must be an instance of OpticalSystem2D")
            else: 
                optical_system = OpticalSystem2D.init_from_3D(optical_system)
        
        super().__init__(illumination=illumination,
                            optical_system=optical_system,
                            kernel=kernel,
                            phase_modulation_patterns=phase_modulation_patterns,
                        )
        
    def _reconstruct_slice(self, sim_images_slice):
        sliced_reconstructed_image = np.zeros(sim_images_slice.shape[2:], dtype=np.float64)
        for r in range(sim_images_slice.shape[0]):
            image1rotation = np.zeros(sim_images_slice.shape[2:], dtype=np.float64)
            for n in range(sim_images_slice.shape[1]):
                image_convolved = hpc_utils.convolve2d(
                    sim_images_slice[r, n], self.kernel.conjugate(), mode='same', boundary='wrap'
                )
                temp = self.modulation_patterns[r, n] * image_convolved
                image1rotation += temp
                del image_convolved, temp  # free intermediates immediately
            sliced_reconstructed_image += image1rotation
            del image1rotation
        return sliced_reconstructed_image

    def reconstruct(self, sim_images, backend='gpu'):
        reconstructed_image = np.zeros(sim_images.shape[2:], dtype=np.float64)

        hpc_utils.pick_backend('cpu' if backend == 'cpu' else 'gpu')

        if backend != 'cpu':
            for z in range(sim_images.shape[-1]):
                reconstructed_image[:, :, z] = self._reconstruct_slice(sim_images[:, :, :, :, z])
                cupy.get_default_memory_pool().free_all_blocks()
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
        
def estimate_gain_and_offset(sim_images, edge_pixels=30):
    # Shapes and constants
    H, W = sim_images.shape[-2:]
    image_size = H * W  # FIX 1: correct number of pixels per image

    # Edge “strip” pixel count — matches your numerator's double-counting
    e = int(edge_pixels)
    if not (1 <= e <= min(H, W)//2):
        raise ValueError(f"edge_pixels must be in 1..{min(H,W)//2}")
    mask = np.zeros((H, W), dtype=bool)
    e = int(edge_pixels)
    mask[:e, :] = True
    mask[-e:, :] = True
    mask[:, :e] |= True
    mask[:, -e:] |= True
    edge_pixel_number = int(mask.sum())

    # Accumulators
    sI = np.zeros(sim_images.shape[:2], dtype=float)
    N  = np.zeros(sim_images.shape[:2], dtype=float)

    # FIX 2: correct sample-count check
    if sim_images.shape[0] * sim_images.shape[1] < 2:
        raise ValueError(
            "Impossible to estimate gain and offset with less than 2 images by this method. "
            "Use as many images as possible."
        )

    # Per-image stats
    for r in range(sim_images.shape[0]):
        for n in range(sim_images.shape[1]):
            image = sim_images[r, n]
            sI[r, n] = float(np.sum(image))

            image_ft  = hpc_utils.wrapped_fftn(image)
            image_ft2 = (image_ft * image_ft.conjugate()).real

            N[r, n] = image_ft2[mask].sum() / edge_pixel_number

            # Optional debug:
            print(f"({r},{n}) sum={sI[r,n]:.6g} Nedge={N[r,n]:.6g} std~{N[r,n]**0.5:.6g}")

    # Linear model: sum(I) = alpha * N + offset * (H*W)
    num_imgs = sim_images.shape[0] * sim_images.shape[1]
    X = np.column_stack([
        N.ravel(),
        np.full(num_imgs, image_size, dtype=float)
    ])
    y = sI.ravel()

    theta, residuals_vec, rank, svals = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = theta

    if alpha == 0:
        raise ZeroDivisionError("Estimated alpha is zero; cannot invert to get 'gain'.")

    gain   = 1.0 / alpha
    offset = float(beta)           # this is a per-pixel offset (global across all images)
    f0     = N / (gain**2)         # same shape as (r, n)

    return gain, offset, f0