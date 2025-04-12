import matplotlib.pyplot as plt
import numpy as np
import scipy
from abc import abstractmethod
from OpticalSystems import OpticalSystem, OpticalSystem2D, OpticalSystem3D
import Sources
import wrappers
from Box import BoxSIM, Field
from Illumination import PlaneWavesSIM, IlluminationPlaneWaves2D, IlluminationPlaneWaves3D
from VectorOperations import VectorOperations
from mpl_toolkits.mplot3d import axes3d
from abc import ABC, abstractmethod


class ReconstructorSIM(ABC):

    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem = None,
                 kernel=None,
                 deconvolution=None,
                 phase_modulation_patterns=None,
                 ):
        self.illumination = illumination
        self.optical_system = optical_system
        self.kernel = kernel
        self.deconvolution = deconvolution
        self.phase_modulation_patterns = phase_modulation_patterns

        if phase_modulation_patterns:
            self.phase_modulation_patterns = phase_modulation_patterns
        else:
            if self.optical_system is None:
                raise AttributeError("If phase modulation patterns are not provided, optical system must be provided to compute them.")
            self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

    @abstractmethod
    def reconstruct(self, sim_images):
        ...

    def get_widefield(self, sim_images):
        widefield_image = np.zeros(sim_images.shape[2:])
        for rotation in sim_images:
            test = np.zeros(sim_images.shape[2:])
            for image in rotation:
                widefield_image += image
                test += image
            # plt.imshow(test)
            # plt.show()
        return widefield_image


class ReconstructorFourierDomain(ReconstructorSIM, ABC):
    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem = None,
                 kernel=None,
                 effective_kernels=None,
                 deconvolution=None,
                 phase_modulation_patterns=None,
                 apodization_filter=None,
                 regularization_filter=None,
                 ):
        
        super().__init__(illumination,
                         optical_system,
                         kernel=kernel,
                         deconvolution=deconvolution,
                         phase_modulation_patterns=phase_modulation_patterns)
        if effective_kernels:
            self.effective_kernels = effective_kernels
        else:
            if self.optical_system is None:
                raise AttributeError("If effective kernels are not provided, optical system must be provided to compute them.")
            if kernel is None:
                _, self.effective_kernels = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
            else:
                _, self.effective_kernels = self.illumination.compute_effective_kernels(kernel, self.optical_system.psf_coordinates)
        self.apodization_filter = apodization_filter
        self.regularization_filter = regularization_filter

    def _compute_shifted_image_ft(self, image, r, m):
        phase_shifted = image * self.phase_modulation_patterns[r, m].conjugate()
        shifted_image_ft = wrappers.wrapped_fftn(phase_shifted)
        return shifted_image_ft

    def reconstruct(self, sim_images):
        reconstructed_image_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
        for r in range(sim_images.shape[0]):
            image1rotation_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            for _, m in zip(*self.illumination.get_wavevectors_projected(r)):
                sum_shifts = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                for n in range(sim_images.shape[1]):
                    image_shifted_ft = self._compute_shifted_image_ft(sim_images[r, n], r, m)
                    # plt.imshow(np.log(1 + 10**8 * np.abs(image_shifted_ft)))
                    # plt.show()
                    sum_shifts += self.illumination.phase_matrix[(n, m)].conjugate() * image_shifted_ft
                sum_shifts *= self.effective_kernels[(r, m)]
                image1rotation_ft += sum_shifts
            reconstructed_image_ft += image1rotation_ft
        plt.imshow(np.log(1 + 10**8 * np.abs(reconstructed_image_ft)))
        plt.show()
        if not self.deconvolution is None:  # Provide interface and implement later
            ...
        if not self.regularization_filter is None:
            reconstructed_image_ft /= self.regularization_filter
        if not self.apodization_filter is None:  # Provide interface and rewrite later
            reconstructed_image_ft *= self.apodization_filter
        reconstructed_image = np.abs(wrappers.wrapped_ifftn(reconstructed_image_ft))
        return reconstructed_image


class ReconstructorSpatialDomain(ReconstructorSIM, ABC):
    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem = None,
                 kernel=None,
                 deconvolution=None,
                 phase_modulation_patterns=None,
                 illumination_patterns=None,
                 apodization_kernel=None,
                 ):
        
        super().__init__(illumination,
                         optical_system,
                         kernel=kernel,
                         deconvolution=deconvolution,
                         phase_modulation_patterns=phase_modulation_patterns
                         )

        if kernel is None:
            otf_shape = np.array(self.optical_system.otf.shape, dtype=np.int32)
            kernel = np.zeros(otf_shape, dtype=np.float64)
            kernel[otf_shape[0] // 2 + 1, otf_shape[1] // 2 + 1] = 1
        else:
            shape = np.array(kernel.shape, dtype=np.int32)
            otf_shape = np.array(self.optical_system.otf.shape, dtype=np.int32)

            if ((shape % 2) == 0).any():
                raise ValueError("Size of the kernel must be odd!")

            if (shape > otf_shape).any():
                raise ValueError("Size of the kernel is bigger than of the PSF!")

            if (shape < otf_shape).any():
                kernel_expanded = np.zeros(otf_shape, dtype=kernel.dtype)

                # Build slice objects for each dimension, to center `kernel_new` in `kernel_expanded`.
                slices = []
                for dim in range(len(shape)):
                    center = otf_shape[dim] // 2
                    half_span = shape[dim] // 2
                    start = center - half_span
                    stop = start + shape[dim]
                    slices.append(slice(start, stop))

                kernel_expanded[tuple(slices)] = kernel
                kernel = kernel_expanded

        self.kernel = kernel
        self.apodization_kernel = apodization_kernel

        if illumination_patterns:
            self.illumination_patterns = illumination_patterns
        else:
            self.illumination_patterns = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.float64)

            for r in range(self.illumination.Mr):
                illumination_pattern_one_rotation = np.zeros(self.optical_system.psf.shape, dtype=np.complex128)
                for n in range(self.illumination.Mt):
                    illumination_pattern = np.zeros(self.optical_system.psf.shape, dtype=np.complex128)
                    for wave in self.illumination.waves:
                        m = tuple([wave[dimension] for dimension in range(len(self.illumination.dimensions)) if self.illumination.dimensions[dimension]])
                        illumination_pattern += (self.illumination.phase_matrix[(n, m)] * self.illumination.waves[wave].get_intensity(self.optical_system.x_grid, -self.illumination.angles[r])).conjugate()
                    # illumination_pattern += self.illumination.phase_matrix[r, n, m] * self.phase_modulation_patterns[r, m]
                    # plt.imshow(illumination_pattern.real)
                    # plt.show()
                    # illumination_pattern_one_rotation += illumination_pattern
                    # X, Y = np.meshgrid(*self.optical_system.psf_coordinates)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='3d')
                    # ax.plot_wireframe(X, Y, illumination_pattern_one_rotation)
                    # plt.show()
                    # plt.imshow(illumination_pattern_one_rotation.real)
                    # plt.show()
                    self.illumination_patterns[r, n] = illumination_pattern.real
                # plt.imshow(np.sum(self.illumination_patterns, axis=1))
                plt.show()

    def reconstruct(self, sim_images):
        reconstructed_image = np.zeros(sim_images.shape[2:], dtype=np.float64)
        for r in range(sim_images.shape[0]):
            image1rotation = np.zeros(sim_images.shape[2:], dtype=np.float64)
            for n in range(sim_images.shape[1]):
                # plt.imshow(sim_images[r, n])
                # plt.show()
                # plt.imshow(self.kernel)
                # plt.show()
                image_convolved = scipy.signal.convolve(sim_images[r, n], self.kernel, mode='same')
                image1rotation += self.illumination_patterns[r, n] * image_convolved
                # plt.imshow(image_convolved)
                # plt.show()
                # plt.imshow(self.illumination_patterns[r, n])
                # plt.show()
                # plt.imshow(image1rotation)
                # plt.show()
            reconstructed_image += image1rotation
        if self.deconvolution:  # Provide interface and implement later
            ...
        if self.apodization_kernel:  # Provide interface and rewrite later
            reconstructed_image = scipy.signal.convolve(reconstructed_image, self.apodization_kernel, mode='same')
        return reconstructed_image

class ReconstructorFourierDomain2D(ReconstructorFourierDomain):
    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system: OpticalSystem2D = None,
                 kernel=None,
                 deconvolution=None,
                 phase_modulation_patterns=None,
                 apodization_filter=None,
                 regularization_filter=None,
                 ):
        
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise TypeError("illumination must be an instance of IlluminationPlaneWaves2D")
        
        if not isinstance(optical_system, OpticalSystem2D):
            raise TypeError("optical_system must be an instance of OpticalSystem2D")
        
        super().__init__(illumination=illumination,
                         optical_system=optical_system,
                         kernel=kernel,
                         deconvolution=deconvolution,
                         phase_modulation_patterns=phase_modulation_patterns,
                         apodization_filter=apodization_filter,
                         regularization_filter=regularization_filter)
