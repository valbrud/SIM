import matplotlib.pyplot as plt
import numpy as np
import scipy
from abc import abstractmethod

import Sources
import wrappers
from Box import BoxSIM, Field
from VectorOperations import VectorOperations

class ReconstructorSIM:
    @abstractmethod
    def reconstruct(self, images): ...


class ReconstructorFourierDomain(ReconstructorSIM):

    def __init__(self, illumination, optical_system=None, effective_otfs=None):
        self.illumination = illumination
        self.effective_otfs = effective_otfs
        if effective_otfs is None:
            if optical_system is None:
                raise AttributeError("Provide either an optical system or effective otfs")

            self.effective_otfs = self._compute_effective_otfs()

    def _compute_effective_otfs(self): ...

    def reconstruct(self, images): ...


class ReconstructorSpatialDomain(ReconstructorSIM):
    def __init__(self, illumination, optical_system=None, effecitve_psfs=None, mode='same', amplitudes=None, kernel=None):
        self.illumination = illumination
        self.effective_psfs = effecitve_psfs
        if not effecitve_psfs:
            if optical_system is None:
                raise AttributeError("Provide either an optical system or effective otfs")

            self.effective_psfs = self._compute_effective_psfs()

    def _compute_effective_psfs(self): ...

    def _compute_SDR_coefficients(self, mode, amplitudes=None):
        self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number))
        Cnorm = 1 / (self.illumination.Mr * np.sum((self.illuminations_shifted[0] * self.optical_system.psf)))
        for r in range(self.illumination.Mr):
            wavevectors2d, keys2d = self.illumination.get_wavevectors_projected(r)
            normalized_illumination = []
            for key, wavevector in zip(keys2d, wavevectors2d):
                if amplitudes:
                    amplitude = amplitudes[key]
                elif mode == "same":
                    amplitude = self.illumination.waves[(*key, 0)].amplitude
                elif mode == "uniform":
                    amplitude = 1
                else:
                    raise AttributeError("The mode is not known and amplitudes are not provided")
                source = Sources.IntensityPlaneWave(amplitude, 0, np.array((*wavevector, 0)))
                normalized_illumination.append(Field(source, self.grid, self.source_identifier))

            for n in range(self.illumination.Mt):
                SDR_coefficient = np.zeros(self.point_number, dtype=np.complex128)
                urn = VectorOperations.rotate_vector3d(self.illumination.spatial_shifts[n], np.array((0, 0, 1)), self.illumination.angles[r])
                for field in normalized_illumination:
                    krm = field.source.wavevector
                    phase = np.dot(urn, krm)
                    print(r, phase)
                    SDR_coefficient += field.field * np.exp(-1j * phase)
                # plt.imshow(SDR_coefficient[:, :, 25].real)
                # plt.show()
                self.SDR_coefficients[r, n] = np.real(SDR_coefficient) / Cnorm

    def reconstruct(self, images): ...

class Reconstructor2DFourier(ReconstructorFourierDomain):
    def reconstruct(self, images): ...

class Reconstructor2dSIM2dShiftsSpatial(ReconstructorSpatialDomain):
    def reconstruct(self, images): ...

class Reconstructor3DFourier(ReconstructorFourierDomain):

    def reconstruct(self, sim_images):
        reconstructed_image_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
        for r in range(sim_images.shape[0]):
            image1rotation_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            test_var = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            for krm, m in zip(*self.illumination.get_wavevectors_projected(r)):
                sum_shifts = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                for n in range(sim_images.shape[1]):
                    image_shifted_ft = self._compute_shifted_image_ft(sim_images[r, n], krm)
                    # plt.imshow(np.log(1 + np.abs(wrappers.wrapped_ifftn(image_shifted_ft[:, :, 25]))))
                    # plt.show()
                    print(r, n, m, np.mean(image_shifted_ft))
                    sum_shifts += self.illumination.phase_matrix[(r, n, m)] * image_shifted_ft
                # sum_shifts = np.transpose(sum_shifts, axes=(1, 0, 2))
                sum_shifts *= self.effective_otfs[(r, m)]
                image1rotation_ft += sum_shifts
            plt.imshow(np.log(1 + np.abs(image1rotation_ft[:, :, 25])))
            plt.show()

            reconstructed_image_ft += image1rotation_ft
            # print(r, np.amax(image1rotation_ft))
            # plt.imshow(np.log(1 + np.abs(reconstructed_image_ft[:, :, 25])))
            # plt.show()
        reconstructed_image = np.abs(wrappers.wrapped_ifftn(reconstructed_image_ft))
        return reconstructed_image_ft, reconstructed_image


    def reconstruct_Fourier2d_finite_kernel(self, sim_images, shifted_kernels):
        reconstructed_image_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
        for r in range(sim_images.shape[0]):
            image1rotation_ft = np.zeros(sim_images.shape[2:], dtype=np.complex128)
            for krm, m in zip(*self.illumination.get_wavevectors_projected(r)):
                sum_shifts = np.zeros(sim_images.shape[2:], dtype=np.complex128)
                for n in range(sim_images.shape[1]):
                    image_shifted_ft = self._compute_shifted_image_ft(sim_images[r, n], krm)
                    sum_shifts += self.illumination.phase_matrix[r, n, m] * image_shifted_ft
                sum_shifts *= shifted_kernels[(r, m)].conjugate()
                image1rotation_ft += sum_shifts
            reconstructed_image_ft += image1rotation_ft
            plt.imshow(np.log(1 + np.abs(reconstructed_image_ft[:, :, 25])))
            plt.show()
        reconstructed_image = np.abs(wrappers.wrapped_ifftn(reconstructed_image_ft))
        return reconstructed_image_ft, reconstructed_image


class Reconstructor3DSpatial(ReconstructorSpatialDomain):
    def __init__(self, illumination, optical_system=None, effective_psfs=None, real_kernel=None, mode='same', amplitudes=None):
        super.__init__(illumination)
        self.optical_system = optical_system
        self.illumination = illumination
        self.real_kernel = real_kernel
        self.SDR_coefficients = self._compute_SDR_coefficients(mode, amplitudes)

    def reconstruct(self, sim_images):
        if self.real_kernel:
            low_passed_images = np.zeros(sim_images.shape)
        if not self.SDR_coefficients:
            self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number), dtype=np.float64)
            self._compute_SDR_coefficients(mode=mode, amplitudes=amplitudes)
        reconstructed_image = np.zeros(sim_images.shape[2:])
        for r in range(self.illumination.Mr):
            for n in range(sim_images.shape[1]):
                if self.real_kernel:
                    low_passed_images[r, n] = scipy.signal.convolve(sim_images[r, n], self.kernel, mode='same')
                    reconstructed_image += self.SDR_coefficients[r, n] * low_passed_images[r, n]
                else:
                    reconstructed_image += self.SDR_coefficients[r, n] * sim_images[r, n]
        return reconstructed_image



