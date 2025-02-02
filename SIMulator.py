"""
SIMulator.py

This module contains the SIMulator class for simulating raw
structured illumination microscopy (SIM) images and/or reconstructing
the super resolution images from the raw SIM images.

This class will be probably split into two classes in the future. The detailed documentation will be provided in the further release.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

import Sources
import wrappers
from Box import BoxSIM, Field
from VectorOperations import VectorOperations


class SIMulator(BoxSIM):
    def __init__(self, illumination, optical_system, box_size=10, point_number=100, readout_noise_variance=0, additional_info=None):
        super().__init__(illumination, box_size, point_number, additional_info)
        self.optical_system = optical_system
        self.SDR_coefficients = None
        self.effective_psfs = {}
        self.effective_otfs = {}
        self.effective_illumination = np.zeros((self.illumination.Mr, self.illumination.Mt, len(self.illumination.waves), *self.optical_system.psf.shape))
        self.readout_noise_variance = readout_noise_variance

    def _compute_effective_psfs_and_otfs(self):
        self.effective_psfs, self.effective_otfs = self.optical_system.compute_effectve_psfs_and_otfs(self.illumination)
        self.otf_sim = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs:
            self.otf_sim += self.effective_otfs[m]

    def _compute_SDR_coefficients(self, mode):
        self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number))
        Cnorm = 1 / (self.illumination.Mr * np.sum((self.illuminations_shifted[0] * self.optical_system.psf)))
        for r in range(self.illumination.Mr):
            wavevectors2d, keys2d = self.illumination.get_wavevectors_projected(r)
            normalized_illumination = []
            for key, wavevector in zip(keys2d, wavevectors2d):
                if mode == "same":
                    amplitude = self.illumination.waves[(*key, 0)].amplitude
                elif mode == "uniform":
                    amplitude = 1
                elif mode == "custom":
                    ...
                else:
                    raise AttributeError("The mode is not known")
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

    def _compute_effective_psfs(self):
        waves = self.illumination.waves
        X, Y, Z = np.meshgrid(*self.optical_system.psf_coordinates)
        indices = self.illumination.rearranged_indices
        for m in self.illumination.indices2d:
            effective_psf = 0
            for z_index in indices[m]:
                wavevector = waves[(*m, z_index)].wavevector
                kp = wavevector[2]
                amplitude = waves[(*m, z_index)].amplitude
                phase_shifted = np.exp(1j * Z * kp) * self.optical_system.psf
                effective_psf += amplitude * phase_shifted
            self.effective_psfs[m] = effective_psf
            # plt.imshow(np.log(1 + 10**4 * np.abs(effective_otfs[(r, w)][:, :, 50])))
            # plt.show()

    def generate_sim_images(self, object):
        np.random.seed(1234)
        if not self.effective_psfs:
            self._compute_effective_psfs()
        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number), dtype=np.complex128)
        for r in range(self.illumination.Mr):
            wavevectors2d, _ = self.illumination.get_wavevectors_projected(r)
            for w in range(len(wavevectors2d)):  #to parallelize
                m = self.illumination.indices2d[w]
                wavevector = np.array((*wavevectors2d[w], 0))
                effective_illumination = np.exp(1j * np.einsum('ijkl,l ->ijk', self.grid, wavevector))
                for n in range(self.illumination.Mt):
                    effective_illumination_phase_shifted = effective_illumination * self.illumination.phase_matrix[(r, n, m)]
                    sim_images[r, n] += scipy.signal.convolve(effective_illumination_phase_shifted * object, self.effective_psfs[m], mode='same')
        sim_images = np.abs(sim_images)
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                # print(r, n, np.sum(sim_images[r, n]))
                # plt.imshow(sim_images[r, n][:, :, 25])
                # plt.show()

                poisson = np.random.poisson(sim_images[r, n])
                gaussian = np.random.normal(0, scale=self.readout_noise_variance, size=object.shape)
                sim_images[r, n] = poisson + gaussian

        return sim_images

    def generate_sim_images2d(self, object):
        np.random.seed(1234)
        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number[:2]), dtype=np.complex128)
        for r in range(self.illumination.Mr):
            wavevectors, keys = self.illumination.get_wavevectors_projected(r)
            for wavevector, m in zip(wavevectors, keys):  #to parallelize
                effective_illumination = np.exp(1j * np.einsum('ijk,k ->ij', self.grid[:, :, 0, :2], wavevector))
                for n in range(self.illumination.Mt):
                    effective_illumination_phase_shifted = effective_illumination * self.illumination.phase_matrix[(r, n, m)]
                    sim_images[r, n] += scipy.signal.convolve(effective_illumination_phase_shifted * object, self.optical_system.psf, mode='same')
        sim_images = np.abs(sim_images)
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                poisson = np.random.poisson(sim_images[r, n])
                gaussian = np.random.normal(0, scale=self.readout_noise_variance, size=object.shape)
                sim_images[r, n] = poisson + gaussian

        return sim_images

    def generate_widefield(self, sim_images):
        widefield_image = np.zeros(sim_images.shape[2:])
        for rotation in sim_images:
            for image in rotation:
                widefield_image += image
        return widefield_image

    def reconstruct_real_space(self, sim_images, mode="same"):
        if not self.SDR_coefficients:
            self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number), dtype=np.float64)
            self._compute_SDR_coefficients(mode=mode)
        reconstructed_image = np.zeros(sim_images.shape[2:])
        for r in range(self.illumination.Mr):
            for n in range(sim_images.shape[1]):
                reconstructed_image += self.SDR_coefficients[r, n] * sim_images[r, n]
        return reconstructed_image

    def reconstruct_real2d_finite_kernel(self, sim_images, kernel, mode='same'):
        low_passed_images = np.zeros(sim_images.shape)
        if not self.SDR_coefficients:
            self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number), dtype=np.float64)
            self._compute_SDR_coefficients(mode)
        reconstructed_image = np.zeros(sim_images.shape[2:])
        for r in range(self.illumination.Mr):
            for n in range(sim_images.shape[1]):
                # plt.imshow(kernel[:, :, 0])
                # plt.show()
                low_passed_images[r, n] = scipy.ndimage.convolve(sim_images[r, n], kernel, mode='constant', cval=0.0, origin=0)
                # plt.imshow(low_passed_images[r, n, :, :, 25])
                # plt.show()
                reconstructed_image += self.SDR_coefficients[r, n][:, :, 0] * low_passed_images[r, n]
        return reconstructed_image

    def _compute_shifted_image_ft(self, image, kmr):
        kmr = np.array((*kmr, 0))
        phase_shifted = np.exp(1j * np.einsum('ijkl,l ->ijk', self.grid, kmr)) * image
        shifted_image_ft = wrappers.wrapped_fftn(phase_shifted)
        return shifted_image_ft

    def reconstruct_Fourier_space(self, sim_images):
        if not self.effective_otfs:
            self.effective_otfs = self.optical_system.compute_effective_otfs_projective_3dSIM(self.illumination)
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
