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

