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
from OpticalSystems import OpticalSystem
from abc import abstractmethod
import Sources
import wrappers
from Box import BoxSIM, Field
from Illumination import PlaneWavesSIM
from Camera import Camera
from VectorOperations import VectorOperations


class SIMulator:
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0):
        self.optical_system = optical_system
        self.illumination = illumination
        self.readout_noise_variance = readout_noise_variance
        self.camera = camera
        self.readout_noise_variance = 0
        self.effective_psfs = {}
        self.phase_modulation_patterns = {}

    def generate_sim_images(self, ground_truth, effective_psfs=None):
        if not effective_psfs:
            if not self.effective_psfs:
                self.effective_psfs, _ = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        else:
            self.effective_psfs = effective_psfs

        if not self.phase_modulation_patterns:
            self.phase_modulation_patterns = self.illumination.get_phase_modulation_patterns(self.optical_system.psf_coordinates)

        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        for r in range(self.illumination.Mr):
            for sim_index in self.illumination.rearranged_indices:
                projective_index = self.illumination.rearranged_indices[sim_index][0] if self.illumination.rearranged_indices[sim_index] else ()
                index = self.illumination.glue_indices(sim_index, projective_index, self.illumination.dimensions)
                wavevector = self.illumination.waves[index].wavevector
                wavevector[np.bool(1 - np.array(self.illumination.dimensions))] = 0
                for n in range(self.illumination.Mt):
                    total_phase_modulation = self.phase_modulation_patterns[r, sim_index] * self.illumination.phase_matrix[(r, n, sim_index)]
                    sim_images[r, n] += scipy.signal.convolve(total_phase_modulation * ground_truth, self.effective_psfs[r, sim_index], mode='same')
        sim_images = np.real(sim_images)
        return sim_images

    def add_noise(self, image):
        if self.camera:
            image = self.camera.get_image(image, self.readout_noise_variance)
        else:
            image = (np.random.poisson(image) +
                                  np.sqrt(self.readout_noise_variance))
            sigma = np.sqrt(self.readout_noise_variance)
            image += np.random.normal(loc=0.0, scale=sigma, size=image.shape)
        return image

    def generate_noisy_images(self, sim_images):
        noisy_images = np.zeros(sim_images.shape)
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                noisy_images[r, n] = self.add_noise(sim_images[r, n])

        return noisy_images
    def generate_widefield(self, image):
        widefield_image = scipy.signal.convolve(image, self.optical_system.psf, mode='same')
        return widefield_image


class SIMulator2D(SIMulator):
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0):
        if not len(optical_system.psf.shape) == 2:
            raise ValueError("The PSF must be 2D for 2D SIM simulations.")
        super().__init__(illumination, optical_system, camera, readout_noise_variance)


class SIMulator3D(SIMulator):
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0):
        if not len(optical_system.psf.shape) == 3:
            raise ValueError("The PSF must be 3D for 3D SIM simulations.")
        super().__init__(illumination, optical_system, camera, readout_noise_variance)

