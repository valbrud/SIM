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
        self.effective_illumination_patterns = {}
        self._compute_effective_illumination_patterns()

    @abstractmethod
    def generate_sim_images(self, object, effective_psfs=None):
        ...

    @abstractmethod
    def compute_effective_illumination_patterns(self, object, effective_psfs=None):
        ...

    def generate_widefield(self, sim_images):
        widefield_image = np.zeros(sim_images.shape[2:])
        for rotation in sim_images:
            for image in rotation:
                widefield_image += image
        return widefield_image


class SIMulator2D(SIMulator):
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0):
        if not len(optical_system.psf.shape) == 2:
            raise ValueError("The PSF must be 2D for 2D SIM simulations.")
        super().__init__(illumination, optical_system, camera, readout_noise_variance)


    def generate_sim_images(self, object, effective_psfs=None):
        if not effective_psfs:
            if not self.effective_psfs:
                self.effective_psfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        else:
            self.effective_psfs = effective_psfs

        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        X, Y = np.meshgrid(*self.optical_system.psf_coordinates)
        grid = np.stack((X, Y), axis=-1)
        for r in range(self.illumination.Mr):
            for sim_index in self.illumination.rearranged_indices:
                projective_index = self.illumination.rearranged_indices[sim_index][0]
                index = self.illumination.glue_indices(sim_index, projective_index)
                wavevector = np.array(self.illumination.waves[index])
                wavevector[np.bool(1 - np.array(self.illumination.dimensions))] = 0
                effective_illumination = np.exp(1j * np.einsum('ijkl,l ->ijk', grid, wavevector))
                for n in range(self.illumination.Mt):
                    effective_illumination_phase_shifted = effective_illumination * self.illumination.phase_matrix[(r, n, sim_index)]
                    sim_images[r, n] += scipy.signal.convolve(effective_illumination_phase_shifted * object, self.effective_psfs[sim_index], mode='same')
        sim_images = np.real(sim_images)

        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                if self.camera:
                    sim_images[r, n] = self.camera.get_image(sim_images[r, n])
                else:
                    poisson = np.random.poisson(sim_images[r, n])
                    gaussian = np.random.normal(0, scale=self.readout_noise_variance, size=object.shape)
                    sim_images[r, n] = poisson + gaussian

        return sim_images


class SIMulator3D(SIMulator):
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 camera: Camera = None,
                 readout_noise_variance=0):
        if not len(optical_system.psf.shape) == 3:
            raise ValueError("The PSF must be 3D for 3D SIM simulations.")
        super().__init__(illumination, optical_system, camera, readout_noise_variance)

    def generate_sim_images(self, object, effective_psfs=None):
        if not effective_psfs:
            if not self.effective_psfs:
                self.effective_psfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        else:
            self.effective_psfs = effective_psfs

        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.optical_system.psf.shape), dtype=np.complex128)
        X, Y, Z = np.meshgrid(*self.optical_system.psf_coordinates)
        grid = np.stack((X, Y, Z), axis=-1)
        for r in range(self.illumination.Mr):
            for sim_index in self.illumination.rearranged_indices:
                projective_index = self.illumination.rearranged_indices[sim_index][0]
                index = self.illumination.glue_indices(sim_index, projective_index)
                wavevector = np.array(self.illumination.waves[index])
                wavevector[np.bool(1 - np.array(self.illumination.dimensions))] = 0
                effective_illumination = np.exp(1j * np.einsum('ijkl,l ->ijk', grid, wavevector))
                for n in range(self.illumination.Mt):
                    effective_illumination_phase_shifted = effective_illumination * self.illumination.phase_matrix[(r, n, sim_index)]
                    sim_images[r, n] += scipy.signal.convolve(effective_illumination_phase_shifted * object, self.effective_psfs[sim_index], mode='same')
        sim_images = np.real(sim_images)

        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                if self.camera:
                    sim_images[r, n] = self.camera.get_image(sim_images[r, n])
                else:
                    poisson = np.random.poisson(sim_images[r, n])
                    gaussian = np.random.normal(0, scale=self.readout_noise_variance, size=object.shape)
                    sim_images[r, n] = poisson + gaussian

        return sim_images
