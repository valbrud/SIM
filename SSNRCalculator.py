"""
SSNRCalculator.py

This module contains classes for calculating the (image-independent) spectral signal-to-noise ratio (SSNR)
for a given system optical system and illumination.

Mathematical details will be provided in the later documentation versions and in the corresponding papers.
"""

import numpy
import numpy as np

from Illumination import PlaneWavesSIM, IlluminationPlaneWaves2D, IlluminationPlaneWaves3D
import OpticalSystems
import wrappers
from stattools import average_rings2d
import VectorOperations
import matplotlib.pyplot as plt
from abc import abstractmethod


class SSNRBase:
    def __init__(self, optical_system):
        self._optical_system = optical_system
        self._ssnri = None

    @property
    def ssnri(self):
        return self._ssnri

    @property
    def optical_system(self):
        return self._optical_system

    @optical_system.setter
    def optical_system(self, new_optical_system):
        if not isinstance(new_optical_system, OpticalSystems.OpticalSystem):
            raise AttributeError("Trying to set optical system with a wrong object!")
        if not new_optical_system.otf.shape == self.optical_system.otf.shape:
            raise AttributeError("Trying to set optical system with a wrong OTF shape!")
        self._optical_system = new_optical_system
        self._compute_ssnri()

    def ring_average_ssnri(self, number_of_samples=None):
        q_axes = 2 * np.pi * self.optical_system.otf_frequencies
        ssnri = np.copy(self.ssnri)
        if q_axes.shape[0] == 2:
            return average_rings2d(ssnri, q_axes, number_of_samples=number_of_samples)
        elif q_axes.shape[0] != 3:
            raise AttributeError("PSF dimension is not equal to 2 or 3!")

        averaged_slices = []
        for i in range(ssnri.shape[2]):
            averaged_slices.append(average_rings2d(ssnri[:, :, i], (q_axes[0], q_axes[1]), number_of_samples=number_of_samples))
        return np.array(averaged_slices).T

    def compute_ssnri_volume(self, factor=10, volume_element=1):
        return np.sum(np.abs(self.ssnri)) * volume_element * factor

    def compute_ssnri_entropy(self, factor=100):
        noise_filtered = self.ssnri[self.ssnri > 10 ** (-10) * np.amax(self.ssnri)]
        sum = np.sum(noise_filtered)
        probabilities = noise_filtered / sum
        S = -np.sum(probabilities * np.log(probabilities))
        return S.real * factor

    def compute_radial_ssnri_entropy(self, factor=100):
        ssnr_ra = self.ring_average_ssnri()
        ssnr_ra = ssnr_ra[~np.isnan(ssnr_ra.real) & ~np.isnan(ssnr_ra.imag)]
        noise_filtered = ssnr_ra[ssnr_ra > 10 ** (-12) * np.amax(ssnr_ra)]
        sum = np.sum(noise_filtered)
        probabilities = noise_filtered / sum
        S = -np.sum(probabilities * np.log(probabilities))
        return S.real * factor

    @abstractmethod
    def _compute_ssnri(self):
        ...

    @abstractmethod
    def compute_full_ssnr(self, object_ft):
        ...

class SSNRPointScanning(SSNRBase):
    def __init__(self, optical_system):
        super().__init__(optical_system)
        self._compute_ssnri()

    def _compute_ssnri(self):
        self._ssnri = np.abs(self.optical_system.otf) ** 2

    def compute_full_ssnr(self, object_ft):
        return np.abs(object_ft) ** 2 / np.amax(np.abs(object_ft)) * self.ssnri


class SSNRPointScanning2D(SSNRPointScanning):
    def __init__(self, optical_system):
        if len(optical_system.otf.shape) == 3:
            raise AttributeError("Trying to initialize 2D Confocal Calculator with 3D OTF!")
        if not len(optical_system.otf.shape) == 2:
            raise AttributeError("Trying to initialize 2D Confocal Calculator with wrong OTF!")
        super().__init__(optical_system)


class SSNRPointScanning3D(SSNRPointScanning):
    def __init__(self, optical_system):
        if len(optical_system.otf.shape) == 2:
            raise AttributeError("Trying to initialize 3D Confocal Calculator with 2D OTF!")
        if not len(optical_system.otf.shape) == 3:
            raise AttributeError("Trying to initialize 3D Confocal Calculator with wrong OTF!")
        super().__init__(optical_system)


SSNRConfocal2D = SSNRPointScanning2D
SSNRConfocal3D = SSNRPointScanning3D

SSNRRCM2D = SSNRPointScanning2D
SSNRRCM3D = SSNRPointScanning3D


class SSNRWidefield(SSNRBase):
    def __init__(self, optical_system):
        super().__init__(optical_system)
        self._compute_ssnri()

    def _compute_ssnri(self):
        self._ssnr = np.abs(self.optical_system.otf) ** 2 / np.amax(np.abs(self.optical_system.otf))

    def compute_full_ssnr(self, object_ft):
        return np.abs(object_ft) ** 2 / np.amax(np.abs(object_ft)) * self.ssnri


class SSNRSIM(SSNRBase):
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system, kernel=None,
                 readout_noise_variance=0,
                 save_memory=False):
        if not isinstance(illumination, PlaneWavesSIM):
            raise AttributeError("Illumination data is not of SIM type!")
        super().__init__(optical_system)
        self._illumination = illumination
        self.vj = None
        self.dj = None
        self.effective_otfs = {}
        self.otf_sim = None
        self.effective_kernels_ft = {}
        self._kernel = None
        self._kernel_ft = None
        self.save_memory = save_memory
        self.readout_noise_variance = readout_noise_variance

        if optical_system.otf is None:
            raise AttributeError("Optical system otf is not computed")

        self._compute_effective_otfs()

        if kernel:
            self.kernel = kernel
            self._compute_effective_kernels_ft()

        self._compute_ssnri()

    @property
    def optical_system(self):
        return self._optical_system

    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.effective_otfs = {}
        self._compute_effective_otfs()
        self._compute_ssnri()

    @property
    def illumination(self):
        return self._illumination

    @illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self._compute_effective_otfs()
        if self.kernel:
            self._compute_effective_kernels_ft()
        self._compute_ssnri()

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel_new):
        shape = np.array(kernel_new.shape, dtype=np.int32)
        otf_shape = np.array(self.optical_system.otf.shape, dtype=np.int32)

        if ((shape % 2) == 0).any():
            raise ValueError("Size of the kernel must be odd!")

        if (shape > otf_shape).any():
            raise ValueError("Size of the kernel is bigger than of the PSF!")

        if (shape < otf_shape).any():
            kernel_expanded = np.zeros(otf_shape, dtype=kernel_new.dtype)

            # Build slice objects for each dimension, to center `kernel_new` in `kernel_expanded`.
            slices = []
            for dim in range(len(shape)):
                center = otf_shape[dim] // 2
                half_span = shape[dim] // 2
                start = center - half_span
                stop = start + shape[dim]
                slices.append(slice(start, stop))

            # Assign kernel_new into the center of kernel_expanded
            kernel_expanded[tuple(slices)] = kernel_new
            kernel_new = kernel_expanded

        self.kernel_ft = wrappers.wrapped_ifftn(kernel_new)
        self.kernel_ft /= np.amax(self.kernel_ft)
        self._kernel = kernel_new
        self.effective_kernels_ft = {}
        self._compute_effective_kernels_ft()

        self._compute_ssnri()

    def _compute_effective_otfs(self):
        _, self.effective_otfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def _compute_effective_kernels_ft(self):
        _, self.effective_kernels_ft = self.illumination.compute_effective_kernels(self.kernel, self.optical_system.psf_coordinates)

    def _compute_Dj(self):
        if not self.effective_kernels_ft:
            effective_kernels_ft = self.effective_otfs
        else:
            effective_kernels_ft = self.effective_kernels_ft

        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * effective_kernels_ft[m].conjugate()
        d_j *= self.illumination.Mt
        # plt.title("Dj")
        # plt.imshow(np.log(1 + 10**8 * np.abs(d_j)[:, :, 50]))
        # plt.show()
        return np.abs(d_j)

    def _compute_Vj(self):
        if not self.effective_kernels_ft:
            effective_kernels_ft = self.effective_otfs
        else:
            effective_kernels_ft = self.effective_kernels_ft

        center = np.array(self.optical_system.otf.shape, dtype=np.int32) // 2
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)

        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.rearranged_indices:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = effective_kernels_ft[idx1]
                otf2 = effective_kernels_ft[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        # plt.title("Vj")
        # plt.imshow(np.log(1 + 10**8 * np.abs(v_j)[:, :, 50]))
        # plt.show()
        return np.abs(v_j)

    def _compute_ssnri(self):
        # Only needed if effective kernels/otfs were deleted in a memory efficient mode
        if not self.effective_otfs:
            self._compute_effective_otfs()
        if self.kernel and not self.effective_kernels_ft:
            self._compute_effective_kernels_ft()

        self.dj = self._compute_Dj()
        self.vj = self._compute_Vj()
        ssnr = np.zeros(self.dj.shape, dtype=np.float64)
        mask = (self.vj != 0) * (self.dj != 0)
        numpy.putmask(ssnr, mask, np.abs(self.dj) ** 2 / self.vj)
        self._ssnri = ssnr
        if self.save_memory:
            self.effective_otfs = {}
            self.effective_kernels_ft = {}

    def compute_full_ssnr(self, object_ft):
        return ((self.dj * np.abs(object_ft)) ** 2 /
                (np.amax(np.abs(object_ft)) * self.vj + self.optical_system.otf.size * self.readout_noise_variance * self.dj))

    def compute_analytic_ssnri_volume(self, factor=10, volume_element=1):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        g0 = np.abs(np.amax(self.optical_system.otf))
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted2sum = np.sum(weights * weights.conjugate()).real
        volume = ((self.illumination.Mt * self.illumination.Mr) ** 2 * weighted2sum * g2 /
                  g0 * volume_element * factor)
        return volume

    def compute_total_signal_to_noise(self, factor=10, volume_element=1):
        mask = (self.dj != 0) * (self.vj != 0)
        total_signal_power = np.sum(np.abs(self.dj[mask]) ** 2)
        total_noise_power = np.sum(np.abs(self.vj[mask]))
        return total_signal_power / total_noise_power * volume_element * factor

    def compute_total_analytic_signal_to_noise(self, factor=10, volume_element=1):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        g4 = np.sum(self.optical_system.otf ** 2 * self.optical_system.otf.conjugate() ** 2).real
        g0 = np.abs(np.amax(self.optical_system.otf))
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted2sum = np.sum(weights * weights.conjugate()).real
        weighted4sum = np.sum(weights ** 2 * weights.conjugate() ** 2).real
        total = ((self.illumination.Mt * self.illumination.Mr) ** 2 * weighted4sum * g4 /
                 weighted2sum / g2 / g0 * volume_element * factor)
        return total

    def _find_threshold_value(self, stock, max, min, noise_level, ssnr_widefield):
        average = (max + min) / 2
        less_min = ssnr_widefield[(ssnr_widefield < min) * (self.ssnri > noise_level)]
        less_max = ssnr_widefield[(ssnr_widefield < max) * (self.ssnri > noise_level)]
        if less_max.size == less_min.size:
            return average
        less = ssnr_widefield[(ssnr_widefield < average) * (self.ssnri > noise_level)]
        sum_less = np.sum(less)
        fill = less.size * max - sum_less
        if fill > stock:
            return self._find_threshold_value(stock, average, min, noise_level, ssnr_widefield)
        else:
            return self._find_threshold_value(stock, max, average, noise_level, ssnr_widefield)

    def compute_maximum_resolved_lateral(self):
        fR = 2 * self.optical_system.n * np.sin(self.optical_system.alpha)
        fourier_peaks_wavevectors = np.array([spatial_wave.wavevector for spatial_wave in self.illumination.waves.values()])
        fI = np.max(np.array([(wavevector[0] ** 2 + wavevector[1] ** 2) ** 0.5 for wavevector in fourier_peaks_wavevectors]))
        return fR + fI

    def compute_ssnr_waterline_measure(self, factor=10):
        ssnr_widefield = SSNRWidefield(self.optical_system).ssnr
        diff = np.sum(self.ssnri - ssnr_widefield).real
        upper_estimate = np.abs(np.amax(self.ssnri - ssnr_widefield))
        noise_level = 10 ** -10 * np.abs(np.amax(self.ssnri))
        threshold = self._find_threshold_value(diff, upper_estimate, 0, noise_level, ssnr_widefield)
        measure = np.where((np.abs(ssnr_widefield) < threshold) * (np.abs(self.ssnri) > noise_level), np.abs(self.ssnri - ssnr_widefield), 0)
        measure = np.where(measure < threshold, measure, threshold)
        return np.sum(measure) * factor, threshold


class SSNRSIM2D(SSNRSIM):
    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system, kernel=None, readout_noise_variance=0, save_memory=False):
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise AttributeError("Illumination data is not of the valid dimension!")
        if len(optical_system.otf.shape) == 3:
            raise AttributeError("Trying to initialize 2D SIM Calculator with 3D OTF!")
        if not len(optical_system.otf.shape) == 2:
            raise AttributeError("Trying to initialize 2D SIM Calculator with wrong OTF!")
        super().__init__(illumination, optical_system, kernel=kernel, readout_noise_variance=readout_noise_variance, save_memory=save_memory)

    def plot_effective_kernel_and_otf(self):
        Nx, Ny = self.optical_system.otf.shape
        fig, ax = plt.subplots()
        ax.plot(self.optical_system.otf_frequencies[0] / (2 * self.optical_system.NA), self.kernel_ft[:, Ny // 2], label="Kernel")
        ax.plot(self.optical_system.otf_frequencies[0] / (2 * self.optical_system.NA), self.optical_system.otf[:, Ny // 2], label="OTF")
        ax.set_title("Kernel vs OTF")
        ax.set_xlabel("$f_r, \\frac{2NA}{\lambda}$")
        ax.set_ylabel("OTF/K, u.e.")
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid()

    def imshow_effective_kernel_and_otf(self):
        Nx, Ny = self.optical_system.otf.shape
        scaled_frequencies = self.optical_system.otf_frequencies / (2 * self.optical_system.NA)
        fig, ax = plt.subplots(2)
        ax[0].set_title("OTF")
        ax[0].imshow(self.optical_system.otf[:, Ny // 2],
                     extent=(scaled_frequencies[0][0], scaled_frequencies[0][-1], scaled_frequencies[1][0], scaled_frequencies[1][-1]))
        ax[1].set_title("Kernel")
        ax[1].imshow(self.kernel_ft[:, Ny // 2],
                     extent=(scaled_frequencies[0][0], scaled_frequencies[0][-1], scaled_frequencies[1][0], scaled_frequencies[1][-1]))


class SSNRSIM3D(SSNRSIM):
    def __init__(self, illumination, optical_system, kernel=None, readout_noise_variance=0, save_memory=False):
        if len(optical_system.otf.shape) == 2:
            raise AttributeError("Trying to initialize 3D SIM Calculator with 2D OTF!")
        if not len(optical_system.otf.shape) == 3:
            raise AttributeError("Trying to initialize 3D SIM Calculator with wrong OTF!")
        super().__init__(illumination, optical_system, kernel=kernel, readout_noise_variance=readout_noise_variance, save_memory=save_memory)
