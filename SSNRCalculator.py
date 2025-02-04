"""
SSNRCalculator.py

This module contains classes for calculating the (image-independent) spectral signal-to-noise ratio (SSNR)
for a given system optical system and illumination.

Mathematical details will be provided in the later documentation versions and in the corresponding papers.
"""

import numpy
import numpy as np

import wrappers
from stattools import average_rings2d
import VectorOperations
import matplotlib.pyplot as plt
from abc import abstractmethod


class SSNRBase:
    def __init__(self, optical_system):
        self._optical_system = optical_system
        self.ssnr = None

    @abstractmethod
    def compute_ssnr(self):
        ...

    @property
    def optical_system(self):
        return self._optical_system

    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.ssnr = None

    @abstractmethod
    def ring_average_ssnr(self, number_of_samples=None):
        q_axes = 2 * np.pi * self.optical_system.otf_frequencies
        ssnr = np.copy(self.ssnr)
        if q_axes.shape[0] == 2:
            return average_rings2d(ssnr, q_axes, number_of_samples=number_of_samples)
        elif q_axes.shape[0] != 3:
            raise AttributeError("PSF dimension is not equal to 2 or 3!")

        averaged_slices = []
        for i in range(ssnr.shape[2]):
            averaged_slices.append(average_rings2d(ssnr[:, :, i], (q_axes[0], q_axes[1]), number_of_samples=number_of_samples))
        return np.array(averaged_slices).T

    def compute_ssnr_volume(self, factor=10, volume_element=1):
        return np.sum(np.abs(self.ssnr)) * volume_element * factor

    def compute_true_ssnr_entropy(self, factor=100):
        noise_filtered = self.ssnr[self.ssnr > 10 ** (-10) * np.amax(self.ssnr)]
        sum = np.sum(noise_filtered)
        probabilities = noise_filtered / sum
        S = -np.sum(probabilities * np.log(probabilities))
        return S.real * factor

    def compute_radial_ssnr_entropy(self, factor=100):
        ssnr_ra = self.ring_average_ssnr()
        ssnr_ra = ssnr_ra[~np.isnan(ssnr_ra.real) & ~np.isnan(ssnr_ra.imag)]
        noise_filtered = ssnr_ra[ssnr_ra > 10 ** (-12) * np.amax(ssnr_ra)]
        sum = np.sum(noise_filtered)
        probabilities = noise_filtered / sum
        S = -np.sum(probabilities * np.log(probabilities))
        return S.real * factor


class SSNRWidefield(SSNRBase):
    def __init__(self, optical_system):
        super().__init__(optical_system)
        self.ssnr = self.compute_ssnr()

    def compute_ssnr(self):
        ssnr = np.abs(self.optical_system.otf) ** 2 / np.amax(np.abs(self.optical_system.otf))
        return ssnr


class SSNRConfocal(SSNRBase):
    def __init__(self, optical_system):
        super().__init__(optical_system)
        self.ssnr = self.compute_ssnr()

    def compute_ssnr(self):
        Nx, Ny, Nz = self.optical_system.psf.shape
        otf_confocal = np.abs(wrappers.wrapped_fftn(np.abs(self.optical_system.psf) ** 2))
        otf_confocal /= np.amax(otf_confocal)
        ssnr = otf_confocal ** 2
        return ssnr


class SSNRSIM(SSNRBase):
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        super().__init__(optical_system)
        self._illumination = illumination
        self._optical_system = optical_system
        self.vj = None
        self.dj = None
        self.otf_sim = None
        self.maximum_resolved = 0
        self.readout_noise_variance = readout_noise_variance

        if optical_system.otf is None:
            raise AttributeError("Optical system otf is not computed")

    @property
    def optical_system(self):
        return self._optical_system

    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self.ssnr = None  # to avoid heavy computations where they are not needed

    @property
    def illumination(self):
        return self._illumination

    @illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.otf_sim = None
        self._compute_effective_otfs()
        self.ssnr = None
        # to avoid heavy computations where they are not needed

    @abstractmethod
    def _compute_effective_otfs(self):
        ...

    @abstractmethod
    def _Dj(self):
        ...

    @abstractmethod
    def _Vj(self):
        ...

    def _compute_Dj(self, effective_kernels_ft=None):
        if effective_kernels_ft is None:
            effective_kernels_ft = self.effective_otfs
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * effective_kernels_ft[m].conjugate()
        d_j *= self.illumination.Mt
        return np.abs(d_j)

    def _compute_Vj(self, effective_kernels_ft=None):
        if effective_kernels_ft is None:
            effective_kernels_ft = self.effective_otfs
        center = np.array(self.optical_system.otf.shape, dtype=np.int32) // 2
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)

        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = effective_kernels_ft[idx1]
                otf2 = effective_kernels_ft[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return np.abs(v_j)

    def compute_ssnr(self):
        dj = self._Dj()
        ssnr = np.zeros(dj.shape, dtype=np.complex128)
        vj = self._Vj()
        mask = (vj != 0) * (dj != 0)
        numpy.putmask(ssnr, mask, np.abs(dj) ** 2 / vj)
        self.dj = dj
        self.vj = vj
        self.ssnr = ssnr
        return np.abs(ssnr)

    def compute_analytic_ssnr_volume(self, factor=10, volume_element=1):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        g0 = np.abs(np.amax(self.optical_system.otf))
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted2sum = np.sum(weights * weights.conjugate()).real
        volume = ((self.illumination.Mt * self.illumination.Mr) ** 2 * weighted2sum * g2 /
                  g0 * volume_element * factor)
        return volume

    def compute_total_ssnr(self, factor=10, volume_element=1):
        mask = (self.dj != 0) * (self.vj != 0)
        total_signal_power = np.sum(np.abs(self.dj[mask]) ** 2)
        total_noise_power = np.sum(np.abs(self.vj[mask]))
        return total_signal_power / total_noise_power * volume_element * factor

    def compute_analytic_total_ssnr(self, factor=10, volume_element=1):
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
        less_min = ssnr_widefield[(ssnr_widefield < min) * (self.ssnr > noise_level)]
        less_max = ssnr_widefield[(ssnr_widefield < max) * (self.ssnr > noise_level)]
        if less_max.size == less_min.size:
            return average
        less = ssnr_widefield[(ssnr_widefield < average) * (self.ssnr > noise_level)]
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
        diff = np.sum(self.ssnr - ssnr_widefield).real
        upper_estimate = np.abs(np.amax(self.ssnr - ssnr_widefield))
        noise_level = 10 ** -10 * np.abs(np.amax(self.ssnr))
        threshold = self._find_threshold_value(diff, upper_estimate, 0, noise_level, ssnr_widefield)
        measure = np.where((np.abs(ssnr_widefield) < threshold) * (np.abs(self.ssnr) > noise_level), np.abs(self.ssnr - ssnr_widefield), 0)
        measure = np.where(measure < threshold, measure, threshold)
        return np.sum(measure) * factor, threshold


class SSNR2dSIM(SSNRSIM):
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        if len(optical_system.otf.shape) == 3:
            raise AttributeError("Trying to initialize 2D SIM Calculator with 3D OTF!")
        if not len(optical_system.otf.shape) == 2:
            raise AttributeError("Trying to initialize 2D SIM Calculator with wrong OTF!")
        if not len(illumination.indices3d) == len(illumination.indices2d):
            raise AttributeError("2D SIM requireds 2D illumination!")
        super().__init__(illumination, optical_system, readout_noise_variance=0)
        self.effective_otfs = {}
        self._compute_effective_otfs()

    def _compute_effective_otfs(self):
        self.effective_otfs = self.optical_system.compute_effective_otfs_2dSIM(self.illumination)
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def ring_average_ssnr(self, number_of_samples=None):
        q_axes = 2 * np.pi * self.optical_system.otf_frequencies
        ssnr = np.copy(self.ssnr)
        if q_axes[0].size != ssnr.shape[0] or q_axes[1].size != ssnr.shape[1]:
            raise ValueError("Wrong axes are provided for the ssnr")
        ssnr_ra = average_rings2d(ssnr, (q_axes[0], q_axes[1]))
        return ssnr_ra

    def _Dj(self):
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
        d_j *= self.illumination.Mt
        return d_j

    def _Vj(self):
        size_x, size_y = self.optical_system.otf.shape
        center = (size_x // 2, size_y // 2)
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = self.effective_otfs[idx1]
                otf2 = self.effective_otfs[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return v_j


class SSNR2dSIMFiniteKernel(SSNR2dSIM):
    def __init__(self, illumination, optical_system, kernel, readout_noise_variance=0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.kernel_ft = None
        self.kernel = kernel

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
            kernel_expanded = np.zeros(otf_shape)
            kernel_expanded[otf_shape[0] // 2 - shape[0] // 2:otf_shape[0] // 2 + shape[0] // 2 + 1,
            otf_shape[1] // 2 - shape[1] // 2:otf_shape[1] // 2 + shape[1] // 2 + 1,
            ] = kernel_new
            kernel_new = kernel_expanded
        self._kernel = kernel_new
        self.kernel_ft = wrappers.wrapped_ifftn(kernel_new)
        self.kernel_ft /= np.amax(self.kernel_ft)
        self._kernel = kernel_new
        self.effective_kernels_ft = {}
        self._compute_effective_kernels_ft()
        self.compute_ssnr()

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

    @SSNR2dSIM.illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.otf_sim = None
        self._compute_effective_otfs()
        self._compute_effective_kernels_ft()
        self.ssnr = None

    def _compute_effective_kernels_ft(self):
        waves = self.illumination.waves
        X, Y = np.meshgrid(*self.optical_system.psf_coordinates)
        for r in range(self.illumination.Mr):
            wavevectors, indices = self.illumination.get_wavevectors_projected(r)
            for wavevector, index in zip(wavevectors, indices):
                amplitude = waves[(*index, 0)].amplitude
                phase_shifted = np.exp(1j * np.einsum('ijk,i ->jk', np.array((X, Y)), wavevector)) * self.kernel
                effective_real_kernel = amplitude * phase_shifted
                self.effective_kernels_ft[(r, index)] = wrappers.wrapped_fftn(effective_real_kernel)

    def _Dj(self):
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_kernels_ft[m].conjugate()
        d_j *= self.illumination.Mt
        return np.abs(d_j)

    def _Vj(self):
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        size_x, size_y = self.optical_system.otf.shape
        center = np.array((size_x, size_y), dtype=np.int32) // 2
        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = self.effective_kernels_ft[idx1]
                otf2 = self.effective_kernels_ft[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return np.abs(v_j)


class SSNR3dSIMBase(SSNRSIM):
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        if len(optical_system.otf.shape) == 2:
            raise AttributeError("Trying to initialize 3D SIM Calculator with 2D OTF!")
        if not len(optical_system.otf.shape) == 3:
            raise AttributeError("Trying to initialize 3D SIM Calculator with wrong OTF!")
        super().__init__(illumination, optical_system, readout_noise_variance=0)


class SSNR3dSIM3dShifts(SSNR3dSIMBase):
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.effective_otfs = {}
        self._compute_effective_otfs()

    def _compute_effective_otfs(self):
        self.effective_otfs = self.optical_system.compute_effective_otfs_true_3dSIM(self.illumination)
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def _Dj(self):
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
        d_j *= self.illumination.Mt
        return d_j

    def _Vj(self):
        size_x, size_y, size_z = self.optical_system.otf.shape
        center = (size_x // 2, size_y // 2, size_z // 2)
        v_j = np.zeros(self.optical_system.otf_frequencies.shape, dtype=np.complex128)
        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = self.effective_otfs[idx1]
                otf2 = self.effective_otfs[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return v_j


class SSNR3dSIM2dShifts(SSNR3dSIMBase):
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.effective_otfs = {}
        self._compute_effective_otfs()

    def _compute_effective_otfs(self):
        self.effective_otfs = self.optical_system.compute_effective_otfs_projective_3dSIM(self.illumination)
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def _Dj(self):
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
            # plt.gca().set_title(f'{m}')
            # plt.imshow(np.abs(self.effective_otfs[m])[:, :, 50])
            # plt.show()
        d_j *= self.illumination.Mt
        return np.abs(d_j)

    def _Vj(self):
        size_x, size_y, size_z = self.optical_system.otf.shape
        center = np.array((size_x, size_y, size_z), dtype=np.int32) // 2
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = self.effective_otfs[idx1]
                otf2 = self.effective_otfs[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return np.abs(v_j)


class SSNR3dSIM2dShiftsFiniteKernel(SSNR3dSIM2dShifts):
    def __init__(self, illumination, optical_system, kernel, readout_noise_variance=0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.kernel_ft = None
        self.kernel = kernel

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
            kernel_expanded = np.zeros(otf_shape)
            kernel_expanded[otf_shape[0] // 2 - shape[0] // 2:otf_shape[0] // 2 + shape[0] // 2 + 1,
            otf_shape[1] // 2 - shape[1] // 2:otf_shape[1] // 2 + shape[1] // 2 + 1,
            otf_shape[2] // 2 - shape[2] // 2:otf_shape[2] // 2 + shape[2] // 2 + 1,
            ] = kernel_new
            kernel_new = kernel_expanded
        self._kernel = kernel_new
        self.kernel_ft = wrappers.wrapped_ifftn(kernel_new)
        self.kernel_ft /= np.amax(self.kernel_ft)
        self._kernel = kernel_new
        self.effective_kernels_ft = {}
        self._compute_effective_kernels_ft()
        self.compute_ssnr()

    def plot_effective_kernel_and_otf(self):
        Nx, Ny, Nz = self.optical_system.otf.shape
        fig, ax = plt.subplots()
        ax.plot(self.optical_system.otf_frequencies[0], self.kernel_ft[:, Ny // 2, Nz // 2], label="Kernel")
        ax.plot(self.optical_system.otf_frequencies[0], self.optical_system.otf[:, Ny // 2, Nz // 2], label="OTF")
        ax.set_title("Kernel vs OTF")
        ax.set_xlabel("$f_r, \\frac{2NA}{\lambda}$")
        ax.set_ylabel("OTF/K, u.e.")
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid()

    @SSNR3dSIM2dShifts.illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.otf_sim = None
        self._compute_effective_otfs()
        self._compute_effective_kernels_ft()
        self.ssnr = None

    def _compute_effective_kernels_ft(self):
        waves = self.illumination.waves
        X, Y, Z = np.meshgrid(*self.optical_system.psf_coordinates)
        for r in range(self.illumination.Mr):
            angle = self.illumination.angles[r]
            indices = self.illumination.rearranged_indices
            for xy_indices in self.illumination.indices2d:
                effective_real_kernel = 0
                for z_index in indices[xy_indices]:
                    wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                        waves[(*xy_indices, z_index)].wavevector, np.array((0, 0, 1)), angle)
                    amplitude = waves[(*xy_indices, z_index)].amplitude
                    phase_shifted = np.exp(1j * np.einsum('ijkl,i ->jkl', np.array((X, Y, Z)), wavevector)) * self.kernel
                    effective_real_kernel += amplitude * phase_shifted
                self.effective_kernels_ft[(r, xy_indices)] = wrappers.wrapped_fftn(effective_real_kernel)

    def _Dj(self):
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_kernels_ft[m].conjugate()
        d_j *= self.illumination.Mt
        return np.abs(d_j)

    def _Vj(self):
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        size_x, size_y, size_z = self.optical_system.otf.shape
        center = np.array((size_x, size_y, size_z), dtype=np.int32) // 2
        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idx_diff = (idx1[0], m21)
                otf1 = self.effective_kernels_ft[idx1]
                otf2 = self.effective_kernels_ft[idx2]
                otf3 = self.effective_otfs[idx_diff][*center]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return np.abs(v_j)


class SSNR3dSIMUniversal(SSNR3dSIMBase): ...
