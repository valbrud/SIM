import numpy
import numpy as np
from stattools import average_rings2d
import VectorOperations
import matplotlib.pyplot as plt
from abc import abstractmethod
class SSNRCalculator3dSIM:
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        self._illumination = illumination
        self._optical_system = optical_system
        self.ssnr_widefield = None
        self.ssnr = None
        self.vj = None
        self.dj = None
        self.otf_sim = None
        self.maximum_resolved = 0
        self.readout_noise_variance = readout_noise_variance

        if optical_system.otf is None:
            raise AttributeError("Optical system otf is not computed")

        self._compute_ssnr_widefield()

    def _compute_ssnr_widefield(self):
        self.ssnr_widefield = np.abs(self.optical_system.otf) ** 2 / np.amax(np.abs(self.optical_system.otf))

    @property
    def optical_system(self):
        return self._optical_system

    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()
        self._compute_ssnr_widefield()
        self.ssnr = None  # to avoid heavy computations where they are not needed
    @property
    def illumination(self):
        return self._illumination
    @illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.otf_sim = None
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()
        self.ssnr = None
        # to avoid heavy computations where they are not needed
    @abstractmethod
    def _compute_effective_otfs(self): ...

    @abstractmethod
    def _compute_otfs_at_point(self): ...
    @abstractmethod
    def _Dj(self, q_grid): ...

    @abstractmethod
    def _Vj(self, q_grid): ...

    def compute_ssnr(self):
        q_axes = 2 * np.pi * self.optical_system.otf_frequencies
        qx, qy, qz = np.array(q_axes[0]), np.array(q_axes[1]), np.array(q_axes[2])
        q_vectors = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -2], q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, qz.size, 3)
        dj = self._Dj(q_grid)
        ssnr = np.zeros(dj.shape, dtype=np.complex128)
        vj = self._Vj(q_grid)
        mask = (vj != 0) * (dj != 0)
        numpy.putmask(ssnr, mask, np.abs(dj) ** 2 / vj)
        self.dj = dj
        self.vj = vj
        # approximation_quality = np.abs(self.dj / self.vj * self.effective_otfs[(0, 0, 0)][50, 50, 50])
        self.ssnr = ssnr
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted2sum = np.sum(weights * weights.conjugate()).real
        expected = self.illumination.Mt * self.illumination.Mr * g2 * weighted2sum
        observed = np.abs(np.sum(dj))
        # print(observed / expected)
        # print(np.amin(approximation_quality), np.abs(np.sum(approximation_quality * np.abs(self.ssnr)) / np.sum(self.ssnr)), np.amax(approximation_quality))
        # plt.imshow(np.log(1 + 10**4 * np.abs(ssnr[:, :, 50])))
        # plt.show()
        return np.abs(ssnr)

    def ring_average_ssnr(self):
        q_axes = 2 * np.pi * self.optical_system.otf_frequencies
        ssnr = np.copy(self.ssnr)
        if q_axes[0].size != ssnr.shape[0] or q_axes[1].size != ssnr.shape[1]:
            raise ValueError("Wrong axes are provided for the ssnr")
        averaged_slices = []
        for i in range(ssnr.shape[2]):
            averaged_slices.append(average_rings2d(ssnr[:, :, i], (q_axes[0], q_axes[1])))
        return np.array(averaged_slices).T

    def compute_ssnr_volume(self, factor=10, volume_element=1):
        return np.sum(np.abs(self.ssnr)) * volume_element * factor

    def compute_analytic_ssnr_volume(self, factor=10 , volume_element=1):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        g0 = np.abs(np.amax(self.optical_system.otf))
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted2sum = np.sum(weights * weights.conjugate()).real
        volume = ((self.illumination.Mt * self.illumination.Mr) ** 2 * weighted2sum * g2 /
                  g0 * volume_element * factor)
        return volume

    def compute_total_ssnr(self, factor=10 , volume_element=1):
        mask = (self.dj != 0) * (self.vj != 0)
        total_signal_power = np.sum(np.abs(self.dj[mask]) ** 2)
        total_noise_power = np.sum(np.abs(self.vj[mask]))
        return total_signal_power / total_noise_power * volume_element * factor

    def compute_analytic_total_ssnr(self, factor=10 , volume_element=1):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        g4 = np.sum(self.optical_system.otf ** 2 * self.optical_system.otf.conjugate() ** 2).real
        g0 = np.abs(np.amax(self.optical_system.otf))
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted2sum = np.sum(weights * weights.conjugate()).real
        weighted4sum = np.sum(weights ** 2 * weights.conjugate() ** 2).real
        total = ((self.illumination.Mt * self.illumination.Mr) ** 2 * weighted4sum * g4 /
                 weighted2sum / g2 / g0 * volume_element * factor)
        return total

    def _find_threshold_value(self, stock, max, min, noise_level):
        average = (max + min) / 2
        less_min = self.ssnr_widefield[(self.ssnr_widefield < min) * (self.ssnr > noise_level)]
        less_max = self.ssnr_widefield[(self.ssnr_widefield < max) * (self.ssnr > noise_level)]
        if less_max.size == less_min.size:
            return average
        less = self.ssnr_widefield[(self.ssnr_widefield < average) * (self.ssnr > noise_level)]
        sum_less = np.sum(less)
        fill = less.size * max - sum_less
        if fill > stock:
            return self._find_threshold_value(stock, average, min, noise_level)
        else:
            return self._find_threshold_value(stock, max, average, noise_level)

    def compute_maximum_resolved_lateral(self):
        fR = 2 * self.optical_system.n * np.sin(self.optical_system.alpha)
        fourier_peaks_wavevectors = np.array([spacial_wave.wavevector for spacial_wave in self.illumination.waves.values()])
        fI = np.max(np.array([(wavevector[0] ** 2 + wavevector[1] ** 2) ** 0.5 for wavevector in fourier_peaks_wavevectors]))
        return fR + fI

    def compute_ssnr_waterline_measure(self, factor=10):
        diff = np.sum(self.ssnr - self.ssnr_widefield).real
        upper_estimate = np.abs(np.amax(self.ssnr - self.ssnr_widefield))
        noise_level = 10 ** -10 * np.abs(np.amax(self.ssnr))
        threshold = self._find_threshold_value(diff, upper_estimate, 0, noise_level)
        measure = np.where((np.abs(self.ssnr_widefield) < threshold) * (np.abs(self.ssnr) > noise_level), np.abs(self.ssnr - self.ssnr_widefield), 0)
        measure = np.where(measure < threshold, measure, threshold)
        return np.sum(measure) * factor, threshold

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

class SSNRCalculatorTrue3dSIM(SSNRCalculator3dSIM):
    def __init__(self, illumination, optical_system, readout_noise_variance=0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()

    def _compute_effective_otfs(self):
        self.effective_otfs = self.optical_system.compute_effective_otfs_true_3dSIM(self.illumination)
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def _compute_otfs_at_point(self):
        indices = self.effective_otfs.keys()
        size_x, size_y, size_z = self.optical_system.otf.shape
        for index in indices:
            self.effective_otfs_at_point_k_diff[index] = self.effective_otfs[index][size_x // 2, size_y // 2, size_z // 2]

    def _Dj(self, q_grid):
        d_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
        d_j *= self.illumination.Mt
        return d_j

    def _Vj(self, q_grid):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m1 in self.effective_otfs.keys():
            for m2 in self.effective_otfs.keys():
                if m1[3] != m2[3]:
                    continue
                idx_diff = np.array(m2)[:3] - np.array(m1)[:3]
                idx_diff = (*idx_diff, m1[3])
                if idx_diff not in self.effective_otfs_at_point_k_diff.keys():
                    continue
                otf1 = self.effective_otfs[m1]
                otf2 = self.effective_otfs[m2]
                otf3 = self.effective_otfs_at_point_k_diff[idx_diff]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        return v_j


class SSNRCalculatorProjective3dSIM(SSNRCalculator3dSIM):
    def __init__(self, illumination, optical_system, readout_noise_variance = 0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()

    def _compute_effective_otfs(self):
        self.effective_otfs = self.optical_system.compute_effective_otfs_projective_3dSIM(self.illumination)
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def _compute_otfs_at_point(self):
        indices = self.effective_otfs.keys()
        size_x, size_y, size_z = self.optical_system.otf.shape
        for index in indices:
            self.effective_otfs_at_point_k_diff[index] = self.effective_otfs[index][size_x // 2, size_y // 2, size_z // 2]

    def _Dj(self, q_grid):
        d_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
        d_j *= self.illumination.Mt
        return np.abs(d_j)

    def _Vj(self, q_grid):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if m21 not in self.illumination.indices2d:
                    continue
                idxdiff = (idx1[0], m21)
                otf1 = self.effective_otfs[idx1]
                otf2 = self.effective_otfs[idx2]
                otf3 = self.effective_otfs_at_point_k_diff[idxdiff]
                term = otf1 * otf2.conjugate() * otf3
                v_j += term
        v_j *= self.illumination.Mt
        # plt.imshow(np.log(1 + 10**4 * np.abs(v_j[:, :, 50])))
        # plt.show()
        return np.abs(v_j)

