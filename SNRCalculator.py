import numpy
import numpy as np
from stattools import average_ring
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
    @abstractmethod
    def optical_system(self, new_optical_system):
        ...

    @property
    def illumination(self):
        return self._illumination

    @illumination.setter
    @abstractmethod
    def illumination(self, new_illumination):
        ...

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
        return np.abs(ssnr)

    def ring_average_ssnr(self):
        q_axes = 2 * np.pi * self.optical_system.otf_frequencies
        ssnr = np.copy(self.ssnr)
        if q_axes[0].size != ssnr.shape[0] or q_axes[1].size != ssnr.shape[1]:
            raise ValueError("Wrong axes are provided for the ssnr")
        averaged_slices = []
        for i in range(ssnr.shape[2]):
            averaged_slices.append(average_ring(ssnr[:, :, i], (q_axes[0], q_axes[1])))
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

    def compute_maximum_resolved(self):
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

    def compute_true_ssnr_entropy(self, factor=1):
        noise_filtered = self.ssnr[self.ssnr > 10 ** (-10) * np.amax(self.ssnr)]
        sum = np.sum(noise_filtered)
        probabilities = noise_filtered / sum
        S = -np.sum(probabilities * np.log(probabilities))
        return S.real * factor

    def compute_radial_ssnr_entropy(self, factor=1):
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


    @SSNRCalculator3dSIM.optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()
        self._compute_ssnr_widefield()
        self.ssnr = None  # to avoid heavy computations where they are not needed
    @SSNRCalculator3dSIM.illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()
        self.ssnr = None
        # to avoid heavy computations where they are not needed

    def _compute_effective_otfs(self):
        waves = self.illumination.waves
        plt.show()
        for angle in self.illumination.angles:
            for index in waves.keys():
                wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                    waves[index].wavevector, np.array((0, 0, 1)), angle)
                amplitude = waves[index].amplitude
                # interpolated_otf = self.optical_system.interpolate_otf(wavevector)
                # print(xy_indices, " ", z_index, " ", np.sum(interpolated_otf), " ", amplitude)
                shifted_otf = amplitude * self.optical_system.interpolate_otf(wavevector)
                # plt.imshow(np.log10(1 + 10 ** 16 * np.abs(effective_otf[50, :, :].T)))
                # print(np.abs(np.amin(effective_otf)), " ", np.abs(np.amax(effective_otf)))
                self.effective_otfs[(*index, angle)] = shifted_otf

    def _compute_otfs_at_point(self):
        indices = self.effective_otfs.keys()
        size_x, size_y, size_z = self.optical_system.otf.shape
        for index in indices:
            self.effective_otfs_at_point_k_diff[index] = self.effective_otfs[index][size_x // 2, size_y // 2, size_z // 2]

    def _Dj(self, q_grid):
        d_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
            # plt.imshow(np.log10(1 + 10**16 * d_j[:, :, 50].real))
            # plt.imshow(np.log10(self.effective_otfs[m] * self.effective_otfs[m].conjugate())[:, :, 50].real)
            # plt.show()

            # print(m)
        # print(self.effective_otfs[(0, 0, 0.0)][45:55, 50, 50])
        # print((d_j * self.effective_otfs[(0, 0, 0.0)][50, 50, 50])[45:55, 50, 50])
        d_j *= self.illumination.Mt
        return d_j

    def _Vj(self, q_grid):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        test_main = 0
        test_additional = 0
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
                # print(m2, m1)
                term = otf1 * otf2.conjugate() * otf3
                if m1 == m2:
                    # print(m1)
                    # print(otf3)
                    test_main += term
                else:
                    test_additional += term
                v_j += term
                # print(otf1, otf2, otf3)
        # print((test_main[45:55, 50, 50]))
        v_j *= self.illumination.Mt
        return v_j


class SSNRCalculatorProjective3dSIM(SSNRCalculator3dSIM):
    def __init__(self, illumination, optical_system, readout_noise_variance = 0):
        super().__init__(illumination, optical_system, readout_noise_variance)
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()


    @SSNRCalculator3dSIM.optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()
        self._compute_ssnr_widefield()
        self.ssnr = None  # to avoid heavy computations where they are not needed
    @SSNRCalculator3dSIM.illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()
        self.ssnr = None
        # to avoid heavy computations where they are not needed
    def _rearrange_indices(self, indices):
        result_dict = {}
        for index in indices:
            key = index[:2]
            value = index[2]
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
        result_dict = {key: tuple(values) for key, values in result_dict.items()}
        # print(result_dict)
        return result_dict

    def _compute_effective_otfs(self):
        waves = self.illumination.waves
        plt.show()
        for angle in self.illumination.angles:
            indices = self._rearrange_indices(waves.keys())
            for xy_indices in indices.keys():
                effective_otf = 0
                for z_index in indices[xy_indices]:
                    wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                        waves[(*xy_indices, z_index)].wavevector, np.array((0, 0, 1)), angle)
                    amplitude = waves[(*xy_indices, z_index)].amplitude
                    # interpolated_otf = self.optical_system.interpolate_otf(wavevector)
                    # print(xy_indices, " ", z_index, " ", np.sum(interpolated_otf), " ", amplitude)
                    effective_otf += amplitude * self.optical_system.interpolate_otf(wavevector)
                # plt.imshow(np.log10(1 + 10 ** 16 * np.abs(effective_otf[50, :, :].T)))
                # plt.show()
                # print(np.abs(np.amin(effective_otf)), " ", np.abs(np.amax(effective_otf)))
                self.effective_otfs[(*xy_indices, angle)] = effective_otf

    def _compute_otfs_at_point(self):
        indices = self.effective_otfs.keys()
        size_x, size_y, size_z = self.optical_system.otf.shape
        for index in indices:
            self.effective_otfs_at_point_k_diff[index] = self.effective_otfs[index][size_x // 2, size_y // 2, size_z // 2]

    def _Dj(self, q_grid):
        d_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
            # plt.imshow(np.log10(1 + 10**16 * d_j[:, :, 50].real))
            # plt.imshow(np.log10(self.effective_otfs[m] * self.effective_otfs[m].conjugate())[:, :, 50].real)
            # plt.show()

            # print(m)
        # print(self.effective_otfs[(0, 0, 0.0)][45:55, 50, 50])
        # print((d_j * self.effective_otfs[(0, 0, 0.0)][50, 50, 50])[45:55, 50, 50])
        d_j *= self.illumination.Mt
        return np.abs(d_j)

    def _Vj(self, q_grid):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        test_main = 0
        test_additional = 0
        for m1 in self.effective_otfs.keys():
            for m2 in self.effective_otfs.keys():
                if m1[2] != m2[2]:
                    continue
                xy_idx_diff = np.array(m2)[:2] - np.array(m1)[:2]
                idx_diff = (*xy_idx_diff, m1[2])
                if idx_diff not in self.effective_otfs_at_point_k_diff.keys():
                    continue
                otf1 = self.effective_otfs[m1]
                otf2 = self.effective_otfs[m2]
                otf3 = self.effective_otfs_at_point_k_diff[idx_diff]
                # print(m2, m1)
                term = otf1 * otf2.conjugate() * otf3
                if m1 == m2:
                    # print(m1)
                    # print(otf3)
                    test_main += term
                else:
                    test_additional += term
                v_j += term
                # print(otf1, otf2, otf3)
        # print((test_main[45:55, 50, 50]))
        v_j *= self.illumination.Mt
        return np.abs(v_j)

