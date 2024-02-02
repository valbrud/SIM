import numpy
import numpy as np
from stattools import average_ring
import VectorOperations


class SNRCalculator:
    def __init__(self, illumination, optical_system):
        self._illumination = illumination
        self._optical_system = optical_system
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}

        if optical_system.otf is None:
            raise AttributeError("Optical system otf is not computed")

        self._compute_effective_otfs()
        self._compute_otfs_at_point()

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

    @property
    def illumination(self):
        return self._illumination

    @illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.effective_otfs = {}
        self.effective_otfs_at_point_k_diff = {}
        self._compute_effective_otfs()
        self._compute_otfs_at_point()

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
        for angle in self.illumination.angles:
            indices = self._rearrange_indices(waves.keys())
            for xy_indices in indices.keys():
                otf = 0
                for z_index in indices[xy_indices]:
                    wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                        waves[(*xy_indices, z_index)].wavevector, np.array((0, 0, 1)), angle)
                    amplitude = waves[(*xy_indices, z_index)].amplitude
                    otf += amplitude * self.optical_system.interpolate_otf(wavevector)
                self.effective_otfs[(*xy_indices, angle)] = otf

    def _compute_otfs_at_point(self):
        indices = self.effective_otfs.keys()
        size_x, size_y, size_z = self.optical_system.otf.shape
        for index in indices:
            self.effective_otfs_at_point_k_diff[index] = self.effective_otfs[index][size_x//2, size_y//2, size_z//2]

    def Dj(self, q_grid):
        d_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_otfs[m].conjugate()
        d_j *= self.illumination.Mt
        return d_j

    def Vj(self, q_grid):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
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
                v_j += otf1 * otf2.conjugate() * otf3
                # if (k_m1 == np.array((0, 0, 0))).all() and (k_m2 == np.array((0, 0, 0))).all():
                #     print(k_m1, k_m2, term[25,25,25])
        v_j *= self.illumination.Mt
        return v_j

    def SSNR(self, q_axes):
        qx, qy, qz = np.array(q_axes[0]), np.array(q_axes[1]), np.array(q_axes[2])
        q_vectors = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -2], q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, qz.size, 3)
        dj = self.Dj(q_grid)
        ssnr = np.zeros(dj.shape, dtype=np.complex128)
        vj = self.Vj(q_grid)
        mask = (vj != 0) * (dj != 0)
        numpy.putmask(ssnr, mask, np.abs(dj) ** 2 / vj)
        return ssnr

    def ring_average_SSNR(self, q_axes, SSNR):
        SSNR = np.copy(SSNR)
        if q_axes[0].size != SSNR.shape[0] or q_axes[1].size != SSNR.shape[1]:
            raise ValueError("Wrong axes are provided for the SSNR")
        averaged_slices = []
        for i in range(SSNR.shape[2]):
            averaged_slices.append(average_ring(SSNR[:, :, i], (q_axes[0], q_axes[1])))
        return np.array(averaged_slices).T

    def compute_SSNR_volume(self, SSNR, volume_element, factor = 10**8):
        return np.sum(np.log10(1 + factor * np.abs(SSNR))) * volume_element

    def compute_analytic_SSNR_sum(self):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate().real)
        g0 = np.abs(np.amax(self.optical_system.otf))
        a0 = 1 / self.illumination.Mt
        weights = np.array([wave.amplitude for wave in self.illumination.waves.values()])
        weighted_sum = np.sum(weights * weights.conjugate().real)
        volume = (self.illumination.Mt * self.illumination.Mr)**2 * weighted_sum * g2 / g0
        return volume.real