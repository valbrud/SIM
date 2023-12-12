import numpy
import numpy as np
from stattools import average_ring
import VectorOperations


class SNRCalculator:
    def __init__(self, illumination, optical_system):
        self._illumination = illumination
        self._optical_system = optical_system
        self.vj_parameters = {}
        self.vj_otf_diffs = {}

        if optical_system.otf is None:
            raise AttributeError("Optical system otf is not computed")

        self._compute_parameters_for_Vj()
        self._compute_wfdiff_otfs_for_Vj()

    @property
    def optical_system(self):
        return self._optical_system

    @optical_system.setter
    def optical_system(self, new_optical_system):
        self._optical_system = new_optical_system
        self.vj_parameters = {}
        self.vj_otf_diffs = {}
        self._compute_parameters_for_Vj()
        self._compute_wfdiff_otfs_for_Vj()

    @property
    def illumination(self):
        return self._illumination

    @illumination.setter
    def illumination(self, new_illumination):
        self._illumination = new_illumination
        self.vj_parameters = {}
        self.vj_otf_diffs = {}
        self._compute_parameters_for_Vj()
        self._compute_wfdiff_otfs_for_Vj()
    class VjParrametersHolder:
        def __init__(self, a_m, otf, wavevector2d):
            self.a_m = a_m
            self.otf = otf
            self.wavevector = wavevector2d

    def Dj(self, q_grid, method="Fourier"):
        d_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for m in self.vj_parameters.keys():
            d_j += np.abs(self.vj_parameters[m].a_m)**2 * np.abs(self.vj_parameters[m].otf)**2
        d_j *= self.illumination.M_t
        print(d_j[25, 25, 25])
        return d_j
    def _compute_parameters_for_Vj(self):
        waves = self.illumination.waves
        for angle in self.illumination.angles:
            for indices in waves.keys():
                indices_dual = (indices[0], indices[1], -indices[2])
                wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                    waves[indices].wavevector, np.array((0, 0, 1)), angle)
                wavevector2d = wavevector[:2]
                wavevector_dual = np.array((wavevector[0], wavevector[1], -wavevector[2]))

                if indices[2] == 0:
                    otf = self.optical_system.interpolate_otf(wavevector)
                    self.vj_parameters[(indices[0], indices[1], angle)] = (
                        self.VjParrametersHolder(waves[indices].amplitude, otf, wavevector2d))
                else:
                    if indices[:2] not in self.vj_parameters.keys():
                        coeff = waves[indices_dual].amplitude/waves[indices].amplitude

                        otf = (self.optical_system.interpolate_otf(wavevector) +
                               coeff * self.optical_system.interpolate_otf(wavevector_dual))

                        indices_in_plane = (indices[0], indices[1], 0)
                        if indices_in_plane in waves.keys():
                            coeff_in_plane = waves[indices_in_plane].amplitude/waves[indices].amplitude
                            wavevector_in_plane = np.array((wavevector[0], wavevector[1], 0))
                            otf += coeff_in_plane * self.optical_system.interpolate_otf(wavevector_in_plane)

                        self.vj_parameters[(indices[0], indices[1], angle)] = (
                            self.VjParrametersHolder(waves[indices].amplitude, otf, wavevector2d))


    def _compute_wfdiff_otfs_for_Vj(self):
        indices = self.vj_parameters.keys()
        for angle in self.illumination.angles:
            for index1 in indices:
                for index2 in indices:
                    idx_diff = np.array(index1[:2]) - np.array(index2[:2])
                    idx_diff = (idx_diff[0], idx_diff[1], angle)
                    if idx_diff not in indices:
                        continue
                    wavevector1 = VectorOperations.VectorOperations.rotate_vector2d(
                        self.vj_parameters[index1].wavevector, angle)
                    wavevector2 = VectorOperations.VectorOperations.rotate_vector2d(
                        self.vj_parameters[index2].wavevector,  angle)
                    wvdiff = wavevector2 - wavevector1
                    wvdiff3d = np.array((wvdiff[0], wvdiff[1], 0))

                    index = (index1[0], index1[1], index2[0], index2[1], angle)
                    self.vj_otf_diffs[index] = self.optical_system.interpolate_otf_at_one_point(wvdiff3d, self.vj_parameters[idx_diff].otf)


    def Vj(self, q_grid):
        v_j = np.zeros((q_grid.shape[0], q_grid.shape[1], q_grid.shape[2]), dtype=np.complex128)
        for angle in self.illumination.angles:
            for m1 in self.vj_parameters.keys():
                for m2 in self.vj_parameters.keys():
                    if m1[2] != angle or m2[2] != angle:
                        continue
                    a_m1 = self.vj_parameters[m1].a_m
                    a_m2 = self.vj_parameters[m2].a_m
                    idx_diff = np.array(m1) - np.array(m2)
                    idx = (idx_diff[0], idx_diff[1], angle)
                    if idx not in self.vj_parameters.keys():
                        continue
                    a_m12 = self.vj_parameters[idx].a_m
                    otf1 = self.vj_parameters[m1].otf
                    otf2 = self.vj_parameters[m2].otf
                    otf3 = self.vj_otf_diffs[(m1[0], m1[1], m2[0], m2[1], angle)]
                    term = a_m1 * a_m2.conjugate() * a_m12 * otf1.conjugate() * otf2 * otf3
                    v_j += term
                    # if (k_m1 == np.array((0, 0, 0))).all() and (k_m2 == np.array((0, 0, 0))).all():
                    #     print(k_m1, k_m2, term[25,25,25])
        v_j *= self.illumination.M_t
        print(v_j[25, 25, 25])
        return v_j

    def SSNR(self, q_axes):
        qx, qy, qz = np.array(q_axes[0]), np.array(q_axes[1]), np.array(q_axes[2])
        q_vectors = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
        q_sorted = q_vectors[np.lexsort((q_vectors[:, -2], q_vectors[:, -1], q_vectors[:, 0]))]
        q_grid = q_sorted.reshape(qx.size, qy.size, qz.size, 3)
        dj = self.Dj(q_grid)
        ssnr = np.zeros(dj.shape, dtype=np.complex128)
        print(ssnr.shape)
        vj = self.Vj(q_grid)
        mask = (vj != 0) * (dj != 0)
        numpy.putmask(ssnr, mask, np.abs(dj) ** 2 / vj)
        print(ssnr[25, 25, 25])
        return ssnr

    def ring_average_SSNR(self, q_axes, SSNR):
        SSNR = np.copy(SSNR)
        if q_axes[0].size != SSNR.shape[0] or q_axes[1].size != SSNR.shape[1]:
            raise ValueError("Wrong axes are provided for the SSNR")
        averaged_slices = []
        for i in range(SSNR.shape[2]):
            averaged_slices.append(average_ring(SSNR[:, :, i], (q_axes[0], q_axes[1])))
        return np.array(averaged_slices).T


