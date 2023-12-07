import numpy
import numpy as np
import scipy as sp
import scipy.interpolate
import wrappers
from stattools import average_ring
import VectorOperations, ApodizationFilters
from abc import abstractmethod

class OpticalSystem:
    # Other scipy interpolation methods are not directly supported, because they require too much computation time.
    # Nevertheless, just add them to the list if needed
    supported_interpolation_methods = ["linear", "Fourier"]
    def __init__(self, interpolation_method):
        self.psf = None
        self.otf = None
        self._otf_frequencies = None
        self._psf_coordinates = None
        self._interpolation_method = None
        self.interpolation_method = interpolation_method
        self._shifted_otfs = {}
        self._wvdiff_otfs = {}

    def compute_psf_and_otf_cordinates(self, psf_size, N):

        dx = psf_size[0] / N
        dy = psf_size[1] / N
        dz = psf_size[2] / N

        x = np.arange(-psf_size[0] / 2, psf_size[0] / 2, dx)
        y = np.arange(-psf_size[1] / 2, psf_size[1] / 2, dy)
        z = np.arange(-psf_size[2] / 2, psf_size[2] / 2, dz)

        self.psf_coordinates = np.array((x, y, z))

    @property
    def psf_coordinates(self):
        return self._psf_coordinates

    @psf_coordinates.setter
    def psf_coordinates(self, new_coordinates):
        self._psf_coordinates = new_coordinates
        x, y, z = new_coordinates
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        dfx = 1 / (2 * np.abs(x[0]))
        dfy = 1 / (2 * np.abs(y[0]))
        dfz = 1 / (2 * np.abs(z[0]))

        fx = np.arange(-1 / (2 * dx), 1 / (2 * dx) - dfx/10, dfx)
        fy = np.arange(-1 / (2 * dy), 1 / (2 * dy) - dfx/10, dfy)
        fz = np.arange(-1 / (2 * dz), 1 / (2 * dz) - dfx/10, dfz)
        self._otf_frequencies = np.array((fx, fy, fz))

    @property
    def otf_frequencies(self):
        return self._otf_frequencies

    @property
    def interpolation_method(self):
        return self._interpolation_method

    @interpolation_method.setter
    def interpolation_method(self, new_method):
        if new_method not in OpticalSystem.supported_interpolation_methods:
            raise AttributeError("Interpolation method ", new_method, " is not supported")
        self._interpolation_method = new_method

    @abstractmethod
    def compute_psf_and_otf(self): ...
    def prepare_Fourier_interpolation(self, wavevectors):
        self._compute_shifted_otfs(wavevectors)
        self._compute_wvdiff_otfs(wavevectors)

    def _compute_shifted_otfs(self, wavevectors):
        x, y, z = self.psf_coordinates[0], self.psf_coordinates[1], self.psf_coordinates[2]
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        for wavevector in wavevectors:
            if np.sum(np.abs(wavevector)) == 0:
                self._shifted_otfs[tuple(wavevector)] = self.otf
            else:
                phases = np.einsum('ij, j -> i', c_vectors_sorted, wavevector)
                phases = phases.reshape((len(x), len(y), len(z)))
                psf_phase_shifted = self.psf * np.exp(-1j * phases)
                self._shifted_otfs[tuple(wavevector)] = np.abs(wrappers.wrapped_ifftn(psf_phase_shifted))
        return self._shifted_otfs

    def _compute_wvdiff_otfs(self, wv_group1, wv_group2=None):
        if not wv_group2:
            wv_group2 = np.copy(wv_group1)

        x, y, z = self.psf_coordinates[0], self.psf_coordinates[1], self.psf_coordinates[2]
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        for wv1 in wv_group1:
            for wv2 in wv_group2:
                wvdiff = wv2 - wv1
                if not tuple(wvdiff) in self._wvdiff_otfs.keys():
                    phases = np.dot(c_vectors_sorted, wvdiff)
                    phases = phases.reshape((len(x), len(y), len(z)))
                    psf_phase_shifted = self.psf * np.exp(1j * phases)
                    self._wvdiff_otfs[tuple(wvdiff)] = np.sum(psf_phase_shifted) / (len(x) * len(y) * len(z))
        return self._wvdiff_otfs

    def interpolate_otf(self, k_shift):
        if self.interpolation_method == "Fourier":
            if tuple(k_shift) not in self._shifted_otfs.keys():
                print("Shifted otf for a wavevector ", k_shift,
                      " is not stored for this optical system. Computing...")
                self._compute_shifted_otfs((k_shift, ))
            return self._shifted_otfs[tuple(k_shift)]

        else:
            axes = 2 * np.pi * self.otf_frequencies
            otf = self.otf
            interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method=self._interpolation_method,
                                                                     bounds_error=False,
                                                                     fill_value=0.)
            qx = (2 * np.pi * self.otf_frequencies[0] - k_shift[0])
            qy = (2 * np.pi * self.otf_frequencies[1] - k_shift[1])
            qz = (2 * np.pi * self.otf_frequencies[2] - k_shift[2])
            interpolation_points = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
            interpolation_points = interpolation_points[
                np.lexsort((interpolation_points[:, 2], interpolation_points[:, 1],
                            interpolation_points[:, 0]))]
            otf_interpolated = interpolator(interpolation_points)
            otf_interpolated = otf_interpolated.reshape(self.otf_frequencies[0].size, self.otf_frequencies[1].size,
                                                        self.otf_frequencies[2].size)
            return otf_interpolated

    def interpolate_otf_at_one_point(self, q, otf=None):
        axes = 2 * np.pi * self.otf_frequencies
        if otf is None:
            otf = self.otf

        if self.interpolation_method == "Fourier":
            if tuple(q) not in self._wvdiff_otfs:
                print("Shifted otf for a wavevector ", q, " is not stored for this optical system. Computing...")
                self._compute_wvdiff_otfs((q, np.array((0, 0, 0))))
            return self._wvdiff_otfs[tuple(q)]

        else:
            interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method=self._interpolation_method,
                                                                     bounds_error=False,
                                                                     fill_value=0.)
            otf_interpolated = interpolator(q)
            return otf_interpolated

class Lens(OpticalSystem):
    def __init__(self, alpha=np.pi/4, regularization_parameter=0.01, interpolation_method="linear"):
        super().__init__(interpolation_method)
        self.alpha = alpha
        self.e = regularization_parameter / (4 * np.sin(self.alpha / 2) ** 2)

    def PSF(self, c_vectors, apodization_filter = None):
        r = (c_vectors[:, :, :, 0] ** 2 + c_vectors[:, :, :, 1] ** 2) ** 0.5
        z = c_vectors[:, :, :, 2]
        v = 2 * np.pi * r * np.sin(self.alpha)
        u = 8 * np.pi * z * np.sin(self.alpha / 2) ** 2

        def integrand(rho):
            return np.exp(- 1j * (u[:, :, :, None] / 2 * rho ** 2)) * sp.special.j0(
                rho * v[:, :, :, None]) * 2 * np.pi * rho

        rho = np.linspace(0, 1, 100)
        integrands = integrand(rho)
        h = sp.integrate.simpson(integrands, rho)
        I = (h * h.conjugate()).real
        if apodization_filter:
            shape = np.array(c_vectors.shape[:-1])
            mask = apodization_filter(shape, np.amin(shape)//5)
            I *= mask
        return I

    # Could not get good numbers yet
    def regularized_analytic_OTF(self, f_vector):
        f_r = (f_vector[0] ** 2 + f_vector[1] ** 2) ** 0.5
        f_z = f_vector[2]
        l = f_r / np.sin(self.alpha)
        s = f_z / (4 * np.sin(self.alpha / 2) ** 2)
        if l > 2:
            return 0

        def p_max(theta):
            D = 4 - l ** 2 * (1 - np.cos(theta) ** 2)
            return (-l * np.cos(theta) + D ** 0.5) / 2

        def integrand(p, theta):
            denum = self.e ** 2 + (abs(s) - p * l * np.cos(theta)) ** 2
            return 8 * self.e * p / denum

        otf, _ = sp.integrate.dblquad(integrand, 0, np.pi / 2, lambda x: 0, p_max)
        return (otf)

    def compute_psf_and_otf(self, parameters = None, apodization_filter = None):
        if not self.psf_coordinates and not parameters:
            raise AttributeError("Compute psf first or provide psf parameters")
        elif parameters:
            psf_size, N = parameters
            self.compute_psf_and_otf_cordinates(psf_size, N)
        self._wvdiff_otfs = {}
        self._shifted_otfs = {}
        x, y, z = self.psf_coordinates
        N = x.size
        c_vectors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
        c_vectors_sorted = c_vectors[np.lexsort((c_vectors[:, 2], c_vectors[:, 1], c_vectors[:, 0]))]
        grid = c_vectors_sorted.reshape((len(x), len(y), len(z), 3))
        psf = self.PSF(grid, apodization_filter)
        self.psf = psf / np.sum(psf[:, :, int(N / 2)])
        self.otf = wrappers.wrapped_ifftn(self.psf)






class Illumination:
    def __init__(self, intensity_plane_waves, spacial_shifts=(np.array((0, 0, 0)), ), M_r = 1):
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / M_r)
        self._spacial_shifts = spacial_shifts
        self._M_r = M_r
        self.M_t = len(self.spacial_shifts)
        self.waves = intensity_plane_waves

    @property
    def M_r(self):
        return self._M_r

    @M_r.setter
    def M_r(self, new_M_r):
        self.M_r = new_M_r
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / new_M_r)

    @property
    def spacial_shifts(self):
        return self._spacial_shifts

    @spacial_shifts.setter
    def spacial_shifts(self, new_spacial_shifts):
        self._spacial_shifts = new_spacial_shifts
        self.M_t = len(new_spacial_shifts)

    def get_wavevectors(self):
        wavevectors = []
        for angle in self.angles:
            for spacial_wave in self.waves.values():
                wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                    spacial_wave.wavevector, np.array((0, 0, 1)), angle)
                wavevectors.append(wavevector)
        return wavevectors


class NoiseEstimator:
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


