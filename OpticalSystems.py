import numpy as np
import scipy as sp
import scipy.interpolate
import wrappers
import ApodizationFilters
from abc import abstractmethod
class OpticalSystem:
    # Other scipy interpolation methods are not directly supported, because they require too much computation time.
    # Nevertheless, just add them to the list if needed
    supported_interpolation_methods = ["linear", "Fourier"]
    def __init__(self, interpolation_method):
        self.psf = None
        self.otf = None
        self.interpolator = None
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

    def _prepare_interpolator(self):
        if self.otf_frequencies is None or self.otf is None:
            raise AttributeError("OTF or axes are not computed yet. This method can not be called at this stage")
        axes = 2 * np.pi * self.otf_frequencies
        otf = self.otf
        self.interpolator = scipy.interpolate.RegularGridInterpolator(axes, otf, method=self._interpolation_method,
                                                                 bounds_error=False,
                                                                 fill_value=0.)

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
            qx = (2 * np.pi * self.otf_frequencies[0] - k_shift[0])
            qy = (2 * np.pi * self.otf_frequencies[1] - k_shift[1])
            qz = (2 * np.pi * self.otf_frequencies[2] - k_shift[2])
            interpolation_points = np.array(np.meshgrid(qx, qy, qz)).T.reshape(-1, 3)
            interpolation_points = interpolation_points[
                np.lexsort((interpolation_points[:, 2], interpolation_points[:, 1],
                            interpolation_points[:, 0]))]
            if self.interpolator is None:
                raise AttributeError("Interpolator does not exist. Compute OTF to prepare the interpolator")

            otf_interpolated = self.interpolator(interpolation_points)
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
    def __init__(self, alpha=np.pi/4, refractive_index=1,  regularization_parameter=0.01, interpolation_method="linear"):
        super().__init__(interpolation_method)
        self.n = refractive_index
        self.alpha = alpha
        self.e = regularization_parameter / (4 * np.sin(self.alpha / 2) ** 2)

    def PSF(self, c_vectors, apodization_filter = None):
        r = (c_vectors[:, :, :, 0] ** 2 + c_vectors[:, :, :, 1] ** 2) ** 0.5
        z = c_vectors[:, :, :, 2]
        v = 2 * np.pi * r * self.n * np.sin(self.alpha)
        u = 8 * np.pi * z * self.n * np.sin(self.alpha / 2) ** 2

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

    def compute_psf_and_otf(self, parameters=None, apodization_filter=None):
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
        self.otf = np.abs(wrappers.wrapped_ifftn(self.psf)).astype(complex)
        self._prepare_interpolator()

