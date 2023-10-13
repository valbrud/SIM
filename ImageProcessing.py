import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import wrappers
import multiprocessing as mp
import time
n = 0
class Lens:
    def __init__(self, alpha = 0.3, regularization_parameter = 0.01):
        self.alpha = alpha
        self.e = regularization_parameter / (4 * np.sin(self.alpha/2)**2)

        self.psf = None
        self.otf = None
        self.otf_frequencies = None
        self.psf_coordinates = None
        self.shifted_otfs = {}
        self.wvdiff_otfs = {}
    def compute_PSF_and_OTF(self, psf_size, N):

        dx = psf_size[0]/N
        dy = psf_size[1]/N
        dz = psf_size[2]/N

        x = np.arange(-psf_size[0]/2, psf_size[0]/2, dx)
        y = np.arange(-psf_size[1]/2, psf_size[1]/2, dy)
        z = np.arange(-psf_size[2]/2, psf_size[2]/2, dz)

        self.psf_coordinates = [x, y, z]

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / psf_size[0], N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / psf_size[1], N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / psf_size[2], N)

        self.otf_frequencies = [fx, fy, fz]

        psf = np.zeros((N, N, N))

        for i, j, k in [(i, j, k) for i in range(len(x)) for j in range(len(y)) for k in range(len(z))]:
            c_vector = np.array((x[i], y[j], z[k]))
            psf[i, j, k] = self.PSF(c_vector)

        self.psf = psf / np.sum(psf[:, :, int(N/2)])
        self.otf = np.abs(wrappers.wrapped_ifftn(self.psf))

    def compute_wvdiff_otfs(self, wv_group1, wv_group2 = None):
        begin = time.time()
        if not wv_group2:
            wv_group2 = np.copy(wv_group1)
        for wv1 in wv_group1:
            for wv2 in wv_group2:
                wvdiff = wv2 - wv1
                if not tuple(wvdiff) in self.wvdiff_otfs.keys():
                    x = self.psf_coordinates[0]
                    y = self.psf_coordinates[1]
                    z = self.psf_coordinates[2]
                    psf_phase_shifted = np.zeros((len(x), len(y), len(z)), dtype=np.complex128)
                    for i, j, k in [(i, j, k) for i in range(len(x)) for j in range(len(y)) for k in range(len(z))]:
                        c_vector = np.array((x[i], y[j], z[k]))
                        psf_phase_shifted[i, j, k] = self.psf[i, j, k] * np.exp(1j * np.dot(wvdiff, c_vector))
                    self.wvdiff_otfs[tuple(wvdiff)] = (
                        np.abs(wrappers.wrapped_ifftn(psf_phase_shifted)))[int(len(x)/2), int(len(y)/2), int(len(z)/2)]
        end = time.time()
        return self.wvdiff_otfs
    def compute_shifted_otf(self, wavevectors):
        for wavevector in wavevectors:
            if np.sum(np.abs(wavevector)) == 0:
                self.shifted_otfs[tuple(wavevector)] = self.otf
            else:
                x = self.psf_coordinates[0]
                y = self.psf_coordinates[1]
                z = self.psf_coordinates[2]
                psf_phase_shifted = np.zeros((len(x), len(y), len(z)), dtype = np.complex128)
                for i, j, k in [(i, j, k) for i in range(len(x)) for j in range(len(y)) for k in range(len(z))]:
                    c_vector = np.array((x[i], y[j], z[k]))
                    psf_phase_shifted[i, j, k] = self.psf[i, j, k] * np.exp(1j * np.dot(wavevector, c_vector))
                self.shifted_otfs[tuple(wavevector)] = np.abs(wrappers.wrapped_ifftn(psf_phase_shifted))
        return self.shifted_otfs

    def get_otf(self, q_j):
        f_j = q_j / (2 * np.pi)
        f_x = f_j[0]
        f_y = f_j[1]
        f_z = f_j[2]

        if abs(f_x) >= abs(self.otf_frequencies[0][-1]) \
                or abs(f_y) >= abs(self.otf_frequencies[1][-1]) \
                or abs(f_z) >= abs(self.otf_frequencies[2][-1]):
            return 0

        idx = np.abs(self.otf_frequencies[0] - f_x).argmin()
        idy = np.abs(self.otf_frequencies[1] - f_y).argmin()
        idz = np.abs(self.otf_frequencies[2] - f_z).argmin()

        # if idy == 19:
        #     ...
        diff_x = f_x - self.otf_frequencies[0][idx]
        diff_y = f_y - self.otf_frequencies[1][idy]
        diff_z = f_z - self.otf_frequencies[2][idz]

        if self.otf_frequencies[0][idx] < f_x:
            der_x = ((self.otf[idx + 1, idy, idz] - self.otf[idx, idy, idz])
                     / (self.otf_frequencies[0][idx + 1] - self.otf_frequencies[0][idx]))
        else:
            der_x = ((self.otf[idx, idy, idz] - self.otf[idx - 1, idy, idz])
                     / (self.otf_frequencies[0][idx] - self.otf_frequencies[0][idx - 1]))

        if self.otf_frequencies[0][idy] < f_y:
            der_y = ((self.otf[idx, idy + 1, idz] - self.otf[idx, idy, idz])
                     / (self.otf_frequencies[0][idy + 1] - self.otf_frequencies[0][idy]))
        else:
            der_y = ((self.otf[idx, idy, idz] - self.otf[idx, idy - 1, idz])
                     / (self.otf_frequencies[0][idy] - self.otf_frequencies[0][idy - 1]))

        if self.otf_frequencies[0][idz] < f_z:
            der_z = ((self.otf[idx, idy, idz + 1] - self.otf[idx, idy, idz])
                     / (self.otf_frequencies[0][idz + 1] - self.otf_frequencies[0][idz]))
        else:
            der_z = ((self.otf[idx, idy, idz] - self.otf[idx, idy, idz - 1])
                     / (self.otf_frequencies[0][idz] - self.otf_frequencies[0][idz - 1]))

        otf = self.otf[idx, idy, idz] + der_x * diff_x + der_y * diff_y + der_z * diff_z

        return otf


    def PSF(self, c_vector):
        r = (c_vector[0]**2 + c_vector[1]**2)**0.5
        z = c_vector[2]
        v = 2 * np.pi * r * np.sin(self.alpha)
        u = 2 * np.pi * z * np.sin(self.alpha/2)**2

        def integrand(rho):
            return np.exp(- 1j * (u / 2 * rho**2)) * sp.special.j0(rho * v) * 2 * np.pi * rho

        rho = np.linspace(0, 1, 100)
        integrands = integrand(rho)
        h = sp.integrate.simpson(integrands, rho)
        I = (h * h.conjugate()).real
        return I

    #Could not get good numbers yet
    def regularized_analytic_OTF(self, f_vector):
        f_r = (f_vector[0] ** 2 + f_vector[1] ** 2) ** 0.5
        f_z = f_vector[2]
        l = f_r / np.sin(self.alpha)
        s = f_z / (4 * np.sin(self.alpha/2)**2)
        if l > 2:
            return 0
        def p_max(theta):
            D = 4 - l ** 2 * (1 - np.cos(theta)**2)
            return (-l * np.cos(theta) + D**0.5)/2

        def integrand(p, theta):
            denum = self.e ** 2 + (abs(s) - p * l * np.cos(theta))**2
            return 8 * self.e * p / denum

        otf, _ = sp.integrate.dblquad(integrand, 0, np.pi / 2, lambda x: 0, p_max)
        return(otf)




class Illumination:
    def __init__(self, intensity_plane_waves, spacial_shifts):
        self.spacial_shifts = spacial_shifts
        self.M_t = len(self.spacial_shifts)
        self.waves = intensity_plane_waves

class NoiseEstimator:
    def __init__(self, illumination, optical_system):
        self.illumination = illumination
        self.optical_system = optical_system
    def Dj(self, q_j, indices):
        d_j = 0
        if np.isclose(q_j[2], 0) and np.isclose(q_j[1], 0) and np.abs(q_j[0] - 9) < 1:
            print(q_j[0]/(2 * np.pi))
            ...
        for m in range(len(self.illumination.waves)):
            a_m = self.illumination.waves[m].amplitude
            k_m = self.illumination.waves[m].wavevector
            if not indices:
                d_j += np.abs(a_m)**2 * np.abs(self.optical_system.get_otf(q_j - k_m))**2
            else:
                d_j += np.abs(a_m)**2 * np.abs(self.optical_system.shifted_otfs[tuple(k_m)][indices])**2
        d_j *= self.illumination.M_t
        return d_j

    def Vj(self, q_j, indices):
        v_j = 0
        for m1 in range(len(self.illumination.waves)):
            for m2 in range(len(self.illumination.waves)):
                a_m1 = self.illumination.waves[m1].amplitude
                a_m2 = self.illumination.waves[m2].amplitude
                a_m12 = self.illumination.waves[m1 - m2].amplitude
                k_m1 = self.illumination.waves[m1].wavevector
                k_m2 = self.illumination.waves[m2].wavevector
                if indices:
                    otf1 = self.optical_system.shifted_otfs[tuple(k_m1)][indices]
                    otf2 = self.optical_system.shifted_otfs[tuple(k_m2)][indices]
                    otf3 = self.optical_system.wvdiff_otfs[tuple(k_m2 - k_m1)]
                else:
                    otf1 = self.optical_system.get_otf(q_j - k_m1)
                    if otf1 == 0:
                        continue
                    otf2 = self.optical_system.get_otf(q_j - k_m2)
                    if otf2 == 0:
                        continue
                    otf3 = self.optical_system.get_otf(k_m2 - k_m1)
                    if otf3 == 0:
                        continue

                v_j += a_m1 * a_m2.conjugate() * a_m12 * otf1.conjugate() * otf2 * otf3
        v_j *= self.illumination.M_t
        return v_j

    def SSNR(self,q_j, indices = None):
        dj = self.Dj(q_j, indices)
        if dj == 0:
            return 0
        vj = self.Vj(q_j, indices)
        if vj == 0:
            return 0
        ssnr = np.abs(dj)**2 / vj
        return ssnr