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
from Dimensions import DimensionMeta
import stattools

class SSNRBase(metaclass=DimensionMeta):
    """
    Base class for SSNR calculators. 
    

    Attributes:
        optical_system: OpticalSystem object, the optical system used in the experiment.
        ssnri: numpy.ndarray, the spectral signal-to-noise ratio (SSNR) image.
    Methods:
        ring_average_ssnri(number_of_samples=None): Computes the ring-averaged SSNR.
        compute_ssnri_volume(factor=10, volume_element=1): Computes the SSNR volume.
        compute_ssnri_entropy(factor=100): Computes the SSNR entropy.
        compute_radial_ssnri_entropy(factor=100): Computes the radial SSNR entropy.
        compute_full_ssnr(object_ft): Computes the full SSNR for a given object Fourier transform.        
    """
    def __init__(self, optical_system, readout_noise_variance=0):
        self._optical_system = optical_system
        self._ssnri = None
        self.readout_noise_variance = 0

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
        q_axes = self.optical_system.otf_frequencies
        ssnri = np.copy(self.ssnri)
        if len(q_axes) == 2:
            return average_rings2d(ssnri, q_axes, number_of_samples=number_of_samples)
        elif len(q_axes) != 3:
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

    @staticmethod
    def estimate_ssnr_from_image_binomial_splitting(
        data: np.ndarray,
        n_iter: int = 1,
        radial: bool = False,
        return_freq: bool = False,
        rng: np.random.Generator | None = None,
    ):
        """
        Spectral Signal-to-Noise Ratio (SSNR) based on ½–½ binomial pixel splitting.

        * Single Poisson image  → pixels are split once per iteration
        * Stack of images      → **each frame** is split independently and the
                                resulting SSNR maps are averaged.
        """
        rng = np.random.default_rng() if rng is None else rng
        if data.ndim not in (2, 3):
            raise ValueError("`data` must be 2-D or 3-D (stack).")

        Ny, Nx = data.shape[-2:]
        ssnr_accum = np.zeros((Ny, Nx), float)

        # Fourier-space radius grid for optional radial averaging
        if radial:
            u = np.fft.fftfreq(Nx) * Nx
            v = np.fft.fftfreq(Ny) * Ny
            R = np.hypot(*np.meshgrid(u, v))
            r_int = R.astype(int)
            r_max = r_int.max()
            radial_counts = np.bincount(r_int.ravel())

        n_frames = 1 if data.ndim == 2 else data.shape[0]

        # ------------------------------------------------------------------------
        for _ in range(n_iter):
            # ---- SINGLE IMAGE ---------------------------------------------------
            if data.ndim == 2:
                frames = [data]
            # ---- IMAGE STACK ----------------------------------------------------
            else:
                frames = data                                 # iterate over frames

            for frame in frames:
                nA = rng.binomial(frame.astype(int), 0.5)
                nB = frame - nA

                # wrappers for centred FFT / IFFT
                e1 = wrappers.wrapped_fftn(nA, axes=(-2, -1))
                e2 = wrappers.wrapped_fftn(nB, axes=(-2, -1))

                num = np.abs(e1 + e2) ** 2        # |ê₁+ê₂|²
                den = np.abs(e1 - e2) ** 2        # |ê₁–ê₂|²
                with np.errstate(divide="ignore", invalid="ignore"):
                    ssnr_map = num / den - 1.0
                ssnr_map[np.isnan(ssnr_map) | np.isinf(ssnr_map)] = 0.0
                # plt.imshow(np.log1p(10**4 * ssnr_map), cmap='gray')
                # plt.show()
                ssnr_accum += ssnr_map

        # average over (iterations × frames)
        ssnr_mean = ssnr_accum / (n_iter * n_frames)

        # ---------------- optional radial average -------------------------------
        if not radial:
            return ssnr_mean

        radial_accum = np.bincount(r_int.ravel(), weights=ssnr_mean.ravel())
        radial_profile = radial_accum / radial_counts
        radial_profile[np.isnan(radial_profile)] = 0.0

        if return_freq:
            return radial_profile, np.arange(r_max + 1) / max(Nx, Ny)
        return radial_profile

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
        self._ssnri = np.abs(self.optical_system.otf) ** 2 / np.amax(np.abs(self.optical_system.otf))

    def compute_full_ssnr(self, object_ft):
        ((np.abs(object_ft)) ** 2 /
                (np.amax(np.abs(object_ft)) + self.optical_system.otf.size * self.readout_noise_variance))


class SSNRPointScanning2D(SSNRPointScanning):
    dimensionality = 2
    def __init__(self, optical_system):
        if not isinstance(optical_system, OpticalSystems.OpticalSystem2D):	
            raise AttributeError("Trying to initialize 2D SSNR class with the wrong OTF!")
        super().__init__(optical_system)


class SSNRPointScanning3D(SSNRPointScanning):
    dimensionality = 2
    def __init__(self, optical_system):
        if not isinstance(optical_system, OpticalSystems.OpticalSystem3D):
            raise AttributeError("Trying to initialize 3D SSNR class with the wrong OTF!")
        super().__init__(optical_system)


SSNRConfocal2D = SSNRPointScanning2D
SSNRConfocal3D = SSNRPointScanning3D

SSNRRCM2D = SSNRPointScanning2D
SSNRRCM3D = SSNRPointScanning3D

SSNRWidefield2D = SSNRPointScanning2D
SSNRWidefield3D = SSNRPointScanning3D

class SSNRSIM(SSNRBase):
    def __init__(self, illumination: PlaneWavesSIM,
                 optical_system:OpticalSystems.OpticalSystem,
                 kernel: np.ndarray=None,
                 readout_noise_variance:float=0,
                 effective_otfs={},
                 effective_kernels_ft={},
                 save_memory=False, 
                 illumination_reconstruction=None
                 ):
        
        super().__init__(optical_system, readout_noise_variance)
        self._illumination = illumination
        self._illumination_reconstruction = illumination_reconstruction if not illumination_reconstruction is None else illumination
        self.vj = None
        self.dj = None

        self.effective_otfs = {} if not effective_otfs else effective_otfs
        self.otf_sim = None
        if self.effective_otfs:
            self._compute_otf_sim()

        self.effective_kernels_ft = {} if not effective_kernels_ft else effective_kernels_ft
        self._kernel = None
        self._kernel_ft = None

        self.save_memory = save_memory
        self.readout_noise_variance = readout_noise_variance

        if optical_system.otf is None:
            raise AttributeError("Optical system otf is not computed")

        if not self.effective_otfs:
            self._compute_effective_otfs()

        if not self.effective_kernels_ft:
            if not kernel is None:
                self.kernel = kernel
            else:
                self.kernel = self.optical_system.psf

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
        if self._illumination is self._illumination_reconstruction:
            self._illumination_reconstruction = None 
        self._illumination = new_illumination
        self.effective_otfs = {}
        self._compute_effective_otfs()
        if self.illumination_reconstruction is None:
            self.illumination_reconstruction = new_illumination
        else:
            self._compute_ssnri()

    @property 
    def illumination_reconstruction(self):
        return self._illumination_reconstruction
    
    @illumination_reconstruction.setter
    def illumination_reconstruction(self, new_illumination_reconstruction):
        self._illumination_reconstruction = new_illumination_reconstruction
        self.effective_kernels_ft = {}
        self._compute_effective_kernels_ft()
        self._compute_ssnri()

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel_new):
        kernel_new = stattools.expand_kernel(kernel_new, self.optical_system.psf.shape)

        self.kernel_ft = wrappers.wrapped_ifftn(kernel_new)
        self.kernel_ft /= np.amax(self.kernel_ft)
        self._kernel = kernel_new
        self.effective_kernels_ft = {}
        self._compute_effective_kernels_ft()

        self._compute_ssnri()

    def _compute_effective_otfs(self):
        _, self.effective_otfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        self._compute_otf_sim()

    def _compute_otf_sim(self):
        self.otf_sim = np.zeros(self.optical_system.otf.shape)
        for m in self.effective_otfs:
            self.otf_sim += np.abs(self.effective_otfs[m])

    def _compute_effective_kernels_ft(self):
        if np.isclose(self.kernel, self.optical_system.psf).all() and self.illumination_reconstruction is self.illumination:
            self.effective_kernels_ft = self.effective_otfs
        else:
            _, self.effective_kernels_ft = self.illumination_reconstruction.compute_effective_kernels(self.kernel, self.optical_system.psf_coordinates)

    def _compute_Dj(self):
        d_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs.keys():
            d_j += self.effective_otfs[m] * self.effective_kernels_ft[m].conjugate()
        d_j *= self.illumination.Mt
        # plt.title("Dj")
        # plt.imshow(np.log(1 + 10**8 * np.abs(d_j)[:, :, 50]))
        # plt.show()
        return np.abs(d_j)

    def _compute_Vj(self):
        center = np.array(self.optical_system.otf.shape, dtype=np.int32) // 2
        v_j = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)

        for idx1 in self.effective_otfs.keys():
            for idx2 in self.effective_otfs.keys():
                if idx1[0] != idx2[0]:
                    continue
                r = idx1[0]
                m1 = idx1[1]
                m2 = idx2[1]
                m21 = tuple(xy2 - xy1 for xy1, xy2 in zip(m1, m2))
                if (r, m21) not in self.illumination.rearranged_indices:
                    continue
                idx_diff = (r, m21)
                otf1 = self.effective_kernels_ft[idx1]
                otf2 = self.effective_kernels_ft[idx2]
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

    def ring_average__ssnri_approximated(self, number_of_samples=None):
        """
        Compute ring avergaged ssnri as <Dj^2> / <Vj> instead of <Dj^2 / Vj>
        """
        q_axes = self.optical_system.otf_frequencies
        dj = np.copy(self.dj)
        vj = np.copy(self.vj)
        if len(q_axes) == 2:
            return average_rings2d(dj**2, q_axes, number_of_samples=number_of_samples) / average_rings2d(vj, q_axes, number_of_samples=number_of_samples)
        elif len(q_axes) != 3:
            raise AttributeError("PSF dimension is not equal to 2 or 3!")

        averaged_slices = []
        for i in range(dj.shape[2]):
            averaged_slices.append(average_rings2d(dj[i]**2, q_axes, number_of_samples=number_of_samples) / average_rings2d(vj[i], q_axes, number_of_samples=number_of_samples))
        return np.array(averaged_slices).T

    def compute_analytic_ssnri_volume(self, factor=10, volume_element=1):
        g2 = np.sum(self.optical_system.otf * self.optical_system.otf.conjugate()).real
        g0 = np.abs(np.amax(self.optical_system.otf))
        weights = np.array([harmonic.amplitude for harmonic in self.illumination.harmonics.values()])
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
        weights = np.array([harmonic.amplitude for harmonic in self.illumination.harmonics.values()])
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
        Widefield = SSNRWidefield2D if self.dimensionality == 2 else SSNRWidefield3D
        ssnr_widefield = Widefield(self.optical_system).ssnri
        diff = np.sum(self.ssnri - ssnr_widefield).real
        upper_estimate = np.abs(np.amax(self.ssnri - ssnr_widefield))
        noise_level = 10 ** -10 * np.abs(np.amax(self.ssnri))
        threshold = self._find_threshold_value(diff, upper_estimate, 0, noise_level, ssnr_widefield)
        measure = np.where((np.abs(ssnr_widefield) < threshold) * (np.abs(self.ssnri) > noise_level), np.abs(self.ssnri - ssnr_widefield), 0)
        measure = np.where(measure < threshold, measure, threshold)
        return np.sum(measure) * factor, threshold


class SSNRSIM2D(SSNRSIM):
    dimensionality = 2
    def __init__(self,
                 illumination: IlluminationPlaneWaves2D,
                 optical_system,
                 kernel=None,
                 readout_noise_variance=0,
                 save_memory=False,
                 illumination_reconstruction=None):
        if not isinstance(illumination, IlluminationPlaneWaves2D):
            raise AttributeError("Illumination data is not of the valid dimension!")
        if not isinstance(optical_system, OpticalSystems.OpticalSystem2D):
            raise AttributeError("Optical system data is not of the valid dimension!")
        if not isinstance(illumination_reconstruction, IlluminationPlaneWaves2D) and not illumination_reconstruction is None:
            raise AttributeError("Illumination reconstruction data is not of the valid dimension!")
        
        super().__init__(illumination, optical_system, kernel=kernel, readout_noise_variance=readout_noise_variance, save_memory=save_memory, illumination_reconstruction=illumination_reconstruction)

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
    dimensionality = 3
    def __init__(self,
                 illumination,
                 optical_system,
                 kernel=None,
                 readout_noise_variance=0,
                 save_memory=False, 
                 illumination_reconstruction=None):
        if not isinstance(illumination, IlluminationPlaneWaves3D):
            raise AttributeError("Illumination data is not of the valid dimension!")
        if not isinstance(optical_system, OpticalSystems.OpticalSystem3D):
            raise AttributeError("Optical system data is not of the valid dimension!")
        if not isinstance(illumination_reconstruction, IlluminationPlaneWaves3D) and not illumination_reconstruction is None:
            raise AttributeError("Illumination reconstruction data is not of the valid dimension!")
        super().__init__(illumination, optical_system, kernel=kernel, readout_noise_variance=readout_noise_variance, save_memory=save_memory, illumination_reconstruction=illumination_reconstruction)
