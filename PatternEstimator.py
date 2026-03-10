from __future__ import annotations
"""
illumination_pattern_estimator.py

Data‑driven estimation of the illumination parameters (phase offsets, k‑vectors
and modulation depths *aₘ*) directly from a raw SIM stack

The module currently provides two estimation strategies that can be naturally mixed:

1. **Cross‑correlation**: The illumination pattern is estimated by estimating cross-correlation
coefficients following the method of Gustaffson (2000). Maxima of the absolute values of the 
coefficients correspond to the peaks positions of the illumination pattern, while complex phases 
of the coefficients correspond to the phase offsets.

2. **Interpolation**: The illumination pattern is estimated by first finding peaks of the Fourier images
I(k) = OTF(k) f(k - kₘ) and then correcting for the difference between max I(k) and max f(k-kₘ), achieved at k = kₘ.
This method is not, up to our knowledge, found in the literature, though it is possible it was used in the past.
Assumptions here are that max I(k) exist, i. e., k_max < k_cutoff, and  that k_max is close to kₘ, so that the 
Taylor series expansion is justified. Phases are found as arg I(k_max).

Under the high illumination condition, the performance of the two methods is similar. Cross-correlation method is 
more universal, as it relies on the less number of assumptions. The interpolation method is faster and seems to be 
more robust to noise for peak positions determination and less robust to determination of the phase offsets.

Both methods converge to peaks by repeated interpolation in the grid near the estimated peak positions. Reasonable 
initial guess is required. 

Several methods are available for the estiamtion of the modulation depth coefficients, but the difference in the results, as
well as the effect on the reconstruction is typically negligible.

By default, we recommend to use the interpolation method for the peak positions estimation and the cross-correlation method
for the phase offsets estimation.

IlluminationPatternEstimator classes return the Illumination object with estimated parameters.   

"""

from copy import deepcopy
from typing import Dict, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import math 
from VectorOperations import VectorOperations

from wrappers import wrapped_fftn, wrapped_ifftn
from Dimensions import DimensionMetaAbstract
from abc import abstractmethod
from utils import off_grid_ft
import hpc_utils

from Illumination import (
    PlaneWavesSIM,
    IlluminationPlaneWaves2D,
    IlluminationPlaneWaves3D,
)

from Sources import IntensityHarmonic
from OpticalSystems import OpticalSystem, OpticalSystem2D, OpticalSystem3D
from SSNRCalculator import SSNRBase


class IlluminationPatternEstimator(metaclass=DimensionMetaAbstract):
    """
    Class for estimating the illumination parameters from a raw SIM stack.
    """

    dimensionality = None  

    peak_estimation_methods = ('interpolation', 'cross_correlation')
    phase_estimation_methods = ('peak_phases', 'autocorrelation', 'cross_correlation')
    modulation_coefficients_methods = ('default', 'least_squares', 'peak_height_ratio')
    debug_info_levels = (0, 1, 2)

    def __init__(self, illumination: PlaneWavesSIM, optical_system: OpticalSystem):
        self.illumination = illumination
        self.optical_system = optical_system
        if not self.illumination.dimensionality == self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: {self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )

    def estimate_illumination_parameters(
        self,
        stack: np.ndarray,
        peak_estimation_method='interpolation', 
        phase_estimation_method='autocorrelation',
        modulation_coefficients_method='default',
        peak_search_area_size: int = 11,
        zooming_factor: int = 100, 
        max_iterations: int = 2, 
        debug_info_level = 0,
        first_order_phases_only: bool = True,
        correct_peak_position: bool = True,
    ) -> PlaneWavesSIM:
        
        """Return a **new illumination object** whose amplitudes and phases come
        from the image data.
        """


        Mr, Mt = self.illumination.Mr, self.illumination.Mt
        if stack.shape[0] != Mr:
            raise ValueError(
                f"Stack rotations={stack.shape[0]} differ from Illumination.Mr={Mr}")
        if stack.shape[1] != Mt:
            raise ValueError(
                f"Stack translations={stack.shape[1]} differ from Illumination.Mt={Mt}")

        if not peak_estimation_method in self.peak_estimation_methods:
            raise ValueError(
                f"Unknown method of peaks estimation {peak_estimation_method}. Available methods are {self.peak_estimation_methods}"
            )
        if not phase_estimation_method in self.phase_estimation_methods:
            raise ValueError(
                f"Unknown method of phase estimation {phase_estimation_method}. Available methods are {self.phase_estimation_methods}"
            )
        if not modulation_coefficients_method in self.modulation_coefficients_methods:
            raise ValueError(
                f"Unknown method of modulation coefficients estimation {modulation_coefficients_method}. Available methods are {self.modulation_coefficients_methods}"
            )
    
        peaks, rotation_angles = self.estimate_peaks(peak_estimation_method, stack, peak_search_area_size, zooming_factor, max_iterations, debug_info_level=debug_info_level, correct_peak_position=correct_peak_position)
        
        if debug_info_level > 0:
            for r in range(Mr):
                print('r = ', r,  'rotation_angle = ', np.round(rotation_angles[r] / np.pi * 180, 1), 'degrees')
            for sim_index in sorted(peaks.keys()):
                print('r = ', sim_index[0], 'm = ', sim_index[1], 'wavevector = ', np.round((peaks[sim_index]), 3), '1 / lambda')

        phase_matrix = self.build_phase_matrix(phase_estimation_method, stack, peaks, debug_info_level)

        if debug_info_level > 0:
            for sim_index in sorted(phase_matrix.keys()):
                print('r = ', sim_index[0], 'n = ', sim_index[1], 'm = ', sim_index[2], 'phase = ', np.round(np.angle(phase_matrix[sim_index]) / np.pi * 180, 1), 'degrees')

        illumination_estimated = self.build_illumination_object(rotation_angles, peaks, phase_matrix, debug_info_level)
        
        if modulation_coefficients_method != 'default':
            illumination_estimated.estimate_modulation_coefficients(stack, self.optical_system.psf, self.optical_system.x_grid, 
                                                                    method = modulation_coefficients_method, update=True)
            
        if debug_info_level > 0:
            amplitudes, indices = illumination_estimated.get_all_amplitudes()
            for amplitude, index in zip(amplitudes, indices):
                print('r = ', index[0], 'm = ', index[1], 'amplitude = ', round(amplitude, 3))
        
        if debug_info_level > 2:
            image_ft_reference = wrapped_fftn(stack[0, 0])
            plt.imshow(np.log1p(np.abs(image_ft_reference)).T, cmap='gray', origin='lower')
            for peak in illumination_estimated.harmonics.keys():
                qx, qy = self.optical_system.otf_frequencies
                sizex, sizey = qx.size, qy.size
                approximate_peak = illumination_estimated.harmonics[peak].wavevector / (2 * np.pi)
                plt.plot(approximate_peak[0]/qx[-1] * sizex//2 + sizex//2, approximate_peak[1]/qx[-1] * sizey//2 + sizey//2, 'rx')
            plt.show()

        return illumination_estimated
    
    @abstractmethod
    def estimate_peaks(self,
                        peak_estimation_method: str,
                        stack: np.ndarray,
                        peak_search_area_size: int,
                        zooming_factor: int,
                        max_iterations: int,
                        debug_info_level: int = 0) -> Tuple[np.ndarray, dict]:
        
        """
        Estimate the peaks postitions of the illumination pattern from the stack.
        """

    def _make_phase_estimator(
        self,
        illumination: PlaneWavesSIM,
        optical_system: OpticalSystem,
        method_name: str,
        first_order_phases_only: bool = True,
    ) -> callable:
        """
        Instantiate the correct PhasesEstimator subclass for the given
        illumination / optical_system pair and return the bound method
        that corresponds to *method_name*.

        The caller receives a plain callable ``fn(stack, refined_wavevectors)``
        and never has to pass illumination or optical_system explicitly.
        """
        if optical_system.dimensionality == 3 and illumination.dimensions[2] == 1:
            estimator = PhasesEstimator3D(illumination, optical_system, first_order_phases_only)
        else:
            estimator = PhasesEstimator2D(illumination, optical_system, first_order_phases_only)

        match method_name:
            case 'peak_phases':
                return estimator.phase_matrix_peak_values
            case 'autocorrelation':
                return estimator.phase_matrix_autocorrelation
            case 'cross_correlation':
                return estimator.phase_matrix_cross_correlation
            case _:
                raise ValueError(
                    f"Unknown method of phase estimation '{method_name}'. "
                    f"Available methods are {self.phase_estimation_methods}"
                )

    @abstractmethod
    def build_phase_matrix(self,
                           phase_estimation_method: str = 'autocorrelation',
                           stack: np.ndarray = None,
                           peaks: dict = None,
                           debug_info_level: int = 0) -> dict:
        """
        Build the phase matrix from the image stack and estimated peak positions.
        """

         
    def _compose_harmonics_dict(self, refined_wavevectors: dict) -> Dict[int, IntensityHarmonic]:
        """
        Compose the harmonics dictionary from the refined wavevectors.
        """
        harmonic_class = type(next(iter(self.illumination.harmonics.values())))
        harmonics_dict = {}
        for index in refined_wavevectors.keys():
            wavevector = np.copy(refined_wavevectors[index])
            harmonics_dict[index] = harmonic_class(
                amplitude=self.illumination.harmonics[index].amplitude,
                phase=0,
                wavevector=wavevector,
            )
        return harmonics_dict
    
    def build_illumination_object(self, 
                                   rotation_angles: np.ndarray,
                                   refined_wavevectors: dict,
                                   phase_matrix: dict,
                                   debug_info_level: int = 0
                                   ) -> PlaneWavesSIM:
        
        """
        Build the Illumination object from the parameters.
        """

        Illumination = type(self.illumination)

        harmonics_dict = self._compose_harmonics_dict(refined_wavevectors)
        illumination_estimated = Illumination(
                 intensity_harmonics_dict = harmonics_dict,
                 dimensions= self.illumination.dimensions,
                 Mr=self.illumination.Mr,
                 spatial_shifts=None,
                 angles=rotation_angles,
                 )
        
        illumination_estimated.Mt = self.illumination.Mt
        illumination_estimated.phase_matrix = phase_matrix
        illumination_estimated.source_electromagnetic_plane_waves = self.illumination.source_electromagnetic_plane_waves
        illumination_estimated.normalize_spatial_waves()
        for plane_wave in illumination_estimated.source_electromagnetic_plane_waves:
            scaling_factor = np.sum(self.illumination.get_base_vectors(0)**2)**0.5 / np.sum(illumination_estimated.get_base_vectors(0)**2)**0.5
            if debug_info_level > 0:
                print('scaling factor', scaling_factor)
            plane_wave.wavevector *= scaling_factor

        return illumination_estimated

    def compute_ssnr(self, stack: np.ndarray, iterations: int) -> np.ndarray:
        """
        Compute the SSNR directly from the image stack.
        """

        ssnr_estimated = SSNRBase.estimate_ssnr_from_image_binomial_splitting(stack, n_iter=iterations, radial=False)
        ssnr_estimated[ssnr_estimated < 0] = 0
        # plt.imshow(np.log1p(ssnr_estimated), cmap='gray')
        # plt.show()
        # plt.plot(self.optical_system.otf_frequencies[0], ssnr_estimated[:, 50], label='ssnr_estimated')
        # plt.ylim(0, 100)
        # plt.plot(2.45, 0, 'cx')
        # plt.show()
        print('ssnr_estimated', ssnr_estimated[50, 50])
        # exit()
        return ssnr_estimated

class IlluminationPatternEstimator2D(IlluminationPatternEstimator):
    """
    Class for estimating the illumination parameters from a raw SIM stack in 2D.
    """
    dimensionality = 2

    def estimate_peaks(self,
                        peak_estimation_method: str,
                        stack: np.ndarray,
                        peak_search_area_size: int,
                        zooming_factor: int,
                        max_iterations: int,
                        debug_info_level: int = 0,
                        correct_peak_position: bool = True,
                        first_order_phases_only: bool = True,
                        ) -> Tuple[np.ndarray, dict]:

        if peak_estimation_method == 'cross_correlation':
            PeaksEstimator = PeaksEstimatorCrossCorrelation2D
        elif peak_estimation_method == 'interpolation':
            PeaksEstimator = PeaksEstimatorInterpolation2D
        else:
            raise ValueError(f"Unknown peaks estimation method: {peak_estimation_method}")

        peak_estimator = PeaksEstimator(self.illumination, self.optical_system)
        peaks, rotation_angles = peak_estimator.estimate_peaks(stack,
                                                               peak_search_area_size,
                                                               zooming_factor,
                                                               max_iterations,
                                                               debug_info_level=debug_info_level,
                                                               correct_peak_position=correct_peak_position)
        return peaks, rotation_angles

    def build_phase_matrix(self,
                           phase_estimation_method: str = 'autocorrelation',
                           stack: np.ndarray = None,
                           peaks: dict = None,
                           first_order_phases_only: bool = True,
                           debug_info_level: int = 0) -> dict:
        """
        Build the phase matrix from the image stack and estimated peak positions.
        Direct 2-D call: the whole stack is a single 2-D image per (r, n).
        """
        phase_fn = self._make_phase_estimator(self.illumination, self.optical_system, phase_estimation_method, first_order_phases_only)
        return phase_fn(stack, peaks)


class IlluminationPatternEstimator3D(IlluminationPatternEstimator):
    """
    Class for estimating the illumination parameters from a raw SIM stack in 3D.
    """
    dimensionality = 3

    estimation_strategies = ('plane_by_plane', 'true_3D')

    def estimate_peaks(self,
                        peak_estimation_method: str,
                        stack: np.ndarray,
                        peak_search_area_size: int,
                        zooming_factor: int,
                        max_iterations: int,
                        debug_info_level: int = 0, 
                        estimation_strategy: str = "plane_by_plane", 
                        reference_expansion: tuple = (1, 0, 1),
                        correct_peak_position: bool = True,
                        first_order_phases_only: bool = True,
                        ) -> Tuple[np.ndarray, dict]:

        if not estimation_strategy in self.estimation_strategies:
            raise ValueError(f"Unknown estimation strategy: {estimation_strategy}")

        if peak_estimation_method == 'cross_correlation':
            PeaksEstimator = PeaksEstimatorCrossCorrelation2D if estimation_strategy == 'plane_by_plane' else PeaksEstimatorCrossCorrelation3D
        elif peak_estimation_method == 'interpolation':
            PeaksEstimator = PeaksEstimatorInterpolation2D if estimation_strategy == 'plane_by_plane' else PeaksEstimatorInterpolation3D
        else:
            raise ValueError(f"Unknown peaks estimation method: {peak_estimation_method}")

        if PeaksEstimator.dimensionality == 3:
            peak_estimator = PeaksEstimator(self.illumination, self.optical_system)
        else: 
            illumination2d = IlluminationPlaneWaves2D.init_from_3D(self.illumination, force=True)
            optical_system2d = self.optical_system.project_in_2D()
            peak_estimator = PeaksEstimator(illumination2d, optical_system2d)
        
        if estimation_strategy == 'true_3D':
            peaks, rotation_angles = peak_estimator.estimate_peaks(stack, 
                                                                peak_search_area_size, 
                                                                zooming_factor, 
                                                                max_iterations, 
                                                                debug_info_level=debug_info_level, 
                                                                correct_peak_position=correct_peak_position)

        else:
            if debug_info_level:
                print('3D volume split into planes. Averaging FT magnitudes or complex FTs along the z-axis')

            projected_stack = self._project_stack(stack)

            peaks2d, rotation_angles = peak_estimator.estimate_peaks(projected_stack, 
                                                        peak_search_area_size, 
                                                        zooming_factor, 
                                                        max_iterations, 
                                                        debug_info_level=debug_info_level, 
                                                        correct_peak_position=correct_peak_position)
            print(peaks2d, rotation_angles)
            for peak in peaks2d: 
                illumination2d.harmonics[peak].wavevector = peaks2d[peak]
            illumination2d.angles = rotation_angles
            peaks = self.illumination3d_from_illuminaton2d_and_total_wavelength(illumination2d, reference_expansion=reference_expansion)
        return peaks, rotation_angles

    def _project_stack(self, stack: np.ndarray) -> np.ndarray:
        """
        Projects a 3D raw stack (Mr, Mt, Nx, Ny, Nz) to a 2D stack (Mr, Mt, Nx, Ny) 
        by summing the Fourier transforms along the z-axis.

        If dimensions[2] == 1, sum the absolute values of the FTs.
        If dimensions[2] == 0, sum the complex FTs directly.
        Returns the IFFT of the sum, which is fed into the 2D estimator.
        """
        n_planes = stack.shape[4]
        
        if not self.illumination.dimensions[2]:
            return np.sum(stack, axis=4)
        else:
            projected_stack_ft = np.zeros_like(wrapped_fftn(stack[..., 0]), dtype=complex)
            for z in range(n_planes):
                substack_ft = np.abs(wrapped_fftn(stack[..., z]))
                projected_stack_ft += substack_ft
            return wrapped_ifftn(projected_stack_ft).real

    def illumination3d_from_illuminaton2d_and_total_wavelength(self,
                                                               illumination2d: IlluminationPlaneWaves2D,
                                                               reference_expansion: tuple = (1, 0, 1), 
                                                               debug_info: bool = True) -> IlluminationPlaneWaves3D:
        """
        Assuming we found 2d illumination pattern, we can find 3d illumination pattern if we know the total wavelength, 
        which is typically the case. It is assumed for now, that the missing direction is z-direction, in the same fashion as it is assumed, 
        that we rotate the illumination pattern around the z-axis.
        """
        # target_dimensions = target_illumination_3d.dimensions
        # target_illumination_3d.dimensions = tuple([1 if i != missing_dimension else 0 for i in range(3)])
        k_vector_length = 2 * np.pi * self.optical_system.nm
        peaks3d = {}
        base_vectors3d = np.zeros((illumination2d.Mr, 3))
        for r in range(illumination2d.Mr):
            base_vectors2d = illumination2d.get_base_vectors(r)
            missing_base = (k_vector_length **2 - np.sum(base_vectors2d**2 * np.array(reference_expansion[:2])**2))**0.5 / reference_expansion[2]
            base_vectors3d[r, :2] = base_vectors2d
            base_vectors3d[r, 2] = missing_base
            if debug_info:
                print("r", r, "missing_base length", missing_base)

        for key in self.illumination.harmonics.keys():
            r = key[0]
            wavevector = np.zeros(3)
            for i in range(3):
                wavevector[i] = key[1][i] * base_vectors3d[r][i]
            peaks3d[key] = VectorOperations.rotate_vector3d(wavevector, np.array((0, 0, 1)), illumination2d.angles[r])

        return peaks3d

    def build_phase_matrix(self,
                           phase_estimation_method: str = 'autocorrelation',
                           stack: np.ndarray = None,
                           peaks: dict = None,
                           first_order_phases_only: bool = True,
                           debug_info_level: int = 0) -> dict:
        """
        Build the phase matrix for a 3-D stack. 

        If dimensions[2] == 1, compute 3D phases using the full stack and natively.
        If dimensions[2] == 0, build a quasi 2D 'superimage' (by summing complex FTs), create a quasi-2D illumination, and compute
        using 2D estimators along with projected peaks.
        """
        if self.illumination.dimensions[2]:
            phase_fn = self._make_phase_estimator(self.illumination, self.optical_system, phase_estimation_method, first_order_phases_only)
            return phase_fn(stack, peaks)
        else:
            optical_system2d = self.optical_system.project_in_2D()
            illumination2d = IlluminationPlaneWaves2D.init_from_3D(self.illumination.project_in_quasi_2D())
            projected_stack = self._project_stack(stack)
            peaks2d = {(peak[0], peak[1][:2]): peaks[peak][:2] for peak in peaks.keys()}
            phase_fn = self._make_phase_estimator(illumination2d, optical_system2d, phase_estimation_method, first_order_phases_only)
            phase_matrix2d = phase_fn(projected_stack, peaks2d)
            phase_matrix3d = {}
            for peak in self.illumination.project_in_quasi_2D().harmonics.keys():
                r, m = peak
                for n in range(self.illumination.Mt):
                    phase_matrix3d[peak[0], n, peak[1]] = phase_matrix2d[(r, n, m[:2])]
            return phase_matrix3d

class PeaksEstimator(metaclass=DimensionMetaAbstract):
    def __init__(self, illumination: PlaneWavesSIM, optical_system: OpticalSystem):
        self.illumination = illumination
        self.optical_system = optical_system
        if self.illumination.dimensionality != self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: "
                f"{self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )
        self._m = self._select_m()

    @abstractmethod
    def estimate_peaks(
        self,
        stack: np.ndarray,
        peak_search_area_size: int,
        zooming_factor: int,
        max_iterations: int,
        debug_info_level: int = 0,
        **kwargs,
    ):
        """
        Implementations should return (refined_wavevectors, rotation_angles) like your existing API.
        """
        ...

    # ----------------------------
    # Shared utilities
    # ----------------------------
    def _select_m(self):
        m_unique = set()
        for m in self.illumination.harmonics.keys():
            if np.isclose(np.sum(np.abs(np.array(m[1]))), 0):
                continue
            # Generate the inverse harmonic index pair
            m_inv = (m[0], tuple(-np.array(m[1])))
            
            # Deterministically choose the canonical representation (e.g. lexicographically larger)
            canonical_m = m if m > m_inv else m_inv
            m_unique.add(canonical_m)
            
        # Ensure deterministic iteration order returned
        return tuple(sorted(list(m_unique), key=lambda x: (x[0], sum(abs(v) for v in x[1]), x[1])))

    def get_stack_ft(self, stack: np.ndarray) -> np.ndarray:
        return np.array([np.array([wrapped_fftn(image) for image in stack[r]]) for r in range(stack.shape[0])])

    def merit_function(self, Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray]) -> np.ndarray:
        return sum([C * C.conjugate() for C in Cmn1n2.values()])

    def _czt_nd(self, sig_stack: np.ndarray, q_coords_cycles: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        Minimal CZT wrapper used by multiple estimators.
        - q_coords_cycles: tuple of 1D coordinate arrays in cycles / unit
        - returns numpy array
        """
        x_coords = self.optical_system.psf_coordinates
        q_coords_rad = tuple((2.0 * np.pi) * np.asarray(qc) for qc in q_coords_cycles)

        D = self.optical_system.dimensionality
        spatial_axes = tuple(range(sig_stack.ndim - D, sig_stack.ndim))

        out = hpc_utils.czt_nd_fourier(
            np.asarray(sig_stack),
            x_coords,
            q_coords_rad,
            axes=spatial_axes,
            rtol=1e-6,
            atol=1e-12,
        )
        return hpc_utils._to_numpy(out)

    def _argmax_coords(self, arr: np.ndarray, q_coords: tuple[np.ndarray, ...]) -> np.ndarray:
        idx = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
        return np.array([q_coords[d][idx[d]] for d in range(len(q_coords))], dtype=np.float64)

    def _zoom_coords(
        self,
        q_coords: tuple[np.ndarray, np.ndarray],
        zooming_factor: int,
        center: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        CZT-uniform zoom/refine (2D):

        Assumptions:
        - q_coords axes are uniform (linspace-like), as used for CZT evaluation.

        Behavior:
        - Let dq_x, dq_y be the uniform steps of the current grid.
        - Shrink the coordinate span to one-step neighborhood around center:
                [cx - dq_x, cx + dq_x], [cy - dq_y, cy + dq_y]
        - Increase sampling density by returning:
                n = 2*zooming_factor + 1 samples per axis
            meaning: insert zooming_factor points between center and each neighbor.
        """
        qx = np.asarray(q_coords[0], dtype=np.float64)
        qy = np.asarray(q_coords[1], dtype=np.float64)

        z = int(zooming_factor)
        if z < 1:
            raise ValueError(f"zooming_factor must be >= 1, got {zooming_factor}")

        if center is None:
            cx, cy = 0.0, 0.0
        else:
            cx, cy = float(center[0]), float(center[1])

        # Degenerate grids: nothing to refine
        if qx.size < 2 or qy.size < 2:
            return (qx.copy(), qy.copy())

        dq_x = float(qx[1] - qx[0])
        dq_y = float(qy[1] - qy[0])

        n = 2 * z + 1
        qx_new = np.linspace(cx - dq_x, cx + dq_x, n, dtype=np.float64)
        qy_new = np.linspace(cy - dq_y, cy + dq_y, n, dtype=np.float64)
        return (qx_new, qy_new)
    
    def _q_window_around(self, q0: np.ndarray, size: int) -> tuple[np.ndarray, ...]:
        """
        Build coordinate tuple by taking a window on each otf_frequencies axis
        around the nearest bin to q0[d].
        """
        half = size // 2
        out = []
        for d, axis in enumerate(self.optical_system.otf_frequencies):
            idx = int(np.argmin(np.abs(axis - q0[d])))
            lo = max(0, idx - half)
            hi = min(axis.size, idx + half + 1)
            out.append(axis[lo:hi])
        return tuple(out)

    # ----------------------------
    # Post-estimation refinement (shared)
    # ----------------------------
    def estimate_rotation_angles(self, estimated_peaks: dict) -> np.ndarray:
        N = np.zeros(self.illumination.Mr, dtype=np.float64)
        sum_expected = np.zeros(self.illumination.Mr, dtype=np.float64)
        sum_observed = np.zeros(self.illumination.Mr, dtype=np.float64)
        angles = np.zeros(self.illumination.Mr, dtype=np.float64)

        for index in estimated_peaks.keys():
            r = index[0]
            m = index[1][:2]
            angle_expected = math.atan2(m[1], m[0])
            wavevector = np.copy(estimated_peaks[index])[:2]
            angle_observed = math.atan2(wavevector[1], wavevector[0])
            sum_expected[r] += angle_expected
            sum_observed[r] += angle_observed
            N[r] += 1

        for r in range(self.illumination.Mr):
            angles[r] = (sum_observed[r] - sum_expected[r]) / N[r]
        return angles

    def refine_base_vectors(self, averaged_maxima: dict, angles: np.ndarray) -> np.ndarray:
        refined_base_vectors = np.zeros((self.illumination.Mr, self.dimensionality))
        base_vectors_sum = np.zeros((self.illumination.Mr, self.dimensionality), dtype=np.float64)
        index_sum = np.zeros((self.illumination.Mr, self.optical_system.dimensionality), dtype=np.int16)

        for index in self._m:
            r, m = index[0], index[1][:2]
            base_vectors_sum[r] += averaged_maxima[index]
            index_sum[r] += np.array(m)

        for r in range(self.illumination.Mr):
            vector_sum_aligned = VectorOperations.rotate_vector2d(base_vectors_sum[r], -angles[r])
            refined_base_vectors[r] = vector_sum_aligned / np.where(index_sum[r], index_sum[r], np.inf)

        return refined_base_vectors

    def refine_wavevectors(self, refined_base_vectors: np.ndarray, angles) -> dict[tuple[int, ...], np.ndarray]:
        refined_wavevectors = {}
        for sim_index in self.illumination.rearranged_indices.keys():
            r = sim_index[0]
            base_vectors = refined_base_vectors[r]
            wavevector_aligned = np.array(
                [2 * np.pi * sim_index[1][dim] * base_vectors[dim] for dim in range(len(sim_index[1]))]
            )
            refined_wavevectors[sim_index] = VectorOperations.rotate_vector2d(wavevector_aligned, angles[r])
        return refined_wavevectors

class PeaksEstimatorIterative(PeaksEstimator):
    """
    2D-only inside estimator.
    - Tracks all peak positions in cycles/unit.
    - Parent owns the whole window->iterate->zoom cycle + debug logic.
    - Children implement: one iteration update + debug payload for debug>=3.
    """
    
    def estimate_peaks(
        self,
        stack: np.ndarray,
        peak_search_area_size: int,
        zooming_factor: int,
        max_iterations: int,
        debug_info_level: int = 0,
        **kwargs,
    ):
        correct_peak_position = kwargs.get("correct_peak_position", False)
        peak_guesses = self._initial_peak_guesses_cycles()
        state = self._init_global_state(stack, peak_guesses)

        estimated_cycles: dict = {}
        initial_cycles: dict = {}

        for peak in self._m:
            r = peak[0]

            q_coords = self._q_window_around(peak_guesses[peak], peak_search_area_size)
            peak_state = self._init_peak_state(stack, r, peak, peak_guesses, state)

            initial_cycles[peak] = np.array(peak_state["q"], dtype=np.float64)

            for it in range(1, max_iterations + 1):
                peak_state, q_coords, dbg = self._iterate_one(
                    stack=stack,
                    r=r,
                    peak=peak,
                    q_coords=q_coords,
                    peak_state=peak_state,
                    zooming_factor=zooming_factor,
                    debug_info_level=debug_info_level,
                )

                # Preserve your printing behavior
                if debug_info_level > 1:
                    dqs = [float(q_coords[d][1] - q_coords[d][0]) for d in range(2)]
                    print("iteration", it, peak, dbg, "dq", dqs)


                # debug>=3: show per-iteration per-peak image on the interpolated q-grid
                if debug_info_level >= 3:
                    self._debug_show_iteration_surface_2d(
                        stack=stack,
                        r=r,
                        peak=peak,
                        q_coords=q_coords,
                        peak_state=peak_state,
                        it=it,
                    )

            estimated_cycles[peak] = self._final_peak_cycles(peak_state)
        if correct_peak_position:
            estimated_cycles = self._postprocess_estimated_cycles(estimated_cycles, stack)

        # debug>=2: show initial & final peak positions on top of averaged stack FT
        if debug_info_level >= 2:
            self._debug_show_initial_final_on_avg_ft_2d(stack, initial_cycles, estimated_cycles)

        rotation_angles = self.estimate_rotation_angles(estimated_cycles)
        refined_base_vectors = self.refine_base_vectors(estimated_cycles, rotation_angles)
        refined_wavevectors = self.refine_wavevectors(refined_base_vectors, rotation_angles)
        return refined_wavevectors, rotation_angles

    # ----------------------------
    # Initial guesses (parent)
    # ----------------------------
    def _initial_peak_guesses_cycles(self) -> dict:
        """
        Default initial guesses from illumination.
        Returns dict[peak_index -> q_guess(2,)] in cycles/unit.
        """
        wavevectors, indices = self.illumination.get_all_wavevectors_projected()
        return {idx: (wv / (2 * np.pi)) for idx, wv in zip(indices, wavevectors)}

    # ----------------------------
    # Hooks for children
    # ----------------------------
    def _init_global_state(self, stack: np.ndarray, peak_guesses: dict) -> dict[str, Any]:
        return {}

    def _init_peak_state(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        peak_guesses: dict,
        global_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {"q": np.array(peak_guesses[peak], dtype=np.float64)}

    @abstractmethod
    def _iterate_one(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        q_coords: tuple[np.ndarray, np.ndarray],
        peak_state: dict[str, Any],
        zooming_factor: int,
        debug_info_level: int,
        **kwargs,
    ) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray], str]:
        """One refinement iteration for one peak."""
        ...

    def _final_peak_cycles(self, peak_state: dict[str, Any]) -> np.ndarray:
        return np.array(peak_state["q"], dtype=np.float64)

    def _postprocess_estimated_cycles(self, estimated_cycles: dict, stack: np.ndarray) -> dict:
        return estimated_cycles

    # ----------------------------
    # Debug: debug>=2
    # ----------------------------
    def _debug_show_initial_final_on_avg_ft_2d(self, stack: np.ndarray, initial_cycles: dict, final_cycles: dict):
        import matplotlib.pyplot as plt

        qx, qy = self.optical_system.otf_frequencies  # 1D axes in cycles/unit
        q_coords = (np.asarray(qx), np.asarray(qy))

        # Cache avg|FT| per r once (avoid recomputing per peak)
        ft_avg_per_r: dict[int, np.ndarray] = {}
        for r in range(self.illumination.Mr):
            ft_stack = self._czt_nd(stack[r], q_coords)   # (Mt, qx, qy)
            ft_avg_per_r[r] = np.mean(np.abs(ft_stack), axis=0)

        for peak in self._m:
            r = peak[0]
            ft_avg = ft_avg_per_r[r]

            q0 = initial_cycles[peak]
            q1 = final_cycles[peak]

            fig, ax = plt.subplots()
            ax.set_title(f"r={r}, peak={peak}: avg |FT| (initial x, final o)")
            ax.imshow(
                np.log1p(10**4 * ft_avg.T),
                origin="lower",
                aspect="auto",
                extent=[q_coords[0].min(), q_coords[0].max(), q_coords[1].min(), q_coords[1].max()],
            )
            ax.plot(q0[0], q0[1], marker="x", linestyle="None")
            ax.plot(q1[0], q1[1], marker="o", linestyle="None")
            ax.set_xlabel("qx (cycles/unit)")
            ax.set_ylabel("qy (cycles/unit)")
            plt.show()

    # ----------------------------
    # Debug: debug>=3
    # ----------------------------
    def _debug_show_iteration_surface_2d(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        q_coords: tuple[np.ndarray, np.ndarray],
        peak_state: dict[str, Any],
        it: int,
    ):
        import matplotlib.pyplot as plt

        payload = self._debug_iteration_payload(stack, r, peak, q_coords, peak_state)
        if not payload:
            return

        img = payload["image"]
        qmax = payload.get("max_q", None)
        title = payload.get("title", f"r={r}, peak={peak}, it={it}")

        qc = payload.get("q_coords", None)
        if qc is None:
            qc = q_coords

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.imshow(
            img.T,
            origin="lower",
            aspect="auto",
            extent=[qc[0].min(), qc[0].max(), qc[1].min(), qc[1].max()],
        )
        if qmax is not None:
            ax.plot(qmax[0], qmax[1], marker="o", linestyle="None")
        ax.set_xlabel("qx (cycles/unit)")
        ax.set_ylabel("qy (cycles/unit)")
        plt.show()

    @abstractmethod
    def _debug_iteration_payload(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        q_coords: tuple[np.ndarray, np.ndarray],
        peak_state: dict[str, Any],
    ) -> dict | None:
        """
        Return a dict like:
          {'image': (qx,qy) array, 'max_q': (2,) optional, 'title': str optional}
        for debug>=3, or None to skip.
        """
        ...

class PeaksEstimatorInterpolation(PeaksEstimatorIterative):
    """
    2D-only estimator.
    Peak positions are tracked in cycles/unit.
    """

    def _iterate_one(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        q_coords: tuple[np.ndarray, np.ndarray],
        peak_state: dict[str, Any],
        zooming_factor: int,
        debug_info_level: int,
        **kwargs,
    ):
        # Evaluate FT of each image on current q-grid
        ft_stack = self._czt_nd(stack[r], q_coords)       # (Mt, qx, qy)
        ft_avg = np.mean(np.abs(ft_stack), axis=0)        # (qx, qy)

        # Peak estimate is argmax of mean magnitude
        q_center = self._argmax_coords(ft_avg, q_coords)  # (2,) cycles/unit
        peak_state["q"] = q_center

        # Debug>=3: stash the surface + argmax
        if debug_info_level >= 3:
            peak_state["_dbg_img"] = ft_avg
            peak_state["_dbg_qmax"] = q_center

        # Zoom around the found maximum (Interpolation behavior)
        q_coords = self._zoom_coords(q_coords, zooming_factor, center=q_center)

        return peak_state, q_coords, f"max {q_center}"

    def _debug_iteration_payload(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        q_coords: tuple[np.ndarray, np.ndarray],
        peak_state: dict[str, Any],
    ) -> dict | None:
        img = peak_state.get("_dbg_img", None)
        if img is None:
            return None
        return {
            "title": f"Interpolation: r={r}, peak={peak}, mean |FT| on current grid",
            "image": img,
            "max_q": peak_state.get("_dbg_qmax", None),
        }

    def _postprocess_estimated_cycles(self, estimated_cycles: dict, stack: np.ndarray) -> dict:
        # Optional correction step you already had in your design
        return self._correct_peak_position(estimated_cycles, stack)
    
    def _sample_stack_mean_abs(self, stack_r: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        """
        stack_r: (Mt, H, W)
        q_values: (..., 2) in cycles/unit OR radians/unit depending on off_grid_ft convention
        returns: mean_n |FT(image_n)(q_values)| with shape q_values.shape[:-1]
        """
        grid = self.optical_system.x_grid
        vals = [off_grid_ft(img, grid, q_values) for img in stack_r]
        return np.mean(np.abs(np.array(vals)), axis=0)

    def _sample_otf_abs(self, q_values: np.ndarray) -> np.ndarray:
        """
        q_values: (..., 2)
        returns: |OTF(q_values)| with shape q_values.shape[:-1]
        """
        grid = self.optical_system.x_grid
        psf = self.optical_system.psf
        return np.abs(off_grid_ft(psf, grid, q_values))
    
    @abstractmethod
    def _correct_peak_position(self, estimated_peaks: dict, stack: np.ndarray, dk: float = 1e-4) -> dict:
        """
        Account for non-uniform OTF to correct peak positions, assuming
        estimate is close to the true illumination wavevectors.
        Input/Output: dict[peak_index -> q_cycles(2,)].
        """
        ...

class PeaksEstimatorCrossCorrelation(PeaksEstimatorIterative):
    """
    2D-only estimator.
    Tracks peaks in cycles/unit.
    CC refinement updates q by subtracting the mean argmax location in q-space
    and then zooms the q-grid towards 0 (center=None), matching your CC logic.
    """

    def _initial_peak_guesses_cycles(self) -> dict:
        # make parent center the q-window at 0 (shift-search)
        wvs, idxs = self.illumination.get_all_wavevectors_projected()
        z = np.zeros(2, dtype=np.float64)
        return {idx: z.copy() for idx in idxs}

    def _init_global_state(self, stack: np.ndarray, peak_guesses: dict) -> dict[str, Any]:
        """
        CC global state:
        - abs initial guesses (only to initialize peak_state["q"], not used as "global fit")
        - caches for first-iteration (original-grid) CC results per peak
        """
        wvs, idxs = self.illumination.get_all_wavevectors_projected()
        abs_guesses_cycles = {idx: (wv / (2.0 * np.pi)) for idx, wv in zip(idxs, wvs)}

        return {
            "abs_guesses_cycles": abs_guesses_cycles,
            # first-iteration "fit" caches (on original neighbourhood grid)
            "C0": {},      # peak -> dict[(m,n1,n2)] -> array (optional, heavy)
            "S0": {},      # peak -> S(q) array
            "dq0": {},     # peak -> dq argmax on original grid
            "q0": {},      # peak -> (qx,qy) coords used for S0
        }

    def _init_peak_state(self, stack, r, peak, peak_guesses, global_state):
        return {
            "q": np.array(global_state["abs_guesses_cycles"][peak], dtype=np.float64),  # absolute estimate
            "_first_iter": True,  # tells _iterate_one to cache C/S on the original grid
            "_global_state": global_state,
        }

    def _iterate_one(self, stack, r, peak, q_coords, peak_state, zooming_factor, debug_info_level):
        # --- update modulation pattern from CURRENT absolute estimate ---
        k_rad = (2.0 * np.pi) * peak_state["q"]
        phase = np.einsum("...l,l->...", self.optical_system.x_grid, k_rad)
        estimated_pattern_refined = {peak: np.exp(1j * phase)}  # consider sign flip if needed

        # --- compute CC for this peak on the CURRENT grid (first time: original neighbourhood grid) ---
        C = self.cross_correlation_matrix(
            self.optical_system,
            self.illumination,
            images=stack,
            r=r,
            estimated_modulation_patterns=estimated_pattern_refined,
            q_coords_cycles=q_coords,
        )

        # S(q)=sum_{n1,n2} |C_{n1n2}(q)|^2  (only entries are for this 'peak' anyway)
        Sm = None
        for _, arr in C.items():
            a2 = np.abs(arr) ** 2
            Sm = a2 if Sm is None else (Sm + a2)

        dq = self._argmax_coords(Sm, q_coords)
        peak_state["q"] = peak_state["q"] - dq

        # --- cache first-iteration "fit" on original grid ---
        if peak_state.get("_first_iter", False):
            gs = peak_state["_global_state"]  # we’ll inject this below; see note
            gs["S0"][peak] = Sm
            gs["dq0"][peak] = dq
            gs["q0"][peak] = q_coords
            # Optional: cache C0 too (can be huge). Keep off unless you really need it.
            # gs["C0"][peak] = C
            peak_state["_first_iter"] = False

        # debug surface (use coords that generated Sm!)
        if debug_info_level >= 3:
            peak_state["_dbg_img"] = Sm
            peak_state["_dbg_qmax"] = dq
            peak_state["_dbg_q_coords"] = q_coords

        # refine shift grid around 0 (your zoom_coords already does the correct thing now)
        q_coords = self._zoom_coords(q_coords, zooming_factor, center=None)
        return peak_state, q_coords, f"shift {dq} -> q {peak_state['q']}"

    def _debug_iteration_payload(self, stack, r, peak, q_coords, peak_state):
        img = peak_state.get("_dbg_img", None)
        if img is None:
            return None
        return {
            "title": f"CC: r={r}, peak={peak}, S(q)=sum|C|^2 (shift grid)",
            "image": img,
            "max_q": peak_state.get("_dbg_qmax", None),
            "q_coords": peak_state.get("_dbg_q_coords", None),
        }
    
    @staticmethod
    def phase_modulation_patterns(optical_system: OpticalSystem, illumination_vectors_rad: dict) -> dict:
        """
        illumination_vectors_rad: dict[index -> k_rad] where k_rad is radians/unit
        returns dict[index -> exp(1j * x·k)]
        """
        phase_modulation_patterns = {}
        for index, wavevector_rad in illumination_vectors_rad.items():
            phase = np.einsum("...l,l->...", optical_system.x_grid, wavevector_rad)
            phase_modulation_patterns[index] = np.exp(1j * phase)
        return phase_modulation_patterns
    
    @staticmethod
    def cross_correlation_matrix(
        optical_system: OpticalSystem,
        illumination: PlaneWavesSIM,
        images: np.ndarray,
        r: int,
        estimated_modulation_patterns: dict,
        q_coords_cycles: tuple[np.ndarray, np.ndarray],
    ) -> dict:
        """
        C^{m}_{n1 n2}(q) via CZT (2D).
        Returns dict[(m, n1, n2)] -> ndarray(qx, qy).

        Keys follow: (m_index, n1, n2) where m_index is the illumination harmonic index.
        """
        keys, sigs = [], []
        Mt = int(illumination.spatial_shifts.shape[1])

        for m, pat in estimated_modulation_patterns.items():
            if m[0] != r:
                continue
            for n1 in range(Mt):
                for n2 in range(Mt):
                    sig = images[r, n1] * images[r, n2] * pat
                    if n1 == n2:
                        sig = sig - (images[r, n1] * pat)
                    keys.append((m, n1, n2))
                    sigs.append(sig)

        sig_stack = np.stack(sigs, axis=0)  # (K, H, W)

        # CZT on the stack (convert cycles->radians inside the call)
        x_coords = optical_system.psf_coordinates
        q_coords_rad = tuple((2.0 * np.pi) * np.asarray(qc) for qc in q_coords_cycles)
        D = optical_system.dimensionality
        spatial_axes = tuple(range(sig_stack.ndim - D, sig_stack.ndim))

        out = hpc_utils.czt_nd_fourier(
            np.asarray(sig_stack),
            x_coords,
            q_coords_rad,
            axes=spatial_axes,
            rtol=1e-6,
            atol=1e-12,
        )
        out_np = hpc_utils._to_numpy(out)  # (K, qx, qy)

        return {keys[i]: out_np[i] for i in range(len(keys))}

    @staticmethod
    def autocorrelation_matrix(
        optical_system: OpticalSystem,
        illumination: PlaneWavesSIM,
        images: np.ndarray,
        r: int,
        estimated_modulation_patterns: dict,
        q_coords_cycles: tuple[np.ndarray, np.ndarray],
    ) -> dict:
        """
        A^{m}_{n}(q) via CZT (2D), returned in the SAME KEY/SHAPE convention
        as cross_correlation_matrix:

            dict[(m, n1, n2)] -> ndarray(qx, qy)

        but only diagonal entries exist (n1 == n2 == n).

        Uses your original autocorrelation definition:
            sig = I_n*I_n*pat - I_n*pat
        """
        keys, sigs = [], []
        Mt = int(illumination.spatial_shifts.shape[1])

        for m, pat in estimated_modulation_patterns.items():
            if m[0] != r:
                continue
            for n in range(Mt):
                sig = images[r, n] * images[r, n] * pat
                sig = sig - (images[r, n] * pat)
                keys.append((m, n))
                sigs.append(sig)

        sig_stack = np.stack(sigs, axis=0)  # (K, H, W)

        x_coords = optical_system.psf_coordinates
        q_coords_rad = tuple((2.0 * np.pi) * np.asarray(qc) for qc in q_coords_cycles)
        D = optical_system.dimensionality
        spatial_axes = tuple(range(sig_stack.ndim - D, sig_stack.ndim))

        out = hpc_utils.czt_nd_fourier(
            np.asarray(sig_stack),
            x_coords,
            q_coords_rad,
            axes=spatial_axes,
            rtol=1e-6,
            atol=1e-12,
        )
        out_np = hpc_utils._to_numpy(out)  # (K, qx, qy)

        return {keys[i]: out_np[i] for i in range(len(keys))}
    

class PeaksEstimatorCrossCorrelation2D(PeaksEstimatorCrossCorrelation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)


class PeaksEstimatorCrossCorrelation3D(PeaksEstimatorInterpolation):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)

class PeaksEstimatorInterpolation2D(PeaksEstimatorInterpolation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)

    def _correct_peak_position(self, estimated_peaks: dict, stack: np.ndarray, dk: float = 10**-4) -> dict:
        otf0 = {}
        otf_grad = {}

        for key, peak in estimated_peaks.items():
            q = np.array(
                [
                    peak,
                    peak - np.array((dk, 0.0)),
                    peak + np.array((dk, 0.0)),
                    peak - np.array((0.0, dk)),
                    peak + np.array((0.0, dk)),
                ],
                dtype=np.float64,
            )
            otf = self._sample_otf_abs(q)
            otf0[key] = otf[0]
            otf_grad[key] = np.array([otf[2] - otf[1], otf[4] - otf[3]], dtype=np.float64) / (2.0 * dk)

        # shared local f(q) model around 0-offset (as in your original code)
        q0 = np.array(
            [
                (0.0, 0.0),                          # 0
                (-dk, 0.0), (dk, 0.0),               # 1,2
                (0.0, -dk), (0.0, dk),               # 3,4
                (-dk, -dk), (-dk, dk),               # 5,6
                (dk, -dk), (dk, dk),                 # 7,8
            ],
            dtype=np.float64,
        )

        I = self._sample_stack_mean_abs(stack[0], q0)               # (K,)
        O = self._sample_otf_abs(q0)                                # (K,)

        f = I / O
        f_grad = np.array([f[2] - f[1], f[4] - f[3]], dtype=np.float64) / (2.0 * dk)

        f_hessian = (
            np.array(
                [
                    [f[2] - 2.0 * f[0] + f[1], (f[8] - f[7] - f[6] + f[5]) / 2.0],
                    [(f[8] - f[7] - f[6] + f[5]) / 2.0, f[4] - 2.0 * f[0] + f[3]],
                ],
                dtype=np.float64,
            )
            / (dk**2)
        )
        invH = np.linalg.inv(f_hessian)

        corrected = {}
        for key, peak in estimated_peaks.items():
            otf = otf0[key]
            g = otf_grad[key]
            delta_k = (
                f[0] / otf
                * (1.0 + f[0] / (2.0 * otf**2) * (g.T @ invH @ g))
                * (invH @ g)
            )
            print("correction", delta_k)
            corrected[key] = peak + delta_k

        return corrected

class PeaksEstimatorInterpolation3D(PeaksEstimatorInterpolation):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves3D, optical_system: OpticalSystem3D):
        super().__init__(illumination, optical_system)

    def _correct_peak_position(self, estimated_peaks: dict, stack: np.ndarray, dk: float = 10**-4) -> dict:
        otf0 = {}
        otf_grad = {}

        for key, peak in estimated_peaks.items():
            q = np.array(
                [
                    peak,
                    peak - np.array((dk, 0.0, 0.0)),
                    peak + np.array((dk, 0.0, 0.0)),
                    peak - np.array((0.0, dk, 0.0)),
                    peak + np.array((0.0, dk, 0.0)),
                    peak - np.array((0.0, 0.0, dk)),
                    peak + np.array((0.0, 0.0, dk)),
                ],
                dtype=np.float64,
            )
            otf = self._sample_otf_abs(q)
            otf0[key] = otf[0]
            otf_grad[key] = np.array(
                [otf[2] - otf[1], otf[4] - otf[3], otf[6] - otf[5]],
                dtype=np.float64,
            ) / (2.0 * dk)

        # shared local f(q) model around 0-offset
        q0 = np.array(
            [
                (0.0, 0.0, 0.0),                      # 0
                (-dk, 0.0, 0.0), (dk, 0.0, 0.0),       # 1,2
                (0.0, -dk, 0.0), (0.0, dk, 0.0),       # 3,4
                (0.0, 0.0, -dk), (0.0, 0.0, dk),       # 5,6
                (-dk, -dk, 0.0), (-dk, dk, 0.0),       # 7,8
                (dk, -dk, 0.0), (dk, dk, 0.0),         # 9,10
                (0.0, -dk, -dk), (0.0, -dk, dk),       # 11,12
                (0.0, dk, -dk), (0.0, dk, dk),         # 13,14
                (-dk, 0.0, -dk), (dk, 0.0, -dk),       # 15,16
                (-dk, 0.0, dk), (dk, 0.0, dk),         # 17,18
            ],
            dtype=np.float64,
        )

        I = self._sample_stack_mean_abs(stack[0], q0)               # (K,)
        O = self._sample_otf_abs(q0)                                # (K,)

        f = I / O
        f_grad = np.array([f[2] - f[1], f[4] - f[3], f[6] - f[5]], dtype=np.float64) / (2.0 * dk)

        f_hessian = (
            np.array(
                [
                    [f[2] - 2.0 * f[0] + f[1], (f[10] - f[9] - f[8] + f[7]) / 2.0, (f[16] - f[15] - f[17] + f[18]) / 2.0],
                    [(f[10] - f[9] - f[8] + f[7]) / 2.0, f[4] - 2.0 * f[0] + f[3], (f[14] - f[13] - f[12] + f[11]) / 2.0],
                    [(f[16] - f[15] - f[17] + f[18]) / 2.0, (f[14] - f[13] - f[12] + f[11]) / 2.0, f[6] - 2.0 * f[0] + f[5]],
                ],
                dtype=np.float64,
            )
            / (dk**2)
        )
        invH = np.linalg.inv(f_hessian)

        corrected = {}
        for key, peak in estimated_peaks.items():
            otf = otf0[key]
            g = otf_grad[key]
            delta_k = (
                f[0] / otf
                * (1.0 + f[0] / (2.0 * otf**2) * (g.T @ invH @ g))
                * (invH @ g)
            )
            print("correction", delta_k)
            corrected[key] = peak + delta_k

        return corrected

class PhasesEstimator:
    """
    Estimate phase offsets from a raw SIM stack.

    Interface mirrors PeaksEstimator:
    - constructed with ``(illumination, optical_system)``
    - canonical half of m-values pre-computed once in ``__init__`` via ``_select_m()``
    - public methods are regular instance methods that take only
      ``(stack, refined_wavevectors)``

    Design notes
    ------------
    * **One peak at a time** – ``phase_matrix_autocorrelation`` evaluates the
      autocorrelation matrix for each canonical peak independently.  GPU memory
      is therefore bounded by a single image-sized array at a time rather than
      scaling with ``Mpeaks × Mt``.
    * **Symmetry** – ``_select_m`` (identical contract to
      ``PeaksEstimator._select_m``) returns only the canonical (positive)
      representative of every conjugate-symmetric pair.  The conjugate partner
      is then filled analytically as ``exp(−iφ)``.
    * **Dimension-agnostic** – the array-centre index is derived from the
      output shape at runtime, so the same code path handles 2-D and 3-D stacks
      without any hardcoded ``Nx//2, Ny//2`` indices.
    """

    def __init__(self, illumination: PlaneWavesSIM, optical_system: OpticalSystem, first_order_phases_only: bool = True):
        self.illumination = illumination
        self.optical_system = optical_system
        if self.illumination.dimensionality != self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: "
                f"{self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )
        self.first_order_phases_only = first_order_phases_only
        self._m = self._select_m()
        if first_order_phases_only:
            self._filter_first_orders()

    # -------------------------------------------------------------------------------
    # Leave only first orders which typically ensures more robust phase estimation 
    # -------------------------------------------------------------------------------    
    def _filter_first_orders(self):
        m_first_orders = set()
        non_degenerate_dimensions = self.illumination.find_non_degenerate_dimensions()
        for r in range(self.illumination.Mr):
            harmonics_r_filtered = list(filter(lambda m: m[0] == r, self._m))
            for i in non_degenerate_dimensions:
                harmonics_i_filtered = sorted(filter(lambda m: m[1][i] > 0, harmonics_r_filtered), key=lambda m: 10 ** 8 * (m[1][i]-1) + np.sum(np.abs(np.array(m[1]))))
                m_first_orders.add(harmonics_i_filtered[0])
        self._m = m_first_orders

    def _select_m(self):
        """
        Return a deterministically-ordered tuple of the canonical (positive)
        representative for every conjugate-symmetric harmonic pair, skipping
        the zero-order term.  Mirrors PeaksEstimator._select_m exactly.
        """
        m_unique = set()
        for m in self.illumination.harmonics.keys():
            if np.isclose(np.sum(np.abs(np.array(m[1]))), 0):
                continue
            m_inv = (m[0], tuple(-np.array(m[1])))
            canonical_m = m if m > m_inv else m_inv
            m_unique.add(canonical_m)
        return tuple(sorted(list(m_unique), key=lambda x: (x[0], sum(abs(v) for v in x[1]), x[1])))

    def _compute_full_phase_matrix_from_first_order_phases(self, phase_matrix: dict) -> dict:
        """
        Compute the full phase matrix from the first order phases.
        """
        non_degenerate_dimensions = self.illumination.find_non_degenerate_dimensions()
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                vec_m = []
                vec_phases = []
                for index in phase_matrix:
                    m = index[2]
                    if index[0] == r and index[1] == n and tuple([r, m]) in self._m:
                        vec_m.append(np.array([m[i] for i in non_degenerate_dimensions]))
                        vec_phases.append(np.angle(phase_matrix[index]))
                base_phase_vectors = np.linalg.solve(np.array(vec_m), np.array(vec_phases))
                
                for harmonic in self.illumination.harmonics:
                    r_, m = harmonic
                    if r_ == r:
                        phase = np.dot(np.array([m[i] for i in non_degenerate_dimensions]), np.array(base_phase_vectors))
                        # print(r, m, phase)
                        phase_matrix[r, n, m] = np.exp(1j * phase)
        return phase_matrix
    # ------------------------------------------------------------------
    # Public estimation methods
    # ------------------------------------------------------------------

    def phase_matrix_cross_correlation(
        self,
        stack: np.ndarray,
        refined_wavevectors: dict,
    ) -> dict:
        """
        Estimate phase shifts via cross-correlation (Gustafsson 2000).
        """
        raise NotImplementedError("Full cross-correlation method is not implemented yet")

    def phase_matrix_autocorrelation(
        self,
        stack: np.ndarray,
        refined_wavevectors: dict,
    ) -> dict:
        """
        Estimate phases from the per-peak autocorrelation matrix.

        Memory optimisation: the autocorrelation matrix is evaluated
        **one canonical peak at a time**, so the CZT batch never exceeds
        ``Mt`` images regardless of how many harmonics are present.
        The conjugate-symmetric partner is filled analytically.
        """
        phase_matrix = {}
        D = self.optical_system.dimensionality
        zero_m = tuple([0] * D)

        # Zero-order phase is always unity for every (r, n)
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                phase_matrix[(r, n, zero_m)] = 1.0 + 0.0j

        # Iterate over canonical half only — conjugate is set by symmetry
        for m_idx in self._m:
            r = m_idx[0]
            single_pattern = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(
                self.optical_system, {m_idx: refined_wavevectors[m_idx]}
            )
            Amn = PeaksEstimatorCrossCorrelation.autocorrelation_matrix(
                self.optical_system,
                self.illumination,
                stack,
                r,
                single_pattern,
                self.optical_system.otf_frequencies,
            )

            # Centre index derived from array shape — works for 2-D and 3-D
            sample_arr = next(iter(Amn.values()))
            center = tuple(s // 2 for s in sample_arr.shape)

            for n in range(self.illumination.Mt):
                phase_val = np.angle(Amn[(m_idx, n)][center])
                phase_matrix[(r, n, m_idx[1])]                    = np.exp( 1j * phase_val)
                phase_matrix[(r, n, tuple(-v for v in m_idx[1]))] = np.exp(-1j * phase_val)

        if self.first_order_phases_only:
            phase_matrix = self._compute_full_phase_matrix_from_first_order_phases(phase_matrix)

        return phase_matrix

    def phase_matrix_peak_values(
        self,
        stack: np.ndarray,
        refined_wavevectors: dict,
    ) -> dict:
        """
        Estimate phase shifts as ``arg F[I_n](k_m)``.

        This works because at ``k = k_m`` the sample spectrum
        ``f(k − k_m) = f(0)`` is real and positive, so the phase of
        ``I(k_m)`` equals the phase of the modulation coefficient ``a_m``.
        """
        phase_matrix = {}
        grid = self.optical_system.x_grid
        for n in range(stack.shape[1]):
            for index in self._m:
                r = index[0]
                m = index[1]
                wavevector = refined_wavevectors[index] / (2 * np.pi)
                ft = off_grid_ft(stack[r, n], grid, np.array((wavevector,)))
                phase_matrix[(r, n, m)] = np.exp(-1j * np.angle(ft))
        
        if self.first_order_phases_only:
            phase_matrix = self._compute_full_phase_matrix_from_first_order_phases(phase_matrix)
        return phase_matrix

    def compute_spatial_shifts(
        self,
        phase_matrix: np.ndarray,
        refined_wavevectors: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("compute_spatial_shifts is not implemented yet.")


class PhasesEstimator2D(PhasesEstimator):
    """2-D phase estimator.  All logic lives in PhasesEstimator."""
    pass

class PhasesEstimator3D(PhasesEstimator):
    """3-D phase estimator.  All logic lives in PhasesEstimator."""
    pass


