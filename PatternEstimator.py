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

    peaks_estimation_methods = ('interpolation', 'cross_correlation')
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
        peaks_estimation_method='interpolation', 
        phase_estimation_method='autocorrelation',
        modulation_coefficients_method='default',
        peak_search_area_size: int = 11,
        zooming_factor: int = 3, 
        max_iterations: int = 20,
        ssnr_estimation_iters: int = 10, 
        debug_info_level = 0 
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

        if not peaks_estimation_method in self.peaks_estimation_methods:
            raise ValueError(
                f"Unknown method of peaks estimation {peaks_estimation_method}. Available methods are {self.peaks_estimation_methods}"
            )
        if not phase_estimation_method in self.phase_estimation_methods:
            raise ValueError(
                f"Unknown method of phase estimation {phase_estimation_method}. Available methods are {self.phase_estimation_methods}"
            )
        if not modulation_coefficients_method in self.modulation_coefficients_methods:
            raise ValueError(
                f"Unknown method of modulation coefficients estimation {modulation_coefficients_method}. Available methods are {self.modulation_coefficients_methods}"
            )
    
        peaks, rotation_angles = self.estimate_peaks(peaks_estimation_method, stack, peak_search_area_size, zooming_factor, max_iterations, debug_info_level=debug_info_level)
        
        if debug_info_level > 0:
            for r in range(Mr):
                print('r = ', r,  'rotation_angle = ', np.round(rotation_angles[r] / np.pi * 180, 1), 'degrees')
            for sim_index in peaks:
                print('r = ', sim_index[0], 'm = ', sim_index[1], 'wavevector = ', np.round((peaks[sim_index]), 3), '1 / lambda')

        phase_matrix = self.build_phase_matrix(phase_estimation_method, stack, peaks, debug_info_level)

        if debug_info_level > 0:
            for sim_index in phase_matrix:
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

    def estimate_peaks(self,
                        peaks_estimation_method: str,
                        stack: np.ndarray,
                        peak_search_area_size: int,
                        zooming_factor: int,
                        max_iterations: int,
                        debug_info_level: int = 0) -> Tuple[np.ndarray, dict]:
        
        """
        Estimate the peaks postitions of the illumination pattern from the stack.
        """
        if peaks_estimation_method == 'cross_correlation':
            PeaksEstimator = PeaksEstimatorCrossCorrelation2D if self.dimensionality == 2 else PeaksEstimatorCrossCorrelation3D

        if peaks_estimation_method == 'interpolation':
            PeaksEstimator = PeaksEstimatorInterpolation2D if self.dimensionality == 2 else PeaksEstimatorInterpolation3D

        peaks_estimator = PeaksEstimator(self.illumination, self.optical_system)
        
        if len(stack.shape[2:]) == self.dimensionality:
            peaks, rotation_angles = peaks_estimator.estimate_peaks(stack, 
                                                                peak_search_area_size, 
                                                                zooming_factor, 
                                                                max_iterations, 
                                                                debug_info_level=debug_info_level)
        else:
            averaging_dimensions = range(2 + self.dimensionality, len(stack.shape), 1)
            sizes = [stack.shape[i] for i in averaging_dimensions]
            
            if debug_info_level > 1:
                print('Stack split into substacks. Averaging along the axes', averaging_dimensions, 'of size', sizes)

            sum_peaks = {}
            for idx in np.ndindex(*sizes):
                slicing = tuple([slice(None) for i in range(2 + self.dimensionality)]) + idx
                substack = stack[slicing]
                peaks, _ = peaks_estimator.estimate_peaks(substack, 
                                                          peak_search_area_size, 
                                                          zooming_factor, 
                                                          max_iterations, 
                                                          debug_info_level=debug_info_level)
                
                if debug_info_level > 1:
                    print('substack', idx)
                    for sim_index in peaks:
                        print('r = ', sim_index[0], 'm = ', sim_index[1], 'wavevector = ', round(np.angle(peaks[sim_index])), '1 / lambda')

                # Proper weighting should be added here
                for key in peaks.keys():
                    if not key in sum_peaks.keys():
                        sum_peaks[key] = peaks[key]
                    else: 
                        sum_peaks[key] += peaks[key]


            averaged_peaks = {sum_peaks[key] / np.prod(np.array(sizes)): key for key in sum_peaks.keys()}
            rotation_angles = peaks_estimator.estimate_rotation_angles(averaged_peaks)
            refined_base_vectors = peaks_estimator.refine_base_vectors(averaged_peaks, rotation_angles)
            peaks = peaks_estimator.refine_wavevectors(refined_base_vectors, rotation_angles)

        return peaks, rotation_angles
    
    def build_phase_matrix(self, 
                           phase_estimation_method: str = 'cross_correlation',
                           stack: np.ndarray = None,
                           peaks: dict = None,
                           debug_info_level=0) -> dict:
        
        """
        Estimate the peaks postitions of the illumination pattern from the stack.
        """

        match phase_estimation_method:
            case 'peak_phases':
                phase_estimation_function = PhasesEstimator.phase_matrix_peak_values
            case 'autocorrelation':
                phase_estimation_function = PhasesEstimator.phase_matrix_autocorrelation
            case 'cross_correlation':	
                phase_estimation_function = PhasesEstimator.phase_matrix_cross_correlation
                
        if len(stack.shape[2:]) == self.dimensionality:
            phase_matrix = phase_estimation_function(self.optical_system, self.illumination, stack, peaks)
        
        else:
            averaging_dimensions = range(2 + self.dimensionality, len(stack.shape), 1)
            sizes = [stack.shape[i] for i in averaging_dimensions]

            if debug_info_level > 1:
                print('Stack split into substacks. Averaging along the axes', averaging_dimensions, 'of size', sizes)

            sum_phases = {}
            if debug_info_level > 1:
                print('Stack split into substacks. Averaging along the axes', averaging_dimensions, 'of size', sizes)
            for idx in np.ndindex(*sizes):
                slicing = tuple([slice(None) for i in range(2 + self.dimensionality)]) + idx
                substack = stack[slicing]
                phases = phase_estimation_function(self.optical_system, self.illumination,  substack, peaks)
                if debug_info_level > 1:
                    print('substack', idx)
                    for sim_index in phase_matrix:
                        print('r = ', sim_index[0], 'n = ', sim_index[1], 'm = ', sim_index[2], 'phase = ', round(np.angle(phase_matrix[sim_index]) / np.pi * 180, 1), 'degrees')

                # Proper weighting should be added here

                for key in phases.keys():
                    if not key in sum_phases.keys():
                        sum_phases[key] = phases[key]
                    else: 
                        sum_phases[key] += phases[key]


            phase_matrix = {sum_phases[key] / np.prod(np.array(sizes)): key for key in sum_phases.keys()}

        return phase_matrix
         
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
        illumination_estimated.electric_field_plane_waves = self.illumination.electric_field_plane_waves
        illumination_estimated.normalize_spatial_waves()
        for plane_wave in illumination_estimated.electric_field_plane_waves:
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

class IlluminationPatternEstimator3D(IlluminationPatternEstimator):
    """
    Class for estimating the illumination parameters from a raw SIM stack in 3D.
    """
    dimensionality = 3

    @staticmethod
    def illumination_3d_from_illuminaton_2d_and_total_wavelength(illumination_2d: IlluminationPlaneWaves2D,
                                                                 illumination_3d: IlluminationPlaneWaves3D,
                                                                 total_wavelength: float, 
                                                                 wavevector_expansion: tuple = (1, 0, 1), 
                                                                 debug_info: bool = True) -> IlluminationPlaneWaves3D:
        """
        Assuming we found 2d illumination pattern, we can find 3d illumination pattern if we know the total wavelength, 
        which is typically the case. It is assumed for now, that the missing direction is z-direction, in the same fashion as it is assumed, 
        that we rotate the illumination pattern around the z-axis.
        """
        # target_dimensions = target_illumination_3d.dimensions
        # target_illumination_3d.dimensions = tuple([1 if i != missing_dimension else 0 for i in range(3)])
        k_vector_length = 2 * np.pi / total_wavelength
        if not illumination_3d.Mr == illumination_2d.Mr:
            raise ValueError(
                f"Number of rotations does not coincide between estimated and target illumination: {illumination_3d.Mr} != {illumination_2d.Mr}"
            )
        base_vectors_3d = np.zeros((illumination_2d.Mr, 3))
        for r in range(illumination_2d.Mr):
            base_vectors_2d = illumination_2d.get_base_vectors(r)
            components_2d = wavevector_expansion[0:2]
            missing_base = (k_vector_length **2 - np.sum(base_vectors_2d**2 * components_2d**2))**0.5 / wavevector_expansion[2]
            if debug_info:
                print("r", r, "missing_base length", missing_base)
            base_vectors_3d[r, :2] = base_vectors_2d
            base_vectors_3d[r, 2] = missing_base
            
            illumination_3d.angles[r] = illumination_2d.angles[r]

        for sim_index in illumination_3d.harmonics.keys():
            r = sim_index[0]
            m = sim_index[1][:2]
            illumination_3d.harmonics[sim_index].wavevector = base_vectors_3d[r] * np.array(m)

        illumination_3d.phase_matrix = illumination_2d.phase_matrix
    
        return illumination_3d

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
            if m not in m_unique:
                m_inv = (m[0], tuple(-np.array(m[1])))
                if m_inv not in m_unique:
                    m_unique.add(m)
        return tuple(m_unique)

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

        for index in averaged_maxima.keys():
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
    ):
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
                if debug_info_level > 0:
                    if debug_info_level > 1:
                        dqs = [float(q_coords[d][1] - q_coords[d][0]) for d in range(2)]
                        print("iteration", it, peak, dbg, "dq", dqs)
                    else:
                        print("iteration", it, peak, dbg)

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
                ft_avg.T,
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
        # store absolute initial guesses (cycles/unit) for peak-state init
        wvs, idxs = self.illumination.get_all_wavevectors_projected()
        abs_guesses = {idx: (wv / (2.0 * np.pi)) for idx, wv in zip(idxs, wvs)}
        return {"abs_guesses_cycles": abs_guesses}

    def _init_peak_state(self, stack, r, peak, peak_guesses, global_state):
        return {
            "q": np.array(global_state["abs_guesses_cycles"][peak], dtype=np.float64),  # absolute (cycles/unit)
        }

    def _iterate_one(self, stack, r, peak, q_coords, peak_state, zooming_factor, debug_info_level):
        # --- update modulation pattern from CURRENT absolute estimate ---
        k_rad = (2.0 * np.pi) * peak_state["q"]  # radians/unit
        phase = np.einsum("...l,l->...", self.optical_system.x_grid, k_rad)
        esimtated_pattern_refined = {peak: np.exp(1j * phase)}

        # --- compute CC only for this peak m (not all m) ---
        C = self.cross_correlation_matrix(
            self.optical_system,
            self.illumination,
            images=stack,
            r=r,
            estimated_modulation_patterns=esimtated_pattern_refined,
            q_coords_cycles=q_coords,
        )

        # robust surface: S(q)=sum_{n1,n2} |C_{n1n2}(q)|^2
        Sm = None
        for _, arr in C.items():
            a2 = np.abs(arr) ** 2
            Sm = a2 if Sm is None else (Sm + a2)

        dq = self._argmax_coords(Sm, q_coords)     # shift in cycles/unit
        peak_state["q"] = peak_state["q"] - dq     # update absolute estimate

        if debug_info_level >= 3:
            peak_state["_dbg_img"] = Sm
            peak_state["_dbg_qmax"] = dq
            peak_state["_dbg_q_coords"] = q_coords  # must be used by parent plotter (see point 1)

        # refine shift-grid around 0
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
    @staticmethod
    def compute_spatial_shifts(illumination: PlaneWavesSIM, phase_matrix: np.ndarray, refined_wavevectors: np.ndarray) -> np.ndarray:
        """
        Compute the spatial shifts from the phase matrix.
        """
        raise NotImplementedError("This method is not implemented yet.")
        spatial_shifts = np.zeros((self.illumination.Mr, self.illumination.Mt, self.dimensionality), dtype=np.float64)
        for r in range(self.illumination.Mr):
            phase_array = np.zeros((self.illumination.Mt, self.dimensionality), dtype=np.float64)
            for n in range(self.illumination.Mt):
                for index in refined_wavevectors.keys():
                    m = index[1]
                    wavevector = refined_wavevectors[index]
                    phase = phase_matrix[(r, n, m)]
                    spatial_shifts[r, n] += np.angle(phase) / (2 * np.pi) * wavevector

        return spatial_shifts
        
    @staticmethod
    def phase_matrix_cross_correlation(optical_system: OpticalSystem, illumination: PlaneWavesSIM, stack: np.ndarray, refined_wavevectors: dict) -> np.ndarray: 
        """
        Estimate phase shifts as the phase of the cross-correlation function following ...? 
        """
        raise NotImplementedError("Full cross-correlation method is not implemented yet")
        grid = np.zeros((1, 1, self.optical_system.dimensionality), dtype=np.float64)
        phase_matrix = {}
        for r in range(self.illumination.Mr):
            wavevectors = {index: refined_wavevectors[index] for index in refined_wavevectors.keys() if index[0] == r}
            estimated_modulation_patterns = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(self.optical_system, wavevectors)
            Amn = PeaksEstimatorCrossCorrelation.autocorrelation_matrix(self.optical_system, self.illumination, stack, r, estimated_modulation_patterns, grid)
            for n in range(self.illumination.Mt):
                phase_matrix[(r, n, (0, 0))] = 1. + 0.j
                for m in wavevectors.keys():
                    if m[1] != tuple([0] * self.dimensionality):
                        if m[1][0] >= 0:
                            phase = np.angle(Amn[(m, n)])
                            index = (m, n)
                            max_index = np.unravel_index(np.argmax(np.abs(Amn[index])), Amn[index].shape)
                            phase_matrix[(r, n, m[1])] = np.exp(-1j * phase[0, 0])
                            phase_matrix[(r, n, tuple([mi for mi in m[1]]))] = np.exp(-1j * phase[0, 0])

        return phase_matrix
    
    @staticmethod
    def phase_matrix_autocorrelation(optical_system: OpticalSystem, illumination: PlaneWavesSIM, stack: np.ndarray, refined_wavevectors: dict) -> np.ndarray: 
        phase_matrix = {}
        for r in range(illumination.Mr):
            wavevectors = {index: refined_wavevectors[index] for index in refined_wavevectors.keys() if index[0] == r}
            phase_modulation_patterns = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(optical_system, wavevectors)
            Amn = PeaksEstimatorCrossCorrelation.autocorrelation_matrix(optical_system, illumination, stack, r, phase_modulation_patterns, optical_system.otf_frequencies)
            for n in range(illumination.Mt):
                phase_matrix[(r, n, (0, 0))] = 1. + 0.j
                for m in wavevectors.keys():
                    if m[1] != tuple([0] * optical_system.dimensionality):
                        if m[1][0] >= 0:
                            phase = np.angle(Amn[(m, n)])
                            index = (m, n)
                            phase_matrix[(r, n, m[1])] = np.exp(1j * phase[0, 0])
                            phase_matrix[(r, n, tuple([-mi for mi in m[1]]))] = np.exp(-1j * phase[0, 0])

        return phase_matrix

    @staticmethod
    def phase_matrix_peak_values(optical_system: OpticalSystem, illumination: PlaneWavesSIM, stack: np.ndarray, refined_wavevectors: np.ndarray) -> np.ndarray:
        """
        Estimate phase shifts as the phase of the Fourier transform of the image stack on the peaks positions. 
        It is working because if k = km, then f(k - km) = f(0), which is real and positive. The phase shift hence comes from 
        the phase of the modulation coefficient am. 

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        refined_wavevectors : np.ndarray
            The refined wavevectors.

        Returns
        -------
        phase_matrix : np.ndarray
            The phase matrix.
        """
        phase_matrix = {}
        grid = optical_system.x_grid
        for n in range(stack.shape[1]):
            for index in refined_wavevectors.keys():
                r = index[0]
                m = index[1]
                wavevector = refined_wavevectors[index] / (2 * np.pi)
                ft = off_grid_ft(stack[r, n], grid, np.array((wavevector, )))
                phase = np.angle(ft)
                phase_matrix[(r, n, m)] = np.exp(-1j * phase)

        return phase_matrix

