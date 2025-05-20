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
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import math 
from VectorOperations import VectorOperations

from wrappers import wrapped_fftn, wrapped_ifftn
from Dimensions import DimensionMetaAbstract
from abc import abstractmethod
from stattools import off_grid_ft

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
        phase_estimation_method='cross_correlation',
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
    
        peaks, rotation_angles = self.estimate_peaks(peaks_estimation_method, stack, peak_search_area_size, zooming_factor, max_iterations)
        
        if debug_info_level > 0:
            for r in range(Mr):
                print('r = ', r,  'rotation_angle = ', np.round(rotation_angles[r] / np.pi * 180, 1), 'degrees')
            for sim_index in peaks:
                print('r = ', sim_index[0], 'm = ', sim_index[1], 'wavevector = ', np.round((peaks[sim_index]), 3), '1 / lambda')

        phase_matrix = self.build_phase_matrix(phase_estimation_method, stack, peaks, rotation_angles)

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
        
        if self.dimensionality == self.illumination.dimensionality:
            peaks, rotation_angles = peaks_estimator.estimate_peaks(stack, 
                                                                peak_search_area_size, 
                                                                zooming_factor, 
                                                                max_iterations)
        else:
            projected_dimensions = tuple([i for i in range(len(self.illumination.dimensions)) if self.illumination.dimensions[i] == 0])
            sizes = tuple([stack.shape[i + 2] for i  in projected_dimensions])
            
            if debug_info_level > 1:
                print('Stack split into substacks. Averaging along the axes', projected_dimensions, 'of size', sizes)

            sum_peaks = {}
            for idx in np.ndindex(*sizes):
                slicing = (slice(None), slice(None)) + idx
                substack = stack[slicing]
                peaks, _ = peaks_estimator.estimate_peaks(substack, 
                                                          peak_search_area_size, 
                                                          zooming_factor, 
                                                          max_iterations)
                
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
                           rotation_angles: np.ndarray = None,
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
                
        if self.dimensionality == self.illumination.dimensionality:
            phase_matrix = phase_estimation_function(self.optical_system, self.illumination, stack, peaks)
        
        else:
            projected_dimensions = tuple([i for i in range(len(self.illumination.dimensions)) if self.illumination.dimensions[i] == 0])
            sizes = tuple([stack.shape[i + 2] for i  in projected_dimensions])
            
            sum_phases = {}
            if debug_info_level > 1:
                print('Stack split into substacks. Averaging along the axes', projected_dimensions, 'of size', sizes)
            for idx in np.ndindex(*sizes):
                slicing = (slice(None), slice(None)) + idx
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
        plt.imshow(np.log1p(ssnr_estimated), cmap='gray')
        plt.show()
        plt.plot(self.optical_system.otf_frequencies[0], ssnr_estimated[:, 50], label='ssnr_estimated')
        plt.ylim(0, 100)
        plt.plot(2.45, 0, 'cx')
        plt.show()
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
        if not self.illumination.dimensionality == self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: {self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )
    
    @abstractmethod
    def estimate_peaks(self, 
                       stack, 
                       peak_search_area_size, 
                       zooming_factor, 
                       max_iterations) -> dict:
        """
        Estimate the peaks of the illumination pattern from the stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.

        Returns
        -------
        dict
            The estimated peaks.
        """
        pass     

    

    def merit_function(self, Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray]) -> np.ndarray:
        """
        Compute the merit function for the base vector estimates"""

        return sum([C * C.conjugate() for C in Cmn1n2.values()])  # sum over m,n1,n2
    

    def get_stack_ft(self, stack: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier transform of the stack.
        """
        stack_ft = np.array([np.array([wrapped_fftn(image) for image in stack[r]]) for r in range(stack.shape[0])])
        return stack_ft
    
    def estimate_rotation_angles(self, estimated_peaks: dict) -> np.ndarray:
        """
        Estiamate rotation angles of the SIM pattern.
        """
        #We axpect N to be the same for all rotations, but it is easy to preserve generality
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
        """
        Refine the base vectors using the averaged maxima.
        """

        refined_base_vectors = np.zeros((self.illumination.Mr, self.dimensionality))
        base_vectors_sum = np.zeros((self.illumination.Mr, self.dimensionality), dtype=np.float64)
        index_sum  = np.zeros((self.illumination.Mr, self.optical_system.dimensionality), dtype=np.int16)
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
            wavevector_aligned = np.array([2 * np.pi * sim_index[1][dim] * base_vectors[dim] for dim in range(len(sim_index[1]))])
            refined_wavevectors[sim_index] = VectorOperations.rotate_vector2d(wavevector_aligned, angles[r])
        return refined_wavevectors
    

class PeaksEstimatorCrossCorrelation(PeaksEstimator): 
    def estimate_peaks(self, 
                        stack: np.ndarray,
                        peak_search_area_size: int,
                        zooming_factor: int,
                        max_iterations: int) -> dict:

        Mr = self.illumination.Mr 
        q_grid = self.optical_system.q_grid
        shape = self.optical_system.otf.shape
        slices = []
        for dim in range(len(shape)):
            center = shape[dim] // 2
            half_span = peak_search_area_size // 2
            start = center - half_span
            stop = center + half_span + 1
            slices.append(slice(start, stop))
        q_grid = q_grid[tuple(slices)]

        peaks = {}

        # Split is r is not required but saves memory 
        for r in range(Mr):
            grid = q_grid
            wavevectors, indices = self.illumination.get_wavevectors_projected(r)
            peaks_approximated = {}
            for index, wavevector in zip(indices, wavevectors):
                peaks_approximated[index] = wavevector
            i = 0
            while i < max_iterations:
                i += 1
                estimated_modualtion_patterns = self.phase_modulation_patterns(self.optical_system, peaks_approximated)
                correlation_matrix = self.cross_correlation_matrix(self.optical_system, self.illumination, stack, r, estimated_modualtion_patterns, grid)
                maxima = self._find_maxima(correlation_matrix, grid)
                averaged_maxima = self._average_maxima(maxima)
                peaks_approximated = self._refine_peaks(peaks_approximated, averaged_maxima)

                print(i, 'new', peaks_approximated)
                dq = grid[1, 1] - grid[0, 0]
                print('dq', dq)
                # if (np.sum((refined_base_vectors - base_vectors)**2) < 2 * dq**2).all():
                grid = self._fine_q_grid(grid, zooming_factor)
                # print("zooming in q_grid, new q_grid boundaries", q_grid[0, 0][0], q_grid[-1, -1][0])

                # merit_function = self._merit_function(correlation_matrix)
                # max_index = np.unravel_index(np.argmax(merit_function, axis=0), merit_function.shape)
                # diff = np.sum((refined_base_vectors - base_vectors)**2)


            peaks.update({key:peaks_approximated[key] / (2 * np.pi) for key in peaks_approximated.keys() if key[0] == r})

        rotation_angles = self.estimate_rotation_angles(peaks)
        refined_base_vectors = self.refine_base_vectors(peaks, rotation_angles)
        refined_wavevectors = self.refine_wavevectors(refined_base_vectors, rotation_angles)

        return refined_wavevectors, rotation_angles

    @staticmethod
    def phase_modulation_patterns(optical_system: OpticalSystem, illumination_vectors: dict[tuple(int, ...) : np.array((3), dtype=np.float64)]) -> np.ndarray:
        # Compute the phase modulation patterns for the estimated illumination vectors.
        phase_modulation_patterns = {}
        for index in illumination_vectors.keys():
            wavevector = np.copy(illumination_vectors[index])
            if optical_system.dimensionality == 2:
                phase_modulation_pattern = np.exp(1j * np.einsum('ijl,l ->ij', optical_system.x_grid, wavevector))
            if optical_system.dimensionality == 3:
                phase_modulation_pattern = np.exp(1j * np.einsum('ijk,l ->ij', optical_system.x_grid, wavevector))
            phase_modulation_patterns[index] = phase_modulation_pattern
        return phase_modulation_patterns
    
    @staticmethod
    def cross_correlation_matrix(optical_system: OpticalSystem,
                                illumination: PlaneWavesSIM,
                                images: np.ndarray, r: int,
                                estimated_modualtion_patterns: np.ndarray,
                                fine_q_grid: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Build the image-correlation matrix C^{m}_{n1 n2}(q).

        """
        Cmn1n2 = {}
        x_grid_flat = optical_system.x_grid.reshape(-1, optical_system.dimensionality)
        q_grid_flat = fine_q_grid.reshape(-1, optical_system.dimensionality)
        phase_matrix = q_grid_flat @ x_grid_flat.T
        fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
        for m in illumination.harmonics.keys():
            if np.isclose(np.sum(np.abs(np.array(m[1]))), 0):
                continue
            if m[0] != r:
                continue
            if m[1][0] >= 0:
                for n1 in range(illumination.spatial_shifts.shape[1]):
                    for n2 in range(illumination.spatial_shifts.shape[1]):
                        signal_function = (images[r, n1] * images[r, n2] * estimated_modualtion_patterns[m]).flatten()
                        Cmn1n2[(m, n1, n2)] = fourier_exponents @ signal_function
                        if n1 == n2:
                            noise_function = (images[r, n1] *  estimated_modualtion_patterns[m]).flatten()
                            Cmn1n2[(m, n1, n2)] -= fourier_exponents @ noise_function
                        Cmn1n2[(m, n1, n2)] = Cmn1n2[(m, n1, n2)].reshape(fine_q_grid.shape[:-1])
        return Cmn1n2
    
    @staticmethod
    def autocorrealtion_matrix(optical_system: OpticalSystem,
                                illumination: PlaneWavesSIM,
                                images: np.ndarray, r: int,
                                estimated_modualtion_patterns: np.ndarray,
                                fine_q_grid: np.ndarray) -> Dict[int, np.ndarray]:
        Amn = {}
        x_grid_flat = optical_system.x_grid.reshape(-1, optical_system.dimensionality)
        q_grid_flat = fine_q_grid.reshape(-1, optical_system.dimensionality)
        phase_matrix = q_grid_flat @ x_grid_flat.T
        fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
        for m in illumination.harmonics.keys():
            if np.isclose(np.sum(np.abs(np.array(m[1]))), 0):
                continue
            if m[0] != r:
                continue
            if m[1][0] >= 0:
                for n in range(illumination.spatial_shifts.shape[1]):
                    signal_function = (images[r, n] * images[r, n] * estimated_modualtion_patterns[m]).flatten()
                    noise_function = (images[r, n] *  estimated_modualtion_patterns[m]).flatten()
                    Amn[(m, n)] = fourier_exponents @ (signal_function - noise_function)
                    Amn[(m, n)] = Amn[(m, n)].reshape(fine_q_grid.shape[:-1])
        return Amn        


    def _fine_q_grid(self, q_grid, zooming_factor) -> np.ndarray:
        """
        Build a fine q-grid for the correlation matrix.
        """
        fine_q_coordinates = []
        for dim in range (len(q_grid.shape)-1):
            q_max, q_min = q_grid[..., dim].max(), q_grid[..., dim].min()
            fine_q_coordinates.append(np.linspace(q_min/ zooming_factor, q_max / zooming_factor, q_grid.shape[dim]))
        fine_q_grid = np.stack(np.meshgrid(*tuple(fine_q_coordinates), indexing='ij'), axis=-1)
        return fine_q_grid
    

    def _find_maxima(self, Cmat: Dict[int, np.ndarray], grid: np.ndarray) -> dict:
        """
        Find the maxima in the correlation matrix.
        """
        maxima = {}
        for index in Cmat.keys():
            max_index = np.unravel_index(np.argmax(np.abs(Cmat[index])), Cmat[index].shape)
            # print(index, round(np.abs(Cmat[index][max_index])), round(np.angle(Cmat[index][max_index])* 180/np.pi, 1), grid[max_index])
            q = grid[max_index]
            maxima[index] = q
        return maxima
    
    def _average_maxima(self, maxima: dict[int, np.ndarray]) -> dict:
        """
        Average the maxima to get a new guess for the peaks. 
        """

        averaged_maxima = {}
        for index in maxima.keys():
            m, n1, n2 = index
            if m in averaged_maxima.keys():
                averaged_maxima[m] += maxima[index]
            else:
                averaged_maxima[m] = np.copy(maxima[index])
        for m in averaged_maxima.keys():
            averaged_maxima[m] /= self.illumination.Mt ** 2
        return averaged_maxima

    def _refine_peaks(self, peaks, averaged_maxima: dict) -> dict:
        refined_peaks = {}
        for index in averaged_maxima.keys():
            refined_peaks[index] = peaks[index] - 2 * np.pi * averaged_maxima[index]

        return refined_peaks
    

class PeaksEstimatorCrossCorrelation2D(PeaksEstimatorCrossCorrelation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)

class PeaksEstimatorCrossCorrelation3D(PeaksEstimatorCrossCorrelation):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves3D, optical_system: OpticalSystem3D):
        super().__init__(illumination, optical_system)
 
class PeaksEstimatorInterpolation(PeaksEstimator):
    def estimate_peaks(self,
                        stack: np.ndarray,
                        peak_search_area_size: int = 31,
                        zooming_factor: int = 3,
                        max_iterations: int = 3,
                        debug_info_level: int = 0,
                        ) -> np.ndarray:

        if len(stack.shape[2:]) != self.dimensionality:
            if len(self.illumination.dimensions) != len(stack.shape[2:]):
                raise ValueError(f"Stack dimensions {stack.shape[2:]} do not match illumination dimensions {self.illumination.dimensions}")
            else: 
                ...
        wavevectors, indices = self.illumination.get_all_wavevectors_projected()
        peak_guesses = {index: wavevector / (2 * np.pi) for index, wavevector in zip(indices, wavevectors)}
        peak_search_areas  = self._locate_peaks(peak_guesses, peak_search_area_size)
        stack_ft = self.get_stack_ft(stack)
        stacks_ft = self._crop_stacks(stack_ft, peak_search_areas)
        grids = self._crop_grids(peak_search_areas)
        stacks_ft_averaged = self._average_ft_stacks(stacks_ft)
        averaged_maxima = self._find_maxima(stacks_ft_averaged, grids)
        for i in range(1, max_iterations+1):
            coarse_peak_grids = self._get_coarse_grids(stacks_ft_averaged, grids, zooming_factor)
            interpolated_grids = self._get_fine_grids(coarse_peak_grids, zooming_factor)
            off_grid_ft_stacks, off_grid_otfs = self._off_grid_ft_stacks(stack, interpolated_grids)
            off_grid_ft_averaged = self._average_ft_stacks(off_grid_ft_stacks)
            # plt.imshow(np.log1p(off_grid_ft_averaged[(2, 0)]), cmap='gray')
            # plt.show()
            # plt.imshow(np.log1p(off_grid_otfs[(2, 0)]), cmap='gray')
            # plt.show()
            grids = interpolated_grids
            # plt.imshow(np.log1p(off_grid_ft_averaged[(2, 0)]), cmap='gray')
            # plt.show()
            stacks_ft_averaged = off_grid_ft_averaged
            averaged_maxima = self._find_maxima(off_grid_ft_averaged, grids)
            print('iteration', i, 'averaged_maxima', averaged_maxima)

        if debug_info_level > 0:
            print('before correction', averaged_maxima[(0, (2, 0))] / 2)

        averaged_maxima=self._correct_peak_position(averaged_maxima, stack)

        if debug_info_level > 0:
            print('corrected maxima', averaged_maxima)

        rotation_angles = self.estimate_rotation_angles(averaged_maxima)
        refined_base_vectors = self.refine_base_vectors(averaged_maxima, rotation_angles)
        refined_wavevectors = self.refine_wavevectors(refined_base_vectors, rotation_angles)

        return refined_wavevectors, rotation_angles
    
    def _locate_peaks(self, approximate_peaks: dict[tuple[int], float], peak_search_area: int) -> np.ndarray:
        """
        Label the area around the peaks in the Fourier space.
        """
        peak_search_areas = {}
        for peak in approximate_peaks.keys():
            if np.isclose(np.sum(np.abs(np.array(peak[1]))), 0):
                continue
            if peak[1][0] >= 0:
                grid = self.optical_system.q_grid
                dq = grid[*([1]*self.dimensionality)] - grid[*([0]*self.dimensionality)]
                approximate_peak = approximate_peaks[peak]
                labeling_array = (np.abs((grid - approximate_peak[None, None, :])) <= peak_search_area * dq[None, None, :]).all(axis=-1)
                peak_search_areas[peak] = labeling_array
                # plt.imshow(labeling_array, cmap='gray')
                # plt.show()
        return peak_search_areas
    
    def _crop_stacks(self, stack_ft: np.ndarray, peak_search_areas: np.ndarray) -> np.ndarray:
        """
        Crop the stacks to get small regions around the peaks.
        """
        cropped_stacks = {}
        for peak in peak_search_areas.keys():
            r = peak[0]
            mask = peak_search_areas[peak]
            coords = np.argwhere(mask)
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0) + 1  # slice end is exclusive
            cropped_stacks[peak] = np.array([image_ft[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] for image_ft in stack_ft[r]])
            # plt.imshow((np.abs(stack_ft[r, 0] * mask)).T, cmap='gray', origin='lower')
            # qx, qy = self.optical_system.otf_frequencies
            # approximate_peak = self.illumination.harmonics[peak].wavevector / (2 * np.pi)
            # plt.plot(approximate_peak[0]/qx[-1] * 128 + 128, approximate_peak[1]/qx[-1] * 128 + 128, 'rx')
            # plt.show()
        return cropped_stacks
    
    def _crop_grids(self, peak_search_areas: np.ndarray) -> np.ndarray:
        """
        Crop the global grid to get small regions around the peaks.
        """
        cropped_grids = {}
        for peak in peak_search_areas.keys():
            grid = self.optical_system.q_grid
            mask = peak_search_areas[peak]
            coords = np.argwhere(mask)
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0) + 1  # slice end is exclusive
            grid = grid[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], ...]
            cropped_grids[peak] = grid
        return cropped_grids
    
    def _crop_otf(self, peak_search_areas: np.ndarray) -> np.ndarray:
        """
        Crop the OTF to get small regions around the peaks.
        """
        cropped_otfs = {}
        for peak in peak_search_areas.keys():
            otf = self.optical_system.otf
            otf *= peak_search_areas[peak][...]
            cropped_otfs[peak] = np.abs(otf)
        return cropped_otfs
    

    def _get_coarse_grids(self, stack_ft_averages: np.ndarray, grids: dict[np.ndarray], peak_interpolation_area_size: int) -> np.ndarray:
        """
        Coarse grid around the peaks. 
        """
        coarse_grids = {}
        for peak in grids.keys():
            stack = stack_ft_averages[peak]
            grid = grids[peak]
            max_index = np.unravel_index(np.argmax(stack), stack.shape)
            print('max_index', max_index)
            refined_peak = grid[max_index]
            print('refined peak', refined_peak)
            print('peak_value', stack[max_index])

            slices = tuple([slice(max_index[dim] - peak_interpolation_area_size // 2, max_index[dim] + peak_interpolation_area_size // 2 + 1) for dim in range(len(max_index))])
            coarse_grids[peak] = grid[slices + (slice(None),)]
            dq = grid[*([1] * self.dimensionality)] - grid[*([0]*self.dimensionality)]
            coarse_coords = tuple([np.linspace(refined_peak[dim] - peak_interpolation_area_size//2 * dq[dim],  
                                                refined_peak[dim] + (peak_interpolation_area_size + 1)//2 * dq[dim],
                                                peak_interpolation_area_size) for dim in range(self.dimensionality)])
            coarse_grid = np.stack(np.meshgrid(*coarse_coords), -1)
            coarse_grids[peak] = coarse_grid
            # plt.imshow(np.abs(stack), cmap='gray')
            # plt.plot(max_index[1], max_index[0], 'rx')
            # plt.show()
        return coarse_grids

    def _get_fine_grids(self, peak_interpolation_areas, interpolation_factor: int) -> np.ndarray:
        """
        Interpolate coarse grids to get finer grids aroun the peaks. 
        """
        interpolated_grids = {}
        for peak in peak_interpolation_areas.keys():
            fine_grid = self._get_fine_grid(peak_interpolation_areas[peak], interpolation_factor)
            interpolated_grids[peak] = fine_grid
        return interpolated_grids
    
    def _get_fine_grid(self, coarse_grid: np.ndarray, interpolation_factor: int) -> np.ndarray:
        fine_q_coordinates = []
        for dim in range (len(coarse_grid.shape)-1):
            q_max, q_min = coarse_grid[..., dim].max(), coarse_grid[..., dim].min()
            fine_q_coordinates.append(np.linspace(q_min, q_max, (coarse_grid.shape[dim] - 1) * interpolation_factor + 1))
        fine_q_grid = np.stack(np.meshgrid(*tuple(fine_q_coordinates), indexing='ij'), axis=-1)
        return fine_q_grid
    
    def _off_grid_ft_stacks(self, stack: np.ndarray, fine_q_grids: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier transform of the stacks on the fine q-grids to interpolate them arount the peaks. 
        Nyquist and Dirichlet theorems ensure that this is the only correct way to do it.
        """
        x_grid_flat = self.optical_system.x_grid.reshape(-1, self.optical_system.dimensionality)
        images_ft_dict = {}
        otfs_dict = {}
        for index in fine_q_grids.keys():
            r = index
            fine_q_grid = fine_q_grids[index]
            q_grid_flat = fine_q_grid.reshape(-1, self.optical_system.dimensionality)
            phase_matrix = q_grid_flat @ x_grid_flat.T
            fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
            new_stack = []
            for image in stack[r]:
                image = image.flatten()
                image_ft = fourier_exponents @ image
                new_stack.append(image_ft.reshape(fine_q_grid.shape[:-1]))
            images_ft_dict[index] = np.array(new_stack)
                
            psf = self.optical_system.psf.flatten()
            otf = fourier_exponents @ psf
            otfs_dict[index] = np.abs(otf.reshape(fine_q_grid.shape[:-1]))
            # plt.imshow(np.abs(otfs_dict[index]), cmap='gray')
            # plt.show()
            # plt.imshow(np.abs(otfs_dict[index]), cmap='gray')
            # plt.show()
        return images_ft_dict, otfs_dict
    
    def _average_ft_stacks(self, stacks_ft_dict: dict[np.ndarray]) -> np.ndarray:
        """
        Average the absolute values of Fourier transforms of the raw images for more precise peak estimation.
        """
        averaged_images_ft_dict = {}
        for index in stacks_ft_dict.keys():
            stack = stacks_ft_dict[index]
            averaged_images_ft_dict[index] = np.mean(np.abs(stack), axis=0)
        return averaged_images_ft_dict

    def _find_maxima(self, image_dict: dict[np.ndarray], fine_q_grids: dict[np.ndarray]) -> np.ndarray:
        """
        Find the maxima (i. e., peak approximations) in the image stacks.
        """
        max_dict = {}
        for index in image_dict.keys():
            grid = fine_q_grids[index]
            image = image_dict[index]
            # plt.imshow(image, cmap='gray')
            # plt.show()
            max_index = np.unravel_index(np.argmax(image), image.shape)
            max_dict[index] = grid[max_index]
        return max_dict
    
    @abstractmethod
    def _correct_peak_position(self, estimated_peaks: dict[np.ndarray[3, np.float64]], stack: np.ndarray, dk: float = 10**-4) -> np.ndarray:
        """
        Account for the non-uniform OTF to correct the peaks position assuming it is not far from actual illumination wavevectros. 
        """

class PeaksEstimatorInterpolation2D(PeaksEstimatorInterpolation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)

    def _correct_peak_position(self, estimated_peaks: dict[np.ndarray[2, np.float64]], stack: np.ndarray, dk: float = 10**-4) -> np.ndarray:
        """
        Account for the non-uniform OTF to correct the peaks position assuming it is not far from actual illumination wavevectros. 
        """
        otf_dict = {}
        otf_grad_dict = {}
        grid = self.optical_system.x_grid
        for key in estimated_peaks.keys():
            peak = estimated_peaks[key]
            psf = self.optical_system.psf
            q_values = np.array([peak,
                                 peak - np.array((dk, 0)),
                                 peak + np.array((dk, 0)),
                                 peak - np.array((0, dk)),
                                 peak + np.array((0, dk))])

            otf_values = off_grid_ft(psf, grid, q_values)
            otf_values = np.abs(otf_values)
            otf_dict[key] = otf_values[0]

            otf_grad = np.array([otf_values[2] - otf_values[1], otf_values[4] - otf_values[3]]) / (2 * dk)
            otf_grad_dict[key] = otf_grad

        q_grid_dict = {(0, (0, 0)): np.array([np.array((0, 0)),
                                         np.array((-dk, 0)), np.array((dk, 0)),
                                         np.array((0, -dk)), np.array((0, dk)),
                                         np.array((-dk, -dk)), np.array((-dk, dk)),
                                         np.array((dk, -dk)), np.array((dk, dk))])}
        
        I_values, otf_values = self._off_grid_ft_stacks(stack, q_grid_dict)
        I_values = self._average_ft_stacks(I_values)
        f_values = I_values[(0,(0, 0))] / np.abs(otf_values[(0, (0, 0))])
        f_grad = np.array([f_values[2] - f_values[1], f_values[4] - f_values[3]]) / (2 * dk)

        f_hessian = np.array((
                             np.array([f_values[2] - 2 * f_values[0] + f_values[2],
                                       (f_values[8] - f_values[7] - f_values[6] + f_values[5])/2]),
                             np.array([(f_values[8] - f_values[7] - f_values[6] + f_values[5])/2, 
                                      f_values[4] - 2 * f_values[0] + f_values[3]]),        
                             )) / (dk**2)
        
        corrected_peaks = {}
        for key in otf_grad_dict.keys():
            otf = otf_dict[key]
            otf_grad = otf_grad_dict[key]
            delta_k = f_values[0] / otf * (1  + f_values[0]/(2 * otf **2) * otf_grad.T @ np.linalg.inv(f_hessian) @ otf_grad) * np.linalg.inv(f_hessian) @ otf_grad 
            print('correction ', delta_k)
            corrected_peaks[key] = estimated_peaks[key] + delta_k

        return corrected_peaks


class PeaksEstimatorInterpolation3D(PeaksEstimatorInterpolation):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves3D, optical_system: OpticalSystem3D):
        super().__init__(illumination, optical_system)


    def _correct_peak_position(self, estimated_peaks: dict[np.ndarray[2, np.float64]], stack: np.ndarray, dk: float = 10**-4) -> np.ndarray:
        """
        Account for the non-uniform OTF to correct the peaks position assuming it is not far from actual illumination wavevectros. 
        """
        otf_dict = {}
        otf_grad_dict = {}
        grid = self.optical_system.x_grid
        for key in estimated_peaks.keys():
            peak = estimated_peaks[key]
            psf = self.optical_system.psf
            q_values = np.array([peak, peak - np.array((dk, 0, 0)), peak + np.array((dk, 0, 0)), peak - np.array((0, dk, 0)), peak + np.array((0, dk, 0)),
                                 peak - np.array((0, 0, dk)), peak + np.array((0, 0, dk))])

            otf_values = off_grid_ft(psf, grid, q_values)
            otf_values = np.abs(otf_values)
            otf_dict[key] = otf_values[0]

            otf_grad = np.array([otf_values[2] - otf_values[1], otf_values[4] - otf_values[3]], otf[6] - otf[5]) / (2 * dk)
            otf_grad_dict[key] = otf_grad

        q_grid_dict = {(0, (0, 0)): np.array([np.array((0, 0)), #0
                                         np.array((-dk, 0, 0)), np.array((dk, 0, 0)), #1, 2
                                         np.array((0, -dk, 0)), np.array((0, dk, 0)), #3, 4
                                         np.array((0, 0, -dk)), np.array((0, 0, dk)), #5, 6
                                         np.array((-dk, -dk, 0)), np.array((-dk, dk, 0)), #7, 8
                                         np.array((dk, -dk, 0)), np.array((dk, dk, 0)), #9, 10
                                         np.array((0, -dk, -dk)), np.array((0, -dk, dk)), #11, 12
                                         np.array((0, dk, -dk)), np.array((0, dk, dk)), #13, 14
                                         np.array((-dk, 0, -dk)), np.array((dk, dk, 0)), #15, 16
                                         np.array((-dk, 0, dk)), np.array((-dk, 0, dk))])} #17, 18
        
        I_values, otf_values = self._off_grid_ft_stacks(stack, q_grid_dict)
        I_values = self._average_ft_stacks(I_values)
        f_values = I_values[(0,(0, 0))] / np.abs(otf_values[(0, (0, 0))])
        f_grad = np.array([f_values[2] - f_values[1], f_values[4] - f_values[3]], f_values[6] - f_values[5]) / (2 * dk)

        f_hessian = np.array((
                             np.array([f_values[2] - 2 * f_values[0] + f_values[2],                   #fxx
                                       (f_values[7] - f_values[8] - f_values[9] + f_values[9])/2,     #fxy
                                        (f_values[15] - f_values[16] - f_values[17] + f_values[18])/2 #fxz
                                       ]),
                             np.array([(f_values[7] - f_values[8] - f_values[9] + f_values[9])/2,    #fxy
                                        f_values[4] - 2 * f_values[0] + f_values[3],                  #fyy
                                        (f_values[11] - f_values[12] - f_values[13] + f_values[14])/2 #fyz
                                        ]),
                            np.array([(f_values[15] - f_values[16] - f_values[17] + f_values[18])/2,  #fxz
                                      (f_values[11] - f_values[12] - f_values[13] + f_values[14])/2,  #fyz
                                      f_values[5] - 2 * f_values[0] + f_values[6]/2                  #fzz
                                      ]),
                             )) / (dk**2)
        
        corrected_peaks = {}
        for key in otf_grad_dict.keys():
            otf = otf_dict[key]
            otf_grad = otf_grad_dict[key]
            delta_k = f_values[0] / otf * (1  + f_values[0]/(2 * otf **2) * otf_grad.T @ np.linalg.inv(f_hessian) @ otf_grad) * np.linalg.inv(f_hessian) @ otf_grad 
            print('correction ', delta_k)
            corrected_peaks[key] = estimated_peaks[key] + delta_k

        return corrected_peaks
    
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
            estimated_modualtion_patterns = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(self.optical_system, wavevectors)
            Amn = PeaksEstimatorCrossCorrelation.autocorrealtion_matrix(self.optical_system, self.illumination, stack, r, estimated_modualtion_patterns, grid)
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
        grid = np.zeros((1, 1, optical_system.dimensionality), dtype=np.float64)
        phase_matrix = {}
        for r in range(illumination.Mr):
            wavevectors = {index: refined_wavevectors[index] for index in refined_wavevectors.keys() if index[0] == r}
            estimated_modualtion_patterns = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(optical_system, wavevectors)
            Amn = PeaksEstimatorCrossCorrelation.autocorrealtion_matrix(optical_system, illumination, stack, r, estimated_modualtion_patterns, grid)
            for n in range(illumination.Mt):
                phase_matrix[(r, n, (0, 0))] = 1. + 0.j
                for m in wavevectors.keys():
                    if m[1] != tuple([0] * optical_system.dimensionality):
                        if m[1][0] >= 0:
                            phase = np.angle(Amn[(m, n)])
                            index = (m, n)
                            phase_matrix[(r, n, m[1])] = np.exp(-1j * phase[0, 0])
                            phase_matrix[(r, n, tuple([-mi for mi in m[1]]))] = np.exp(1j * phase[0, 0])

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

