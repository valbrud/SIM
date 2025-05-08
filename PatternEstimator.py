from __future__ import annotations
"""illumination_pattern_estimator.py

Data‑driven estimation of the illumination parameters (phase offsets, k‑vectors
and modulation depths *aₘ*) directly from a raw SIM stack, following the
cross‑correlation algorithm of Gustafsson (2000).

The module provides two concrete estimators that fit naturally into the
existing code base::

    estimator = IlluminationPatternEstimator2D(illumination, optical_system)
    new_illum = estimator.estimate_illumination_parameters(raw_stack)

`new_illum` is a **clone** of the original `IlluminationPlaneWaves2D` object
where:
    * every intensity harmonic has its amplitude replaced by the fitted *aₘ*
    * the internal phase matrix is rebuilt from the measured phase offsets

The same applies in 3‑D with `IlluminationPatternEstimator3D`.
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

class IlluminationPatternEstimator(metaclass=DimensionMetaAbstract):
    """Non‑instantiable base holding common logic."""

    dimensionality = None  # metaclass blocks direct instantiation

    # ---------------------------------------------------------------------
    # constructor
    # ---------------------------------------------------------------------
    def __init__(self, illumination: PlaneWavesSIM, optical_system: OpticalSystem):
        self.illumination = illumination
        self.optical_system = optical_system
        if not self.dimensionality == sum(illumination.dimensions):
            raise ValueError(
                f"Illumination dimensions {illumination.dimensions} do not match the dimensionality {self.dimensionality}."
            )
    @abstractmethod
    def estimate_illumination_parameters(self,
                                         stack,
                                         estimate_modulation_coefficients: bool = True,
                                        **kwargs): 
        pass
    
    def _get_stack_ft(self, stack: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier transform of the stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.

        Returns
        -------
        np.ndarray
            The Fourier transform of the stack.
        """
        stack_ft = np.array([np.array([wrapped_fftn(image) for image in stack[r]]) for r in range(stack.shape[0])])
        return stack_ft
    
    def _estimate_rotation_angles(self, estimated_peaks: dict) -> np.ndarray:
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
    
    def _refine_base_vectors(self, averaged_maxima: dict, angles: np.ndarray) -> np.ndarray:
        """
        Refine the base vectors using the averaged maxima.

        Parameters
        ----------
        averaged_maxima : dict
            The averaged maxima.

        Returns
        -------
        np.ndarray
            The refined base vectors.
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
    
    def _refine_wavevectors(self, refined_base_vectors: np.ndarray, angles) -> dict[tuple[int, ...], np.ndarray]:
        refined_wavevectors = {}
        for sim_index in self.illumination.rearranged_indices.keys():  
            r = sim_index[0]
            base_vectors = refined_base_vectors[r]
            wavevector_aligned = np.array([2 * np.pi * sim_index[1][dim] * base_vectors[dim] for dim in range(len(sim_index[1]))])
            refined_wavevectors[sim_index] = VectorOperations.rotate_vector2d(wavevector_aligned, angles[r])
        return refined_wavevectors

    def _compose_harmonics_dict(self, refined_wavevectors: dict) -> Dict[int, IntensityHarmonic]:
        """
        Compose the harmonics dictionary from the refined wavevectors and base vectors.

        Parameters
        ----------
        refined_wavevectors : dict
            The refined wavevectors.

        Returns
        -------
        dict
            The harmonics dictionary.
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
    
class PatternEstimatorCrossCorrelation(IlluminationPatternEstimator):
    """Cross‑correlation estimator for 2D and 3D SIM."""

    def __init__(self, illumination: PlaneWavesSIM, optical_system: OpticalSystem):
        super().__init__(illumination, optical_system)

    def estimate_illumination_parameters(
        self,
        stack: np.ndarray,
        estimate_modulation_coefficients: bool = True,
        method_for_modulation_coefficients='least_squares',
        peak_neighborhood_size: int = 11,
        zooming_factor: int = 3, 
        max_iterations: int = 20,
        tolerance: float = 1e-3,
        effective_otfs = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, complex]], PlaneWavesSIM]:
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

        if effective_otfs is None:
            _, effective_otfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        
        self.effective_otfs = effective_otfs

        q_grid = self.optical_system.q_grid
        shape = self.optical_system.otf.shape
        slices = []
        for dim in range(len(shape)):
            center = shape[dim] // 2
            half_span = peak_neighborhood_size // 2
            start = center - half_span
            stop = center + half_span + 1
            slices.append(slice(start, stop))
        q_grid = q_grid[tuple(slices)]

        peaks = {}
        # Split is r is not needed but saves memory 
        for r in range(Mr):
            grid = q_grid
            wavevectors, indices = self.illumination.get_wavevectors_projected(r)
            peaks_approximated = {}
            for index, wavevector in zip(indices, wavevectors):
                peaks_approximated[index] = wavevector
            # merit_function = self._merit_function(correlation_matrix)
            # max_index = np.unravel_index(np.argmax(merit_function, axis=0), merit_function.shape)
            # merit_function_new = merit_function[max_index]
            # diff = np.sum((refined_base_vectors - base_vectors)**2)**0.5
            # print(0, 'new', refined_base_vectors, 'initial guess', base_vectors,'diff', diff)
            i = 0
            while i < max_iterations:
                i += 1
                estimated_modualtion_patterns = self._phase_modulation_patterns(peaks_approximated)
                correlation_matrix = self._cross_correlation_matrix(stack, r, estimated_modualtion_patterns, grid)
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

        rotation_angles = self._estimate_rotation_angles(peaks)
        refined_base_vectors = self._refine_base_vectors(peaks, rotation_angles)
        refined_wavevectors = self._refine_wavevectors(refined_base_vectors, rotation_angles)
        print(refined_wavevectors, rotation_angles)
        estimated_modualtion_patterns = self._phase_modulation_patterns(refined_wavevectors)
        correlation_matrix = self._cross_correlation_matrix(stack, 0, estimated_modualtion_patterns, grid)
        # for n in range(Mt):
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(np.abs(correlation_matrix[((0, (2, 0)), n, n)]), cmap='gray')
            # axes[1].imshow(np.angle(correlation_matrix[((0, (2, 0)), n, n)]), cmap='gray')
            # plt.show()
        phase_matrix = self._compute_phase_matrix(stack, refined_wavevectors)
        
        new_illum_class = type(self.illumination)
        harmonics_dict = self._compose_harmonics_dict(refined_wavevectors)
        illumination_estimated = new_illum_class(
                 intensity_harmonics_dict = harmonics_dict,
                 dimensions= self.illumination.dimensions,
                 Mr=self.illumination.Mr,
                 spatial_shifts=None,
                 angles=rotation_angles,
                 )
        illumination_estimated.phase_matrix = phase_matrix
        if estimate_modulation_coefficients:
            illumination_estimated.estimate_modulation_coefficients(stack, self.optical_system.psf, self.optical_system.x_grid, 
                                                                    method = method_for_modulation_coefficients, update=True) 


        return illumination_estimated


    def _fine_q_grid(self, q_grid, zooming_factor) -> np.ndarray:
        """
        Build a fine q-grid for the correlation matrix.

        Parameters
        ----------
        q_grid : ndarray
            The coarse q-grid.
        zooming_factor : int
            The zooming factor to create the fine q-grid.

        Returns
        -------
        ndarray
            The fine q-grid.
        """
        fine_q_coordinates = []
        for dim in range (len(q_grid.shape)-1):
            q_max, q_min = q_grid[..., dim].max(), q_grid[..., dim].min()
            fine_q_coordinates.append(np.linspace(q_min/ zooming_factor, q_max / zooming_factor, q_grid.shape[dim]))
        fine_q_grid = np.stack(np.meshgrid(*tuple(fine_q_coordinates), indexing='ij'), axis=-1)
        return fine_q_grid
    
    def _phase_modulation_patterns(self, shift_vectors: dict[tuple(int, ...) : np.array((3), dtype=np.float64)]) -> np.ndarray:
        # Assuming mentally sane 2D SIM  for now
        phase_modulation_patterns = {}
        for index in shift_vectors.keys():
            wavevector = np.copy(shift_vectors[index])
            phase_modulation_pattern = np.exp(1j * np.einsum('ijl,l ->ij', self.optical_system.x_grid, wavevector))
            phase_modulation_patterns[index] = phase_modulation_pattern
        return phase_modulation_patterns
    
    def _cross_correlation_matrix(self, images: np.ndarray, r: int, estimated_modualtion_patterns: np.ndarray, fine_q_grid: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Build the image-correlation matrix C^{m}_{n1 n2}(q).

        """
        Cmn1n2 = {}
        x_grid_flat = self.optical_system.x_grid.reshape(-1, self.optical_system.dimensionality)
        q_grid_flat = fine_q_grid.reshape(-1, self.optical_system.dimensionality)
        phase_matrix = q_grid_flat @ x_grid_flat.T
        fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
        for m in self.illumination.harmonics.keys():
            if np.isclose(np.sum(np.abs(np.array(m[1]))), 0):
                continue
            if m[0] != r:
                continue
            if m[1][0] >= 0:
                for n1 in range(self.illumination.spatial_shifts.shape[1]):
                    for n2 in range(self.illumination.spatial_shifts.shape[1]):
                        # plt.imshow(estimated_modualtion_patterns[m].real, cmap='gray')
                        # plt.show()
                        signal_function = (images[r, n1] * images[r, n2] * estimated_modualtion_patterns[m]).flatten()
                        Cmn1n2[(m, n1, n2)] = fourier_exponents @ signal_function
                        if n1 == n2:
                            noise_function = (images[r, n1] *  estimated_modualtion_patterns[m]).flatten()
                            Cmn1n2[(m, n1, n2)] -= fourier_exponents @ noise_function
                        Cmn1n2[(m, n1, n2)] = Cmn1n2[(m, n1, n2)].reshape(fine_q_grid.shape[:-1])
                        # if n1== n2:
                            # plt.imshow(np.abs(Cmn1n2[(m, n1, n2)]), cmap='gray',
                            #             extent=(np.amin(fine_q_grid[..., 0]), np.amax(fine_q_grid[..., 0]), np.amin(fine_q_grid[..., 1]), np.amax(fine_q_grid[..., 1])))
                            # plt.title(f"m: {m}, n1: {n1}, n2: {n2}")
                            # plt.show()
        return Cmn1n2
    
    def _find_maxima(self, Cmat: Dict[int, np.ndarray], grid: np.ndarray) -> dict:
        """
        Find the maxima in the correlation matrix.

        Parameters
        ----------
        Cmat : dict[int, np.ndarray]
            The correlation matrix.

        Returns
        -------
        dict
            The maxima in the correlation matrix.
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

        Parameters
        ----------
        maxima : dict[int, np.ndarray]
            The maxima in the correlation matrix.

        Returns
        -------
        dict
            The averaged maxima.
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
    
    # def _refine_base_vectors(self, base_vectors: np.ndarray, r: int, Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray], fine_q_grid) -> np.ndarray:
    #     """
    #     Refine the base vectors using the correlation matrix Cmn1n2.

    #     Parameters
    #     ----------
    #     Cmn1n2 : dict[tuple(int, ...), int, int], np.ndarray]
    #         The correlation matrix.

    #     Returns
    #     -------
    #     np.ndarray
    #         The refined base vectors.
    #     """
    #     nominator = np.zeros((self.optical_system.dimensionality), dtype=np.float64)
    #     denominator = np.zeros((self.optical_system.dimensionality), dtype=np.float64)
    #     qs = {}
    #     weights = {}
    #     otf_shape = np.copy(np.array(self.optical_system.otf.shape))
    #     for index in Cmn1n2.keys():
    #         max_index = np.unravel_index(np.argmax(np.abs(Cmn1n2[index])), Cmn1n2[index].shape)
    #         q = fine_q_grid[max_index]
    #         # print(index, max_index, q)
    #         sim_index = index[0]
    #         if not sim_index in weights.keys():
    #             effective_otf = self.effective_otfs[(r, sim_index)]
    #             weights[sim_index] = np.abs(effective_otf[*otf_shape//2])
    #         qs[index] = q

    #     for index in qs.keys():
    #         component_weights = weights[sim_index] * np.abs(np.array(sim_index))
    #         dk_estimate = qs[index] / np.where(sim_index, sim_index, np.inf)
    #         # print(dk_estimate)
    #         nominator += dk_estimate * component_weights
    #         denominator += component_weights
    #     # print('weighted shigt', nominator / np.where(denominator, denominator, np.inf) )
    #     refined_base_vectors = base_vectors - nominator / np.where(denominator, denominator, np.inf)
    #     return refined_base_vectors
    
    
    def _compute_phase_matrix(self, stack: np.ndarray, refined_wavevectors: dict) -> np.ndarray: 
        grid = np.zeros((1, 1, self.optical_system.dimensionality), dtype=np.float64)
        phase_matrix = {}
        for r in range(self.illumination.Mr):
            wavevectors = {index: refined_wavevectors[index] for index in refined_wavevectors.keys() if index[0] == r}
            estimated_modualtion_patterns = self._phase_modulation_patterns(wavevectors)
            correlation_matrix = self._cross_correlation_matrix(stack, r, estimated_modualtion_patterns, grid)
            Cmat = correlation_matrix
            for n in range(self.illumination.Mt):
                phase_matrix[(r, n, (0, 0))] = 1. + 0.j
                for m in wavevectors.keys():
                    if m[1] != tuple([0] * self.dimensionality):
                        if m[1][0] >= 0:
                            phase = np.angle(correlation_matrix[(m, n, n)])
                            index = (m, n, n)
                            max_index = np.unravel_index(np.argmax(np.abs(Cmat[index])), Cmat[index].shape)
                            print(index, round(np.abs(Cmat[index][max_index])), round(np.angle(Cmat[index][max_index])* 180/np.pi, 1))
                            phase_matrix[(r, n, m[1])] = np.exp(1j * phase[0, 0])
                            phase_matrix[(r, n, tuple([-mi for mi in m[1]]))] = np.exp(-1j * phase[0, 0])

        return phase_matrix
         


    def _merit_function(self, Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray]) -> np.ndarray:
        """
        Compute the merit function for the base vector estimates"""

        return sum([C * C.conjugate() for C in Cmn1n2.values()])  # sum over m,n1,n2
    


from SSNRCalculator import SSNRBase
class PatternEstimatorInterpolation(IlluminationPatternEstimator):
    def estimate_illumination_parameters(self,
                                     stack: np.ndarray,
                                     estimate_modulation_coefficients: bool = True,
                                     method_for_modulation_coefficients='peak_height_ratio',
                                     interpolation_factor: int = 100,  
                                     peak_search_area_size: int = 31,
                                     peak_interpolation_area_size: int = 5,
                                     iteration_number: int = 3,
                                     ssnr_estimation_iters: int = 10, 
                                     deconvolve_stacks = True,
                                     correct_peak_position = False) -> np.ndarray:
        """
        Estimate the illumination patterns using the SSNR method.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        peak_neighborhood_size : int, optional
            The size of the neighborhood around the peak to consider

        Returns
        -------
        np.ndarray
            The estimated illumination patterns.
        """

        if deconvolve_stacks and correct_peak_position:
            raise ValueError("Deconvolution and peak position are mutually exclusive ways to increase peak precision." \
                             "Choose one of them to be False.")
        
        wavevectors, indices = self.illumination.get_all_wavevectors_projected()
        peak_guesses = {index: wavevector / (2 * np.pi) for index, wavevector in zip(indices, wavevectors)}
        peak_search_areas  = self._locate_peaks(peak_guesses, peak_search_area_size, peak_interpolation_area_size)
        stack_ft = self._get_stack_ft(stack)
        stacks_ft = self._crop_stacks(stack_ft, peak_search_areas)
        grids = self._crop_grids(peak_search_areas)
        stacks_ft_averaged = self._average_ft_stacks(stacks_ft)
        averaged_maxima = self._find_maxima(stacks_ft_averaged, grids)
        for i in range(1, iteration_number):
            coarse_peak_grids = self._get_coarse_grids(stacks_ft_averaged, grids, peak_interpolation_area_size)
            interpolated_grids = self._get_fine_grids(coarse_peak_grids, interpolation_factor)
            off_grid_ft_stacks, off_grid_otfs = self._off_grid_ft_stacks(stack, interpolated_grids, compute_otf=correct_peak_position)
            off_grid_ft_averaged = self._average_ft_stacks(off_grid_ft_stacks)
            # plt.imshow(np.log1p(off_grid_ft_averaged[(2, 0)]), cmap='gray')
            # plt.show()
            # plt.imshow(np.log1p(off_grid_otfs[(2, 0)]), cmap='gray')
            # plt.show()
            grids = interpolated_grids
            if deconvolve_stacks and interpolation_factor**i > 10:
                off_grid_ft_averaged = self._deconvolve_stacks(off_grid_ft_averaged, off_grid_otfs)
            # plt.imshow(np.log1p(off_grid_ft_averaged[(2, 0)]), cmap='gray')
            # plt.show()
            stacks_ft_averaged = off_grid_ft_averaged
            averaged_maxima = self._find_maxima(off_grid_ft_averaged, grids)
            print('iteration', i, 'averaged_maxima', averaged_maxima)

        
        if correct_peak_position:
            print('before correction', averaged_maxima[(0, (2, 0))] / 2)
            averaged_maxima=self._correct_peak_position(averaged_maxima, stack)
            print('corrected maxima', averaged_maxima)

        rotation_angles = self._estimate_rotation_angles(averaged_maxima)
        refined_base_vectors = self._refine_base_vectors(averaged_maxima, rotation_angles)
        refined_wavevectors = self._refine_wavevectors(refined_base_vectors, rotation_angles)
        phase_matrix = self._compute_phase_matrix(stack, refined_wavevectors)
        new_illum_class = type(self.illumination)

        harmonics_dict = self._compose_harmonics_dict(refined_wavevectors)
        illumination_estimated = new_illum_class(
                 intensity_harmonics_dict = harmonics_dict,
                 dimensions= self.illumination.dimensions,
                 Mr=self.illumination.Mr,
                 spatial_shifts=None,
                 angles=rotation_angles,
                 )
        illumination_estimated.Mt = self.illumination.Mt
        illumination_estimated.phase_matrix = phase_matrix
        if estimate_modulation_coefficients:
            illumination_estimated.estimate_modulation_coefficients(stack, self.optical_system.psf, self.optical_system.x_grid, 
                                                                    method = method_for_modulation_coefficients, update=True) 

        illumination_estimated.electric_field_plane_waves = self.illumination.electric_field_plane_waves
        illumination_estimated.normalize_spatial_waves()
        for plane_wave in illumination_estimated.electric_field_plane_waves:
            scaling_factor = np.sum(self.illumination.get_base_vectors(0)**2)**0.5 / np.sum(illumination_estimated.get_base_vectors(0)**2)**0.5
            print('scaling factor', scaling_factor)
            plane_wave.wavevector *= scaling_factor
        return illumination_estimated
    
    @abstractmethod
    def _compute_ssnr(self, stack: np.ndarray, peak_neighborhood_size: int) -> np.ndarray:
        """
        Compute the SSNR from the raw image stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        peak_neighborhood_size : int
            The size of the neighborhood around the peak to consider.

        Returns
        -------
        np.ndarray
            The computed SSNR.
        """
        pass

    def _locate_peaks(self, approximate_peaks: dict[tuple[int], float], peak_search_area: int, peak_interpolation_area: int) -> np.ndarray:
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
        Crop the stacks around the peaks.

        Parameters
        ----------
        stacks : np.ndarray
            The raw image stack.
        grids : dict[np.ndarray]
            The grids for each peak.
        approximate_peaks : dict[tuple[int], float]
            The approximate peaks.

        Returns
        -------
        np.ndarray
            The cropped stacks.
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
        Crop the grids around the peaks.

        Parameters
        ----------
        stacks : np.ndarray
            The raw image stack.
        grids : dict[np.ndarray]
            The grids for each peak.
        approximate_peaks : dict[tuple[int], float]
            The approximate peaks.

        Returns
        -------
        np.ndarray
            The cropped grids.
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
        Crop the OTF around the peaks.

        Parameters
        ----------
        stacks : np.ndarray
            The raw image stack.
        grids : dict[np.ndarray]
            The grids for each peak.
        approximate_peaks : dict[tuple[int], float]
            The approximate peaks.

        Returns
        -------
        np.ndarray
            The cropped OTF.
        """
        cropped_otfs = {}
        for peak in peak_search_areas.keys():
            otf = self.optical_system.otf
            otf *= peak_search_areas[peak][...]
            cropped_otfs[peak] = np.abs(otf)
        return cropped_otfs
    
    def _get_coarse_grids(self, stack_ft_averages: np.ndarray, grids: dict[np.ndarray], peak_interpolation_area_size: int) -> np.ndarray:
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
        Interpolate the peaks in the raw image stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        interpolation_factor : int
            The factor by which to interpolate the peaks.

        Returns
        -------
        np.ndarray
            The interpolated peaks.
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
    
    def _off_grid_ft_stacks(self, stack: np.ndarray, fine_q_grids: np.ndarray, compute_otf=True, account_for_pixel_size=True) -> np.ndarray:
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
            if compute_otf:
                psf = self.optical_system.psf.flatten()
                otf = fourier_exponents @ psf
                otfs_dict[index] = np.abs(otf.reshape(fine_q_grid.shape[:-1]))
                # plt.imshow(np.abs(otfs_dict[index]), cmap='gray')
                # plt.show()
                # plt.imshow(np.abs(otfs_dict[index]), cmap='gray')
                # plt.show()
        return images_ft_dict, otfs_dict
    
    def _average_ft_stacks(self, stacks_ft_dict: dict[np.ndarray]) -> np.ndarray:
        averaged_images_ft_dict = {}
        for index in stacks_ft_dict.keys():
            stack = stacks_ft_dict[index]
            averaged_images_ft_dict[index] = np.mean(np.abs(stack), axis=0)
        return averaged_images_ft_dict
    
    def _deconvolve_raw_stack(self, stack: np.ndarray, ssnr: np.ndarray) -> np.ndarray:
        deconvolved_stack = np.zeros_like(stack, dtype=np.complex128)
        for r in range(stack.shape[0]):
            for n in range(stack.shape[1]):
                image_ft = wrapped_fftn(stack[r, n])
                image_filtered = image_ft /np.where(self.optical_system.otf > 10**-10, self.optical_system.otf, np.inf) / (1 + ssnr)
                deconvolved_stack[r, n] =  wrapped_ifftn(image_filtered)
        return deconvolved_stack

    def _deconvolve_stacks(self, stacks_ft_averaged: dict[np.ndarray], otfs_dict: dict[np.ndarray]) -> np.ndarray:
        deconvolved_stacks = {}
        for index in stacks_ft_averaged.keys():
            image = stacks_ft_averaged[index]
            otf = np.abs(otfs_dict[index])
            deconvolved_stack = image / otf
            deconvolved_stacks[index] = deconvolved_stack
        return deconvolved_stacks
    
    def _find_maxima(self, image_dict: dict[np.ndarray], fine_q_grids: dict[np.ndarray]) -> np.ndarray:
        """
        Find the maxima in the image stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.

        Returns
        -------
        np.ndarray
            The found maxima.
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
        
    def _correct_peak_position(self, estimated_peaks: dict[np.ndarray[3, np.float64]], stack: np.ndarray, dk: float = 10**-4) -> np.ndarray:
        """
        Account for the non-uniform OTF to correct the peaks position assuming it is not far from actual illumination wavevectros. 
        """
        otf_dict = {}
        otf_grad_dict = {}
        grid = self.optical_system.x_grid
        for key in estimated_peaks.keys():
            peak = estimated_peaks[key]
            psf = self.optical_system.psf
            q_values = np.array([peak, peak - np.array((dk, 0)), peak + np.array((dk, 0)), peak - np.array((0, dk)), peak + np.array((0, dk))])

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
            # S = 1 if not second_order else 
            delta_k = f_values[0] / otf * (1  + f_values[0]/(2 * otf **2) * otf_grad.T @ np.linalg.inv(f_hessian) @ otf_grad) * np.linalg.inv(f_hessian) @ otf_grad 
            print('correction ', delta_k)
            corrected_peaks[key] = estimated_peaks[key] + delta_k

        return corrected_peaks
    
    def _compute_phase_matrix(self, stack: np.ndarray, refined_wavevectors: np.ndarray) -> np.ndarray:
        """
        Estimate the phase shifts using the refined base vectors.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        refined_base_vectors : np.ndarray
            The refined base vectors.

        Returns
        -------
        np.ndarray
            The estimated phase shifts.
        """
        phase_matrix = {}
        grid = self.optical_system.x_grid
        for n in range(stack.shape[1]):
            for index in refined_wavevectors.keys():
                r = index[0]
                m = index[1]
                wavevector = refined_wavevectors[index] / (2 * np.pi)
                ft = off_grid_ft(stack[r, n], grid, np.array((wavevector, )))
                phase = np.angle(ft)
                if r == 0 and m == (2, 0):
                    print('r', r, 'm', m, 'n', n, 'k', wavevector, 'ft', np.abs(ft),  'angle', phase / np.pi * 180)
                phase_matrix[(r, n, m)] = np.exp(1j * phase)

        return phase_matrix

    def _compute_spatial_shifts(self, phase_matrix, refined_wavevectors: np.ndarray) -> np.ndarray:
        """
        Compute the spatial shifts from the phase matrix.

        Parameters
        ----------
        phase_matrix : np.ndarray
            The phase matrix.
        refined_wavevectors : np.ndarray
            The refined wavevectors.

        Returns
        -------
        np.ndarray
            The computed spatial shifts.
        """
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
    
    def _compute_modulation_depths(self, stack_ft: np.ndarray, phase_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the modulation depths from the phase matrix.

        Parameters
        ----------
        stack_ft : np.ndarray
            The raw image stack in Fourier space.
        phase_matrix : np.ndarray
            The phase matrix.

        Returns
        -------
        np.ndarray
            The computed modulation depths.
        """

        # fourier_orders = {}
        # for index in self.illumination.rearranged_indices.keys():
        #     r, m = index[0], index[1]
        #     fourier_order = np.zeros((stack_ft.shape[2:]), dtype=np.complex128)
        #     for n in range(stack_ft[r].shape[1]):
        #             fourier_order += stack_ft[r, n] * phase_matrix[r, n, m].conjugate()
        #     fourier_orders[index] = fourier_order

        modulation_depths = {}
        # for r in range(self.illumination.Mr):
        #     modulation_depths[r, (0, 0)] = {1. + 0j}

        # for index in fourier_orders.keys():
        #     ...
        # grid = self.optical_system.x_grid
        # for n in range(stack.shape[1]):
        #     for index in refined_wavevectors.keys():
        #         r = index[0]
        #         m = index[1]
        #         wavevector = refined_wavevectors[index] / (2 * np.pi)
        #         ft = off_grid_ft(stack[r, n], grid, np.array((wavevector, )))

        return modulation_depths

    # def _modualation_depth_loss_function(self, fourier_orders: dict[np.ndarray]) -> dict[complex]:
    #     """
    #     Compute the modulation depth loss function.

    #     Parameters
    #     ----------
    #     fourier_orders : dict
    #         The Fourier orders.

    #     Returns
    #     -------
    #     np.ndarray
    #         The computed modulation depths.
    #     """
    #     loss_function = {}
    #     for index in fourier_orders.keys():
    #         r, m = index[0], index[1]
    #         if m == (0, 0):
    #             continue
    #         fourier_order = fourier_orders[index]
    #         loss_function[index] = np.abs(fourier_order) ** 2
    #     return loss_function
    
class PatternEstimatorCrossCorrelation2D(PatternEstimatorCrossCorrelation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)


class PatternEstimatorCrossCorrelation3D(PatternEstimatorCrossCorrelation):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves3D, optical_system: OpticalSystem3D):
        super().__init__(illumination, optical_system)

class PatternEstimatorInterpolation2D(PatternEstimatorInterpolation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)

    def _compute_ssnr(self, stack: np.ndarray, iterations: int) -> np.ndarray:
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
    


            
        
  
        