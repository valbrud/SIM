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

from wrappers import wrapped_fftn, wrapped_ifftn
from Dimensions import DimensionMetaAbstract
from abc import abstractmethod

from Illumination import (
    PlaneWavesSIM,
    IlluminationPlaneWaves2D,
    IlluminationPlaneWaves3D,
)
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

    @abstractmethod
    def estimate_illumination_parameters(self,
                                         stack,
                                        return_as_illumination_object: bool = False,
                                        **kwargs): 
        pass
    
    
    def _refine_wavevectors(self, refined_base_vectors: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        refined_wavevectors = {}
        for sim_index in self.illumination.rearranged_indices.keys():
           refined_wavevectors[sim_index] = np.array([2 * np.pi * sim_index[dim] * refined_base_vectors[dim] for dim in range(len(sim_index))])
        return refined_wavevectors


class PatternEstimatorCrossCorrelation(IlluminationPatternEstimator):
    """Cross‑correlation estimator for 2D and 3D SIM."""

    def __init__(self, illumination: PlaneWavesSIM, optical_system: OpticalSystem):
        super().__init__(illumination, optical_system)

    def estimate_illumination_parameters(
        self,
        stack: np.ndarray,
        return_as_illumination_object: bool = False,
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
                f"Stack rotations={stack.shape[0]} differ from illumination.Mr={Mr}")
        if stack.shape[1] != Mt:
            raise ValueError(
                f"Stack translations={stack.shape[1]} differ from illumination.Mt={Mt}")

        if effective_otfs is None:
            _, effective_otfs = self.illumination.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        
        self.effective_otfs = effective_otfs
        #To minimize memory requirements by avoiding one more dimension in big arras
        for r in range(Mr):
            # correlation matrix & peak search ----------------------------------
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

            wavevectors, indices = self.illumination.get_wavevectors_projected(r)
            initial_guess = {}
            for index, wavevector in zip(indices, wavevectors):
                initial_guess[index] = wavevector

            estimated_modualtion_patterns = self._phase_modulation_patterns(initial_guess)
            correlation_matrix = self._cross_correlation_matrix(stack, r, estimated_modualtion_patterns, q_grid)
            merit_function = self._merit_function(correlation_matrix)
            max_index = np.unravel_index(np.argmax(merit_function, axis=0), merit_function.shape)
            # merit_function_new = merit_function[max_index]
            base_vectors = self.illumination.get_base_vectors()
            base_vectors = np.array((base_vectors[0], 0)) / (2 * np.pi)
            refined_base_vectors = self._refine_base_vectors(base_vectors, correlation_matrix, q_grid)
            refined_wavevectors = self._refine_wavevectors(refined_base_vectors)
            diff = np.sum((refined_base_vectors - base_vectors)**2)**0.5
            print(0, 'new', refined_base_vectors, 'initial guess', base_vectors,'diff', diff)
            i = 0
            while i < max_iterations:
                i += 1
                base_vectors = refined_base_vectors
                wavevectors = refined_wavevectors
                estimated_modualtion_patterns = self._phase_modulation_patterns(refined_wavevectors)
                correlation_matrix = self._cross_correlation_matrix(stack, r, estimated_modualtion_patterns, q_grid)
                refined_base_vectors = self._refine_base_vectors(base_vectors, correlation_matrix, q_grid)
                refined_wavevectors = self._refine_wavevectors(refined_base_vectors)
                print(i, 'new', refined_base_vectors, 'old', base_vectors)
                dq = q_grid[1, 1] - q_grid[0, 0]
                print(i, 'diff', refined_base_vectors - base_vectors, 'dq', dq)
                if (np.sum((refined_base_vectors - base_vectors)**2) < 2 * dq**2).all():
                    q_grid = self._fine_q_grid(q_grid, zooming_factor)
                    print("zooming in q_grid, new q_grid boundaries", q_grid[0, 0][0], q_grid[-1, -1][0])

                # merit_function = self._merit_function(correlation_matrix)
                # max_index = np.unravel_index(np.argmax(merit_function, axis=0), merit_function.shape)
                diff = np.sum((refined_base_vectors - base_vectors)**2)

        return refined_base_vectors


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
        # ensure phase matrix is built    
        x_grid_flat = self.optical_system.x_grid.reshape(-1, self.optical_system.dimensionality)
        q_grid_flat = fine_q_grid.reshape(-1, self.optical_system.dimensionality)
        phase_matrix = q_grid_flat @ x_grid_flat.T
        fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
        for m in self.illumination.waves.keys():
            if np.isclose(np.sum(np.abs(np.array(m))), 0):
                continue
            if m[0] >= 0:
                for n1 in range(self.illumination.spatial_shifts.shape[0]):
                    for n2 in range(self.illumination.spatial_shifts.shape[0]):
                        # plt.imshow(estimated_modualtion_patterns[m].real, cmap='gray')
                        # plt.show()
                        signal_function = (images[r, n1] * images[r, n2] * estimated_modualtion_patterns[m]).flatten()
                        Cmn1n2[(m, n1, n2)] = fourier_exponents @ signal_function
                        if n1 == n2:
                            noise_function = (images[r, n1] *  estimated_modualtion_patterns[m]).flatten()
                            Cmn1n2[(m, n1, n2)] -= fourier_exponents @ noise_function
                        Cmn1n2[(m, n1, n2)] = Cmn1n2[(m, n1, n2)].reshape(fine_q_grid.shape[:-1])
                        # plt.imshow(np.abs(Cmn1n2[(m, n1, n2)]), cmap='gray',
                                    # extent=(np.amin(fine_q_grid[..., 0]), np.amax(fine_q_grid[..., 0]), np.amin(fine_q_grid[..., 1]), np.amax(fine_q_grid[..., 1])))
                        # plt.title(f"m: {m}, n1: {n1}, n2: {n2}")
                        # plt.show()
        return Cmn1n2
    
    def _refine_base_vectors(self, base_vectors: np.ndarray,  Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray], fine_q_grid) -> np.ndarray:
        """
        Refine the base vectors using the correlation matrix Cmn1n2.

        Parameters
        ----------
        Cmn1n2 : dict[tuple(int, ...), int, int], np.ndarray]
            The correlation matrix.

        Returns
        -------
        np.ndarray
            The refined base vectors.
        """
        nominator = np.zeros((self.optical_system.dimensionality), dtype=np.float64)
        denominator = np.zeros((self.optical_system.dimensionality), dtype=np.float64)
        qs = {}
        weights = {}
        otf_shape = np.copy(np.array(self.optical_system.otf.shape))
        for index in Cmn1n2.keys():
            max_index = np.unravel_index(np.argmax(np.abs(Cmn1n2[index])), Cmn1n2[index].shape)
            q = fine_q_grid[max_index]
            # print(index, max_index, q)
            sim_index = index[0]
            if not sim_index in weights.keys():
                effective_otf = self.effective_otfs[(0, sim_index)]
                weights[sim_index] = np.abs(self.illumination.waves[sim_index].amplitude) * np.abs(effective_otf[*otf_shape//2])
            qs[index] = q

        for index in qs.keys():
            component_weights = weights[index[0]] * np.abs(np.array(index[0]))
            dk_estimate = qs[index] / np.where(index[0], index[0], np.inf)
            # print(dk_estimate)
            nominator += dk_estimate * component_weights
            denominator += component_weights
        # print('weighted shigt', nominator / np.where(denominator, denominator, np.inf) )
        refined_base_vectors = base_vectors - nominator / np.where(denominator, denominator, np.inf)
        return refined_base_vectors
    


    def _merit_function(self, Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray]) -> np.ndarray:
        """
        Compute the merit function for the base vector estimates"""

        return sum([C * C.conjugate() for C in Cmn1n2.values()])  # sum over m,n1,n2
    


    def _search_peak(self, Cmat: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Refine the *known* illumination k-vector instead of brute-force search.
        """
        Mr = self.illumination.Mr
        kvec = self._initial_kvec_from_illum()          # (Mr, dim) pixels
        half = 3                                        # ±3-pixel search window

        for r in range(Mr):
            acc = sum(np.abs(arr[r])**2
                    for m, arr in Cmat.items() if m > 0).sum(axis=0)  # (Ny,Nx)

            cy, cx = np.array(acc.shape) // 2
            ky0, kx0 = (cy + int(round(kvec[r, 0])),
                        cx + int(round(kvec[r, 1])))

            # local 7×7 patch around the theoretical k-vector
            ys = slice(ky0 - half, ky0 + half + 1)
            xs = slice(kx0 - half, kx0 + half + 1)
            sub = acc[ys, xs]
            dy, dx = np.unravel_index(sub.argmax(), sub.shape)
            kvec[r, 0] = kvec[r, 0] + dy - half
            kvec[r, 1] = kvec[r, 1] + dx - half

        return kvec

    def _extract_phases(
        self,
        Cmat: Dict[int, np.ndarray],
        kvec: np.ndarray,                       #  <-- pass-in from caller
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute absolute phase of each raw frame.

        *For linear SIM with equal phase spacing.*

        Parameters
        ----------
        Cmat : dict[int, ndarray]
            Correlation planes from `_build_cross_corr`.
        kvec : ndarray
            Peak coordinates returned by `_search_peak`
            (shape (Mr, dimensionality) in pixel units).

        Returns
        -------
        psi_rot, psi_trans : ndarray
            Phase arrays of shape (Mr, Mt).
        """
        Mr, Mt = self.illumination.Mr, self.illumination.Mt
        psi_rot   = np.zeros((Mr, Mt))
        psi_trans = np.zeros((Mr, Mt))

        if 1 not in Cmat:
            return psi_rot, psi_trans

        arr = Cmat[1]                            # (Mr, Mt, *q)
        half = 2                                     # 5×5 window

        for r in range(Mr):
            cy, cx = np.array(arr.shape[-2:]) // 2
            qy0, qx0 = cy + int(kvec[r, 0]), cx + int(kvec[r, 1])

            # build a 5×5 window around the side-band peak
            ys = slice(qy0 - half, qy0 + half + 1)
            xs = slice(qx0 - half, qx0 + half + 1)

            cumulative = 0.0
            for n in range(1, Mt):
                patch = arr[r, n, ys, xs]            # complex 5×5 values
                inc = np.angle(patch.mean())         # robust increment
                cumulative += inc
                psi_trans[r, n] = cumulative

        return psi_rot, psi_trans       

    # modulation depth – least squares
    # ---------------------------------------------------------------------
    def _solve_modulations(self, Iq: np.ndarray, kvec: np.ndarray) -> Dict[int, complex]:
        Mt = self.illumination.Mt
        a_m: Dict[int, complex] = {}
        # naive implementation: mean ratio of side‑band / zero‑order magnitudes
        for m in range(1, Mt):
            sb = np.abs(Iq[:, m:])  # side‑band (n2 > n1)
            zo = np.abs(Iq[:, :-m])  # zero‑order proxy
            ratio = np.mean(sb / (zo + 1e-12))
            a_m[m] = ratio
        return a_m

    def _ensure_otf_is_ready(self):
        if self.optical_system.otf is None:
            self.optical_system.compute_psf_and_otf()


from SSNRCalculator import SSNRBase
class PatternEstimatorInterpolation(IlluminationPatternEstimator):
    def estimate_illumination_parameters(self,
                                     stack: np.ndarray,
                                     return_as_illumination_object: bool = False, 
                                     interpolation_factor: int = 100,  
                                     peak_search_area_size: int = 31,
                                     peak_interpolation_area_size: int = 5,
                                     iteration_number: int = 3,
                                     ssnr_estimation_iters: int = 10, 
                                     real_space_images = True, 
                                     deconvolve_stacks = True,
                                     correct_peak_position = False) -> np.ndarray:
        """
        Estimate the illumination patterns using the SSNR method.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        peak_neighborhood_size : int, optional
            The size of the neighborhood around the peak to consider, by default 11.

        Returns
        -------
        np.ndarray
            The estimated illumination patterns.
        """

        if deconvolve_stacks and correct_peak_position:
            raise ValueError("Deconvolution and peak position are mutually exclusive ways to increase peak precision." \
                             "Choose one of them to be false.")
        
        wavevectors, indices = self.illumination.get_wavevectors_projected(0)
        peak_guesses = {index: wavevector / (2 * np.pi) for index, wavevector in zip(indices, wavevectors)}
        peak_search_areas  = self._crop_peaks(stack, peak_guesses, peak_search_area_size, peak_interpolation_area_size)
        stacks_ft = self._crop_stacks(stack, peak_search_areas)
        cropped_otfs = self._crop_otf(peak_search_areas)
        grids = self._crop_grids(peak_search_areas)
        stacks_ft_averaged = self._average_image_stacks(stacks_ft)
        averaged_maxima = self._find_maxima(stacks_ft_averaged, grids)
        for i in range(1, iteration_number):
            coarse_peak_grids = self._get_coarse_grids(stacks_ft_averaged, grids, peak_interpolation_area_size)
            interpolated_grids = self._get_fine_grids(coarse_peak_grids, interpolation_factor)
            off_grid_ft_stacks, off_grid_otfs = self._off_grid_ft_stacks(stack, interpolated_grids, compute_otf=correct_peak_position)
            off_grid_ft_averaged = self._average_image_stacks(off_grid_ft_stacks)
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
            print('before correction', averaged_maxima[(2, 0)] / 2)
            averaged_maxima=self._correct_peak_position(averaged_maxima, stack)
            print('corrected maxima', averaged_maxima)

        refined_base_vectors = self._refine_base_vectors(averaged_maxima)
        refined_wavevectors = self._refine_wavevectors(refined_base_vectors)

        return refined_base_vectors
    
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
    
    def _stack_ft(self, stack: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier transform of the raw image stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.

        Returns
        -------
        np.ndarray
            The Fourier transformed stack.
        """
        return np.array([wrapped_fftn(image) for image in stack])
    
    def _crop_peaks(self, stack:np.ndarray, approximate_peaks: dict[tuple[int], float], peak_search_area: int, peak_interpolation_area: int) -> np.ndarray:
        peak_search_areas = {}
        for peak in approximate_peaks.keys():
            if np.isclose(np.sum(np.abs(np.array(peak))), 0):
                continue
            if peak[0] >= 0:
                grid = self.optical_system.q_grid
                dq = grid[*([1]*self.dimensionality)] - grid[*([0]*self.dimensionality)]
                approximate_peak = approximate_peaks[peak]
                labeling_array = (np.abs((grid - approximate_peak[None, None, :])) <= peak_search_area * dq[None, None, :]).all(axis=-1)
                peak_search_areas[peak] = labeling_array
                # plt.imshow(labeling_array, cmap='gray')
                # plt.show()
        return peak_search_areas
    
    def _crop_stacks(self, stack: np.ndarray, peak_search_areas: np.ndarray) -> np.ndarray:
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
        stack_ft = np.array([wrapped_fftn(image) for image in stack])
        for peak in peak_search_areas.keys():
            if np.isclose(np.sum(np.abs(np.array(peak))), 0):
                continue
            if peak[0] >= 0:
                area = peak_search_areas[peak]
                cropped_stacks[peak] = np.array([image_ft * area for image_ft in stack_ft])
                plt.imshow(np.abs(cropped_stacks[peak][0]), cmap='gray')
                # plt.plot((approximate_peak[1], approximate_peak[0], 'cx'))
                plt.show()
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
            if np.isclose(np.sum(np.abs(np.array(peak))), 0):
                continue
            if peak[0] >= 0:
                grid = self.optical_system.q_grid
                grid *= peak_search_areas[peak][..., None]
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
            if np.isclose(np.sum(np.abs(np.array(peak))), 0):
                continue
            if peak[0] >= 0:
                otf = self.optical_system.otf
                otf *= peak_search_areas[peak][...]
                cropped_otfs[peak] = np.abs(otf)
        return cropped_otfs
    
    def _get_coarse_grids(self, stack_ft_averages: np.ndarray, grids: dict[np.ndarray], peak_interpolation_area_size: int) -> np.ndarray:
        coarse_grids = {}
        for peak in grids.keys():
            if np.isclose(np.sum(np.abs(np.array(peak))), 0):
                continue
            if peak[0] >= 0:
                stack = stack_ft_averages[peak]
                grid = grids[peak]
                max_index = np.unravel_index(np.argmax(stack), stack.shape)
                print('max_index', max_index)
                refined_peak = grid[max_index]
                print('refined peak', refined_peak)
                print('peak_value', stack[max_index])
                slices = tuple([slice(max_index[dim] - peak_interpolation_area_size // 2, max_index[dim] + peak_interpolation_area_size // 2 + 1) for dim in range(len(max_index))])
                coarse_grids[peak] = grid[slices + (slice(None),)]
                plt.imshow(np.abs(stack), cmap='gray')
                plt.plot(max_index[1], max_index[0], 'rx')
                plt.show()
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
    
    def _refine_base_vectors(self, averaged_maxima: dict):
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
        refined_base_vectors = {}
        base_vectors = np.zeros(self.optical_system.dimensionality, dtype=np.float64)
        index_sum  = np.zeros(self.optical_system.dimensionality, dtype=np.int16)
        for index in averaged_maxima.keys():
            base_vectors += averaged_maxima[index]
            index_sum += np.array(index)

        refined_base_vectors = base_vectors / np.where(index_sum, index_sum, np.inf)
        return refined_base_vectors
    
    def _off_grid_ft(self, array: np.ndarray, x_grid: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        x_grid_flat = self.optical_system.x_grid.reshape(-1, self.optical_system.dimensionality)
        q_grid_flat = q_values.reshape(-1, self.optical_system.dimensionality)
        phase_matrix = q_grid_flat @ x_grid_flat.T
        fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
        array_ft_values = fourier_exponents @ array.flatten()
        return array_ft_values.reshape(q_values.shape[:-1])
    
    def _off_grid_ft_stacks(self, stack: np.ndarray, fine_q_grids: np.ndarray, compute_otf=True) -> np.ndarray:
        x_grid_flat = self.optical_system.x_grid.reshape(-1, self.optical_system.dimensionality)
        images_ft_dict = {}
        otfs_dict = {}
        for index in fine_q_grids.keys():
            fine_q_grid = fine_q_grids[index]
            q_grid_flat = fine_q_grid.reshape(-1, self.optical_system.dimensionality)
            phase_matrix = q_grid_flat @ x_grid_flat.T
            fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
            new_stack = []
            for image in stack:
                image = image.flatten()
                image_ft = fourier_exponents @ image
                new_stack.append(image_ft.reshape(fine_q_grid.shape[:-1]))
            images_ft_dict[index] = np.array(new_stack)
            if compute_otf:
                psf = self.optical_system.psf.flatten()
                otf = fourier_exponents @ psf
                otfs_dict[index] = np.abs(otf.reshape(fine_q_grid.shape[:-1]))
        return images_ft_dict, otfs_dict
    
    def _average_image_stacks(self, images_ft_dict: dict[np.ndarray]) -> np.ndarray:
        averaged_images_ft_dict = {}
        for index in images_ft_dict.keys():
            stack = images_ft_dict[index]
            averaged_images_ft_dict[index] = np.mean(np.abs(stack), axis=0)
        return averaged_images_ft_dict
    
    def _deconvolve_raw_stack(self, stack: np.ndarray, ssnr: np.ndarray) -> np.ndarray:
        deconvolved_stack = np.zeros_like(stack, dtype=np.complex128)
        for i in range(stack.shape[0]):
            image_ft = wrapped_fftn(stack[i])
            image_filtered = image_ft /np.where(self.optical_system.otf > 10**-10, self.optical_system.otf, np.inf) / (1 + ssnr)
            deconvolved_stack[i] =  wrapped_ifftn(image_filtered)
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
        x_grid = self.optical_system.x_grid

        otf_dict = {}
        otf_grad_dict = {}
        for key in estimated_peaks.keys():
            peak = estimated_peaks[key]
            psf = self.optical_system.psf
            q_values = np.array([peak, peak - np.array((dk, 0)), peak + np.array((dk, 0)), peak - np.array((0, dk)), peak + np.array((0, dk))])

            otf_values = self._off_grid_ft(psf, x_grid, q_values)
            otf_values = np.abs(otf_values)
            otf_dict[key] = otf_values[0]

            otf_grad = np.array([otf_values[2] - otf_values[1], otf_values[4] - otf_values[3]]) / (2 * dk)
            otf_grad_dict[key] = otf_grad

        q_grid_dict = {(0, 0): np.array([np.array((0, 0)),
                                         np.array((-dk, 0)), np.array((dk, 0)),
                                         np.array((0, -dk)), np.array((0, dk)),
                                         np.array((-dk, -dk)), np.array((-dk, dk)),
                                         np.array((dk, -dk)), np.array((dk, dk))])}
        
        I_values, otf_values = self._off_grid_ft_stacks(stack, q_grid_dict)
        I_values = self._average_image_stacks(I_values)
        f_values = I_values[(0, 0)] / np.abs(otf_values[(0, 0)])
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
    


            
        
  
        