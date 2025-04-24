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
                        plt.imshow(np.abs(Cmn1n2[(m, n1, n2)]), cmap='gray',
                                    extent=(np.amin(fine_q_grid[..., 0]), np.amax(fine_q_grid[..., 0]), np.amin(fine_q_grid[..., 1]), np.amax(fine_q_grid[..., 1])))
                        plt.title(f"m: {m}, n1: {n1}, n2: {n2}")
                        plt.show()
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
class PatternEstimatorSSNR(IlluminationPatternEstimator):
    def estimate_patterns_using_SSNR(self,
                                     stack: np.ndarray,
                                     peak_neighborhood_size: int = 5, 
                                     interpolation_factor: int = 100,  
                                     ssnr_estimation_iters: int = 10) -> np.ndarray:
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

        # ssnr_from_stack = self._compute_ssnr(stack, ssnr_estimation_iters)
        peak_interpolation_areas = self._localize_peaks(peak_neighborhood_size)
        interpolated_grids = self._get_fine_grids(peak_interpolation_areas)
        off_grid_ft_averaged = self._off_grid_ft_averaged(stack, interpolated_grids)
        averaged_maxima = self._find_maxima(off_grid_ft_averaged)
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

    def _get_fine_grid(coarse_grid: np.ndarray, interpolation_factor: int) -> np.ndarray:
        fine_q_coordinates = []
        for dim in range (len(coarse_grid)-1):
            q_max, q_min = coarse_grid[..., dim].max(), coarse_grid[..., dim].min()
            fine_q_coordinates.append(np.linspace(q_min, q_max, coarse_grid.shape[dim] * interpolation_factor))
        fine_q_grid = np.stack(np.meshgrid(*tuple(fine_q_coordinates), indexing='ij'), axis=-1)
        return fine_q_grid

    @abstractmethod
    def _localize_peaks(self, peak_neighborhood_size: int) -> np.ndarray:
        """
        Localize the peaks in the raw image stack.

        Parameters
        ----------
        stack : np.ndarray
            The raw image stack.
        peak_neighborhood_size : int
            The size of the neighborhood around the peak to consider.

        Returns
        -------
        np.ndarray
            The localized peaks.
        """
        pass
    
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
            base_vectors += averaged_maxima[index] * np.array(index)
            index_sum += np.array(index)
            
        refined_base_vectors = base_vectors / np.where(index_sum, index_sum, np.inf)
        return refined_base_vectors
    @abstractmethod
    def _off_grid_ft_averaged(self, stack: np.ndarray, fine_grid: np.ndarray) -> np.ndarray: ... 


    def _find_maxima(self, image_dict: dict[np.ndarray]) -> np.ndarray:
        """
        Find the maxima in the raw image stack.

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
            image = image_dict[index]
            max_index = np.unravel_index(np.argmax(image), image.shape)
            max_dict = index[max_index]
        return max_dict
        
    
class PatternEstimatorCrossCorrelation2D(PatternEstimatorCrossCorrelation):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)


class PatternEstimatorCrossCorrelation3D(PatternEstimatorCrossCorrelation):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves3D, optical_system: OpticalSystem3D):
        super().__init__(illumination, optical_system)

class PatternEstimatorSSNR2D(PatternEstimatorSSNR):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)

    def _compute_ssnr(self, stack: np.ndarray, interpolation_factor: int) -> np.ndarray:
        ssnr_estimated = SSNRBase.estimate_ssnr_from_image_binomial_splitting(stack, n_iter=100, radial=False)
        ssnr_estimated[ssnr_estimated < 0] = 0
        return ssnr_estimated
    
    def _localize_peaks(self, peak_neighborhood_size: int) -> np.ndarray:
        waves = self.illumination.waves
        grid = self.optical_system.q_grid
        dq = np.amin(grid[1, 1] - grid[0, 0])
        peak_interpolation_areas = {}
        for peak in waves.keys():
            if np.isclose(np.sum(np.abs(np.array(peak))), 0):
                continue
            if peak[0] >= 0:
                peak_approximate = waves[peak].wavevector / (2 * np.pi)
                labeling_array = np.array(self.optical_system.psf.shape, dtype=np.bool)
                labeling_array[np.sum((grid - peak_approximate[None, None, :])**2)**0.5 < peak_neighborhood_size * dq]= True
                peak_grid = grid[labeling_array, :]
                peak_interpolation_areas[peak] = peak_grid
        return peak_interpolation_areas

    def _off_grid_ft_averaged(self, stack: np.ndarray, fine_q_grids: np.ndarray) -> np.ndarray:
        x_grid_flat = self.optical_system.x_grid.reshape(-1, self.optical_system.dimensionality)
        averaged_images_dict = {}
        for index in fine_q_grids.keys():
            fine_q_grid = fine_q_grids[index]
            q_grid_flat = fine_q_grid.reshape(-1, self.optical_system.dimensionality)
            phase_matrix = q_grid_flat @ x_grid_flat.T
            fourier_exponents = np.exp(-1j * 2 * np.pi * phase_matrix)
            image_avg = np.zeros(fine_q_grid.shape[:-1], dtype=np.float64)
            for image in stack:
                image = image.flatten()
                image_ft = fourier_exponents @ image
                image_avg += np.abs(image_ft.reshape(fine_q_grid.shape[:-1]))
            averaged_images_dict[index] = image_avg.reshape() / stack.shape[0]
        return averaged_images_dict

            
        
  
        