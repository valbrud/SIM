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
        self._ensure_otf_is_ready()

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
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
    
    def _refine_wavevectors(self, refined_base_vectors: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        refined_wavevectors = {}
        for sim_index in self.illumination.rearranged_indices.keys():
           refined_wavevectors[sim_index] = np.array([2 * np.pi * sim_index[dim] * refined_base_vectors[dim] for dim in range(len(sim_index))])
        return refined_wavevectors

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

    # ---------------------------------------------------------------------
    # illumination cloning & phase transfer
    # ---------------------------------------------------------------------
    def _construct_data_driven_illumination(
        self,
        a_m: Dict[int, complex],
        psi_rot: np.ndarray,
        psi_trans: np.ndarray,
    ) -> PlaneWavesSIM:
        # deep‑copy to preserve original
        new_illum = deepcopy(self.illumination)

        # --- update amplitudes -------------------------------------------
        # assumes diffraction orders are keyed by integer tuples in waves dict
        for idx, harmonic in new_illum.waves.items():
            m = max(abs(i) for i in idx)  # crude order lookup
            if m in a_m:
                harmonic.amplitude = a_m[m]

        # --- rebuild phase matrix ----------------------------------------
        new_illum.compute_phase_matrix()  # uses spatial_shifts & wavevectors
        # overwrite with measured phases (simplest: store as attribute)
        new_illum._psi_rot = psi_rot  # type: ignore[attr-defined]
        new_illum._psi_trans = psi_trans  # type: ignore[attr-defined]
        return new_illum

    # ---------------------------------------------------------------------
    # internal
    # ---------------------------------------------------------------------
    def _ensure_otf_is_ready(self):
        if self.optical_system.otf is None:
            self.optical_system.compute_psf_and_otf()


###############################################################################
# concrete classes
###############################################################################


class IlluminationPatternEstimator2D(IlluminationPatternEstimator):
    dimensionality = 2

    def __init__(self, illumination: IlluminationPlaneWaves2D, optical_system: OpticalSystem2D):
        super().__init__(illumination, optical_system)


class IlluminationPatternEstimator3D(IlluminationPatternEstimator):
    dimensionality = 3

    def __init__(self, illumination: IlluminationPlaneWaves3D, optical_system: OpticalSystem3D):
        super().__init__(illumination, optical_system)
