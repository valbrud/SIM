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

import numpy as np

from wrappers import wrapped_fftn
from Dimensions import DimensionMetaAbstract

from Illumination import (
    PlaneWavesSIM,
    IlluminationPlaneWaves2D,
    IlluminationPlaneWaves3D,
)
from OpticalSystems import OpticalSystem, OpticalSystem2D, OpticalSystem3D

__all__ = [
    "IlluminationPatternEstimator2D",
    "IlluminationPatternEstimator3D",
]

###############################################################################
# Base class
###############################################################################


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
        *,
        spatial_domain: bool = True,
        return_as_illumination_object: bool = False,
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

        # FFT if required ---------------------------------------------------
        stack_q = (
            self._fft_stack(stack) if spatial_domain else stack  # already fft
        )

        # OTF weighting -----------------------------------------------------
        weight = np.abs(self.optical_system.otf)
        stack_q *= weight  # broadcasting over Mr, Mt axes

        # correlation matrix & peak search ----------------------------------
        Cmat = self._build_cross_corr(stack_q)
        kvec = self._search_peak(Cmat)
        psi_rot, psi_trans = self._extract_phases(Cmat)

        # modulation depths --------------------------------------------------
        a_m = self._solve_modulations(stack_q, kvec)

        if return_as_illumination_object:
            return self._construct_data_driven_illumination(a_m, psi_rot, psi_trans)

        # default: return raw parameter arrays + modulation depths
        dpsi_rot = np.diff(psi_rot[:, 0], prepend=psi_rot[0, 0])
        dpsi_trans = np.diff(psi_trans[0, :], prepend=psi_trans[0, 0])
        return dpsi_rot, dpsi_trans, psi_rot[:, 0], psi_trans[0, :], a_m

    # ---------------------------------------------------------------------
    # helpers – FFT & correlation matrix
    # ---------------------------------------------------------------------
    def _fft_stack(self, stack: np.ndarray) -> np.ndarray:  # noqa: D401
        axes = tuple(range(-self.dimensionality, 0))
        return wrapped_fftn(stack, axes=axes, norm=None)

    def _build_cross_corr(self, Iq: np.ndarray) -> Dict[int, np.ndarray]:
        """C^m_{n1 n2}(q) for *positive* m values."""
        Mr, Mt = self.illumination.Mr, self.illumination.Mt
        Cmat: Dict[int, np.ndarray] = {}
        for m in range(1, Mt):
            I1 = np.conjugate(Iq[:, :-m])
            I2 = Iq[:, m:]
            prod = I1 * I2
            prod = np.pad(prod, ((0, 0), (m, 0)) + ((0, 0),) * self.dimensionality)
            Cmat[m] = prod
        return Cmat

    # ---------------------------------------------------------------------
    # peak search & phase extraction
    # ---------------------------------------------------------------------
    def _search_peak(self, Cmat: Dict[int, np.ndarray]) -> np.ndarray:  
        """Return one spatial k‑vector per rotation (shape (Mr, dimensionality))."""
        Mr = self.illumination.Mr
        kvec = np.zeros((Mr, self.dimensionality))

        for r in range(Mr):
            # accumulate |C|² over all m
            acc = sum(np.abs(arr[r]) ** 2 for arr in Cmat.values())   # (Mt, Ny, Nx)

            # *** critical line: remove the Mt axis ***
            acc = acc.sum(axis=0)                                     # (Ny, Nx) or (Nz, Ny, Nx)

            peak   = np.unravel_index(acc.argmax(), acc.shape)
            centre = np.array([s // 2 for s in acc.shape])
            kvec[r] = np.array(peak) - centre

        return kvec

    def _extract_phases(
        self, Cmat: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recover absolute phase of every raw frame.

        For 2‑D SIM with equal phase spacing we use only the m = 1
        correlation planes and accumulate the phase increments.
        """
        Mr, Mt = self.illumination.Mr, self.illumination.Mt
        psi_rot   = np.zeros((Mr, Mt))               # rotational phases (unused for Mr=1)
        psi_trans = np.zeros((Mr, Mt))               # translational phases

        if 1 not in Cmat:
            return psi_rot, psi_trans                # nothing to do

        arr = Cmat[1]                                # (Mr, Mt, Ny, Nx)
        half = 2                                     # 5‑pixel window
        centre = tuple(s // 2 for s in arr.shape[-self.dimensionality:])
        win = tuple(slice(c - half, c + half + 1) for c in centre)

        for r in range(Mr):
            cumulative = 0.0
            # phase 0 is reference => ψ = 0
            for n in range(1, Mt):
                patch = arr[(r, n) + win]            # product I*_{n‑1} × I_n
                inc = np.angle(patch.mean())         # increment n‑1 → n
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
