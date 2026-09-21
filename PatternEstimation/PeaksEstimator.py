from __future__ import annotations

"""
PeaksEstimator.py

This module contains peak-position estimators used by SIM illumination-pattern
estimation.

Responsibilities:
    PeaksEstimator - Shared peak-estimator configuration and utilities.
    PeaksEstimatorInterpolation* - Iterative interpolation-based peak refinement.
    PeaksEstimatorCrossCorrelation* - Cross-correlation-based peak refinement.
"""

import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import math
from abc import abstractmethod
from typing import Any, Dict, Tuple

import hpc_utils
import matplotlib.pyplot as plt
import numpy as np

import utils
from Dimensions import DimensionMetaAbstract
from Illumination import PlaneWavesSIM
from OpticalSystems import OpticalSystem
from VectorOperations import VectorOperations
from utils import off_grid_ft
from wrappers import wrapped_fftn


class PeaksEstimator(metaclass=DimensionMetaAbstract):
    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 first_order_peaks: bool = True,
                 peak_search_area_size: int = 11,
                 zooming_factor: int = 100,
                 max_iterations: int = 2,
                 debug_info_level: int = 1,
                 correct_peak_position: bool = True):
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            optical_system=(optical_system, OpticalSystem),
            first_order_peaks=(first_order_peaks, bool)
        )
        self.illumination = illumination
        self.optical_system = optical_system
        if self.illumination.dimensionality != self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: "
                f"{self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )
        self.first_order_peaks = first_order_peaks
        self.peak_search_area_size = peak_search_area_size
        self.zooming_factor = zooming_factor
        self.max_iterations = max_iterations
        self.debug_info_level = debug_info_level
        self.correct_peak_position = correct_peak_position
        self._m = utils.select_up_to_inversion(self.illumination.harmonics)
        if first_order_peaks:
            self._m = self.illumination.filter_first_orders(self._m)

    @abstractmethod
    def estimate_peaks(
        self,
        stack: np.ndarray,
    ):
        """
        Implementations should return (refined_wavevectors, rotation_angles) like your existing API.
        """
        ...

    def _select_m(self):
        return utils.select_up_to_inversion(self.illumination.harmonics.keys())

    def get_stack_ft(self, stack: np.ndarray) -> np.ndarray:
        return np.array([np.array([wrapped_fftn(image) for image in stack[r]]) for r in range(stack.shape[0])])

    def merit_function(self, Cmn1n2: Dict[Tuple[Tuple[int, ...], int, int], np.ndarray]) -> np.ndarray:
        return sum([C * C.conjugate() for C in Cmn1n2.values()])

    def _czt_nd(self, sig_stack: np.ndarray, q_coords_cycles: tuple[np.ndarray, ...]) -> np.ndarray:
        x_coords = self.optical_system.psf_coordinates
        q_coords_rad = tuple((2.0 * np.pi) * np.asarray(qc) for qc in q_coords_cycles)

        dimensionality = self.optical_system.dimensionality
        spatial_axes = tuple(range(sig_stack.ndim - dimensionality, sig_stack.ndim))

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
        qx = np.asarray(q_coords[0], dtype=np.float64)
        qy = np.asarray(q_coords[1], dtype=np.float64)

        z = int(zooming_factor)
        if z < 1:
            raise ValueError(f"zooming_factor must be >= 1, got {zooming_factor}")

        if center is None:
            cx, cy = 0.0, 0.0
        else:
            cx, cy = float(center[0]), float(center[1])

        if qx.size < 2 or qy.size < 2:
            return (qx.copy(), qy.copy())

        dq_x = float(qx[1] - qx[0])
        dq_y = float(qy[1] - qy[0])

        n = 2 * z + 1
        qx_new = np.linspace(cx - dq_x, cx + dq_x, n, dtype=np.float64)
        qy_new = np.linspace(cy - dq_y, cy + dq_y, n, dtype=np.float64)
        return (qx_new, qy_new)

    def _q_window_around(self, q0: np.ndarray, size: int) -> tuple[np.ndarray, ...]:
        half = size // 2
        out = []
        for d, axis in enumerate(self.optical_system.otf_frequencies):
            idx = int(np.argmin(np.abs(axis - q0[d])))
            lo = max(0, idx - half)
            hi = min(axis.size, idx + half + 1)
            out.append(axis[lo:hi])
        return tuple(out)

    def estimate_rotation_angles(self, estimated_peaks: dict) -> np.ndarray:
        counts = np.zeros(self.illumination.Mr, dtype=np.float64)
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
            counts[r] += 1

        for r in range(self.illumination.Mr):
            angles[r] = (sum_observed[r] - sum_expected[r]) / counts[r]
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
    def estimate_peaks(self, stack: np.ndarray):
        peak_search_area_size = self.peak_search_area_size
        zooming_factor = self.zooming_factor
        max_iterations = self.max_iterations
        debug_info_level = self.debug_info_level
        correct_peak_position = self.correct_peak_position
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

                if debug_info_level > 1:
                    dqs = [float(q_coords[d][1] - q_coords[d][0]) for d in range(2)]
                    print("iteration", it, peak, dbg, "dq", dqs)

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

        if debug_info_level >= 2:
            self._debug_show_initial_final_on_avg_ft_2d(stack, initial_cycles, estimated_cycles)

        rotation_angles = self.estimate_rotation_angles(estimated_cycles)
        refined_base_vectors = self.refine_base_vectors(estimated_cycles, rotation_angles)
        refined_wavevectors = self.refine_wavevectors(refined_base_vectors, rotation_angles)

        return refined_wavevectors, rotation_angles

    def _initial_peak_guesses_cycles(self) -> dict:
        wavevectors, indices = self.illumination.get_all_wavevectors_projected()
        return {idx: (wv / (2 * np.pi)) for idx, wv in zip(indices, wavevectors)}

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
        ...

    def _final_peak_cycles(self, peak_state: dict[str, Any]) -> np.ndarray:
        return np.array(peak_state["q"], dtype=np.float64)

    def _postprocess_estimated_cycles(self, estimated_cycles: dict, stack: np.ndarray) -> dict:
        return estimated_cycles

    def _debug_show_initial_final_on_avg_ft_2d(self, stack: np.ndarray, initial_cycles: dict, final_cycles: dict):
        qx, qy = self.optical_system.otf_frequencies
        q_coords = (np.asarray(qx), np.asarray(qy))

        ft_avg_per_r: dict[int, np.ndarray] = {}
        for r in range(self.illumination.Mr):
            ft_stack = self._czt_nd(stack[r]**2, q_coords)
            ft_avg_per_r[r] = np.mean(np.abs(ft_stack), axis=0)

        for peak in self._m:
            r = peak[0]
            ft_avg = ft_avg_per_r[r]

            q0 = initial_cycles[peak]
            q1 = final_cycles[peak]

            fig, ax = plt.subplots()
            ax.set_title(f"r={r}, peak={peak}: avg |FT| (initial x, final o)")
            ax.imshow(
                np.log1p(10**8 * ft_avg.T),
                origin="lower",
                aspect="auto",
                extent=[q_coords[0].min(), q_coords[0].max(), q_coords[1].min(), q_coords[1].max()],
            )
            ax.plot(q0[0], q0[1], marker="x", linestyle="None")
            ax.plot(q1[0], q1[1], marker="o", linestyle="None")
            ax.set_xlabel("qx (cycles/unit)")
            ax.set_ylabel("qy (cycles/unit)")
            plt.show()

    def _debug_show_iteration_surface_2d(
        self,
        stack: np.ndarray,
        r: int,
        peak: tuple,
        q_coords: tuple[np.ndarray, np.ndarray],
        peak_state: dict[str, Any],
        it: int,
    ):
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
        ...


class PeaksEstimatorInterpolation(PeaksEstimatorIterative):
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
        ft_stack = self._czt_nd(stack[r], q_coords)
        ft_avg = np.mean(np.abs(ft_stack), axis=0)

        q_center = self._argmax_coords(ft_avg, q_coords)
        peak_state["q"] = q_center

        if debug_info_level >= 3:
            peak_state["_dbg_img"] = ft_avg
            peak_state["_dbg_qmax"] = q_center

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
        return self._correct_peak_position(estimated_cycles, stack)

    def _sample_stack_mean_abs(self, stack_r: np.ndarray, q_values: np.ndarray) -> np.ndarray:
        grid = self.optical_system.x_grid
        vals = [off_grid_ft(img, grid, q_values) for img in stack_r]
        return np.mean(np.abs(np.array(vals)), axis=0)

    def _sample_otf_abs(self, q_values: np.ndarray) -> np.ndarray:
        grid = self.optical_system.x_grid
        psf = self.optical_system.psf
        return np.abs(off_grid_ft(psf, grid, q_values))

    @abstractmethod
    def _correct_peak_position(self, estimated_peaks: dict, stack: np.ndarray, dk: float = 1e-4) -> dict:
        ...


class PeaksEstimatorCrossCorrelation(PeaksEstimatorIterative):
    def _initial_peak_guesses_cycles(self) -> dict:
        wavevectors, indices = self.illumination.get_all_wavevectors_projected()
        zero = np.zeros(2, dtype=np.float64)
        return {idx: zero.copy() for idx in indices}

    def _init_global_state(self, stack: np.ndarray, peak_guesses: dict) -> dict[str, Any]:
        wavevectors, indices = self.illumination.get_all_wavevectors_projected()
        abs_guesses_cycles = {idx: (wv / (2.0 * np.pi)) for idx, wv in zip(indices, wavevectors)}

        return {
            "abs_guesses_cycles": abs_guesses_cycles,
            "C0": {},
            "S0": {},
            "dq0": {},
            "q0": {},
        }

    def _init_peak_state(self, stack, r, peak, peak_guesses, global_state):
        return {
            "q": np.array(global_state["abs_guesses_cycles"][peak], dtype=np.float64),
            "_first_iter": True,
            "_global_state": global_state,
        }

    def _iterate_one(self, stack, r, peak, q_coords, peak_state, zooming_factor, debug_info_level):
        k_rad = (2.0 * np.pi) * peak_state["q"]
        phase = np.einsum("...l,l->...", self.optical_system.x_grid, k_rad)
        estimated_pattern_refined = {peak: np.exp(1j * phase)}

        correlation = self.cross_correlation_matrix(
            self.optical_system,
            self.illumination,
            images=stack,
            r=r,
            estimated_modulation_patterns=estimated_pattern_refined,
            q_coords_cycles=q_coords,
        )

        merit = None
        for arr in correlation.values():
            magnitude_sq = np.abs(arr) ** 2
            merit = magnitude_sq if merit is None else (merit + magnitude_sq)

        dq = self._argmax_coords(merit, q_coords)
        peak_state["q"] = peak_state["q"] - dq

        if peak_state.get("_first_iter", False):
            global_state = peak_state["_global_state"]
            global_state["S0"][peak] = merit
            global_state["dq0"][peak] = dq
            global_state["q0"][peak] = q_coords
            peak_state["_first_iter"] = False

        if debug_info_level >= 3:
            peak_state["_dbg_img"] = merit
            peak_state["_dbg_qmax"] = dq
            peak_state["_dbg_q_coords"] = q_coords

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
        keys, sigs = [], []
        mt = int(illumination.spatial_shifts.shape[1])

        for m, pat in estimated_modulation_patterns.items():
            if m[0] != r:
                continue
            for n1 in range(mt):
                for n2 in range(mt):
                    sig = images[r, n1] * images[r, n2] * pat
                    if n1 == n2:
                        sig = sig - (images[r, n1] * pat)
                    keys.append((m, n1, n2))
                    sigs.append(sig)

        sig_stack = np.stack(sigs, axis=0)

        x_coords = optical_system.psf_coordinates
        q_coords_rad = tuple((2.0 * np.pi) * np.asarray(qc) for qc in q_coords_cycles)
        dimensionality = optical_system.dimensionality
        spatial_axes = tuple(range(sig_stack.ndim - dimensionality, sig_stack.ndim))

        out = hpc_utils.czt_nd_fourier(
            np.asarray(sig_stack),
            x_coords,
            q_coords_rad,
            axes=spatial_axes,
            rtol=1e-6,
            atol=1e-12,
        )
        out_np = hpc_utils._to_numpy(out)

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
        keys, sigs = [], []
        mt = int(illumination.spatial_shifts.shape[1])

        for m, pat in estimated_modulation_patterns.items():
            if m[0] != r:
                continue
            for n in range(mt):
                sig = (images[r, n] * images[r, n] - 1) * pat
                sig = sig - (images[r, n] * pat)
                keys.append((m, n))
                sigs.append(sig)

        sig_stack = np.stack(sigs, axis=0)

        x_coords = optical_system.psf_coordinates
        q_coords_rad = tuple((2.0 * np.pi) * np.asarray(qc) for qc in q_coords_cycles)
        dimensionality = optical_system.dimensionality
        spatial_axes = tuple(range(sig_stack.ndim - dimensionality, sig_stack.ndim))

        out = hpc_utils.czt_nd_fourier(
            np.asarray(sig_stack),
            x_coords,
            q_coords_rad,
            axes=spatial_axes,
            rtol=1e-6,
            atol=1e-12,
        )
        out_np = hpc_utils._to_numpy(out)

        return {keys[i]: out_np[i] for i in range(len(keys))}


class PeaksEstimatorCrossCorrelation2D(PeaksEstimatorCrossCorrelation):
    dimensionality = 2


class PeaksEstimatorCrossCorrelation3D(PeaksEstimatorInterpolation):
    dimensionality = 3


class PeaksEstimatorInterpolation2D(PeaksEstimatorInterpolation):
    dimensionality = 2

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

        q0 = np.array(
            [
                (0.0, 0.0),
                (-dk, 0.0), (dk, 0.0),
                (0.0, -dk), (0.0, dk),
                (-dk, -dk), (-dk, dk),
                (dk, -dk), (dk, dk),
            ],
            dtype=np.float64,
        )

        intensity = self._sample_stack_mean_abs(stack[0], q0)
        otf = self._sample_otf_abs(q0)

        f = intensity / otf
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
        inv_hessian = np.linalg.inv(f_hessian)

        corrected = {}
        for key, peak in estimated_peaks.items():
            otf_value = otf0[key]
            gradient = otf_grad[key]
            delta_k = (
                f[0] / otf_value
                * (1.0 + f[0] / (2.0 * otf_value**2) * (gradient.T @ inv_hessian @ gradient))
                * (inv_hessian @ gradient)
            )
            if self.debug_info_level >= 1:
                print("correction", delta_k)
            corrected[key] = peak + delta_k

        return corrected


class PeaksEstimatorInterpolation3D(PeaksEstimatorInterpolation):
    dimensionality = 3

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

        q0 = np.array(
            [
                (0.0, 0.0, 0.0),
                (-dk, 0.0, 0.0), (dk, 0.0, 0.0),
                (0.0, -dk, 0.0), (0.0, dk, 0.0),
                (0.0, 0.0, -dk), (0.0, 0.0, dk),
                (-dk, -dk, 0.0), (-dk, dk, 0.0),
                (dk, -dk, 0.0), (dk, dk, 0.0),
                (0.0, -dk, -dk), (0.0, -dk, dk),
                (0.0, dk, -dk), (0.0, dk, dk),
                (-dk, 0.0, -dk), (dk, 0.0, -dk),
                (-dk, 0.0, dk), (dk, 0.0, dk),
            ],
            dtype=np.float64,
        )

        intensity = self._sample_stack_mean_abs(stack[0], q0)
        otf = self._sample_otf_abs(q0)

        f = intensity / otf
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
        inv_hessian = np.linalg.inv(f_hessian)

        corrected = {}
        for key, peak in estimated_peaks.items():
            otf_value = otf0[key]
            gradient = otf_grad[key]
            delta_k = (
                f[0] / otf_value
                * (1.0 + f[0] / (2.0 * otf_value**2) * (gradient.T @ inv_hessian @ gradient))
                * (inv_hessian @ gradient)
            )
            if self.debug_info_level >= 1:
                print("correction", delta_k)
            corrected[key] = peak + delta_k

        return corrected


__all__ = [
    'PeaksEstimator',
    'PeaksEstimatorIterative',
    'PeaksEstimatorInterpolation',
    'PeaksEstimatorCrossCorrelation',
    'PeaksEstimatorCrossCorrelation2D',
    'PeaksEstimatorCrossCorrelation3D',
    'PeaksEstimatorInterpolation2D',
    'PeaksEstimatorInterpolation3D',
]