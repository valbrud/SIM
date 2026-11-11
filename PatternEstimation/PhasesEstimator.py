from __future__ import annotations

"""
PhasesEstimator.py

This module contains phase-matrix estimators used by SIM illumination-pattern
estimation.

Responsibilities:
    PhasesEstimator - Shared phase-estimator configuration and helpers.
    PhasesAutocorrelationEstimator - Phase estimation from autocorrelation peaks.
    PhasesCrossCorrelationEstimator - Phase estimation from cross-correlation optimization.
    PhasesPeakValueEstimator - Phase estimation from direct Fourier peak values.
"""

import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from abc import abstractmethod

import numpy as np
from scipy.optimize import minimize

import utils
from hpc_utils import wrapped_ifftn
from Illumination import PlaneWavesSIM
from OpticalSystems import OpticalSystem
from PatternEstimation.PeaksEstimator import PeaksEstimatorCrossCorrelation
from utils import off_grid_ft
import Reconstructor
import matplotlib.pyplot as plt

class PhasesEstimator:
    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 first_order_phases: bool = True,
                 phase_estimation_method: str = 'autocorrelation',
                 debug_info_level: int = 1):
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            optical_system=(optical_system, OpticalSystem),
            first_order_phases=(first_order_phases, bool),
        )
        self.illumination = illumination
        self.optical_system = optical_system
        if self.illumination.dimensionality != self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: "
                f"{self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )
        self.first_order_only = first_order_phases
        self.phase_estimation_method = phase_estimation_method
        self.debug_info_level = debug_info_level
        self._m = utils.select_up_to_inversion(self.illumination.harmonics)
        if first_order_phases:
            self._m = self.illumination.filter_first_orders(self._m)

    def estimate_phases(self,
                        stack: np.ndarray | None = None,
                        refined_wavevectors: dict | None = None,
                        spatial_frequency_orders: dict | None = None) -> dict:
        del spatial_frequency_orders
        return self.estimate_phase_matrix(stack, refined_wavevectors)

    @abstractmethod
    def estimate_phase_matrix(self,
                              stack: np.ndarray | None,
                              refined_wavevectors: dict | None) -> dict:
        ...

    def _compute_full_phase_matrix_from_first_order_phases(self, phase_matrix: dict) -> dict:
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
                        phase_matrix[r, n, m] = np.exp(1j * phase)
        return phase_matrix

    @staticmethod
    def _dict_to_tensor(data: dict, *index_grids) -> np.ndarray:
        shape = tuple(len(g) for g in index_grids)
        tensor = np.zeros(shape, dtype=complex)
        for pos in np.ndindex(shape):
            tensor[pos] = data[tuple(g[i] for g, i in zip(index_grids, pos))]
        return tensor

    def _peak_values(self, stack: np.ndarray, refined_wavevectors: dict) -> dict:
        peak_values = {}
        grid = self.optical_system.x_grid
        for n in range(stack.shape[1]):
            for index in self._m:
                r = index[0]
                m = index[1]
                wavevector = refined_wavevectors[index] / (2 * np.pi)
                ft = off_grid_ft(stack[r, n], grid, np.array((wavevector,)))
                peak_values[(r, n, m)] = np.abs(ft)

        return peak_values

    def _to_images(self, peak_values: dict) -> np.ndarray:
        mt = self.illumination.Mt
        mr = self.illumination.Mr
        peak_images = np.zeros((mr, mt, len(self._m)), dtype=np.float64)
        for r in range(mr):
            for n in range(mt):
                for h, m in enumerate(sorted(m for (r_, m) in self._m if r_ == r)):
                    peak_images[r, n, h] = peak_values[(r, n, m)]
        return peak_images

    def _compute_spatial_frequency_orders(self, stack: np.ndarray | None, illumination: PlaneWavesSIM | None) -> dict:
        if stack is None:
            raise ValueError("Empty stack")
        if illumination is None:
            illumination = self.illumination
        reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if illumination.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D
        reconstructor = reconstructor_class(illumination, self.optical_system, return_ft=True, unitary=False)
        return reconstructor.compute_spatial_frequency_orders(stack)

class PhasesAutocorrelationEstimator(PhasesEstimator):
    def estimate_phase_matrix(self,
                              stack: np.ndarray | None,
                              refined_wavevectors: dict | None) -> dict:
        phase_matrix = {}
        zero_m = tuple([0] * self.optical_system.dimensionality)

        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                phase_matrix[(r, n, zero_m)] = 1.0 + 0.0j

        for m_idx in self._m:
            r = m_idx[0]
            single_pattern = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(
                self.optical_system, {m_idx: refined_wavevectors[m_idx]}
            )
            autocorrelation = PeaksEstimatorCrossCorrelation.autocorrelation_matrix(
                self.optical_system,
                self.illumination,
                stack,
                r,
                single_pattern,
                self.optical_system.otf_frequencies,
            )
            sample_arr = next(iter(autocorrelation.values()))
            center = tuple(s // 2 for s in sample_arr.shape)
            for n in range(self.illumination.Mt):
                phase_val = np.angle(autocorrelation[(m_idx, n)][center])
                phase_matrix[(r, n, m_idx[1])] = np.exp(1j * phase_val)
                phase_matrix[(r, n, tuple(-v for v in m_idx[1]))] = np.exp(-1j * phase_val)

        if self.first_order_only:
            phase_matrix = self._compute_full_phase_matrix_from_first_order_phases(phase_matrix)
        return phase_matrix


class PhasesCrossCorrelationEstimator(PhasesEstimator):
    @staticmethod
    def _loss_function(Cmm1m2: np.ndarray, phimn: np.ndarray, weight: np.ndarray) -> float:
        Dmm1m2 = phimn @ Cmm1m2 @ phimn.conj().T
        return float(np.sum(weight * np.abs(Dmm1m2) ** 0.5))

    @staticmethod
    def _order_weight_tensor(shift_orders: list, full_orders: list) -> np.ndarray:
        weight = np.zeros((len(shift_orders), len(full_orders), len(full_orders)))
        for mi, m in enumerate(shift_orders):
            neg_m = tuple(-v for v in m)
            for i1, m1 in enumerate(full_orders):
                for i2, m2 in enumerate(full_orders):
                    diff = tuple(a - b for a, b in zip(m1, m2))
                    if m1 != m2 and diff != m and diff != neg_m:
                        weight[mi, i1, i2] = 1.0
        return weight


    def estimate_phase_matrix(self,
                              stack: np.ndarray | None,
                              refined_wavevectors: dict | None) -> dict:
        dimensionality = self.optical_system.dimensionality
        zero_m = tuple([0] * dimensionality)
        mt = self.illumination.Mt
        non_deg_dims = self.illumination.find_non_degenerate_dimensions()
        phase_matrix = PhasesAutocorrelationEstimator.estimate_phase_matrix(self, stack, refined_wavevectors)

        for r in range(self.illumination.Mr):
            m_r = sorted(m for (r_, m) in self._m if r_ == r)
            if not m_r:
                continue
            shift_orders = m_r + [zero_m]
            harmonics_r = sorted(m for (r_, m) in self.illumination.harmonics if r_ == r)
            x0 = np.array([np.angle(phase_matrix[(r, n, m)]) for n in range(mt) for m in m_r])

            raw = {}
            for m in shift_orders:
                m_idx = (r, m)
                shift_wavevector = refined_wavevectors[m_idx] if m != zero_m else np.zeros(dimensionality)
                pattern_m = PeaksEstimatorCrossCorrelation.phase_modulation_patterns(
                    self.optical_system, {m_idx: shift_wavevector}
                )
                correlation = PeaksEstimatorCrossCorrelation.cross_correlation_matrix(
                    self.optical_system, self.illumination, stack, r, pattern_m, self.optical_system.otf_frequencies,
                )
                sample_arr = next(iter(correlation.values()))
                center = tuple(s // 2 for s in sample_arr.shape)
                for n1 in range(mt):
                    for n2 in range(mt):
                        raw[(m, n1, n2)] = correlation[(m_idx, n1, n2)][center]
            Cmm1m2 = self._dict_to_tensor(raw, shift_orders, list(range(mt)), list(range(mt)))
            weight = self._order_weight_tensor(shift_orders, harmonics_r)
            vec_m = np.array([[m[i] for i in non_deg_dims] for m in m_r])

            def merit(x, m_r=m_r, harmonics_r=harmonics_r, vec_m=vec_m, Cmm1m2=Cmm1m2, weight=weight):
                x = x.reshape(mt, len(m_r))
                partial = {(r, n, zero_m): 1.0 + 0.0j for n in range(mt)}
                if self.first_order_only:
                    for n in range(mt):
                        base_phase = np.linalg.solve(vec_m, x[n])
                        for m in harmonics_r:
                            phase = np.dot([m[i] for i in non_deg_dims], base_phase)
                            partial[(r, n, m)] = np.exp(1j * phase)
                else:
                    for n in range(mt):
                        for h, m in enumerate(m_r):
                            partial[(r, n, m)] = np.exp(1j * x[n, h])
                            partial[(r, n, tuple(-v for v in m))] = np.exp(-1j * x[n, h])
                mixing_matrix = np.array([[partial[(r, n, m)] for m in harmonics_r] for n in range(mt)])
                phimn = np.linalg.pinv(mixing_matrix)
                return self._loss_function(Cmm1m2, phimn, weight)

            f0 = merit(x0)
            result = minimize(
                merit, x0, method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": max(abs(f0), 1.0) * 1e-6, "maxiter": 10000},
            )
            x_final = result.x if result.fun <= f0 else x0
            x_opt = np.mod(x_final, 2 * np.pi).reshape(mt, len(m_r))
            if self.debug_info_level:
                print("f0:", f0, "result.fun:", result.fun, "improved:", result.fun < f0)
                print("max |x_opt - x0|:", np.max(np.abs(np.mod(result.x - x0 + np.pi, 2 * np.pi) - np.pi)))
            if self.debug_info_level >= 2:
                print("x_final:", x_final)
            for n in range(mt):
                phase_matrix[(r, n, zero_m)] = 1.0 + 0.0j
                for h, m in enumerate(m_r):
                    phase_matrix[(r, n, m)] = np.exp(1j * x_opt[n, h])
                    if not self.first_order_only:
                        phase_matrix[(r, n, tuple(-v for v in m))] = np.exp(-1j * x_opt[n, h])

        if self.first_order_only:
            phase_matrix = self._compute_full_phase_matrix_from_first_order_phases(phase_matrix)
        return phase_matrix


class PhasesPeakValueEstimator(PhasesEstimator):
    def estimate_phase_matrix(self,
                              stack: np.ndarray | None,
                              refined_wavevectors: dict | None) -> dict:
        phase_matrix = {}
        grid = self.optical_system.x_grid
        for n in range(stack.shape[1]):
            for index in self._m:
                r = index[0]
                m = index[1]
                wavevector = refined_wavevectors[index] / (2 * np.pi)
                ft = off_grid_ft(stack[r, n], grid, np.array((wavevector,)))
                phase_matrix[(r, n, m)] = np.exp(-1j * np.angle(ft))
                phase_matrix[(r, n, tuple(-v for v in m))] = np.exp(1j * np.angle(ft))
                phase_matrix[(r, n, tuple([0] * self.optical_system.dimensionality))] = 1.0 + 0.0j

        if self.first_order_only:
            phase_matrix = self._compute_full_phase_matrix_from_first_order_phases(phase_matrix)
        return phase_matrix


class PhasesSidePeakSuppressionEstimator(PhasesEstimator):
    @staticmethod
    def _loss_function(spatial_frequency_orders: dict, harmonics: dict, grid: np.ndarray, extent=None) -> float:
        loss = 0.0
        dimensionality = grid.shape[-1]
        zero_m = tuple([0] * dimensionality)
        # qx = grid[: , 0,  0]
        # qy = grid[0, :,  1]
        # print(qx, qy)
        for key_order in harmonics.keys():
            order_ft = spatial_frequency_orders[key_order]
            order_zero_ft = spatial_frequency_orders[(0, zero_m)]
            order = wrapped_ifftn(order_ft)
            wavevector_order = harmonics[key_order].wavevector / (2 * np.pi)
            for key_harmonics in harmonics.keys():
                wavevector_peak = harmonics[key_harmonics].wavevector / (2 * np.pi)
                if np.allclose(wavevector_order, -wavevector_peak):
                    continue
                wavevector_side = wavevector_peak + wavevector_order
                # plt.imshow(np.log1p(10**4 * np.abs(spatial_frequency_orders[key_order])).T, extent=extent, origin='lower')
                # plt.plot(wavevector_order[1], wavevector_order[0], 'rx')
                # plt.show()
                residue = off_grid_ft(order, grid, np.array((wavevector_side,)))
                loss += np.abs(residue) ** 2
                # if key_order == (0, (1, 0)):
                #     plt.imshow(np.log1p(10**4 * np.abs(order_ft)).T, extent=extent, origin='lower')
                #     plt.plot(wavevector_side[0], wavevector_side[1], 'rx')
                #     plt.colorbar()
                #     plt.title(str(key_order) + str(key_harmonics) + f" Residue: {np.abs(residue) ** 2}")
                #     plt.show()
                # print
        return loss

    def estimate_phase_matrix(self,
                              stack: np.ndarray | None,
                              refined_wavevectors: dict | None) -> dict:
        dimensionality = self.optical_system.dimensionality
        zero_m = tuple([0] * dimensionality)
        mt = self.illumination.Mt
        non_deg_dims = self.illumination.find_non_degenerate_dimensions()

        phase_matrix = PhasesAutocorrelationEstimator.estimate_phase_matrix(self, stack, refined_wavevectors)
        # phase_matrix = {}
        # for n in range(mt):
        #     for r, m in self._m:
        #         phase_matrix[(r, n, zero_m)] = 1.0 + 0.0j
        #         phase_matrix[(r, n, m)] = 1.0 + 0.0j
        #         print(f"Setting phase for (r={r}, n={n}, m={m})")
        #         phase_matrix[(r, n, tuple(-v for v in m))] = 1.0 + 0.0j
        self.illumination.phase_matrix = phase_matrix
    
        for r in range(self.illumination.Mr):
            m_r = sorted(m for (r_, m) in self._m if r_ == r)
            if not m_r:
                continue
            shift_orders = m_r + [zero_m]
            harmonics_r = sorted(m for (r_, m) in self.illumination.harmonics if r_ == r)
            harmonics_dict_r = {
                (0, m): self.illumination.harmonics[(r, m)]
                for m in harmonics_r
            }
            for m in harmonics_r:
                harmonics_dict_r[(0, m)].wavevector = refined_wavevectors[(r, m)]

            stack_r = stack[r:r + 1]
            illumination_r = type(self.illumination)(
                harmonics_dict_r,
                self.illumination.dimensions,
                Mr=1,
                angles=(self.illumination.angles[r],),
            )
            x0 = np.array([np.angle(phase_matrix[(r, n, m)]) for n in range(mt) for m in m_r])


            vec_m = np.array([[m[i] for i in non_deg_dims] for m in m_r])

            def merit(x, m_r=m_r, harmonics_r=harmonics_r, vec_m=vec_m, stack_r=stack_r, illumination_r=illumination_r):
                x = x.reshape(mt, len(m_r))
                phase_matrix_r = {(0, n, zero_m): 1.0 + 0.0j for n in range(mt)}
                if self.first_order_only:
                    for n in range(mt):
                        base_phase = np.linalg.solve(vec_m, x[n])
                        for m in harmonics_r:
                            phase = np.dot([m[i] for i in non_deg_dims], base_phase)
                            phase_matrix_r[(0, n, m)] = np.exp(1j * phase)
                else:
                    for n in range(mt):
                        for h, m in enumerate(m_r):
                            phase_matrix_r[(0, n, m)] = np.exp(1j * x[n, h])
                            phase_matrix_r[(0, n, tuple(-v for v in m))] = np.exp(-1j * x[n, h])
                illumination_r.phase_matrix = phase_matrix_r
                spatial_frequency_orders_r = self._compute_spatial_frequency_orders(stack_r, illumination_r)
                q_grid = self.optical_system.q_grid
                extent = [q_grid[0, 0, 0], q_grid[-1, -1, 0], q_grid[0, 0, 1], q_grid[-1, -1, 1]]
                loss= self._loss_function(
                    spatial_frequency_orders_r,
                    illumination_r.harmonics,
                    self.optical_system.x_grid,
                    extent=extent,
                )
                print("loss:", loss)
                return loss
            

            f0 = merit(x0)
            result = minimize(
                merit,
                x0,
                method="BFGS",
                jac="2-point",
                options={"gtol": 1e-5, "maxiter": 50},
            )
            x_final = result.x if result.fun <= f0 else x0
            x_opt = np.mod(x_final, 2 * np.pi).reshape(mt, len(m_r))
            if self.debug_info_level:
                print("f0:", f0, "result.fun:", result.fun, "improved:", result.fun < f0)
                print("max |x_opt - x0|:", np.max(np.abs(np.mod(result.x - x0 + np.pi, 2 * np.pi) - np.pi)))
            if self.debug_info_level >= 2:
                spatial_frequency_orders_r = self._compute_spatial_frequency_orders(stack_r, illumination_r)
                for key, order in spatial_frequency_orders_r.items():
                    plt.imshow(np.log1p(10**4 * np.abs(order)))
                    plt.colorbar()
                    plt.title(f"Spatial frequency order: {key}")
                    plt.show()

            for n in range(mt):
                phase_matrix[(r, n, zero_m)] = 1.0 + 0.0j
                for h, m in enumerate(m_r):
                    phase_matrix[(r, n, m)] = np.exp(1j * x_opt[n, h])
                    if not self.first_order_only:
                        phase_matrix[(r, n, tuple(-v for v in m))] = np.exp(-1j * x_opt[n, h])

        if self.first_order_only:
            phase_matrix = self._compute_full_phase_matrix_from_first_order_phases(phase_matrix)
        return phase_matrix



__all__ = [
    'PhasesEstimator',
    'PhasesAutocorrelationEstimator',
    'PhasesCrossCorrelationEstimator',
    'PhasesPeakValueEstimator',
    'PhasesSidePeakSuppressionEstimator',
]