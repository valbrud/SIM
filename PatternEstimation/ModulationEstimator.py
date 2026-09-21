from __future__ import annotations

"""
ModulationEstimator.py

This module contains modulation-amplitude estimators used by SIM illumination-pattern
estimation.

Responsibilities:
	ModulationEstimator - Shared modulation-estimator configuration and order preparation.
	ModulationLeastSquaresEstimator - Least-squares modulation fitting.
	ModulationPeakHeightRatioEstimator - Modulation estimation from Fourier peak-height ratios.
	ModulationRatiosInversionEstimator - Modulation estimation by ratio inversion.
"""

import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import numpy as np
import scipy

import Reconstructor
import hpc_utils
import utils
from Illumination import PlaneWavesSIM
from OpticalSystems import OpticalSystem
from utils import off_grid_ft


class ModulationEstimator:
	def __init__(self,
				 illumination: PlaneWavesSIM,
				 optical_system: OpticalSystem,
				 modulation_estimation_method: str = 'least_squares',
				 effective_otfs: dict = None,
				 treat_composite_orders_separately: bool = False,
				 debug_info_level: int = 1):
		utils.validate_init_types(
			illumination=(illumination, PlaneWavesSIM),
			optical_system=(optical_system, OpticalSystem),
		)
		self.optical_system = optical_system
		self.modulation_estimation_method = modulation_estimation_method
		self.debug_info_level = debug_info_level

		if effective_otfs is None:
			for key in illumination.harmonics:
				illumination.harmonics[key].amplitude = 1.0
			_, effective_otfs = illumination.compute_effective_kernels(optical_system.psf, optical_system.psf_coordinates)

		if treat_composite_orders_separately or illumination.dimensionality == 2:
			self.illumination = illumination
		else:
			self.illumination = illumination.project_in_quasi_2D()

		self.effective_otfs = effective_otfs

		if self.illumination.dimensionality != self.optical_system.dimensionality:
			raise ValueError(
				f"Illumination and optical system dimensionality do not match: "
				f"{self.illumination.dimensionality} != {self.optical_system.dimensionality}"
			)

	@staticmethod
	def _as_complex_order(order):
		if np.iscomplexobj(order):
			return order
		if isinstance(order, (tuple, list)) and len(order) == 2:
			return order[0] + 1j * order[1]
		if isinstance(order, np.ndarray) and order.ndim > 0 and order.shape[0] == 2 and not np.iscomplexobj(order):
			return order[0] + 1j * order[1]
		return np.asarray(order, dtype=np.complex128)

	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None) -> dict:
		raise NotImplementedError

	def _compute_spatial_frequency_orders(self, stack: np.ndarray | None, spatial_frequency_orders: dict | None) -> dict:
		if spatial_frequency_orders is not None:
			return spatial_frequency_orders
		if stack is None:
			return {}
		reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if self.illumination.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D
		reconstructor = reconstructor_class(self.illumination, self.optical_system, return_ft=True, unitary=False)
		return reconstructor.compute_spatial_frequency_orders(stack)


class ModulationLeastSquaresEstimator(ModulationEstimator):
	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None) -> dict:
		spatial_frequency_orders = self._compute_spatial_frequency_orders(stack, spatial_frequency_orders)
		am = {}
		a0 = 1
		for r in range(self.illumination.Mr):
			key_zero = (r, tuple([0] * self.illumination.dimensionality))
			am[key_zero] = 1.0 + 0.0j
			I0 = spatial_frequency_orders[key_zero]
			g0 = self.effective_otfs[key_zero]

			if self.debug_info_level >= 2 and self.illumination.dimensionality == 3:
				plt.imshow(hpc_utils.wrapped_ifftn(I0)[..., I0.shape[2] // 2].real, cmap='gray')
				plt.show()

			for key in spatial_frequency_orders.keys():
				if key[0] != r or key == key_zero:
					continue
				I = spatial_frequency_orders[key]
				g = self.effective_otfs[key]
				I0f = np.copy(I0)
				f1 = g * I0f
				f2 = g0 * I

				if self.debug_info_level >= 2 and self.illumination.dimensionality == 3:
					fig, ax = plt.subplots(1, 2)
					ax[0].imshow(hpc_utils.wrapped_ifftn(f1)[..., f1.shape[2] // 2].real, cmap='gray')
					ax[1].imshow(hpc_utils.wrapped_ifftn(f2)[..., f2.shape[2] // 2].real, cmap='gray')
					plt.show()

				if self.debug_info_level >= 1:
					fig, ax = plt.subplots(2, 2)
					if self.illumination.dimensionality == 3:
						ax[0, 0].imshow(np.log1p(np.abs(10**5 * I0f[:, :, I.shape[2] // 2])), cmap='gray')
						ax[0, 1].imshow(np.log1p(np.abs(10**5 * I[:, :, I.shape[2] // 2])), cmap='gray')
						ax[1, 0].imshow(np.log1p(np.abs(10**5 * g0[:, g.shape[1] // 2, :])), cmap='gray')
						ax[1, 1].imshow(np.log1p(np.abs(10**5 * g[:, g.shape[1] // 2, :])), cmap='gray')
					else:
						ax[0, 0].imshow(np.log1p(np.abs(10**2 * I0f)), cmap='gray')
						ax[0, 1].imshow(np.log1p(np.abs(10**2 * I)), cmap='gray')
						ax[1, 0].imshow(np.log1p(np.abs(10**2 * g0)), cmap='gray')
						ax[1, 1].imshow(np.log1p(np.abs(10**2 * g)), cmap='gray')
					for axis in ax.flat:
						axis.axis('off')
						axis.set_aspect('equal')
					plt.tight_layout()
					plt.show()

				def obj(x):
					a = complex(float(x[0]), float(x[1]))
					diff = a0 * g0 * I - a * g * I0f
					return float(np.sum((g0 + g) * np.abs(diff) ** 2))

				def grad(x):
					diff = a0 * g0 * I - complex(float(x[0]), float(x[1])) * g * I0f
					grad_re = -2.0 * np.real(np.sum(np.conj(g) * I0f * diff))
					grad_im = -2.0 * np.imag(np.sum(np.conj(g) * I0f * diff))
					return np.array([grad_re, grad_im], dtype=np.float64)

				optimizer = scipy.optimize.minimize(
					obj,
					x0=[1.0, 0.0],
					method="L-BFGS-B",
					bounds=[(-5, 5), (-5, 5)],
					options={"disp": False},
					jac=grad,
				)
				a = complex(float(optimizer.x[0]), float(optimizer.x[1]))
				am[key] = a
				diff = a0 * g0 * I - a * g * I0f
				if self.debug_info_level >= 1:
					if self.illumination.dimensionality == 3:
						plt.imshow(np.log1p(np.abs(10**5 * diff[:, :, diff.shape[2] // 2])), cmap='gray')
					else:
						plt.imshow(np.log1p(np.abs(10**2 * diff)), cmap='gray')
					plt.title(f"Residual for key {key} with a={a}")
					plt.axis('off')
					plt.show()
		return am


class ModulationPeakHeightRatioEstimator(ModulationEstimator):
	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None) -> dict:
		del spatial_frequency_orders
		grid = self.optical_system.x_grid
		am = {}
		ft0 = off_grid_ft(stack[0, 0], grid, (np.array([0.] * self.illumination.dimensionality)), )

		for r in range(self.illumination.Mr):
			harmonics = {index[1]: harmonic for index, harmonic in self.illumination.harmonics.items() if index[0] == r}
			for harmonic in harmonics.keys():
				if np.isclose(np.sum(np.abs(np.array(harmonic))), 0):
					am[(r, harmonic)] = 1.0 / self.illumination.Mt / self.illumination.Mr
					continue
				wavevector = self.illumination.harmonics[(r, harmonic)].wavevector / (2 * np.pi)
				otf = self.effective_otfs[(r, harmonic)]
				otf_value = otf[tuple([s // 2 for s in otf.shape])]
				ft = off_grid_ft(stack[r, 0], grid, np.array((wavevector, )))
				amplitude = np.abs(ft)[0] / ft0 / np.abs(otf_value)
				if self.debug_info_level:
					print(harmonic, amplitude)
				amplitude /= (self.illumination.Mt * self.illumination.Mr)
				am[(r, harmonic)] = amplitude
		return am


class ModulationRatiosInversionEstimator(ModulationEstimator):
	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None) -> dict:
		spatial_frequency_orders = self._compute_spatial_frequency_orders(stack, spatial_frequency_orders)
		modulations = {}
		canonical_half = utils.select_up_to_inversion(spatial_frequency_orders.keys())

		for r in range(self.illumination.Mr):
			m0 = tuple([0] * self.illumination.dimensionality)
			order0 = self._as_complex_order(spatial_frequency_orders[(r, m0)])
			f0 = np.abs(np.amax(np.abs(order0)))

			for key in canonical_half:
				if key[0] != r or key == (r, m0):
					continue

				order = self._as_complex_order(spatial_frequency_orders[key])
				mask = (self.effective_otfs[key] > 10**-6) * (self.effective_otfs[(r, m0)] > 10**-6)
				ratio = np.where(mask, order / order0 + 1e-12, 0)
				mask2 = (np.abs(order) > 2 * f0**0.5) * (np.abs(order0) > 2 * f0**0.5)
				mask_total = mask * mask2
				ratio = np.where(mask2, ratio, 0)

				wavevector = self.illumination.harmonics[key].wavevector / 2
				phase_modulation_pattern = np.einsum('...l, l->...', self.optical_system.x_grid, wavevector)
				mask_shifted = hpc_utils.wrapped_fftn(hpc_utils.wrapped_ifftn(mask_total) * np.exp(-1j * phase_modulation_pattern))
				ratio_shifted_ft = hpc_utils.wrapped_fftn(hpc_utils.wrapped_ifftn(ratio) * np.exp(-1j * phase_modulation_pattern))
				ratio_shifted_ft = np.where(mask_shifted > 1e-1, ratio_shifted_ft, 0)
				inverse_product = ratio_shifted_ft * np.flip(ratio_shifted_ft)
				modulation_value = np.mean(inverse_product[inverse_product > 10**-6])**0.5
				if np.isnan(modulation_value):
					modulation_value = 0.0
				modulations[key] = modulation_value
				inverse_key = (key[0], tuple(-np.array(key[1])))
				if inverse_key in self.illumination.harmonics:
					modulations[inverse_key] = modulation_value
		return modulations


__all__ = [
	'ModulationEstimator',
	'ModulationLeastSquaresEstimator',
	'ModulationPeakHeightRatioEstimator',
	'ModulationRatiosInversionEstimator',
]