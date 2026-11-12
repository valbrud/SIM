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
				 effective_otfs: dict = None,
				 treat_composite_orders_separately: bool = False,
				 SSNRc: float = 1, 
				 debug_info_level: int = 1):
		utils.validate_init_types(
			illumination=(illumination, PlaneWavesSIM),
			optical_system=(optical_system, OpticalSystem),
		)
		self.optical_system = optical_system
		self.debug_info_level = debug_info_level
		self.SSNRc = SSNRc

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

	def _compute_spatial_frequency_orders(self, stack: np.ndarray | None) -> dict:
		if stack is None:
			return {}
		reconstructor_class = Reconstructor.ReconstructorFourierDomain2D if self.illumination.dimensionality == 2 else Reconstructor.ReconstructorFourierDomain3D
		reconstructor = reconstructor_class(self.illumination, self.optical_system, return_ft=True, unitary=False)
		return reconstructor.compute_spatial_frequency_orders(stack)


class ModulationLeastSquaresEstimator(ModulationEstimator):
	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None) -> dict:
		if spatial_frequency_orders is None:
			spatial_frequency_orders = self._compute_spatial_frequency_orders(stack)
		am = {}
		a0 = 1
		for r in range(self.illumination.Mr):
			key_zero = (r, tuple([0] * self.illumination.dimensionality))
			am[key_zero] = 1.0 + 0.0j
			I0 = spatial_frequency_orders[key_zero]
			# im = plt.imshow(np.log1p(np.abs(I0[..., I0.shape[2] // 2].real)), cmap='gray')
			# plt.colorbar(im)
			# plt.title('I0 (central slice)')
			# plt.show()
			g0 = self.effective_otfs[key_zero]
			# print('Max values of I0 and g0 for key', key_zero, ':', np.amax(I0), np.amax(g0))
			if self.debug_info_level >= 2 and self.illumination.dimensionality == 3:
				fig, ax = plt.subplots(1, 2)
				ax[0].imshow(hpc_utils.wrapped_ifftn(I0)[..., I0.shape[2] // 2].real, cmap='gray')
				ax[0].set_title('I0 (central slice)')
				ax[1].imshow(I0[..., I0.shape[2] // 2].real, cmap='gray')
				ax[1].set_title('I0 ft (central slice)')
				plt.show()

			for key in spatial_frequency_orders.keys():
				if key[0] != r or key == key_zero:
					continue
				I = spatial_frequency_orders[key]
				g = self.effective_otfs[key]
				I0f = np.copy(I0)
				f1 = g * I0f
				f2 = g0 * I
				print('Max values of f1 and f2 for key', key, ':', np.amax(f1), np.amax(f2))

				if self.debug_info_level >= 2 and self.illumination.dimensionality == 3:
					fig, ax = plt.subplots(1, 2)
					ax[0].imshow(hpc_utils.wrapped_ifftn(f1)[..., f1.shape[2] // 2].real, cmap='gray')
					ax[1].imshow(hpc_utils.wrapped_ifftn(f2)[..., f2.shape[2] // 2].real, cmap='gray')
					ax[0].set_title('g I0')
					ax[1].set_title('g0 I')
					plt.show()

				if self.debug_info_level >= 3:
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

				mask = g0 * g > 1e-8
				template = (g * I0f)[mask]
				target = (a0 * g0 * I)[mask]
				denominator = np.vdot(template, template).real
				a = (
					1.0 + 0.0j
					if denominator == 0
					else np.vdot(template, target) / denominator
				)
				am[key] = a
				diff = a0 * g0 * I - a * g * I0f
				if self.debug_info_level >= 2:
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
				if self.debug_info_level >= 2:
					print(f"ft for key {harmonic}: {ft}")
					print(f"ft0: {ft0}")
					print(f"otf_value: {otf_value}")

				if self.debug_info_level >= 3: 
					extent = [self.optical_system.q_grid[..., 0].min(), self.optical_system.q_grid[..., 0].max(),
							  self.optical_system.q_grid[..., 1].min(), self.optical_system.q_grid[..., 1].max()]
					plt.imshow(np.log1p(10**4 * np.abs(otf[..., otf.shape[-1] // 2].real)), extent=extent, cmap='gray', origin='lower')
					plt.plot(0., 0., 'r+', markersize=10)
					plt.title(f"OTF for key {harmonic}")
					plt.axis('off')
					plt.show()

				amplitude = np.abs(ft)[0] / np.abs(ft0) / np.abs(otf_value)
				if self.debug_info_level:
					print(harmonic, amplitude)
				amplitude /= (self.illumination.Mt * self.illumination.Mr)
				am[(r, harmonic)] = amplitude
		return am


class ModulationRatiosInversionEstimator(ModulationEstimator):
	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None) -> dict:
		if spatial_frequency_orders is None:
			spatial_frequency_orders = self._compute_spatial_frequency_orders(stack)
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
				mask2 = (np.abs(order) > self.SSNRc * f0**0.5) * (np.abs(order0) > self.SSNRc * f0**0.5)
				mask_total = mask * mask2
				ratio = np.where(mask2, ratio, 0)

				wavevector = self.illumination.harmonics[key].wavevector / 2
				phase_modulation_pattern = np.einsum('...l, l->...', self.optical_system.x_grid, wavevector)
				mask_shifted = hpc_utils.wrapped_fftn(hpc_utils.wrapped_ifftn(mask_total) * np.exp(-1j * phase_modulation_pattern))
				ratio_shifted_ft = hpc_utils.wrapped_fftn(hpc_utils.wrapped_ifftn(ratio) * np.exp(-1j * phase_modulation_pattern))
				# plt.imshow(mask_shifted.real, origin='lower')
				# plt.colorbar()
				# plt.show()
				ratio_shifted_ft = np.where(np.abs(mask_shifted) > 1e-1, ratio_shifted_ft, 0)
				inverse_product = ratio_shifted_ft * np.flip(ratio_shifted_ft)
				inverse_product_filtered = np.where(np.abs(mask_shifted) > 1e-1, inverse_product, 0)
				# fig, ax = plt.subplots()

				print(np.sum(np.abs(inverse_product[np.abs(inverse_product) > 10**-3])), len(inverse_product[np.abs(inverse_product) > 10**-3]), np.mean(np.abs(inverse_product[np.abs(inverse_product) > 10**-3])))
				if self.debug_info_level >=2:
					plt.hist(np.abs(inverse_product[np.abs(inverse_product) > 10**-3])**0.5, bins=100)
					plt.show()
					fig, ax = plt.subplots(1, 2)
					ax[1].imshow(np.abs(inverse_product)**0.5, origin='lower')
					ax[1].set_title('Inverse Product')
					plt.colorbar(ax[1].imshow(np.abs(inverse_product)**0.5, origin='lower'))
					ax[0].imshow(np.abs(ratio_shifted_ft), origin='lower')
					ax[0].set_title('Shifted Ratio')
					plt.show()

				modulation_value2 = np.mean(np.abs(inverse_product[np.abs(inverse_product) > 10**-3]))
				modulation_value = modulation_value2**0.5
				print(modulation_value2, modulation_value)
				if np.isnan(modulation_value):
					modulation_value = 0.0
				modulations[key] = modulation_value
				inverse_key = (key[0], tuple(-np.array(key[1])))
				if inverse_key in self.illumination.harmonics:
					modulations[inverse_key] = modulation_value
				if self.debug_info_level:
					print(key, modulation_value)
		return modulations

class ModulationEstimatorNoiseVariance(ModulationEstimator):
	def estimate_modulations(self,
							 stack: np.ndarray | None = None,
							 spatial_frequency_orders: dict | None = None, 
							 edge_lat: int = 30, 
							 edge_ax: int = 5) -> dict:
		if spatial_frequency_orders is None:
			spatial_frequency_orders = self._compute_spatial_frequency_orders(stack)
		am = {}
		dim = stack[0,0].ndim
		for key in spatial_frequency_orders:
			print(key)
			if key[1] == tuple([0] * dim):
				am[key] = 1.0
				continue
			r, m = key
			f0 = np.amax(spatial_frequency_orders[(r, tuple([0]* dim))]).real
			pure_noise_region = spatial_frequency_orders[key]
			if dim == 2: 
				pure_noise_region[edge_lat:-edge_lat, edge_ax:-edge_ax] = 0.
			if dim == 3: 
				pure_noise_region[edge_lat:-edge_lat, edge_lat:-edge_lat, edge_ax:-edge_ax] = 0.
			variance = np.mean(pure_noise_region * np.conj(pure_noise_region))
			print(key, variance)
			a = variance / f0
			am[key] = a
		return am
