from __future__ import annotations

"""
PatternEstimator.py

This module defines the facade and orchestration classes for illumination-pattern
estimation from raw SIM stacks.

Responsibilities:
    IlluminationPatternEstimator - Shared constructor configuration and end-to-end estimation flow.
    IlluminationPatternEstimator2D - 2D-specific peak, phase, and modulation orchestration.
    IlluminationPatternEstimator3D - 3D-specific projection and orchestration logic.

Concrete estimator families live in sibling modules:
    PeaksEstimator.py
    PhasesEstimator.py
    ModulationEstimator.py
"""

import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from VectorOperations import VectorOperations

from wrappers import wrapped_fftn, wrapped_ifftn
from Dimensions import DimensionMetaAbstract
from abc import abstractmethod
import utils

from Illumination import (
    PlaneWavesSIM,
    IlluminationPlaneWaves2D,
    IlluminationPlaneWaves3D,
)

from Sources import IntensityHarmonic
from OpticalSystems import OpticalSystem
from PatternEstimation.ModulationEstimator import (
    ModulationEstimator,
    ModulationLeastSquaresEstimator,
    ModulationPeakHeightRatioEstimator,
    ModulationRatiosInversionEstimator,
)
from PatternEstimation.PeaksEstimator import (
    PeaksEstimatorCrossCorrelation2D,
    PeaksEstimatorCrossCorrelation3D,
    PeaksEstimatorInterpolation2D,
    PeaksEstimatorInterpolation3D,
)
from PatternEstimation.PhasesEstimator import (
    PhasesAutocorrelationEstimator,
    PhasesCrossCorrelationEstimator,
    PhasesEstimator,
    PhasesPeakValueEstimator,
    PhasesSidePeakSuppressionEstimator,
)

class IlluminationPatternEstimator(metaclass=DimensionMetaAbstract):
    """
    Class for estimating the illumination parameters from a raw SIM stack.
    """

    dimensionality = None  

    peak_estimation_methods = ('default', 'interpolation', 'cross_correlation')
    phase_estimation_methods = ('default', 'peak_phases', 'autocorrelation', 'cross_correlation', 'side_peak_suppression')
    modulation_coefficients_methods = ('default', 'least_squares', 'peak_height_ratio', 'ratios_inversion')
    debug_info_levels = (0, 1, 2)

    def __init__(self,
                 illumination: PlaneWavesSIM,
                 optical_system: OpticalSystem,
                 peak_estimation_method: str = 'interpolation',
                 phase_estimation_method: str = 'autocorrelation',
                 modulation_coefficients_method: str = 'default',
                 peak_search_area_size: int = 11,
                 zooming_factor: int = 100,
                 max_iterations: int = 2,
                 debug_info_level: int = 1,
                 peak_debug_info_level: int | None = None,
                 phase_debug_info_level: int | None = None,
                 modulation_debug_info_level: int | None = None,
                 correct_peak_position: bool = True,
                 first_order_peaks: bool = True,
                 first_order_phases: bool = True,
                 treat_composite_orders_separately_for_modulation: bool = False,
                 estimation_strategy: str = 'plane_by_plane',
                 reference_expansion: tuple = (1, 0, 1), 
                 SSNRc: float = 1,
                 ):
        utils.validate_init_types(
            illumination=(illumination, PlaneWavesSIM),
            optical_system=(optical_system, OpticalSystem),
        )
        self.illumination = illumination
        self.optical_system = optical_system
        if not self.illumination.dimensionality == self.optical_system.dimensionality:
            raise ValueError(
                f"Illumination and optical system dimensionality do not match: {self.illumination.dimensionality} != {self.optical_system.dimensionality}"
            )
        self.peak_estimation_method = peak_estimation_method
        self.phase_estimation_method = phase_estimation_method
        self.modulation_coefficients_method = modulation_coefficients_method
        self.peak_search_area_size = peak_search_area_size
        self.zooming_factor = zooming_factor
        self.max_iterations = max_iterations
        self.debug_info_level = debug_info_level
        self.peak_debug_info_level = debug_info_level if peak_debug_info_level is None else peak_debug_info_level
        self.phase_debug_info_level = debug_info_level if phase_debug_info_level is None else phase_debug_info_level
        self.modulation_debug_info_level = debug_info_level if modulation_debug_info_level is None else modulation_debug_info_level
        self.correct_peak_position = correct_peak_position
        self.first_order_peaks = first_order_peaks
        self.first_order_phases = first_order_phases
        self.treat_composite_orders_separately_for_modulation = treat_composite_orders_separately_for_modulation
        self.estimation_strategy = estimation_strategy
        self.reference_expansion = reference_expansion
        self.SSNRc = SSNRc

    def estimate_illumination_parameters(
        self,
        stack: np.ndarray,
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

        if not self.peak_estimation_method in self.peak_estimation_methods:
            raise ValueError(
                f"Unknown method of peaks estimation {self.peak_estimation_method}. Available methods are {self.peak_estimation_methods}"
            )
        if not self.phase_estimation_method in self.phase_estimation_methods:
            raise ValueError(
                f"Unknown method of phase estimation {self.phase_estimation_method}. Available methods are {self.phase_estimation_methods}"
            )
        if not self.modulation_coefficients_method in self.modulation_coefficients_methods:
            raise ValueError(
                f"Unknown method of modulation coefficients estimation {self.modulation_coefficients_method}. Available methods are {self.modulation_coefficients_methods}"
            )

        if self.debug_info_level > 2: 
            fig, axes = plt.subplots(Mr, Mt, figsize=(Mt * 3, Mr * 3))
            for r in range(Mr):
                for n in range(Mt):
                    axes[r, n].imshow(stack[r, n], cmap='gray')
                    axes[r, n].set_title(f"r={r}, n={n}")
                    axes[r, n].axis('off')
            plt.tight_layout()
            plt.show()

        # illum = self.illumination if self.illumination.dimensions[]
        mpeaks = utils.select_up_to_inversion(self.illumination.harmonics)
        if self.first_order_peaks:
            mpeaks = self.illumination.filter_first_orders(mpeaks)
        mphases = utils.select_up_to_inversion(self.illumination.rearranged_indices)
        if self.first_order_phases:
            mphases = self.illumination.filter_first_orders(mphases)
        if self.peak_estimation_method == 'default':
            peaks = {harmonic: self.illumination.harmonics[harmonic].wavevector for harmonic in self.illumination.harmonics}
            rotation_angles = self.illumination.angles
        else:
            peaks, rotation_angles = self.estimate_peaks(stack)

        if self.peak_debug_info_level:
            for r in range(Mr):
                print('r = ', r,  'rotation_angle = ', np.round(rotation_angles[r] / np.pi * 180, 2), 'degrees')
            for sim_index in sorted(mpeaks):
                print('r = ', sim_index[0], 'm = ', sim_index[1], 'wavevector = ', np.round((np.linalg.norm(peaks[sim_index])) / (2 * np.pi), 3), '1 / lambda')

        if self.phase_estimation_method == 'default':
            phase_matrix = self.illumination.phase_matrix
        else:
            phase_matrix = self.build_phase_matrix(stack, peaks)

        if self.phase_debug_info_level:
            phases = {sim_index: np.round(np.angle(phase_matrix[sim_index]) / np.pi * 180, 1) for sim_index in phase_matrix}
            for sim_index in sorted(phases):
                if (sim_index[0], sim_index[2]) in mphases:
                    print('r = ', sim_index[0], 'n = ', sim_index[1], 'm = ', sim_index[2], 'phase = ', phases[sim_index], 'degrees')

        illumination_estimated = self.build_illumination_object(rotation_angles, peaks, phase_matrix, self.peak_debug_info_level)
        
        if self.modulation_coefficients_method != 'default':
            am = self.estimate_modulation_coefficients(stack, illumination_estimated)
            illumination_estimated.update_amplitudes(am)
            illumination_estimated.normalize_spatial_waves()

        if self.modulation_debug_info_level:
            amplitudes, indices = illumination_estimated.get_all_amplitudes()
            for amplitude, index in zip(amplitudes, indices):
                print('r = ', index[0], 'm = ', index[1], 'amplitude = ', round(amplitude, 3))
        
        if self.debug_info_level > 2:
            image_ft_reference = wrapped_fftn(stack[0, 0])
            plt.imshow(np.log1p(np.abs(image_ft_reference)).T, cmap='gray', origin='lower')
            for peak in illumination_estimated.harmonics.keys():
                qx, qy = self.optical_system.otf_frequencies
                sizex, sizey = qx.size, qy.size
                approximate_peak = illumination_estimated.harmonics[peak].wavevector / (2 * np.pi)
                plt.plot(approximate_peak[0]/qx[-1] * sizex//2 + sizex//2, approximate_peak[1]/qx[-1] * sizey//2 + sizey//2, 'rx')
            plt.show()

        return illumination_estimated
    
    @abstractmethod
    def estimate_peaks(self, stack: np.ndarray) -> Tuple[np.ndarray, dict]:
        
        """
        Estimate the peaks postitions of the illumination pattern from the stack.
        """

    def _make_phase_estimator(
        self,
        illumination: PlaneWavesSIM,
        optical_system: OpticalSystem,
        method_name: str,
        first_order_phases: bool = True,
        debug_info_level: int = 1,
    ) -> 'PhasesEstimator':
        """
        Instantiate the correct PhasesEstimator subclass for the given
        illumination / optical_system pair and return the bound method
        that corresponds to *method_name*.

        The caller receives a plain callable ``fn(stack, refined_wavevectors)``
        and never has to pass illumination or optical_system explicitly.
        """
        estimator_classes = {
            'peak_phases': PhasesPeakValueEstimator,
            'autocorrelation': PhasesAutocorrelationEstimator,
            'cross_correlation': PhasesCrossCorrelationEstimator,
            'side_peak_suppression': PhasesSidePeakSuppressionEstimator,
        }
        estimator_class = estimator_classes.get(method_name, PhasesAutocorrelationEstimator)
        return estimator_class(
            illumination,
            optical_system,
            first_order_phases=first_order_phases,
            phase_estimation_method=method_name,
            debug_info_level=debug_info_level,
        )

    def _make_modulation_estimator(
        self,
        illumination: PlaneWavesSIM,
        optical_system: OpticalSystem,
        method_name: str,
        treat_composite_orders_separately: bool = False,
        debug_info_level: int = 1,
        SSNRc: float = 1,
    ) -> 'ModulationEstimator':
        estimator_names = {
            'least_squares': 'ModulationLeastSquaresEstimator',
            'peak_height_ratio': 'ModulationPeakHeightRatioEstimator',
            'ratios_inversion': 'ModulationRatiosInversionEstimator',
        }
        estimator_class_name = estimator_names.get(method_name, 'ModulationEstimator')
        estimator_class = globals().get(estimator_class_name)
        return estimator_class(
            illumination,
            optical_system,
            modulation_estimation_method=method_name,
            treat_composite_orders_separately=treat_composite_orders_separately,
            debug_info_level=debug_info_level,
            SSNRc=self.SSNRc,
        )

    @abstractmethod
    def build_phase_matrix(self,
                           stack: np.ndarray,
                           peaks: dict) -> dict:
        """
        Build the phase matrix from the image stack and estimated peak positions.
        """

         
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
        illumination_estimated.source_electromagnetic_plane_waves = self.illumination.source_electromagnetic_plane_waves
        illumination_estimated.normalize_spatial_waves()
        for plane_wave in illumination_estimated.source_electromagnetic_plane_waves:
            scaling_factor = np.sum(self.illumination.get_base_vectors(0)**2)**0.5 / np.sum(illumination_estimated.get_base_vectors(0)**2)**0.5
            if debug_info_level > 0:
                print('scaling factor', scaling_factor)
            plane_wave.wavevector *= scaling_factor

        return illumination_estimated

    @abstractmethod
    def estimate_modulation_coefficients(self,
                                         stack: np.ndarray,
                                         illumination_raw: PlaneWavesSIM,
                                         ) -> dict:
        """
        Estimate modulation coefficients a_m from the image stack and estimated illumination parameters.
        """
        pass
    # def compute_ssnr(self, stack: np.ndarray, iterations: int) -> np.ndarray:
    #     """
    #     Compute the SSNR directly from the image stack.
    #     """

    #     ssnr_estimated = SSNRBase.estimate_ssnr_from_image_binomial_splitting(stack, n_iter=iterations, radial=False)
    #     ssnr_estimated[ssnr_estimated < 0] = 0
    #     # plt.imshow(np.log1p(ssnr_estimated), cmap='gray')
    #     # plt.show()
    #     # plt.plot(self.optical_system.otf_frequencies[0], ssnr_estimated[:, 50], label='ssnr_estimated')
    #     # plt.ylim(0, 100)
    #     # plt.plot(2.45, 0, 'cx')
    #     # plt.show()
    #     print('ssnr_estimated', ssnr_estimated[50, 50])
    #     # exit()
    #     return ssnr_estimated

class IlluminationPatternEstimator2D(IlluminationPatternEstimator):
    """
    Class for estimating the illumination parameters from a raw SIM stack in 2D.
    """
    dimensionality = 2

    def estimate_peaks(self, stack: np.ndarray) -> Tuple[np.ndarray, dict]:

        if self.peak_estimation_method == 'cross_correlation':
            PeaksEstimator = PeaksEstimatorCrossCorrelation2D
            # stack = [[utils.upsample(image) for image in one_rotation] for one_rotation in stack]
        elif self.peak_estimation_method == 'interpolation':
            PeaksEstimator = PeaksEstimatorInterpolation2D
        else:
            raise ValueError(f"Unknown peaks estimation method: {self.peak_estimation_method}")

        peak_estimator = PeaksEstimator(
            self.illumination,
            self.optical_system,
            first_order_peaks=self.first_order_peaks,
            peak_search_area_size=self.peak_search_area_size,
            zooming_factor=self.zooming_factor,
            max_iterations=self.max_iterations,
            debug_info_level=self.peak_debug_info_level,
            correct_peak_position=self.correct_peak_position,
        )
        peaks, rotation_angles = peak_estimator.estimate_peaks(stack)
        return peaks, rotation_angles

    def build_phase_matrix(self,
                           stack: np.ndarray,
                           peaks: dict) -> dict:
        """
        Build the phase matrix from the image stack and estimated peak positions.
        Direct 2-D call: the whole stack is a single 2-D image per (r, n).
        """
        phase_estimator = self._make_phase_estimator(
            self.illumination,
            self.optical_system,
            self.phase_estimation_method,
            self.first_order_phases,
            self.phase_debug_info_level,
        )
        return phase_estimator.estimate_phases(stack, peaks)

    def estimate_modulation_coefficients(self,
                                        stack: np.ndarray,
                                        illumination_raw: IlluminationPlaneWaves2D,
                                        ) -> dict:
            """
            Estimate modulation coefficients a_m from the image stack and estimated illumination parameters.
            """
            modulation_estimator = self._make_modulation_estimator(
                illumination_raw,
                self.optical_system,
                self.modulation_coefficients_method,
                treat_composite_orders_separately=self.treat_composite_orders_separately_for_modulation,
                debug_info_level=self.modulation_debug_info_level,
            )
            return modulation_estimator.estimate_modulations(stack=stack)
            
class IlluminationPatternEstimator3D(IlluminationPatternEstimator):
    """
    Class for estimating the illumination parameters from a raw SIM stack in 3D.
    """
    dimensionality = 3

    estimation_strategies = ('plane_by_plane', 'true_3D')

    def estimate_peaks(self, stack: np.ndarray) -> Tuple[np.ndarray, dict]:

        if not self.estimation_strategy in self.estimation_strategies:
            raise ValueError(f"Unknown estimation strategy: {self.estimation_strategy}")

        if self.peak_estimation_method == 'cross_correlation':
            PeaksEstimator = PeaksEstimatorCrossCorrelation2D if self.estimation_strategy == 'plane_by_plane' else PeaksEstimatorCrossCorrelation3D
        elif self.peak_estimation_method == 'interpolation':
            PeaksEstimator = PeaksEstimatorInterpolation2D if self.estimation_strategy == 'plane_by_plane' else PeaksEstimatorInterpolation3D
        else:
            raise ValueError(f"Unknown peaks estimation method: {self.peak_estimation_method}")

        if PeaksEstimator.dimensionality == 3:
            peak_estimator = PeaksEstimator(
                self.illumination,
                self.optical_system,
                first_order_peaks=self.first_order_peaks,
                peak_search_area_size=self.peak_search_area_size,
                zooming_factor=self.zooming_factor,
                max_iterations=self.max_iterations,
                debug_info_level=self.peak_debug_info_level,
                correct_peak_position=self.correct_peak_position,
            )
        else: 
            illumination2d = IlluminationPlaneWaves2D.init_from_3D(self.illumination, force=True)
            optical_system2d = self.optical_system.project_in_2D()
            peak_estimator = PeaksEstimator(
                illumination2d,
                optical_system2d,
                first_order_peaks=self.first_order_peaks,
                peak_search_area_size=self.peak_search_area_size,
                zooming_factor=self.zooming_factor,
                max_iterations=self.max_iterations,
                debug_info_level=self.peak_debug_info_level,
                correct_peak_position=self.correct_peak_position,
            )
        
        if self.estimation_strategy == 'true_3D':
            peaks, rotation_angles = peak_estimator.estimate_peaks(stack)

        else:
            if self.peak_debug_info_level:
                print('3D volume split into planes. Averaging FT magnitudes or complex FTs along the z-axis')

            projected_stack = self._project_stack(stack)

            peaks2d, rotation_angles = peak_estimator.estimate_peaks(projected_stack)
            if self.peak_debug_info_level >= 2:
                print('Projected 2D peaks', peaks2d)
                print('Projected rotation angles', rotation_angles)
            for peak in peaks2d: 
                illumination2d.harmonics[peak].wavevector = peaks2d[peak]
            illumination2d.angles = rotation_angles
            peaks = self.illumination3d_from_illuminaton2d_and_total_wavelength(
                illumination2d,
                reference_expansion=self.reference_expansion,
                debug_info_level=self.peak_debug_info_level,
            )
        return peaks, rotation_angles

    def _project_stack(self, stack: np.ndarray) -> np.ndarray:
        """
        Projects a 3D raw stack (Mr, Mt, Nx, Ny, Nz) to a 2D stack (Mr, Mt, Nx, Ny) 
        by summing the Fourier transforms along the z-axis.

        If dimensions[2] == 1, sum the absolute values of the FTs.
        If dimensions[2] == 0, sum the complex FTs directly.
        Returns the IFFT of the sum, which is fed into the 2D estimator.
        """
        n_planes = stack.shape[4]
        
        if not self.illumination.dimensions[2]:
            return np.sum(stack, axis=4)
        else:
            projected_stack_ft = np.zeros_like(wrapped_fftn(stack[..., 0]), dtype=complex)
            for z in range(n_planes):
                substack_ft = np.abs(wrapped_fftn(stack[..., z]))
                projected_stack_ft += substack_ft
            return wrapped_ifftn(projected_stack_ft).real

    def illumination3d_from_illuminaton2d_and_total_wavelength(self,
                                                               illumination2d: IlluminationPlaneWaves2D,
                                                               reference_expansion: tuple = (1, 0, 1), 
                                                               debug_info_level: int = 0) -> IlluminationPlaneWaves3D:
        """
        Assuming we found 2d illumination pattern, we can find 3d illumination pattern if we know the total wavelength, 
        which is typically the case. It is assumed for now, that the missing direction is z-direction, in the same fashion as it is assumed, 
        that we rotate the illumination pattern around the z-axis.
        """
        # target_dimensions = target_illumination_3d.dimensions
        # target_illumination_3d.dimensions = tuple([1 if i != missing_dimension else 0 for i in range(3)])
        k_vector_length = 2 * np.pi * self.optical_system.nm
        peaks3d = {}
        base_vectors3d = np.zeros((illumination2d.Mr, 3))
        for r in range(illumination2d.Mr):
            base_vectors2d = illumination2d.get_base_vectors(r)
            missing_base = (k_vector_length **2 - np.sum(base_vectors2d**2 * np.array(reference_expansion[:2])**2))**0.5 / reference_expansion[2]
            missing_base -= k_vector_length * reference_expansion[2]  # subtract the reference expansion to get the missing base vector
            base_vectors3d[r, :2] = base_vectors2d
            base_vectors3d[r, 2] = missing_base
            if debug_info_level > 0:
                print("r", r, "missing_base length", missing_base)

        for key in self.illumination.harmonics.keys():
            r = key[0]
            wavevector = np.zeros(3)
            for i in range(3):
                wavevector[i] = key[1][i] * base_vectors3d[r][i]
            peaks3d[key] = VectorOperations.rotate_vector3d(wavevector, np.array((0, 0, 1)), illumination2d.angles[r])

        return peaks3d

    def build_phase_matrix(self,
                           stack: np.ndarray,
                           peaks: dict) -> dict:
        """
        Build the phase matrix for a 3-D stack. 

        If dimensions[2] == 1, compute 3D phases using the full stack and natively.
        If dimensions[2] == 0, build a quasi 2D 'superimage' (by summing complex FTs), create a quasi-2D illumination, and compute
        using 2D estimators along with projected peaks.
        """
        if self.illumination.dimensions[2]:
            phase_estimator = self._make_phase_estimator(
                self.illumination,
                self.optical_system,
                self.phase_estimation_method,
                self.first_order_phases,
                self.phase_debug_info_level,
            )
            return phase_estimator.estimate_phases(stack, peaks)
        else:
            optical_system2d = self.optical_system.project_in_2D()
            illumination2d = IlluminationPlaneWaves2D.init_from_3D(self.illumination.project_in_quasi_2D())
            projected_stack = self._project_stack(stack)
            peaks2d = {(peak[0], peak[1][:2]): peaks[peak][:2] for peak in peaks.keys()}
            phase_estimator = self._make_phase_estimator(
                illumination2d,
                optical_system2d,
                self.phase_estimation_method,
                self.first_order_phases,
                self.phase_debug_info_level,
            )
            phase_matrix2d = phase_estimator.estimate_phases(projected_stack, peaks2d)
            phase_matrix3d = {}
            for peak in phase_matrix2d.keys():
                r, n, m = peak
                m_inv = tuple(-np.array(m))
                phase_matrix3d[(r, n, (m[0], m[1], 0))] = phase_matrix2d[peak]
                phase_matrix3d[(r, n, (-m[0], m[1], 0))] = phase_matrix2d[peak].conjugate()
            for r in range(self.illumination.Mr):
                for n in range(self.illumination.Mt):
                    phase_matrix3d[(r, n, (0, 0, 0))] = 0
            return phase_matrix3d
        
    def estimate_modulation_coefficients(self,
                                        stack: np.ndarray,
                                        illumination_raw: IlluminationPlaneWaves3D,
                                        ) -> dict:
        """
        Estimate modulation coefficients a_m from the image stack and estimated illumination parameters.
        """
        # if not treat_composite_orders_separately_for_modulation:
        #     illumination_projected = illumination_raw.project_in_quasi_2D()
        # else:
        #     if debug_info_level > 0:
        #         print('Treating composite orders separately for modulation estimation')
        #     illumination_projected = illumination_raw
        modulation_estimator = self._make_modulation_estimator(
            illumination_raw,
            self.optical_system,
            self.modulation_coefficients_method,
            treat_composite_orders_separately=self.treat_composite_orders_separately_for_modulation,
            debug_info_level=self.modulation_debug_info_level,
        )
        return modulation_estimator.estimate_modulations(stack=stack)
        
