import copy
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from PIL import Image

import Apodization
import hpc_utils
from Illumination import IlluminationPlaneWaves2D
from OpticalSystems import System4f2D, System4f3D
from PatternEstimation.ModulationEstimator import (
    ModulationLeastSquaresEstimator,
    ModulationPeakHeightRatioEstimator,
    ModulationRatiosInversionEstimator,
)
from PatternEstimation.PatternEstimator import IlluminationPatternEstimator2D, IlluminationPatternEstimator3D
from PatternEstimation.PhasesEstimator import (
    PhasesAutocorrelationEstimator,
    PhasesCrossCorrelationEstimator,
    PhasesPeakValueEstimator,
    PhasesSidePeakSuppressionEstimator,
)
from SIMulator import SIMulator2D, SIMulator3D
from config.SIM_N100_NA15 import alpha, dx, dz, configurations, nmedium, theta
from windowing import make_mask_cosine_edge2d

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

hpc_utils._detect_gpu = lambda: ('cpu', None)
hpc_utils.pick_backend('cpu')


PHASE_ESTIMATOR_CLASSES = {
    'autocorrelation': PhasesAutocorrelationEstimator,
    'cross_correlation': PhasesCrossCorrelationEstimator,
    'peak_phases': PhasesPeakValueEstimator,
    'side_peak_suppression': PhasesSidePeakSuppressionEstimator,
}

MODULATION_ESTIMATOR_CLASSES = {
    'least_squares': ModulationLeastSquaresEstimator,
    'peak_height_ratio': ModulationPeakHeightRatioEstimator,
    'ratios_inversion': ModulationRatiosInversionEstimator,
}

theta2d = 2. * np.pi / 5
theta3d = 2.1 * np.pi / 5

def build_experimental_illumination_2d():
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta2d * 0.9,
        1,
        0,
        Mr=3,
        angles=(-5 / 180 * np.pi, 62 / 180 * np.pi, 115 / 180 * np.pi),
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    return illum2d


def build_theoretical_illumination_2d():
    illum3d = configurations.get_2_oblique_s_waves_and_s_normal(
        theta2d *0.9,
        1,
        0,
        Mr=3,
        angles=(-5 / 180 * np.pi, 62 / 180 * np.pi, 115 / 180 * np.pi),
    )
    illum2d = IlluminationPlaneWaves2D.init_from_3D(illum3d, dimensions=(1, 1))
    illum2d.set_spatial_shifts_diagonally()
    return illum2d


def build_experimental_illumination_3d():
    illumination = configurations.get_2_oblique_s_waves_and_s_normal(
        theta3d,
        0.9,
        0.5,
        Mr=3,
        angles=(-20 / 180 * np.pi, 40 / 180 * np.pi, 100 / 180 * np.pi),
    )
    illumination.set_spatial_shifts_diagonally()
    return illumination


def build_theoretical_illumination_3d():
    illumination = configurations.get_2_oblique_s_waves_and_s_normal(
        theta3d,
        0.9,
        0.5,
        Mr=3,
        angles=(-20 / 180 * np.pi, 40 / 180 * np.pi, 100 / 180 * np.pi),
    )
    illumination.set_spatial_shifts_diagonally()
    return illumination


class PatternEstimatorAssertionsMixin:
    def assert_phase_matrix_valid(self, phase_matrix: dict):
        self.assertTrue(phase_matrix)
        self.assertTrue(all(np.isfinite(np.real(value)) and np.isfinite(np.imag(value)) for value in phase_matrix.values()))

    def assert_modulations_valid(self, modulations: dict):
        self.assertTrue(modulations)
        self.assertTrue(all(np.isfinite(np.abs(value)) for value in modulations.values()))

    def assert_illumination_valid(self, illumination_estimated, reference_illumination):
        self.assertEqual(len(illumination_estimated.angles), reference_illumination.Mr)
        self.assertEqual(len(illumination_estimated.harmonics), len(reference_illumination.harmonics))
        self.assert_phase_matrix_valid(illumination_estimated.phase_matrix)

    def print_stack_statistics(self, stack: np.ndarray):
        total_photons = float(np.sum(stack))
        spatial_samples = int(np.prod(self.spatial_shape))
        print('stack total photons =', total_photons)
        print('stack average photons =', total_photons / spatial_samples)

    @staticmethod
    def _print_deviation_summary(label: str, deviations: np.ndarray):
        deviations = np.asarray(deviations, dtype=np.float64)
        print(f'{label} mean squared deviation =', float(np.mean(deviations ** 2))**0.5)

    @staticmethod
    def print_expected_plot(description: str):
        print('expected plot:', description)

    def print_estimation_deviations(
        self,
        illumination_estimated,
        reference_illumination,
        report_angles: bool = False,
        report_wavevectors: bool = False,
        report_phases: bool = False,
        report_amplitudes: bool = False,
    ):
        if report_angles:
            angle_deviations_deg = (illumination_estimated.angles - reference_illumination.angles) * 180 / np.pi
            print('angle deviations [deg] =', np.round(angle_deviations_deg, 3))
            self._print_deviation_summary('angles', angle_deviations_deg)

        if report_wavevectors:
            wavevector_deviations = []
            for key in sorted(reference_illumination.harmonics):
                default_wavevector = reference_illumination.harmonics[key].wavevector / (2 * np.pi)
                estimated_wavevector = illumination_estimated.harmonics[key].wavevector / (2 * np.pi)
                deviation = estimated_wavevector - default_wavevector
                wavevector_deviations.append(deviation)
                print('wavevector deviation', key, '=', np.round(deviation, 6))
            self._print_deviation_summary(
                'wavevectors',
                np.array(wavevector_deviations).reshape(-1),
            )

        if report_phases:
            phase_deviations_deg = []
            for key in sorted(reference_illumination.phase_matrix):
                estimated_phase = illumination_estimated.phase_matrix[key]
                default_phase = reference_illumination.phase_matrix[key]
                phase_deviation_deg = np.angle(estimated_phase * np.conjugate(default_phase)) * 180 / np.pi
                phase_deviations_deg.append(phase_deviation_deg)
                print('phase deviation [deg]', key, '=', np.round(phase_deviation_deg, 3))
            self._print_deviation_summary(
                'phases',
                np.array(phase_deviations_deg),
            )

        if report_amplitudes:
            amplitude_deviations = []
            for key in sorted(reference_illumination.harmonics):
                default_amplitude = reference_illumination.harmonics[key].amplitude
                estimated_amplitude = illumination_estimated.harmonics[key].amplitude
                deviation = estimated_amplitude - default_amplitude
                amplitude_deviations.append(deviation)
                print('amplitude deviation', key, '=', np.round(deviation, 6))
            self._print_deviation_summary(
                'amplitudes',
                np.array(amplitude_deviations),
            )

    def report_estimation_summary(
        self,
        stack: np.ndarray,
        illumination_estimated,
        reference_illumination,
        report_angles: bool = False,
        report_wavevectors: bool = False,
        report_phases: bool = False,
        report_amplitudes: bool = False,
    ):
        self.print_stack_statistics(stack)
        self.print_estimation_deviations(
            illumination_estimated,
            reference_illumination,
            report_angles=report_angles,
            report_wavevectors=report_wavevectors,
            report_phases=report_phases,
            report_amplitudes=report_amplitudes,
        )


class PatternEstimator2DFixture(PatternEstimatorAssertionsMixin):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234)
        N = 501
        max_r = N // 2 * dx
        psf_size = 2 * np.array((max_r, max_r))

        cls.optical_system = System4f2D(alpha=alpha - 0.2, refractive_index=nmedium)
        cls.optical_system.compute_psf_and_otf((psf_size, N))
        apodization = Apodization.AutocorrelationApodizationWidefield(cls.optical_system)
        cls.optical_system.otf = np.where(apodization.apodization_function > 1e-16, cls.optical_system.otf, 0)

        cls.experimental_illumination = build_experimental_illumination_2d()
        cls.theoretical_illumination = build_theoretical_illumination_2d()
        cls.theoretical_illumination.phase_matrix = copy.deepcopy(cls.experimental_illumination.phase_matrix)
        cls.simulator = SIMulator2D(cls.experimental_illumination, cls.optical_system)

        cls.sample = Image.open('resolution_templates/1951usaf_test_target.jpg').convert('L')
        cls.sample = np.array(cls.sample, dtype=np.float32)[80:N+80, 45:N+45]
        cls.sample -= cls.sample.min()
        cls.sample /= cls.sample.max()
        Nph = 10**5
        cls.sample *= Nph
        cls.spatial_shape = cls.sample.shape
        # cls.sample = np.zeros((N, N)) + 10**4
        # cls.sample[N // 2, N // 2 - 12:N // 2 + 13] = 10**8
        # cls.sample[N // 2 - 12:N // 2 + 13, N // 2] = 10**8
        # plt.imshow(cls.sample, cmap='gray')
        # plt.show()
        cls.raw_stack = cls.simulator.generate_noisy_sim_images(cls.sample)
        cls.raw_stack *= make_mask_cosine_edge2d(cls.sample.shape, 12)[None, None, :, :]
        cls.refined_wavevectors = {
            key: harmonic.wavevector for key, harmonic in cls.experimental_illumination.harmonics.items()
        }


class PatternEstimator3DFixture(PatternEstimatorAssertionsMixin):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1234)
        N = (65, 65, 9)
        max_r = N[0] // 2 * dx
        max_z = N[2] // 2 * dz
        psf_size = 2 * np.array((max_r, max_r, max_z))

        cls.optical_system = System4f3D(
            alpha=alpha,
            refractive_index_medium=nmedium,
            refractive_index_sample=nmedium,
            vectorial=True,
            high_NA=True,
        )
        cls.optical_system.compute_psf_and_otf((psf_size, N))
        apodization = Apodization.AutocorrelationApodizationWidefield(cls.optical_system)
        cls.optical_system.otf = np.where(apodization.apodization_function > 1e-16, cls.optical_system.otf, 0)

        cls.experimental_illumination = build_experimental_illumination_3d()
        cls.theoretical_illumination = build_theoretical_illumination_3d()
        cls.theoretical_illumination.phase_matrix = copy.deepcopy(cls.experimental_illumination.phase_matrix)
        cls.simulator = SIMulator3D(cls.experimental_illumination, cls.optical_system)

        cls.sample = np.zeros(N) + 10**3
        cls.spatial_shape = cls.sample.shape
        cls.sample[N[0] // 2, :, N[2] // 2] = 10**7
        cls.sample[:, N[1] // 2, N[2] // 2] = 10**7
        cls.sample[N[0] // 2, N[1] // 2, :] = 10**7
        cls.sample[N[0] // 2 - 3:N[0] // 2 + 4, N[1] // 2 - 3:N[1] // 2 + 4, N[2] // 2] = 10**7
        cls.raw_stack = cls.simulator.generate_noiseless_sim_images(cls.sample)
        cls.raw_stack *= make_mask_cosine_edge2d(cls.sample.shape[:2], 10)[None, None, :, :, None]
        cls.refined_wavevectors = {
            key: harmonic.wavevector for key, harmonic in cls.experimental_illumination.harmonics.items()
        }


class TestPatternEstimator2DMethods(PatternEstimator2DFixture, unittest.TestCase):
    def test_peak_estimation_methods(self):
        cases = ['interpolation', 'cross_correlation']
        for peak_method in cases:
            print("PEAK ESTIMATION METHOD:", peak_method)
            self.print_expected_plot('peak locations or refinement diagnostics only')
            with self.subTest(peak_method=peak_method):
                estimator = IlluminationPatternEstimator2D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method=peak_method,
                    phase_estimation_method='default',
                    modulation_coefficients_method='default',
                    peak_search_area_size=20,
                    zooming_factor=20,
                    max_iterations=2,
                    debug_info_level=0,
                    peak_debug_info_level=2,
                    correct_peak_position=False,
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_angles=True,
                    report_wavevectors=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_phase_estimation_methods(self):
        for phase_method in ['peak_phases', 'autocorrelation', 'cross_correlation']:
            print("PHASE ESTIMATION METHOD:", phase_method)
            # self.print_expected_plot('phase offsets only')
            with self.subTest(phase_method=phase_method):
                estimator = IlluminationPatternEstimator2D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='default',
                    phase_estimation_method=phase_method,
                    modulation_coefficients_method='default',
                    peak_search_area_size=7,
                    zooming_factor=20,
                    max_iterations=3,
                    debug_info_level=0,
                    phase_debug_info_level=1,
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_phases=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_modulation_estimation_methods(self):
        for modulation_method in ['peak_height_ratio', 'ratios_inversion']:
            print("MODULATION ESTIMATION METHOD:", modulation_method)
            self.print_expected_plot('modulation amplitudes only')
            with self.subTest(modulation_method=modulation_method):
                estimator = IlluminationPatternEstimator2D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='default',
                    phase_estimation_method='default',
                    modulation_coefficients_method=modulation_method,
                    treat_composite_orders_separately_for_modulation=False,
                    debug_info_level=0,
                    modulation_debug_info_level=1,
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_amplitudes=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

class TestPatternEstimator2DParameters(PatternEstimator2DFixture, unittest.TestCase):
    def test_boolean_parameters(self):
        cases = [
            {'first_order_peaks': True, 'first_order_phases': True, 'correct_peak_position': True},
            {'first_order_peaks': False, 'first_order_phases': False, 'correct_peak_position': False},
        ]
        for case in cases:
            with self.subTest(case=case):
                estimator = IlluminationPatternEstimator2D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='interpolation',
                    phase_estimation_method='autocorrelation',
                    modulation_coefficients_method='least_squares',
                    peak_search_area_size=7,
                    zooming_factor=20,
                    max_iterations=1,
                    debug_info_level=0,
                    **case,
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_constructor_parameter_storage(self):
        estimator = IlluminationPatternEstimator2D(
            copy.deepcopy(self.theoretical_illumination),
            self.optical_system,
            peak_estimation_method='cross_correlation',
            phase_estimation_method='cross_correlation',
            modulation_coefficients_method='peak_height_ratio',
            peak_search_area_size=9,
            zooming_factor=15,
            max_iterations=2,
            debug_info_level=0,
            peak_debug_info_level=1,
            phase_debug_info_level=2,
            modulation_debug_info_level=3,
            correct_peak_position=False,
            first_order_peaks=False,
            first_order_phases=False,
            treat_composite_orders_separately_for_modulation=True,
        )
        self.assertEqual(estimator.peak_search_area_size, 9)
        self.assertEqual(estimator.zooming_factor, 15)
        self.assertEqual(estimator.max_iterations, 2)
        self.assertEqual(estimator.peak_debug_info_level, 1)
        self.assertEqual(estimator.phase_debug_info_level, 2)
        self.assertEqual(estimator.modulation_debug_info_level, 3)
        self.assertFalse(estimator.correct_peak_position)
        self.assertFalse(estimator.first_order_peaks)
        self.assertFalse(estimator.first_order_phases)
        self.assertTrue(estimator.treat_composite_orders_separately_for_modulation)


class TestPatternEstimator3DMethods(PatternEstimator3DFixture, unittest.TestCase):
    def test_peak_estimation_methods(self):
        for peak_method in ['default', 'interpolation', 'cross_correlation']:
            print("3D peak estimation method:", peak_method)
            self.print_expected_plot('3D peak locations or projected peak diagnostics only')
            with self.subTest(peak_method=peak_method):
                estimator = IlluminationPatternEstimator3D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method=peak_method,
                    phase_estimation_method='default',
                    modulation_coefficients_method='default',
                    peak_search_area_size=30,
                    zooming_factor=100,
                    max_iterations=2,
                    debug_info_level=0,
                    peak_debug_info_level=1,
                    estimation_strategy='plane_by_plane',
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_angles=True,
                    report_wavevectors=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_peak_estimation_strategies(self):
        for strategy in ['plane_by_plane', 'true_3D']:
            print("3D peak estimation strategy:", strategy)
            self.print_expected_plot('strategy-specific peak diagnostics only')
            with self.subTest(estimation_strategy=strategy):
                estimator = IlluminationPatternEstimator3D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='interpolation',
                    phase_estimation_method='default',
                    modulation_coefficients_method='default',
                    peak_search_area_size=5,
                    zooming_factor=10,
                    max_iterations=1,
                    debug_info_level=0,
                    peak_debug_info_level=1,
                    estimation_strategy=strategy,
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_angles=True,
                    report_wavevectors=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_phase_estimation_methods(self):
        for phase_method in ['default', 'peak_phases', 'autocorrelation', 'cross_correlation', 'side_peak_suppression']:
            print("3D phase estimation method:", phase_method)
            self.print_expected_plot('3D phase offsets only')
            with self.subTest(phase_method=phase_method):
                estimator = IlluminationPatternEstimator3D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='default',
                    phase_estimation_method=phase_method,
                    modulation_coefficients_method='default',
                    peak_search_area_size=5,
                    zooming_factor=10,
                    max_iterations=1,
                    debug_info_level=0,
                    phase_debug_info_level=1,
                    estimation_strategy='plane_by_plane',
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_phases=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_modulation_estimation_methods(self):
        for modulation_method in ['default', 'least_squares', 'peak_height_ratio', 'ratios_inversion']:
            print("3D modulation estimation method:", modulation_method)
            self.print_expected_plot('3D modulation amplitudes only')
            with self.subTest(modulation_method=modulation_method):
                estimator = IlluminationPatternEstimator3D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='default',
                    phase_estimation_method='default',
                    modulation_coefficients_method=modulation_method,
                    treat_composite_orders_separately_for_modulation=True,
                    debug_info_level=0,
                    estimation_strategy='plane_by_plane',
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.report_estimation_summary(
                    self.raw_stack,
                    illumination_estimated,
                    self.theoretical_illumination,
                    report_amplitudes=True,
                )
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)


class TestPatternEstimator3DParameters(PatternEstimator3DFixture, unittest.TestCase):
    def test_boolean_and_projection_parameters(self):
        cases = [
            {
                'first_order_peaks': True,
                'first_order_phases': True,
                'correct_peak_position': True,
                'treat_composite_orders_separately_for_modulation': False,
                'estimation_strategy': 'plane_by_plane',
                'reference_expansion': (1, 0, 1),
            },
            {
                'first_order_peaks': False,
                'first_order_phases': False,
                'correct_peak_position': False,
                'treat_composite_orders_separately_for_modulation': True,
                'estimation_strategy': 'true_3D',
                'reference_expansion': (1, 0, 2),
            },
        ]
        for case in cases:
            with self.subTest(case=case):
                estimator = IlluminationPatternEstimator3D(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    peak_estimation_method='interpolation',
                    phase_estimation_method='autocorrelation',
                    modulation_coefficients_method='least_squares',
                    peak_search_area_size=5,
                    zooming_factor=10,
                    max_iterations=1,
                    debug_info_level=0,
                    **case,
                )
                illumination_estimated = estimator.estimate_illumination_parameters(self.raw_stack)
                self.assert_illumination_valid(illumination_estimated, self.theoretical_illumination)

    def test_constructor_parameter_storage(self):
        estimator = IlluminationPatternEstimator3D(
            copy.deepcopy(self.theoretical_illumination),
            self.optical_system,
            peak_estimation_method='cross_correlation',
            phase_estimation_method='cross_correlation',
            modulation_coefficients_method='ratios_inversion',
            peak_search_area_size=5,
            zooming_factor=10,
            max_iterations=2,
            debug_info_level=0,
            peak_debug_info_level=1,
            phase_debug_info_level=2,
            modulation_debug_info_level=3,
            correct_peak_position=False,
            first_order_peaks=False,
            first_order_phases=False,
            treat_composite_orders_separately_for_modulation=True,
            estimation_strategy='true_3D',
            reference_expansion=(1, 0, 2),
        )
        self.assertEqual(estimator.peak_debug_info_level, 1)
        self.assertEqual(estimator.phase_debug_info_level, 2)
        self.assertEqual(estimator.modulation_debug_info_level, 3)
        self.assertFalse(estimator.correct_peak_position)
        self.assertFalse(estimator.first_order_peaks)
        self.assertFalse(estimator.first_order_phases)
        self.assertTrue(estimator.treat_composite_orders_separately_for_modulation)
        self.assertEqual(estimator.estimation_strategy, 'true_3D')
        self.assertEqual(estimator.reference_expansion, (1, 0, 2))


class TestDirectPhaseEstimators2D(PatternEstimator2DFixture, unittest.TestCase):
    def test_every_phase_estimator_class(self):
        for name, estimator_class in PHASE_ESTIMATOR_CLASSES.items():
            with self.subTest(phase_estimator=name):
                estimator = estimator_class(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    first_order_phases=False,
                    debug_info_level=0,
                )
                phase_matrix = estimator.estimate_phase_matrix(self.raw_stack, self.refined_wavevectors)
                self.assert_phase_matrix_valid(phase_matrix)
                self.assertIn((0, 0, (0, 0)), phase_matrix)


class TestDirectPhaseEstimators3D(PatternEstimator3DFixture, unittest.TestCase):
    def test_every_phase_estimator_class(self):
        for name, estimator_class in PHASE_ESTIMATOR_CLASSES.items():
            with self.subTest(phase_estimator=name):
                estimator = estimator_class(
                    copy.deepcopy(self.theoretical_illumination),
                    self.optical_system,
                    first_order_phases=False,
                    debug_info_level=0,
                )
                phase_matrix = estimator.estimate_phase_matrix(self.raw_stack, self.refined_wavevectors)
                self.assert_phase_matrix_valid(phase_matrix)
                self.assertIn((0, 0, (0, 0, 0)), phase_matrix)


class TestDirectModulationEstimators2D(PatternEstimator2DFixture, unittest.TestCase):
    def test_every_modulation_estimator_class(self):
        for name, estimator_class in MODULATION_ESTIMATOR_CLASSES.items():
            with self.subTest(modulation_estimator=name):
                estimator = estimator_class(
                    copy.deepcopy(self.experimental_illumination),
                    self.optical_system,
                    debug_info_level=0,
                )
                modulations = estimator.estimate_modulations(stack=self.raw_stack)
                self.assert_modulations_valid(modulations)


class TestDirectModulationEstimators3D(PatternEstimator3DFixture, unittest.TestCase):
    def test_every_modulation_estimator_class(self):
        for name, estimator_class in MODULATION_ESTIMATOR_CLASSES.items():
            with self.subTest(modulation_estimator=name):
                estimator = estimator_class(
                    copy.deepcopy(self.experimental_illumination),
                    self.optical_system,
                    debug_info_level=0,
                    treat_composite_orders_separately=True,
                )
                modulations = estimator.estimate_modulations(stack=self.raw_stack)
                self.assert_modulations_valid(modulations)


if __name__ == '__main__':
    unittest.main(verbosity=2)
