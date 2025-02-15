"""
Illumination.py

This module contains the Illumination class, which handles the simulation and analysis of illumination patterns in optical systems.

Classes:
    Illumination: Manages the properties and behavior of illumination patterns, including wavevectors and spatial shifts.
"""
# from collections.abc import dict_values
from typing import Dict, Tuple, Any, List

import numpy as np
from numpy import ndarray, dtype
from skimage.color.rgb_colors import dimgray

import Sources
import wrappers
from abc import abstractmethod
from Sources import IntensityHarmonic3D
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools


class Illumination:

    @abstractmethod
    def get_illumination_density(self, **kwargs): ...


class PeriodicStructure:

    @abstractmethod
    def get_elementary_cell(self): ...


class IlluminationArray2D(PeriodicStructure): ...


class IlluminationArray3D(PeriodicStructure): ...


class PlaneWavesSIM(PeriodicStructure):
    """
    Manages the properties and behavior of illumination patterns in SIM with a finite number of plane waves interference.

    Attributes:
        angles (np.ndarray): Array of rotation angles.
        _spatial_shifts (list): List of spatial shifts.
        _Mr (int): Number of rotations.
        Mt (int): Number of spatial shifts.
        waves (dict): Dictionary of intensity plane waves.
        phase_matrix (dict): Dictionary of all phase the relevant phase shifts.
    """

    def __init__(self, intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic], dimensions: tuple[int, ...], Mr=1, spatial_shifts=[]):
        """
        Collect the information describing the SIM experiment

        Args:
            intensity_harmonics_dict (dict): Dictionary of intensity plane waves.
            Mr (int): Number of rotations.
        """
        self.angles = np.arange(0, np.pi, np.pi / Mr)
        self._Mr = Mr

        self._spatial_shifts = spatial_shifts
        self.Mt = len(self.spatial_shifts)

        self.waves = {key: intensity_harmonics_dict[key] for key in intensity_harmonics_dict.keys() if not np.isclose(intensity_harmonics_dict[key].amplitude, 0)}
        self.dimensions = dimensions
        self.rearranged_indices = self._rearrange_indices(dimensions)

        self.sim_indices, self.projected_wavevectors = self.get_wavevectors_projected(0)

        self.phase_matrix = {}
        self.compute_phase_matrix()

    @classmethod
    def init_from_list(cls,
                       intensity_harmonics_list: list[Sources.IntensityHarmonic],
                       base_vector_lengths: tuple[float, ...],
                       dimensions, Mr=1,
                       spatial_shifts=[]):
        """
        Class method to initialize Illumination from a list of intensity plane waves.

        Args:
            intensity_harmonics_list (list): List of intensity plane waves.
            base_vector_lengths (tuple): Base vector lengths of the illumination Fourier space Bravais lattice.
            Mr (int): Number of rotations.

        Returns:
            Illumination: Initialized Illumination object.
        """
        intensity_harmonics_dict = cls.index_frequencies(intensity_harmonics_list, base_vector_lengths)
        return cls(intensity_harmonics_dict, dimensions, Mr=Mr)

    @property
    def Mr(self):
        return self._Mr

    @Mr.setter
    def Mr(self, new_Mr):
        self.Mr = new_Mr
        self.angles = np.arange(0, np.pi, np.pi / new_Mr)
        self.normalize_spatial_waves()

    @property
    def spatial_shifts(self):
        return self._spatial_shifts

    @spatial_shifts.setter
    def spatial_shifts(self, new_spatial_shifts):
        self._spatial_shifts = new_spatial_shifts
        self.Mt = len(new_spatial_shifts)
        self.normalize_spatial_waves()
        self.compute_phase_matrix()

    def _rearrange_indices(self, dimensions) -> dict[tuple[int, ...], tuple[tuple[int, ...]]]:
        """
        Rearrange indices for the computation of effective OTFs, required in SIM.

        Returns:
            dict: Dictionary of rearranged indices.
        """
        indices = self.waves.keys()
        result_dict = {}
        for index in indices:
            key = tuple([index[dim] for dim in range(len(dimensions)) if dimensions[dim]])
            value = tuple([index[dim] for dim in range(len(dimensions)) if not dimensions[dim]])
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
        result_dict = {key: tuple(values) for key, values in result_dict.items()}
        return result_dict

    @staticmethod
    @abstractmethod
    def index_frequencies(waves_list: list[Sources.IntensityHarmonic], base_vector_lengths: tuple[float, ...]) -> dict[tuple[int, ...], Sources.IntensityHarmonic]:
        """
        Index the frequencies of the intensity harmonics.

        Args:
            waves_list (list): List of intensity harmonics.
            base_vector_lengths (tuple): Base vector lengths of the illumination Fourier space Bravais lattice.

        Returns:
            dict: Dictionary of indexed intensity harmonics.
        """
        pass

    @staticmethod
    def glue_indices(sim_index, projected_index, dimensions) -> tuple[int, ...]:
        i, j = 0, 0
        index = []
        for dim in range(len(dimensions)):
            index.append(sim_index[i]) if dimensions[dim] else index.append(projected_index[j])
            if dimensions[dim]:
                i += 1
            else:
                j += 1
        return tuple(index)


    def get_elementary_cell(self):
        ...

    @abstractmethod
    def normalize_spatial_waves(self):
        """
        Normalize the spatial waves on zero peak (i.e., a0 = 1).

        Raises:
            AttributeError: If zero wavevector is not found.
        """
        pass

    @abstractmethod
    def compute_expanded_lattice(self) -> set[tuple[int, ...]]:
        ...

    def get_wavevectors(self, r: int) -> tuple[list[np.ndarray, ...], list[tuple[int, ...]]]:
        """
         Get the wavevectors and indices for a given rotation.

         Args:
             r (int): Rotation index.

         Returns:
             tuple: List of wavevectors and list of indices.
         """
        wavevectors = []
        angle = self.angles[r]
        indices = []
        for index in self.waves:
            indices.append(index)
            wavevector = VectorOperations.rotate_vector3d(
                self.waves[index].wavevector, np.array((0, 0, 1)), angle)
            wavevectors.append(wavevector)
        return wavevectors, indices

    def get_all_wavevectors(self) -> list[np.ndarray]:
        """
        Get all wavevectors for all rotations.

        Returns:
            list: List of all wavevectors.
        """
        wavevectors = []
        for r in range(self.Mr):
            wavevectors_r, _ = self.get_wavevectors(r)
            wavevectors.extend(wavevectors_r)
        return wavevectors

    def get_wavevectors_projected(self, r: int) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
        """
        Get the projected wavevectors and indices for a given rotation.

        Args:
            r (int): Rotation index.

        Returns:
            tuple: List of projected wavevectors and list of indices.
        """
        angle = self.angles[r]
        wavevectors_projected = []
        sim_indices = []
        for sim_index in self.rearranged_indices:
            index = self.glue_indices(sim_index, self.rearranged_indices[sim_index][0], self.dimensions)
            # index = (*index, 0) if len(self.dimensions) == 2 else index
            # dimensions = np.array((*self.dimensions, 0)) if len(self.dimensions) == 2 else self.dimensions
            wavevector = self.waves[index].wavevector
            wavevector[:2] = VectorOperations.rotate_vector2d(
                self.waves[index].wavevector[:2], angle)
            wavevectors_projected.append(wavevector[np.bool(self.dimensions)])
            sim_indices.append(sim_index)
        return wavevectors_projected, sim_indices

    def get_all_wavevectors_projected(self):
        """
        Get all projected wavevectors for all rotations.

        Returns:
            list: List of all projected wavevectors.
        """
        wavevectors2d = []
        for r in range(self.Mr):
            wavevectors2d_r, _ = self.get_wavevectors_projected(r)
            wavevectors2d.extend(wavevectors2d_r)
        return wavevectors2d

    def compute_effective_kernels(self, kernel: np.ndarray, coordinates: tuple[3, np.ndarray]) -> tuple[
        dict[tuple[int, tuple[int, ...]], np.ndarray], dict[tuple[int, tuple[int, ...]], np.ndarray]]:
        """
        Compute effective kernels for SIM computations

        Args:
            kernel(np.ndarray): SIM reconstruction kernel, e.g., OTF.
            coordinates(tuple): coordinates
            dimensions(tuple): defines which dimensions are fixed w.r.t. a sample (1)
            and which are fixed w.r.t. focal plane (0), i.e. projected

        Returns:
            tuple: Effective kernels and their Fourier transform.
        """
        waves = self.waves
        effective_kernels = {}
        effective_kernels_ft = {}
        grid = np.stack(np.meshgrid(*coordinates), -1)
        for r in range(self.Mr):
            angle = self.angles[r]
            indices = self._rearrange_indices(self.dimensions)
            for sim_index in indices:
                effective_kernel = 0
                for projected_index in indices[sim_index]:
                    index = self.glue_indices(sim_index, projected_index, self.dimensions)
                    wavevector = waves[index].wavevector.copy()
                    wavevector[:2] = VectorOperations.rotate_vector2d(
                        waves[index].wavevector[:2], angle)
                    # print(angle, sim_index, wavevector)
                    amplitude = waves[index].amplitude
                    if len(self.dimensions) == 2:
                        phase_shifted = np.exp(1j * np.einsum('ijl,l ->ij', grid, wavevector)) * kernel
                    elif len(self.dimensions) == 3:
                        phase_shifted = np.transpose(np.exp(1j * np.einsum('ijkl,l ->ijk', grid, wavevector)), axes=(1, 0, 2)) * kernel
                    else:
                        raise ValueError("The number of dimensions is meaningless in the context of microscopy!")
                    effective_kernel += amplitude * phase_shifted
                effective_kernels[(r, sim_index)] = effective_kernel
                effective_kernels_ft[(r, sim_index)] = wrappers.wrapped_fftn(effective_kernel)
        return effective_kernels, effective_kernels_ft

    def get_phase_modulation_patterns(self, coordinates):
        phase_modulation_patterns = {}
        grid = np.stack(np.meshgrid(*coordinates), axis=-1)
        for r in range(self.Mr):
            for sim_index in self.rearranged_indices:
                projective_index = self.rearranged_indices[sim_index][0]
                index = self.glue_indices(sim_index, projective_index, self.dimensions)
                wavevector = self.waves[index].wavevector
                wavevector[np.bool(1 - np.array(self.dimensions))] = 0
                if len(self.dimensions) == 2:
                    phase_modulation = np.exp(1j * np.einsum('ijl,l ->ij', grid, wavevector))
                elif len(self.dimensions) == 3:
                    phase_modulation = np.exp(1j * np.einsum('ijkl,l ->ijk', grid, wavevector))
                else:
                    raise ValueError("The number of dimensions is meaningless in the context of microscopy!")
                phase_modulation_patterns[r, sim_index] = phase_modulation
        return phase_modulation_patterns

    def compute_phase_matrix(self):
        """
        Compute the dictionary of all the relevant phase shifts
         (products of spatial shifts and illumination pattern spatial frequencies).
        """
        self.phase_matrix = {}
        for r in range(self.Mr):
            for n in range(self.Mt):
                urn = self.spatial_shifts[n]
                urn[:2] = VectorOperations.rotate_vector2d(urn[:2], self.angles[r])
                wavevectors, indices = self.get_wavevectors_projected(r)
                for i in range(len(wavevectors)):
                    wavevector = wavevectors[i]
                    self.phase_matrix[(r, n, indices[i])] = np.exp(-1j * np.dot(urn[np.bool(np.array(self.dimensions))], wavevector))


class IlluminationPlaneWaves3D(PlaneWavesSIM):
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 dimensions: tuple[bool, bool, bool] = (1, 1, 0),
                 Mr: int = 1, spatial_shifts=np.array([(0, 0, 0)])):
        super().__init__(intensity_harmonics_dict, dimensions, Mr, spatial_shifts)

    @classmethod
    def init_from_list(cls, intensity_harmonics_list: list[Sources.IntensityHarmonic3D], base_vector_lengths: tuple[float, ...], dimensions=(1, 1, 0), Mr: int = 1,
                       spatial_shifts=np.array([(0, 0, 0)])):
        return super().init_from_list(intensity_harmonics_list, base_vector_lengths, dimensions, Mr, spatial_shifts)

    def normalize_spatial_waves(self):
        if not (0, 0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0, 0].amplitude * self.Mt * self.Mr
        for spatial_wave in self.waves.values():
            spatial_wave.amplitude /= norm

    @staticmethod
    def index_frequencies(waves_list: list[Sources.IntensityHarmonic3D], base_vector_lengths: tuple[float, float, float]) -> dict[tuple[int, int, int],
    Sources.IntensityHarmonic3D]:
        intensity_harmonics_dict = {}
        for wave in waves_list:
            wavevector = wave.wavevector
            m1, m2, m3 = int(round(wavevector[0] / base_vector_lengths[0])), int(round(wavevector[1] / base_vector_lengths[1])), \
                int(round(wavevector[2] / base_vector_lengths[2]))
            if not (m1, m2, m3) in intensity_harmonics_dict.keys():
                intensity_harmonics_dict[(m1, m2, m3)] = wave
            else:
                intensity_harmonics_dict[(m1, m2, m3)].amplitude += wave.amplitude
        return intensity_harmonics_dict

    @staticmethod
    def find_ipw_from_pw(plane_waves) -> tuple[IntensityHarmonic3D, ...]:
        """
        Static method to find intensity plane waves
         (i.e. Fourier transform of the illumination pattern) from plane waves.

        Args:
            plane_waves (list): List of plane waves.

        Returns:
            list: List of intensity plane waves.
        """
        intensity_harmonics = {}
        for projection_index in (0, 1, 2):
            for plane_wave1 in plane_waves:
                amplitude1p = plane_wave1.field_vectors[0][projection_index]
                amplitude1s = plane_wave1.field_vectors[1][projection_index]
                amplitude1 = amplitude1s + amplitude1p
                if np.abs(amplitude1) < 10 ** -3:
                    continue
                wavevector1 = plane_wave1.wavevector

                for plane_wave2 in plane_waves:
                    amplitude2p = plane_wave2.field_vectors[0][projection_index]
                    amplitude2s = plane_wave2.field_vectors[1][projection_index]
                    amplitude2 = amplitude2s + amplitude2p
                    if np.abs(amplitude2) < 10 ** -3:
                        continue
                    wavevector2 = plane_wave2.wavevector
                    wavevector_new = tuple(wavevector1 - wavevector2)
                    if not wavevector_new in intensity_harmonics.keys():
                        intensity_harmonics[wavevector_new] = (Sources.IntensityHarmonic3D(amplitude1 * amplitude2.conjugate(), 0,
                                                                                           np.array(wavevector_new)))
                    else:
                        intensity_harmonics[wavevector_new].amplitude += amplitude1 * amplitude2.conjugate()
        return intensity_harmonics.values()

    def compute_expanded_lattice(self) -> set[tuple[int, int, int]]:
        """
        Compute the expanded 3D lattice of Fourier peaks
         (autoconvoluiton of Fourier transform of the illumination pattern).

        Returns:
            set: Set of expanded 3D lattice peaks.
        """
        fourier_peaks = set(self.waves.keys())
        expanded_lattice3d = set()
        for peak1 in fourier_peaks:
            for peak2 in fourier_peaks:
                expanded_lattice3d.add((peak1[0] - peak2[0], peak1[1] - peak2[1], peak1[2] - peak2[2]))
        print(len(expanded_lattice3d))
        return expanded_lattice3d

    def set_spatial_shifts_diagonally(self, number: int, base_vectors: tuple[float, float, float]):
        """
        Set the spatial shifts diagonally (i.e., all the spatial shifts are assumed to be on the same lin).
        This is the most common use in practice.
        Appropriate shifts for a given illumination pattern can be computed in the module 'compute_optimal_lattices.py'

        Args:
            number (int): Number of shifts.
            base_vectors (tuple): Base vectors for the shifts.
        """
        kx, ky = base_vectors[0], base_vectors[1]
        shiftsx = np.arange(0, number) / number / kx
        shiftsy = np.arange(0, number) / number / ky
        self.spatial_shifts = np.array([(shiftsx[i], shiftsy[i], 0) for i in range(number)])

    def get_illumination_density(self, grid=None, coordinates=None, depth=None, r=0, n=0):
        if not grid and not coordinates:
            raise ValueError("Either grid or coordinates must be provided!")
        if not grid and coordinates:
            X, Y, Z = np.meshgrid(coordinates)
            grid = np.stack((X, Y, Z), axis=-1)
        if depth:
            grid[:, :, :, 2] -= depth

        illumination_density = np.zeros(grid.shape[:3], dtype=np.complex128)
        wavevectors, indices = self.get_wavevectors(r)
        for i in range(len(wavevectors)):
            phase = self.phase_matrix[(r, n, indices[i])]
            amplitude = self.waves[indices[i]].amplitude
            wavevector = wavevectors[i]
            illumination_density += Sources.IntensityHarmonic3D(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density

    def get_elementary_cell(self):
        ...


class IlluminationPlaneWaves2D(PlaneWavesSIM):

    def __init__(self, intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic2D], dimensions: tuple[int, int] = (1, 1), Mr: int = 1, spatial_shifts=np.array([(0, 0)])):
        super().__init__(intensity_harmonics_dict, dimensions, Mr, spatial_shifts)

    @classmethod
    def init_from_3D(cls, illumination_3d: IlluminationPlaneWaves3D, dimensions: tuple[int, int] = (1, 1)):
        for wave in illumination_3d.waves:
            if wave[2] != 0:
                raise ValueError("The 3D illumination pattern cannot be converted to 2D unless it's not changing in z direction!")

        intensity_harmonics_dict = {tuple(wave[:2]): Sources.IntensityHarmonic2D.init_from_3D(illumination_3d.waves[wave])
                                    for wave in illumination_3d.waves}
        spatial_shifts = illumination_3d.spatial_shifts[:, :2]
        return cls(intensity_harmonics_dict, dimensions, illumination_3d.Mr, spatial_shifts)

    @staticmethod
    def index_frequencies(waves_list: list[Sources.IntensityHarmonic2D], base_vector_lengths: tuple[float, float]) -> dict[tuple[int, int],
    Sources.IntensityHarmonic2D]:
        intensity_harmonics_dict = {}
        for wave in waves_list:
            wavevector = wave.wavevector
            m1, m2 = int(round(wavevector[0] / base_vector_lengths[0])), int(round(wavevector[1] / base_vector_lengths[1]))
            if not (m1, m2) in intensity_harmonics_dict.keys():
                intensity_harmonics_dict[(m1, m2)] = wave
            else:
                intensity_harmonics_dict[(m1, m2)].amplitude += wave.amplitude
        return intensity_harmonics_dict

    def compute_expanded_lattice(self) -> set[tuple[int, int]]:
        """
        Compute the expanded 2D lattice of Fourier peaks
         (autoconvoluiton of Fourier transform of the illumination pattern).

        Returns:
            set: Set of expanded 2D lattice peaks.
        """
        self.xy_fourier_peaks = set((mx, my) for mx, my, mz in self.waves.keys())
        expanded_lattice2d = set()
        for peak1 in self.xy_fourier_peaks:
            for peak2 in self.xy_fourier_peaks:
                expanded_lattice2d.add((peak1[0] - peak2[0], peak1[1] - peak2[1]))
        print(len(expanded_lattice2d))
        return expanded_lattice2d

    def set_spatial_shifts_diagonally(self, number: int, base_vectors: tuple[float, float]):
        """
        Set the spatial shifts diagonally (i.e., all the spatial shifts are assumed to be on the same lin).
        This is the most common use in practice.
        Appropriate shifts for a given illumination pattern can be computed in the module 'compute_optimal_lattices.py'

        Args:
            number (int): Number of shifts.
            base_vectors (tuple): Base vectors for the shifts.
        """
        kx, ky = base_vectors[0], base_vectors[1]
        shiftsx = np.arange(0, number) / number / kx
        shiftsy = np.arange(0, number) / number / ky
        self.spatial_shifts = np.array([(shiftsx[i], shiftsy[i]) for i in range(number)])

    def normalize_spatial_waves(self):
        if not (0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0].amplitude * self.Mt * self.Mr
        for spatial_wave in self.waves.values():
            spatial_wave.amplitude /= norm

    def get_illumination_density(self, grid=None, coordinates=None, r=0, n=0):
        if not grid and not coordinates:
            raise ValueError("Either grid or coordinates must be provided!")
        if not grid and coordinates:
            if not len(coordinates) == 2:
                raise ValueError("Coordinates must be 2D for 2D illumination!")
            X, Y = np.meshgrid(coordinates)
            grid = np.stack((X, Y), axis=-1)

        else:
            if not len(grid.shape) == 3:
                raise ValueError("Grid must be 2D for 2D illumination!")

        illumination_density = np.zeros(grid.shape[:2], dtype=np.complex128)
        wavevectors, indices = self.get_wavevectors(r)
        for i in range(len(wavevectors)):
            phase = self.phase_matrix[(r, n, indices[i])]
            amplitude = self.waves[indices[i]].amplitude
            wavevector = wavevectors[i]
            illumination_density += Sources.IntensityHarmonic2D(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density

    def get_elementary_cell(self):
        ...


class PlaneWavesSIMNonlinear(PlaneWavesSIM):
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nonlinear_expansion_coefficients: tuple[int, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=[]):
        intensity_harmonics_dict_nonlinear = self.get_intensity_harmonics_nonlinear(intensity_harmonics_dict, nonlinear_expansion_coefficients)
        super().__init__(intensity_harmonics_dict_nonlinear, dimensions, Mr, spatial_shifts)

    @staticmethod
    def convolve_harmonics_expansions(harmonics_dict1, harmonics_dict2):
        harmonics_dict = {}
        for index1 in harmonics_dict1:
            for index2 in harmonics_dict2:
                index = tuple([index1[p] + index2[p] for p in range(len(index1))])
                harmonic = Sources.multiply_harmonics(harmonics_dict1[index1], harmonics_dict2[index2])
                if index not in harmonics_dict:
                    harmonics_dict[index] = harmonic
                else:
                    harmonics_dict[index] = Sources.add_harmonics(harmonics_dict[index], harmonic)
        return harmonics_dict

    @staticmethod
    def harmonics_to_the_power(harmonics_dict, power):
        harmonics_dict_power = {(0, 0, 0): Sources.IntensityHarmonic3D(1, 0, np.array([0, 0, 0]))}
        if not isinstance(power, int):
            raise ValueError("The operation for fractional power is not supported!")

        if power == 0:
            return harmonics_dict_power
        else:
            for i in range(power, 0, -1):
                harmonics_dict_power = PlaneWavesSIMNonlinear.convolve_harmonics_expansions(harmonics_dict_power, harmonics_dict)

        return harmonics_dict_power

    @staticmethod
    def scale_harmonics(harmonics_dict, scale_factor):
        for harmonic in harmonics_dict.values():
            harmonic.amplitude *= scale_factor
        return harmonics_dict

    @staticmethod
    def add_harmonics(harmonics_dict1, harmonics_dict2):
        harmonics_dict = harmonics_dict1.copy()
        for harmonic in harmonics_dict2:
            if harmonic in harmonics_dict:
                harmonics_dict[harmonic] = Sources.add_harmonics(harmonics_dict[harmonic], harmonics_dict2[harmonic])
            else:
                harmonics_dict[harmonic] = harmonics_dict2[harmonic]
        return harmonics_dict

    @staticmethod
    def get_intensity_harmonics_nonlinear(intensity_harmonics_dict, nonlinearity_expansion_coefficients):
        intensity_polynomial = {}
        for order in range(len(nonlinearity_expansion_coefficients)):
            harmonics_monomial = PlaneWavesSIMNonlinear.harmonics_to_the_power(intensity_harmonics_dict, order)
            harmonics_monomial = PlaneWavesSIMNonlinear.scale_harmonics(harmonics_monomial, nonlinearity_expansion_coefficients[order])
            intensity_polynomial = PlaneWavesSIMNonlinear.add_harmonics(intensity_polynomial, harmonics_monomial)
        return intensity_polynomial

    @classmethod
    def init_from_list(cls,
                       intensity_harmonics_list: list[Sources.IntensityHarmonic3D],
                       base_vector_lengths: tuple[float, ...],
                       nonlinear_expansion_coefficients: tuple[int, ...],
                       dimensions: tuple[int, ...],
                       Mr=1,
                       spatial_shifts=[]):
        intensity_harmonics_dict = cls.index_frequencies(intensity_harmonics_list, base_vector_lengths)
        return cls(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr=Mr)

    @classmethod
    def init_from_linear_illumination(cls,
                                      illumination: PlaneWavesSIM,
                                      nonlinear_expansion_coefficients: tuple[int, ...]):
        intensity_harmonics_dict = illumination.waves
        return cls(intensity_harmonics_dict, nonlinear_expansion_coefficients, illumination.dimensions, illumination.Mr, illumination.spatial_shifts)


class IlluminationNonLinearSIM2D(PlaneWavesSIMNonlinear, IlluminationPlaneWaves2D):
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nonlinear_expansion_coefficients: tuple[int, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=[]):
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)
        for wave in self.waves.keys():
            if wave[2] != 0:
                raise ValueError("Non-zero z-component of the wavevector is not allowed in 2D illumination!")
        self.waves = {key[:2]: self.waves[key] for key in self.waves}


class IlluminationNonLinearSIM3D(PlaneWavesSIMNonlinear, IlluminationPlaneWaves3D):
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nonlinear_expansion_coefficients: tuple[int, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=[]):
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)


class IlluminationNPhotonSIM2D(IlluminationNonLinearSIM2D):
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nphoton: int,
                 dimensions: tuple[int, int],
                 Mr=1,
                 spatial_shifts=((0, 0))):
        nonlinear_expansion_coefficients = tuple([0] * (nphoton - 1) + [1])
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)


class IlluminationNPhotonSIM3D(IlluminationNonLinearSIM3D):
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nphoton: int,
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=((0, 0, 0))):
        nonlinear_expansion_coefficients = tuple([0] * nphoton + [1])
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)
