"""
Illumination.py

This module contains the Illumination class, which handles the simulation and analysis of illumination patterns in optical systems.

Classes:
    Illumination: Manages the properties and behavior of illumination patterns, including wavevectors and spatial shifts.
"""

import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)


from typing import Dict, Tuple, Any, List

import numpy as np
from numpy import ndarray, dtype
from skimage.color.rgb_colors import dimgray

import Sources
import wrappers
from abc import abstractmethod, ABC
from Sources import IntensityHarmonic3D
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools
from ShiftsFinder import ShiftsFinder3d, ShiftsFinder2d
from Dimensions import *
from Dimensions import DimensionMetaAbstract

class Illumination(ABC, metaclass=DimensionMetaAbstract):
    """
    Abstract class for managing illumination patterns.
    """

    dimensionality = None

    @abstractmethod
    def get_illumination_density(self, **kwargs): ...


class PeriodicStructure():

    @abstractmethod
    def get_elementary_cell(self): ...


class IlluminationArray2D(PeriodicStructure, Illumination):
    dimensionality=2

class IlluminationArray3D(PeriodicStructure, Illumination): 
    dimensionality=3

class PlaneWavesSIM(Illumination, PeriodicStructure):
    """
    Manages the properties and behavior of illumination patterns in SIM 
    with a finite number of plane waves interfering.
    The base class implements all the functionality but cannot be implemented.
    Use dimensional children classes instead.


    Attributes:
        angles (np.ndarray): Array of rotation angles.
        _spatial_shifts (list): List of spatial shifts.
        _Mr (int): Number of rotations.
        Mt (int): Number of spatial shifts.
        waves (dict): Dictionary of intensity plane waves.
        phase_matrix (dict): Dictionary of all phase the relevant phase shifts.

    methods:
        init_from_list: Class method to initialize Illumination from a list of intensity plane waves.
        index_frequencies: Index the frequencies of the intensity harmonics.
        glue_indices: Glue the indices of the SIM and projected indices.
        get_wavevectors: Get the wavevectors for a given rotation.
        get_all_wavevectors: Get all wavevectors for all rotations.
        get_wavevectors_projected: Get the projected wavevectors for a given rotation.
        get_all_wavevectors_projected: Get all projected wavevectors for all rotations.
        get_base_vectors: Get the base vectors of the illumination Fourier space Bravais lattice.
        compute_effective_kernels: Compute effective kernels for SIM computations.
    """
    dimensionality = None  # Base class should not define dimensionality

    def __init__(self, intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic], dimensions: tuple[int, ...], Mr=1, spatial_shifts=[]):
        """
        Collect the information describing the SIM experiment

        Args:
            intensity_harmonics_dict (dict): Dictionary of intensity plane waves.
            Mr (int): Number of rotations.
        """
        self.angles = np.arange(0, np.pi, np.pi / Mr)
        self._Mr = Mr
        self.waves = {key: intensity_harmonics_dict[key] for key in intensity_harmonics_dict.keys() if not np.isclose(intensity_harmonics_dict[key].amplitude, 0)}

        self._spatial_shifts = spatial_shifts
        self.Mt = len(self.spatial_shifts)

        self.dimensions = dimensions
        self.rearranged_indices = self._rearrange_indices(dimensions)

        self.sim_indices, self.projected_wavevectors = self.get_wavevectors_projected(0)

        self.phase_matrix = {}
        self.compute_phase_matrix()

        self.electric_field_plane_waves = []

    @classmethod
    def init_from_list(cls,
                       intensity_harmonics_list: list[Sources.IntensityHarmonic],
                       base_vector_lengths: tuple[float, ...],
                       dimensions,
                       Mr=1,
                       spatial_shifts=None):
        """
        Class method to initialize Illumination from a list of intensity plane waves.

        Args:
            intensity_harmonics_list (list): List of intensity plane waves.
            base_vector_lengths (tuple): Base vector lengths of the illumination Fourier space Bravais lattice.
            dimensions (tuple): Indicates which SIM dimensions are projective(0) and true(1). In this notation
            'true' means that illumination shifts are necessary for disentanglement in this direction. For example,
            standard 3D SIM has dimensions (1, 1, 0), as there are now shifts in the z-direction.
            Mr (int): Number of rotations.
            spatial_shifts np.ndarray: Spatial shifts for the illumination pattern.
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
    
    def get_illumination_density(self, grid=None, coordinates=None, depth=None, r=0, n=0):
        """
        Get the illumination density for a given grid or coordinates.

        Args:
            grid (np.ndarray): Grid of coordinates.
            coordinates (tuple): Coordinates.
            depth (float): Depth of the illumination.
            r (int): Rotation index.
            n (int): Spatial shift index.

        Returns:
            np.ndarray: Illumination density.
        """
        pass

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

    def get_amplitudes(self) -> tuple[list[complex, ...], list[tuple[int, ...]]]:
        """
         Get the amplitudes and indices of the harmonics.

         Returns:
             tuple: List of amplitudes and list of indices.
         """
        amplitudes = [self.waves[index].amplitude for index in self.waves.keys()]
        indices = [index for index in self.waves.keys()]
        return amplitudes, indices

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
            wavevector = np.copy(self.waves[index].wavevector)
            wavevector = VectorOperations.rotate_vector2d(wavevector[:2], angle)
            wavevectors.append(wavevector)
        return np.array(wavevectors), tuple(indices)

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
        return np.array(wavevectors)

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
        return np.array(wavevectors_projected), tuple(sim_indices)

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
        return np.array(wavevectors2d)

    def get_base_vectors(self) -> tuple[float, ...]:
        """
        Get the base vectors of the illumination Fourier space Bravais lattice.

        Returns:
            tuple: Base vectors of the illumination Fourier space Bravais lattice.
        """
        base_vectors = np.zeros(len(self.dimensions))
        for i in range(len(self.dimensions)):
            for wave in self.waves.keys():
                if wave[i] != 0:
                    base_vectors[i] = self.waves[wave].wavevector[i] / wave[i]
                else :
                    base_vectors[i] = 0
        
        return tuple(base_vectors)

    @abstractmethod
    def set_spatial_shifts_diagonally(self, number: int = 0):
        """
        Set the spatial shifts diagonally (i.e., all the spatial shifts are assumed to be on the same line).
        This is the most common use in practice.
        Appropriate shifts for a given illumination pattern can be computed in the module 'ShiftsFinder.py'

        Args:
            number (int): Number of shifts.
        """
        pass

    def compute_effective_kernels(self, kernel: np.ndarray, coordinates: tuple[3, np.ndarray]) -> tuple[
        dict[tuple[int, tuple[int, ...]], np.ndarray], dict[tuple[int, tuple[int, ...]], np.ndarray]]:
        """
        Compute effective kernels for SIM computations

        Args:
            kernel(np.ndarray): SIM reconstruction kernel, e.g., OTF.
            coordinates(tuple): coordinates
            and which are fixed w.r.t. focal plane (0), i.e. projected

        Returns:
            tuple: Effective kernels and their Fourier transform.
        """
        waves = self.waves
        effective_kernels = {}
        effective_kernels_ft = {}
        grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), -1)
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
                        raise ValueError(f"{len(self.dimensions)} dimensions is meaningless in the context of microscopy!")
                    effective_kernel += amplitude * phase_shifted
                effective_kernels[(r, sim_index)] = effective_kernel
                effective_kernels_ft[(r, sim_index)] = wrappers.wrapped_fftn(effective_kernel)
        return effective_kernels, effective_kernels_ft

    def get_phase_modulation_patterns(self, coordinates):
        phase_modulation_patterns = {}
        grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)
        for r in range(self.Mr):
            for sim_index in self.rearranged_indices:
                projective_index = self.rearranged_indices[sim_index][0]
                index = self.glue_indices(sim_index, projective_index, self.dimensions)
                wavevector = np.copy(self.waves[index].wavevector)
                wavevector[:2] = VectorOperations.rotate_vector2d(wavevector[:2], self.angles[r])
                wavevector[np.bool(1 - np.array(self.dimensions))] = 0
                if len(self.dimensions) == 2:
                    phase_modulation = np.exp(1j * np.einsum('ijl,l ->ij', grid, wavevector))
                elif len(self.dimensions) == 3:
                    phase_modulation = np.exp(1j * np.einsum('ijkl,l ->ijk', grid, wavevector))
                else:
                    raise ValueError("The number of dimensions is meaningless in the context of microscopy!")
                phase_modulation_patterns[r, sim_index] = phase_modulation
            # plt.imshow(np.real(phase_modulation_patterns[r, sim_index].real), cmap='gray')
            # plt.show()
        return phase_modulation_patterns

    def compute_phase_matrix(self):
        """
        Compute the dictionary of all the relevant phase shifts
         (products of spatial shifts and illumination pattern spatial frequencies).
        """
        self.phase_matrix = {}
        for n in range(self.Mt):
            urn = self.spatial_shifts[n]
            wavevectors, indices = self.get_wavevectors_projected(0)
            for i in range(len(wavevectors)):
                wavevector = wavevectors[i]
                test = np.dot(urn[np.bool(np.array(self.dimensions))], wavevector)
                self.phase_matrix[(n, indices[i])] = np.exp(-1j * np.dot(urn[np.bool(np.array(self.dimensions))], wavevector))


class IlluminationPlaneWaves3D(PlaneWavesSIM):
    dimensionality=3
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 dimensions: tuple[bool, bool, bool] = (1, 1, 0),
                 Mr: int = 1, spatial_shifts=np.array([(0, 0, 0)])):
        super().__init__(intensity_harmonics_dict, dimensions, Mr, spatial_shifts)

    @classmethod
    def init_from_list(cls,
                       intensity_harmonics_list: list[Sources.IntensityHarmonic3D],
                       base_vector_lengths: tuple[float, ...],
                       dimensions=(1, 1, 0),
                       Mr: int = 1,
                       spatial_shifts=np.array([(0, 0, 0)])):
        return super().init_from_list(intensity_harmonics_list, base_vector_lengths, dimensions, Mr, spatial_shifts)

    @classmethod
    def init_from_plane_waves(cls, 
                              plane_waves: list[Sources.PlaneWave],
                              base_vector_lengths: tuple[float, ...],
                              dimensions,
                              Mr=1,
                              spatial_shifts=np.array([(0, 0, 0)]), 
                              store_plane_waves=False):
        """
        Class method to initialize Illumination from a list of plane waves.

        Args:
            plane_waves_list (list): List of plane waves.
            base_vector_lengths (tuple): Base vector lengths of the illumination Fourier space Bravais lattice.
            dimensions (tuple): Indicates which SIM dimensions are projective(0) and true(1). In this notation
            'true' means that illumination shifts are necessary for disentanglement in this direction. For example,
            standard 3D SIM has dimensions (1, 1, 0), as there are now shifts in the z-direction.
            Mr (int): Number of rotations.
            spatial_shifts np.ndarray: Spatial shifts for the illumination pattern.
        Returns:
            Illumination: Initialized Illumination object.
        """
        intensity_harmonics_list = IlluminationPlaneWaves3D.find_ipw_from_pw(plane_waves)
        intensity_harmonics_dict = cls.index_frequencies(intensity_harmonics_list, base_vector_lengths)
        illumination = cls(intensity_harmonics_dict, dimensions, Mr=Mr)
        if store_plane_waves:
            illumination.electric_field_plane_waves = plane_waves
        return illumination
    

    def normalize_spatial_waves(self):
        if not (0, 0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0, 0].amplitude * self.Mt * self.Mr
        for spatial_wave in self.waves.values():
            spatial_wave.amplitude = spatial_wave.amplitude * np.exp(1j * spatial_wave.phase)
            spatial_wave.phase = 0
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
    def find_ipw_from_pw(plane_waves) -> dict[tuple[Any], IntensityHarmonic3D].values:
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
                    if wavevector_new not in intensity_harmonics.keys():
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

    def set_spatial_shifts_diagonally(self, number: int = 0):
        expanded_lattice = self.compute_expanded_lattice()
        shift_ratios = ShiftsFinder3d.get_shift_ratios(expanded_lattice)
        bases = sorted(list(shift_ratios.keys()))
        base, ratios = bases[number], list(shift_ratios[bases[number]])[0]
        base_vectors = np.array(self.get_base_vectors())
        shifts = 2 * np.pi / np.where(base_vectors, base_vectors, np.inf) * ratios / base
        self.spatial_shifts = np.array([shifts * i for i in range(base)])

    def get_illumination_density(self, grid=None, coordinates=None, depth=None, r=0, n=0):
        if grid is None and coordinates is None:
            raise ValueError("Either grid or coordinates must be provided!")
        if grid is None and not coordinates is None:
            X, Y, Z = np.meshgrid(*coordinates)
            grid = np.stack((X, Y, Z), axis=-1)
        if depth:
            grid[:, :, :, 2] -= depth

        illumination_density = np.zeros(grid.shape[:3], dtype=np.complex128)
        wavevectors, indices = self.get_wavevectors(r)
        for i in range(len(wavevectors)):
            phase = self.phase_matrix[(n, indices[i])]
            amplitude = self.waves[indices[i]].amplitude
            wavevector = wavevectors[i]
            illumination_density += Sources.IntensityHarmonic3D(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density.real

    def get_elementary_cell(self):
        ...


class IlluminationPlaneWaves2D(PlaneWavesSIM):
    dimensionality=2
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
        illumination = cls(intensity_harmonics_dict, dimensions, illumination_3d.Mr, spatial_shifts)
        if illumination_3d.electric_field_plane_waves:
            illumination.electric_field_plane_waves = illumination_3d.electric_field_plane_waves
        return illumination
    
    @staticmethod
    def index_frequencies(waves_list: list[Sources.IntensityHarmonic2D], base_vector_lengths: tuple[float, float]) -> dict[tuple[int, int], Sources.IntensityHarmonic2D]:
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
        self.xy_fourier_peaks = set((mx, my) for mx, my in self.waves.keys())
        expanded_lattice2d = set()
        for peak1 in self.xy_fourier_peaks:
            for peak2 in self.xy_fourier_peaks:
                expanded_lattice2d.add((peak1[0] - peak2[0], peak1[1] - peak2[1]))
        print(len(expanded_lattice2d))
        return expanded_lattice2d

    def set_spatial_shifts_diagonally(self, number: int = 0):
        expanded_lattice = self.compute_expanded_lattice()
        shift_ratios = ShiftsFinder2d.get_shift_ratios(expanded_lattice)
        bases = sorted(list(shift_ratios.keys()))
        base, ratios = bases[number], list(shift_ratios[bases[number]])[0]
        base_vectors = np.array(self.get_base_vectors())
        shifts = 2 * np.pi / np.where(base_vectors, base_vectors, np.inf) * ratios / base
        self.spatial_shifts = np.array([shifts * i for i in range(base)])

    def normalize_spatial_waves(self):
        if not (0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0].amplitude * self.Mt * self.Mr
        for spatial_wave in self.waves.values():
            spatial_wave.amplitude /= norm

    def get_illumination_density(self, grid=None, coordinates=None, r=0, n=0):
        if grid is None and coordinates is None:
            raise ValueError("Either grid or coordinates must be provided!")
        if grid is None and not  coordinates is None:
            if not len(coordinates) == 2:
                raise ValueError("Coordinates must be 2D for 2D illumination!")
            X, Y = np.meshgrid(*coordinates, indexing='ij')
            grid = np.stack((X, Y), axis=-1)

        else:
            if not len(grid.shape) == 3:
                raise ValueError("Grid must be 2D for 2D illumination!")

        illumination_density = np.zeros(grid.shape[:2], dtype=np.complex128)
        wavevectors, indices = self.get_wavevectors(r)
        for i in range(len(wavevectors)):
            phase = self.phase_matrix[(n, indices[i])]
            amplitude = self.waves[indices[i]].amplitude
            wavevector = wavevectors[i]
            illumination_density += Sources.IntensityHarmonic2D(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density.real

    def get_elementary_cell(self):
        ...


class PlaneWavesSIMNonlinear(PlaneWavesSIM):

    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic],
                 nonlinear_expansion_coefficients: tuple[float, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=None):
        intensity_harmonics_dict_nonlinear = self.get_intensity_harmonics_nonlinear(intensity_harmonics_dict, nonlinear_expansion_coefficients)
        super().__init__(intensity_harmonics_dict_nonlinear, dimensions, Mr, spatial_shifts)

    @staticmethod
    def convolve_harmonics_expansions(harmonics_dict1, harmonics_dict2):
        harmonics_dict = {}
        for index1 in harmonics_dict1:
            for index2 in harmonics_dict2:
                index = tuple([index1[p] + index2[p] for p in range(len(index1))])
                harmonic = harmonics_dict1[index1] * harmonics_dict2[index2]
                if index not in harmonics_dict:
                    harmonics_dict[index] = harmonic
                else:
                    harmonics_dict[index] += harmonic
        return harmonics_dict

    @staticmethod
    def harmonics_to_the_power(harmonics_dict, power):
        if len(next(iter(harmonics_dict.keys()))) == 2:
            harmonics_dict_power = {(0, 0): Sources.IntensityHarmonic2D(1, 0, np.array([0, 0]))}
        elif len(next(iter(harmonics_dict.keys()))) == 3:
            harmonics_dict_power = {(0, 0, 0): Sources.IntensityHarmonic3D(1, 0, np.array([0, 0, 0]))}
        else:
            raise ValueError("The number of dimensions is meaningless in the context of microscopy!")

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
                harmonics_dict[harmonic] += harmonics_dict2[harmonic]
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
                       intensity_harmonics_list: list[Sources.IntensityHarmonic],
                       base_vector_lengths: tuple[float, ...],
                       dimensions: tuple[int, ...],
                       Mr=1,
                       spatial_shifts=[],
                       nonlinear_expansion_coefficients: tuple[float, ...] = (0, 1),
                       ):
        intensity_harmonics_dict = cls.index_frequencies(intensity_harmonics_list, base_vector_lengths)
        return cls(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr=Mr)

    @classmethod
    def init_from_linear_illumination(cls,
                                      illumination: PlaneWavesSIM,
                                      nonlinear_expansion_coefficients: tuple[float, ...]):
        intensity_harmonics_dict = illumination.waves
        return cls(intensity_harmonics_dict, nonlinear_expansion_coefficients, illumination.dimensions, illumination.Mr, illumination.spatial_shifts)

    @classmethod
    def init_from_linear_illumination_no_taylor(cls,
                                                illumination: PlaneWavesSIM,
                                                nonlinear_dependence: lambda x: float):
        ...


class IlluminationNonLinearSIM2D(PlaneWavesSIMNonlinear, IlluminationPlaneWaves2D):
    dimensionality=2
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nonlinear_expansion_coefficients: tuple[int, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=[]):
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)


class IlluminationNonLinearSIM3D(PlaneWavesSIMNonlinear, IlluminationPlaneWaves3D):
    dimensionality=3
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nonlinear_expansion_coefficients: tuple[int, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=None):
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)


class IlluminationNPhotonSIM2D(IlluminationNonLinearSIM2D):
    dimensionality=2
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nphoton: int,
                 dimensions: tuple[int, int],
                 Mr=1,
                 spatial_shifts=np.array((0, 0))):
        nonlinear_expansion_coefficients = tuple([0] * (nphoton - 1) + [1])
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)


class IlluminationNPhotonSIM3D(IlluminationNonLinearSIM3D):
    dimensionality=3
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nphoton: int,
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=np.array((0, 0, 0))):
        nonlinear_expansion_coefficients = tuple([0] * nphoton + [1])
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts)
