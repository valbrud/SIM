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

import Sources
import wrappers
from abc import abstractmethod
from Sources import IntensityPlaneWave
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools


class Illumination:

    @abstractmethod
    def get_illumination_density(self, **kwargs): ...


class PeriodicStructure:

    @abstractmethod
    def get_elementary_cell(self): ...


class Illumination2D(Illumination):
    def get_illumination_density(self, coordinates):
        if len(object.shape) != 2:
            raise ValueError("Object must be 2D!")


class Illumination3D(Illumination):
    def get_illumination_density(self, grid):
        if len(object.shape) != 3:
            raise ValueError("Object must be 3D!")

class IlluminationArray2D(Illumination2D, PeriodicStructure): ...


class IlluminationArray3D(Illumination2D, PeriodicStructure): ...


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

    def __init__(self, intensity_plane_waves_dict: dict[tuple[int, ...], Sources.IntensityPlaneWave], dimensions: tuple[int, ...], Mr=1, spatial_shifts=[]):
        """
        Collect the information describing the SIM experiment

        Args:
            intensity_plane_waves_dict (dict): Dictionary of intensity plane waves.
            Mr (int): Number of rotations.
        """
        self.angles = np.arange(0, np.pi, np.pi / Mr)
        self._Mr = Mr

        self._spatial_shifts = spatial_shifts
        self.Mt = len(self.spatial_shifts)

        self.waves = {key: intensity_plane_waves_dict[key] for key in intensity_plane_waves_dict.keys() if not np.isclose(intensity_plane_waves_dict[key].amplitude, 0)}
        self.dimensions = dimensions
        self.rearranged_indices = self._rearrange_indices(dimensions)

        self.sim_indices, self.projected_wavevectors = self.get_wavevectors_projected(0)

        self.phase_matrix = {}
        self.compute_phase_matrix()

    @classmethod
    def init_from_list(cls, intensity_plane_waves_list: list[Sources.IntensityPlaneWave], base_vector_lengths: tuple[float, ...], dimensions, Mr=1, spatial_shifts=[]):
        """
        Class method to initialize Illumination from a list of intensity plane waves.

        Args:
            intensity_plane_waves_list (list): List of intensity plane waves.
            base_vector_lengths (tuple): Base vector lengths of the illumination Fourier space Bravais lattice.
            Mr (int): Number of rotations.

        Returns:
            Illumination: Initialized Illumination object.
        """
        intensity_plane_waves_dict = cls.index_frequencies(intensity_plane_waves_list, base_vector_lengths)
        return cls(intensity_plane_waves_dict, dimensions, Mr=Mr)

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
    def glue_indices(sim_index, projected_index, dimensions=(1, 1, 0)) -> tuple[int, ...]:
        i, j = 0, 0
        index = []
        for dim in range(len(dimensions)):
            index.append(sim_index[i]) if dimensions[dim] else index.append(projected_index[j])
            if dimensions[dim]:
                i += 1
            else:
                j += 1
        return tuple(index)

    @staticmethod
    def index_frequencies(waves_list: list[Sources.IntensityPlaneWave], base_vector_lengths: tuple[float, float, float]) -> dict[tuple[int, int, int],
                                            Sources.IntensityPlaneWave]:
        intensity_plane_waves_dict = {}
        for wave in waves_list:
            wavevector = wave.wavevector
            m1, m2, m3 = int(round(wavevector[0] / base_vector_lengths[0])), int(round(wavevector[1] / base_vector_lengths[1])), \
                int(round(wavevector[2] / base_vector_lengths[2]))
            if not (m1, m2, m3) in intensity_plane_waves_dict.keys():
                intensity_plane_waves_dict[(m1, m2, m3)] = wave
            else:
                intensity_plane_waves_dict[(m1, m2, m3)].amplitude += wave.amplitude
        return intensity_plane_waves_dict


    @abstractmethod
    def compute_phase_matrix(self):
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
            index = self.glue_indices(sim_index, self.rearranged_indices[sim_index][0])
            wavevector = VectorOperations.rotate_vector3d(
                self.waves[index].wavevector, np.array((0, 0, 1)), angle)
            wavevectors_projected.append(wavevector[np.bool(np.array(self.dimensions))])
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

    @abstractmethod
    def compute_effective_kernels(self, psf, psf_coordinates):
        print("Child classes of SIM interface can compute effective kernels.")


class IlluminationPlaneWaves2D(Illumination2D, PlaneWavesSIM):
    def __init__(self, intensity_plane_waves_dict: dict[tuple[int, ...], Sources.IntensityPlaneWave], dimensions: tuple[int, int] = (1, 1), Mr: int = 1, spatial_shifts = np.array([(0, 0)])
):
        super().__init__(intensity_plane_waves_dict, dimensions, Mr, spatial_shifts)
        for wave in self.waves.keys():
            if wave[2] != 0:
                raise ValueError("Non-zero z-component of the wavevector is not allowed in 2D illumination!")
        self.waves = {key[:2]: self.waves[key] for key in self.waves}

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
        self.spatial_shifts = np.array([(shiftsx[i], shiftsy[i]) for i in range(number)])

    def compute_effective_kernels(self, kernel: np.ndarray[tuple[int, int], np.complex128], coordinates: tuple[2, np.ndarray]) -> tuple[
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
        X, Y = np.meshgrid(coordinates)
        for r in range(self.Mr):
            angle = self.angles[r]
            indices = self._rearrange_indices(self.dimensions)
            for sim_index in indices:
                effective_kernel = 0
                for projected_index in indices[sim_index]:
                    index = self.glue_indices(sim_index, projected_index, self.dimensions)
                    wavevector = VectorOperations.rotate_vector2d(
                        waves[index].wavevector[np.bool(np.array(self.dimensions))], angle)
                    amplitude = waves[index].amplitude
                    phase_shifted = np.transpose(np.exp(1j * np.einsum('ijk,i ->jk', np.array((X, Y)), wavevector)), axes=(1, 0)) * kernel
                    effective_kernel += amplitude * phase_shifted
                effective_kernels[(r, sim_index)] = effective_kernel
                effective_kernels_ft[(r, sim_index)] = wrappers.wrapped_fftn(effective_kernel)
        return effective_kernels, effective_kernels_ft
    def compute_phase_matrix(self):
        """
        Compute the dictionary of all the relevant phase shifts
         (products of spatial shifts and illumination pattern spatial frequencies).
        """
        self.phase_matrix = {}
        for r in range(self.Mr):
            for n in range(self.Mt):
                urn = VectorOperations.rotate_vector2d(self.spatial_shifts[n], self.angles[r])
                wavevectors, indices = self.get_wavevectors_projected(r)
                for i in range(len(wavevectors)):
                    wavevector = wavevectors[i]
                    self.phase_matrix[(r, n, indices[i])] = np.exp(-1j * np.dot(urn, wavevector))

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
            illumination_density += Sources.IntensityPlaneWave(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density

class IlluminationPlaneWaves3D(Illumination3D, PlaneWavesSIM):
    def __init__(self, intensity_plane_waves_dict: dict[tuple[int, ...], Sources.IntensityPlaneWave], dimensions: tuple[bool, bool, bool] = (1, 1, 0), Mr: int = 1, spatial_shifts=np.array([(0, 0, 0)])):
        super().__init__(intensity_plane_waves_dict, dimensions, Mr, spatial_shifts)

    @classmethod
    def init_from_list(cls, intensity_plane_waves_list: list[Sources.IntensityPlaneWave], base_vector_lengths: tuple[float, ...], dimensions = (1, 1, 0), Mr: int = 1, spatial_shifts=np.array([(0, 0, 0)])):
        return super().init_from_list(intensity_plane_waves_list, base_vector_lengths, dimensions, Mr, spatial_shifts)

    def normalize_spatial_waves(self):
        if not (0, 0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0, 0].amplitude * self.Mt * self.Mr
        for spatial_wave in self.waves.values():
            spatial_wave.amplitude /= norm

    @staticmethod
    def find_ipw_from_pw(plane_waves) -> tuple[IntensityPlaneWave, ...]:
        """
        Static method to find intensity plane waves
         (i.e. Fourier transform of the illumination pattern) from plane waves.

        Args:
            plane_waves (list): List of plane waves.

        Returns:
            list: List of intensity plane waves.
        """
        intensity_plane_waves = {}
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
                    if not wavevector_new in intensity_plane_waves.keys():
                        intensity_plane_waves[wavevector_new] = (Sources.IntensityPlaneWave(amplitude1 * amplitude2.conjugate(), 0,
                                                                                            np.array(wavevector_new)))
                    else:
                        intensity_plane_waves[wavevector_new].amplitude += amplitude1 * amplitude2.conjugate()
        return intensity_plane_waves.values()

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
        X, Y, Z = np.meshgrid(*coordinates)
        for r in range(self.Mr):
            angle = self.angles[r]
            indices = self._rearrange_indices(self.dimensions)
            for sim_index in indices:
                effective_kernel = 0
                for projected_index in indices[sim_index]:
                    index = self.glue_indices(sim_index, projected_index, self.dimensions)
                    wavevector = VectorOperations.rotate_vector3d(
                        waves[index].wavevector, np.array((0, 0, 1)), angle)
                    amplitude = waves[index].amplitude
                    phase_shifted = np.transpose(np.exp(1j * np.einsum('ijkl,i ->jkl', np.array((X, Y, Z)), wavevector)), axes=(1, 0, 2)) * kernel
                    effective_kernel += amplitude * phase_shifted
                effective_kernels[(r, sim_index)] = effective_kernel
                effective_kernels_ft[(r, sim_index)] = wrappers.wrapped_fftn(effective_kernel)
        return effective_kernels, effective_kernels_ft

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

    def compute_phase_matrix(self):
        """
        Compute the dictionary of all the relevant phase shifts
         (products of spatial shifts and illumination pattern spatial frequencies).
        """
        self.phase_matrix = {}
        for r in range(self.Mr):
            for n in range(self.Mt):
                urn = VectorOperations.rotate_vector3d(self.spatial_shifts[n], np.array((0, 0, 1)), self.angles[r])[np.bool(np.array(self.dimensions))]
                wavevectors, indices = self.get_wavevectors_projected(r)
                for i in range(len(wavevectors)):
                    wavevector = wavevectors[i]
                    self.phase_matrix[(r, n, indices[i])] = np.exp(-1j * np.dot(urn, wavevector))

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
            illumination_density += Sources.IntensityPlaneWave(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density

