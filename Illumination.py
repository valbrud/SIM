"""
Illumination.py

This module contains the Illumination class, which handles the simulation and analysis of illumination patterns in optical systems.

Classes:
    Illumination: Manages the properties and behavior of illumination patterns, including wavevectors and spatial shifts.
"""

import numpy as np
import Sources
from Sources import IntensityPlaneWave
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools
class Illumination:
    """
    Manages the properties and behavior of illumination patterns, including wavevectors and spatial shifts.

    Attributes:
        angles (np.ndarray): Array of rotation angles.
        _spatial_shifts (list): List of spatial shifts.
        _Mr (int): Number of rotations.
        Mt (int): Number of spatial shifts.
        waves (dict): Dictionary of intensity plane waves.
        wavevectors2d (list): List of 2D wavevectors.
        indices2d (list): List of 2D indices.
        wavevectors3d (list): List of 3D wavevectors.
        indices3d (list): List of 3D indices.
        rearranged_indices (dict): Dictionary of rearranged indices.
        xy_fourier_peaks (set): Set of 2D Fourier peaks.
        phase_matrix (dict): Dictionary of all phase the relevant phase shifts.
    """
    def __init__(self, intensity_plane_waves_dict: dict[tuple[int, int, int], Sources.IntensityPlaneWave], Mr = 1):
        """
        Initialize the Illumination object with given intensity plane waves and number of rotations.

        Args:
            intensity_plane_waves_dict (dict): Dictionary of intensity plane waves.
            Mr (int): Number of rotations.
        """
        self.angles = np.arange(0, np.pi, np.pi / Mr)
        self._spatial_shifts = [np.array((0, 0, 0)), ]
        self._Mr = Mr
        self.Mt = len(self.spatial_shifts)
        self.waves = {key : intensity_plane_waves_dict[key] for key in intensity_plane_waves_dict.keys() if not np.isclose(intensity_plane_waves_dict[key].amplitude, 0)}
        self.wavevectors2d, self.indices2d = self.get_wavevectors_projected(0)
        self.wavevectors3d, self.indices3d = self.get_wavevectors(0)
        self.rearranged_indices = self._rearrange_indices()
        self.xy_fourier_peaks = None
        self.phase_matrix = None
        self.compute_phase_matrix()

    @classmethod
    def init_from_list(cls, intensity_plane_waves_list: list[Sources.IntensityPlaneWave], base_vector_lengths: tuple[float, float, float], Mr = 1):
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
        return cls(intensity_plane_waves_dict, Mr = Mr)

    @staticmethod
    def index_frequencies(waves_list: list[Sources.IntensityPlaneWave], base_vector_lengths: tuple[float, float, float]) -> dict[tuple[int, int, int], Sources.IntensityPlaneWave]:
        """
        Static method to automatically index intensity plane waves given corresponding base vector lengths.

        Args:
            waves_list (list): List of plane waves.
            base_vector_lengths (tuple): Base vector lengths.

        Returns:
            dict: Dictionary of indexed frequencies.
        """
        intensity_plane_waves_dict = {}
        for wave in waves_list:
            wavevector = wave.wavevector
            m1, m2, m3 = int(round(wavevector[0]/base_vector_lengths[0])), int(round(wavevector[1]/base_vector_lengths[1])), \
                int(round(wavevector[2]/base_vector_lengths[2]))
            if not (m1, m2, m3) in intensity_plane_waves_dict.keys():
                intensity_plane_waves_dict[(m1, m2, m3)] = wave
            else:
                intensity_plane_waves_dict[(m1, m2, m3)].amplitude += wave.amplitude
        return intensity_plane_waves_dict

    def _rearrange_indices(self) -> dict[tuple[int, int], tuple[int]]:
        """
        Rearrange indices for the computation of effective OTFs, required in SIM.

        Returns:
            dict: Dictionary of rearranged indices.
        """
        indices = self.waves.keys()
        result_dict = {}
        for index in indices:
            key = index[:2]
            value = index[2]
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
        result_dict = {key: tuple(values) for key, values in result_dict.items()}
        return result_dict

    @staticmethod
    def find_ipw_from_pw(plane_waves) -> list[Sources.IntensityPlaneWave]:
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
        shiftsx = np.arange(0, number)/number/kx
        shiftsy = np.arange(0, number)/number/ky
        self.spatial_shifts = np.array([(shiftsx[i], shiftsy[i], 0) for i in range(number)])

    def get_wavevectors(self, r: int) -> tuple[list[np.ndarray], list[tuple[int]]]:
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
        for spatial_wave in self.waves.values():
            indices.append(spatial_wave)
            wavevector = VectorOperations.rotate_vector3d(
                spatial_wave.wavevector, np.array((0, 0, 1)), angle)
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

    def get_wavevectors_projected(self, r: int) -> tuple[list[np.ndarray], list[tuple[int]]]:
        """
        Get the projected wavevectors and indices for a given rotation.

        Args:
            r (int): Rotation index.

        Returns:
            tuple: List of projected wavevectors and list of indices.
        """
        wavevectors2d = []
        angle = self.angles[r]
        indices2d = []
        for spatial_wave in self.waves:
            if not spatial_wave[:2] in indices2d:
                indices2d.append(spatial_wave[:2])
                wavevector = VectorOperations.rotate_vector3d(
                    self.waves[spatial_wave].wavevector, np.array((0, 0, 1)), angle)
                wavevectors2d.append(wavevector[:2])
        return wavevectors2d, indices2d

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

    def normalize_spatial_waves(self):
        """
        Normalize the spatial waves on zero peak (i.e., a0 = 1).

        Raises:
            AttributeError: If zero wavevector is not found.
        """
        if not (0, 0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0, 0].amplitude * self.Mt * self.Mr
        for spatial_wave in self.waves.values():
            spatial_wave.amplitude /= norm

    def compute_expanded_lattice2d(self) -> set[tuple[int, int]]:
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

    def compute_expanded_lattice3d(self) -> set[tuple[int, int, int]]:
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

    def compute_phase_matrix(self):
        """
        Compute the dictionary of all the relevant phase shifts
         (products of spatial shifts and illumination pattern spatial frequencies).
        """
        self.phase_matrix = {}
        for r in range(self.Mr):
            for n in range(self.Mt):
                urn = VectorOperations.rotate_vector2d(self.spatial_shifts[n][:2], self.angles[r])
                wavevectors2d, keys2d = self.get_wavevectors_projected(r)
                for i in range(len(wavevectors2d)):
                    wavevector = wavevectors2d[i]
                    self.phase_matrix[(r, n, keys2d[i])] = np.exp(-1j * np.dot(urn, wavevector))