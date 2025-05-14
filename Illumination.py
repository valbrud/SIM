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
from warnings import warn

import numpy as np
from numpy import ndarray, dtype
from skimage.color.rgb_colors import dimgray
import scipy 
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
import copy 
from stattools import off_grid_ft

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
        index_harmonics: Index the frequencies of the intensity harmonics.
        glue_indices: Glue the indices of the SIM and projected indices.
        get_wavevectors: Get the wavevectors for a given rotation.
        get_all_wavevectors: Get all wavevectors for all rotations.
        get_wavevectors_projected: Get the projected wavevectors for a given rotation.
        get_all_wavevectors_projected: Get all projected wavevectors for all rotations.
        get_base_vectors: Get the base vectors of the illumination Fourier space Bravais lattice.
        compute_effective_kernels: Compute effective kernels for SIM computations.
    """
    dimensionality = None  # Base class should not define dimensionality

    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...],
                 Sources.IntensityHarmonic], 
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=None,
                 angles=None,
                 ):
        """
        Collect the information describing the SIM experiment

        Args:
            intensity_harmonics_dict (dict): Dictionary of intensity plane waves.
            Mr (int): Number of rotations.
        """
        self._Mr = Mr
        self.angles = np.array(angles) if angles is not None else np.arange(0, np.pi, np.pi / Mr)

        key = list(intensity_harmonics_dict.keys())[0]
        if type(key[0]) is int and  type(key[1]) is tuple:
            self.harmonics = {key: intensity_harmonics_dict[key] for key in intensity_harmonics_dict.keys() if not np.isclose(intensity_harmonics_dict[key].amplitude, 0)}
        else:
            if not type(key) is tuple:
                raise ValueError("Urecognizable indexing of the intensity harmonics!")
            else:
                self.harmonics = self._compute_harmonics_for_all_rotations(intensity_harmonics_dict)

        self._spatial_shifts = spatial_shifts if not spatial_shifts is None else self._get_zero_shifts()
        self.Mt = self.spatial_shifts.shape[1]

        self.dimensions = dimensions
        self.rearranged_indices = self._rearrange_indices(dimensions)

        self.sim_indices, self.projected_wavevectors = self.get_wavevectors_projected(0)

        self._phase_matrix = {}
        self.compute_phase_matrix()

        self.electric_field_plane_waves = []

    def _compute_harmonics_for_all_rotations(self, intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic]) -> dict[tuple[int, tuple[int, ...]], Sources.IntensityHarmonic]:
        """
        Compute the wavevectors for all rotations.

        Returns:
            list: List of wavevectors for all rotations.
        """
        intensity_harmonics_dict_full = {}
        intensity_harmonics_dict_filtered = {key: intensity_harmonics_dict[key] for key in intensity_harmonics_dict.keys() if not np.isclose(intensity_harmonics_dict[key].amplitude, 0)}
        for r in range(self.Mr):
            for key in intensity_harmonics_dict_filtered.keys():
                wavevector = np.copy(intensity_harmonics_dict[key].wavevector)
                wavevector[:2] = VectorOperations.rotate_vector2d(wavevector[:2], self.angles[r])
                harmonic_rotated = copy.deepcopy(intensity_harmonics_dict[key])
                harmonic_rotated.wavevector = wavevector
                intensity_harmonics_dict_full[(r, key)] = harmonic_rotated
        return intensity_harmonics_dict_full
    
    def _get_zero_shifts(self):
        shifts = np.zeros((self.Mr, 1, self.dimensionality))
        return shifts
    
    @classmethod
    def init_from_list(cls,
                       intensity_harmonics_list: list[Sources.IntensityHarmonic],
                       base_vector_lengths: tuple[float, ...],
                       dimensions,
                       Mr=1,
                       spatial_shifts=None, 
                       angles=None,
                       ) -> 'PlaneWavesSIM':
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
        intensity_harmonics_dict = cls.index_harmonics(intensity_harmonics_list, base_vector_lengths)
        return cls(intensity_harmonics_dict, dimensions, Mr, spatial_shifts, angles)
    
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
        if not new_spatial_shifts.shape[-1] == self.dimensionality:
            raise ValueError(f"Spatial shifts must be of shape (Mr, Mt, {self.dimensionality}) or (Mt, {self.dimensionality})!")
        if not len(new_spatial_shifts.shape) == 3 and not len(new_spatial_shifts.shape) == 2:
            raise ValueError(f"Spatial shifts must be of shape (Mr, Mt, {self.dimensionality}) or (Mt, {self.dimensionality})!")
        
        if len(new_spatial_shifts.shape) == 3:
            self.Mt = new_spatial_shifts.shape[1]
            if not new_spatial_shifts.shape[0] == self.Mr:
                raise ValueError(f"Spatial shifts Mr dimension should either be equal to the illumination.Mr one or be absent)!")
            self._spatial_shifts = new_spatial_shifts

        if len(new_spatial_shifts.shape) == 2:
            self.Mt = new_spatial_shifts.shape[0]
            self._spatial_shifts = np.zeros((self.Mr, self.Mt, self.dimensionality))
            for i in range(self.Mr):
                self._spatial_shifts[i] = new_spatial_shifts

        self.normalize_spatial_waves()
        self.compute_phase_matrix()
    
    @property
    def phase_matrix(self):
        return self._phase_matrix
    
    @phase_matrix.setter
    def phase_matrix(self, new_phase_matrix):
        shift_indices = set(index[1] for index in new_phase_matrix.keys())
        self.Mt = len(shift_indices)
        self._phase_matrix = new_phase_matrix

    def _rearrange_indices(self, dimensions) -> dict[tuple[int, ...], tuple[tuple[int, ...]]]:
        """
        Rearrange indices for the computation of effective OTFs, required in SIM.

        Returns:
            dict: Dictionary of rearranged indices.
        """
        indices = self.harmonics.keys()
        result_dict = {}
        for index in indices:
            r = index[0]
            sim_index = index[1]
            key = (r, tuple([sim_index[dim] for dim in range(len(dimensions)) if dimensions[dim]]))
            value = tuple([sim_index[dim] for dim in range(len(dimensions)) if not dimensions[dim]])
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
        result_dict = {key: tuple(values) for key, values in result_dict.items()}
        return result_dict

    @staticmethod
    @abstractmethod
    def index_harmonics(waves_list: list[Sources.IntensityHarmonic], base_vector_lengths: tuple[float, ...]) -> dict[tuple[int, ...], Sources.IntensityHarmonic]:
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
            index.append(sim_index[1][i]) if dimensions[dim] else index.append(projected_index[j])
            if dimensions[dim]:
                i += 1
            else:
                j += 1
        return (sim_index[0], tuple(index))

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

    def normalize_spatial_waves(self):
        for r in range(self.Mr):
            zero_index = (r, tuple([0] * len(self.dimensions)))
            if not zero_index in self.harmonics.keys():
                raise AttributeError("Zero wavevector is not found! No constant power in the illumination!")
            norm = self.harmonics[zero_index].amplitude * self.Mt * self.Mr
            for harmonic in self.harmonics.keys():
                if harmonic[0] == r:
                    spatial_wave = self.harmonics[harmonic]
                    spatial_wave.amplitude = spatial_wave.amplitude * np.exp(1j * spatial_wave.phase)
                    spatial_wave.phase = 0
                    spatial_wave.amplitude /= norm


    def compute_expanded_lattice(self, r=0, ignore_projected_dimensions=True) -> set[tuple[int, int]]:
        """
        Compute the expanded lattice of Fourier peaks
         (autoconvoluiton of Fourier transform of the illumination pattern).
        Parameters:
            ignore_projected_dimensions (bool): If True, lattice is only computed for the dimensions, in which the illumination pattern is shifted.
        Returns:
            set: Set of expanded lattice peaks.
        """
        fourier_peaks = tuple(key[1] for key in self.harmonics.keys() if key[0] == 0)
        lattice = set()
        if ignore_projected_dimensions:
            for peak in fourier_peaks:
                peak_projected = tuple(peak[dim] for dim in range(len(self.dimensions)) if self.dimensions[dim])
                lattice.add(peak_projected)
        else:
            lattice = set(fourier_peaks)

        expanded_lattice = set()

        for peak1 in lattice:
            for peak2 in lattice:
                expanded_lattice.add(tuple([peak1[i] - peak2[i] for i in range(len(peak1))]))
        print(len(expanded_lattice))
        return expanded_lattice

    def get_amplitudes(self, r=0) -> tuple[list[complex], list[tuple[int, ...]]]:
        """
         Get the amplitudes and indices of the harmonics.

         Returns:
             tuple: List of amplitudes and list of indices.
         """
        amplitudes = [self.harmonics[index].amplitude for index in self.harmonics.keys() if index[0] == r]
        indices = [index for index in self.harmonics.keys() if index[0] == r]
        return amplitudes, indices

    def get_all_amplitudes(self) -> list[complex]:
        """
         Get all amplitudes for all rotations.

         Returns:
             list: List of all amplitudes.
         """
        amplitudes = []
        for r in range(self.Mr):
            amplitudes_r, _ = self.get_amplitudes(r)
            amplitudes.extend(amplitudes_r)
        return np.array(amplitudes)

    def get_wavevectors(self, r: int) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
        """
         Get the wavevectors and indices for a given rotation.

         Args:
             r (int): Rotation index.

         Returns:
             tuple: List of wavevectors and list of indices.
         """
        wavevectors = [self.harmonics[index].wavevector for index in self.harmonics.keys() if index[0] == r]
        indices = [index for index in self.harmonics.keys() if index[0] == r]
        return np.array(wavevectors), tuple(indices)

    def get_all_wavevectors(self) -> list[np.ndarray]:
        """
        Get all wavevectors for all rotations.

        Returns:
            list: List of all wavevectors.
        """
        wavevectors = []
        indices = []
        for r in range(self.Mr):
            wavevectors_r, indices_r = self.get_wavevectors(r)
            wavevectors.extend(wavevectors_r)
            indices.extend(indices_r)
        return np.array(wavevectors), tuple(indices)

    def get_wavevectors_projected(self, r: int) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
        """
        Get the projected wavevectors and indices for a given rotation.

        Args:
            r (int): Rotation index.

        Returns:
            tuple: List of projected wavevectors and list of indices.
        """
        sim_indices = [key for key in self.rearranged_indices.keys() if key[0] == r]
        wavevectors_projected = []
        for sim_index in sim_indices:
            index = self.glue_indices(sim_index, self.rearranged_indices[sim_index][0], self.dimensions)
            wavevector = self.harmonics[index].wavevector
            wavevectors_projected.append(wavevector[np.bool(self.dimensions)])
        return np.array(wavevectors_projected), tuple(sim_indices)
        
    def get_all_wavevectors_projected(self):
        """
        Get all projected wavevectors for all rotations.

        Returns:
            list: List of all projected wavevectors.
        """
        wavevectors_proj = []
        sim_indices_proj = []
        for r in range(self.Mr):
            wavevectors_r, indices_r = self.get_wavevectors_projected(r)
            wavevectors_proj.extend(wavevectors_r)
            sim_indices_proj.extend(indices_r)
        return np.array(wavevectors_proj), tuple(sim_indices_proj)

    def get_base_vectors(self, r) -> tuple[float, ...]:
        """
        Get the base vectors of the illumination Fourier space Bravais lattice.

        Returns:
            tuple: Base vectors of the illumination Fourier space Bravais lattice.
        """
        base_vectors = np.zeros(self.dimensionality)
        for i in range(self.dimensionality):
            for index in self.harmonics.keys():
                if index[0] == r:
                    wavevector = np.copy(self.harmonics[index].wavevector)
                    aligned_wavevector = VectorOperations.rotate_vector2d(wavevector[:2], -self.angles[r])
                    if index[1][i] != 0:
                        base_vectors[i] = aligned_wavevector[i] / index[1][i]
                        break
        return base_vectors

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
        harmonics = self.harmonics
        effective_kernels = {}
        effective_kernels_ft = {}
        grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), -1)
        indices = self.rearranged_indices
        for sim_index in indices:
            effective_kernel = 0
            for projected_index in indices[sim_index]:
                index = self.glue_indices(sim_index, projected_index, self.dimensions)
                wavevector = harmonics[index].wavevector.copy()
                amplitude = harmonics[index].amplitude
                if self.dimensionality == 2:
                    phase_shifted = np.exp(-1j * np.einsum('ijl,l ->ij', grid, wavevector)) * kernel
                elif self.dimensionality == 3:
                    phase_shifted = np.transpose(np.exp(-1j * np.einsum('ijkl,l ->ijk', grid, wavevector)), axes=(1, 0, 2)) * kernel
                effective_kernel += amplitude * phase_shifted
            
            # effective_kernel /= np.sum(np.abs(effective_kernel))
            effective_kernels[sim_index] = effective_kernel
            effective_kernels_ft[sim_index] = wrappers.wrapped_fftn(effective_kernel)
            # effective_kernels_ft[sim_index] /= np.amax(np.abs(effective_kernels_ft[sim_index]))
            # plt.imshow(np.abs(effective_kernels_ft[sim_index]).T, cmap='gray', origin='lower')
            # plt.title(f"Effective kernel {sim_index}")
            # plt.show()
        return effective_kernels, effective_kernels_ft

    def get_phase_modulation_patterns(self, coordinates):
        phase_modulation_patterns = {}
        grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)
        wavevectors, indices = self.get_all_wavevectors_projected()
        for sim_index, wavevector in zip(indices, wavevectors):
            if self.dimensionality == 2:
                phase_modulation = np.exp(-1j * np.einsum('ijl,l ->ij', grid, wavevector))
            elif self.dimensionality == 3:
                phase_modulation = np.exp(-1j * np.einsum('ijkl,l ->ijk', grid, wavevector))
            phase_modulation_patterns[sim_index] = phase_modulation
        # plt.imshow(np.real(phase_modulation_patterns[r, sim_index].real), cmap='gray')
        # plt.show()
        return phase_modulation_patterns

    def compute_phase_matrix(self):
        """
        Compute the dictionary of all the relevant phase shifts
         (products of spatial shifts and illumination pattern spatial frequencies).
        """
        self._phase_matrix = {}
        for r in range(self.Mr):
            wavevectors, indices = self.get_wavevectors_projected(r)
            for n in range(self.Mt):
                urn = self.spatial_shifts[r, n]
                for wavevector, index in zip(wavevectors, indices):
                    test = np.dot(urn[np.bool(np.array(self.dimensions))], wavevector)
                    self.phase_matrix[r, n, index[1]] = np.exp(-1j * np.dot(urn[np.bool(np.array(self.dimensions))], wavevector))

    def estimate_modulation_coefficients(self, stack, psf, grid, update=False, method='least_squares'): 
        """
        Estimate the modulation coefficients from a given image.

        Args:
            image (np.ndarray): Image.
            grid (np.ndarray): Grid of coordinates.
            update (bool): If True, update the coefficients of the illumination harmonics.

        Returns:
            np.ndarray: Estimated modulation coefficients.
        """
        
        flat_grid = grid.reshape(-1, self.dimensionality)
        am = {}
        for r in range(self.Mr):
            harmonics = {index[1]: harmonic for index, harmonic in self.harmonics.items() if index[0] == r}
            wavevectors = np.zeros((len(harmonics), self.dimensionality))
            for i, harmonic in enumerate(harmonics.keys()):
                wavevectors[i] = harmonics[harmonic].wavevector
            
            amr = np.zeros((len(harmonics)), dtype=np.complex128)
            otfs = off_grid_ft(psf, grid, wavevectors / (2 * np.pi))

            if method == 'least_squares':
                phases = flat_grid @ wavevectors.T

                for n in range(self.Mt):
                    phase_vector = np.zeros((wavevectors.shape[0]), dtype=np.complex128)
                    for i in range(len(wavevectors)):
                        phase_vector[i] = self._phase_matrix[r, n, tuple(harmonics.keys())[i]]
                    phase_modulation = np.exp(1j * phases) * phase_vector[None, :]
                    # M = np.hstack((np.cos(phase_modulation), np.sin(phase_modulation)))
                    image = stack[r, n]
                    flat_image = image.ravel()
                    amr, *_ = scipy.linalg.lstsq(phase_modulation, flat_image, cond=None) 

            if method == 'peak_height_ratio':
                for n in range(self.Mt):
                    ft = off_grid_ft(stack[r, n], grid, np.array(wavevectors / (2 * np.pi)))
                    amr = np.abs(ft)

            amr /= np.abs((otfs * self.Mt))
            amr /= np.amax(np.abs(amr))
            for i in range(len(harmonics)):
                am[(r, tuple(harmonics.keys())[i])] = amr[i]

        if update:
            for harmonic in self.harmonics.keys():
                self.harmonics[harmonic].amplitude = am[harmonic]
        return am

    
class IlluminationPlaneWaves3D(PlaneWavesSIM):
    dimensionality=3
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 dimensions: tuple[bool, bool, bool] = (1, 1, 0),
                 Mr: int = 1,
                 spatial_shifts=None,
                 angles=None):
        super().__init__(intensity_harmonics_dict, dimensions, Mr, spatial_shifts, angles)

    @classmethod
    def init_from_list(cls,
                       intensity_harmonics_list: list[Sources.IntensityHarmonic3D],
                       base_vector_lengths: tuple[float, ...],
                       dimensions=(1, 1, 0),
                       Mr: int = 1,
                       spatial_shifts=None,
                       angles=None,
                       ):
        return super().init_from_list(intensity_harmonics_list, base_vector_lengths, dimensions, Mr, spatial_shifts, angles)

    @classmethod
    def init_from_plane_waves(cls, 
                              plane_waves: list[Sources.PlaneWave],
                              base_vector_lengths: tuple[float, ...],
                              dimensions,
                              Mr=1,
                              spatial_shifts=None,
                              angles=None, 
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
        intensity_harmonics_dict = cls.index_harmonics(intensity_harmonics_list, base_vector_lengths)
        illumination = cls(intensity_harmonics_dict, dimensions, Mr, spatial_shifts, angles)
        if store_plane_waves:
            illumination.electric_field_plane_waves = plane_waves
        return illumination
    

    @staticmethod
    def index_harmonics(waves_list: list[Sources.IntensityHarmonic3D], base_vector_lengths: tuple[float, float, float]) -> dict[tuple[int, int, int],
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

    def set_spatial_shifts_diagonally(self, number: int = 0):
        expanded_lattice = self.compute_expanded_lattice()
        shift_ratios = ShiftsFinder3d.get_shift_ratios(expanded_lattice)
        bases = sorted(list(shift_ratios.keys()))
        base, ratios = bases[number], list(shift_ratios[bases[number]])[0]
        spatial_shifts = np.zeros((self.Mr, base, 3))
        for r in range(self.Mr):
            base_vectors = self.get_base_vectors(r)
            shift = 2 * np.pi / np.where(base_vectors, base_vectors, np.inf) * ratios / base
            shift_rotated = VectorOperations.rotate_vector2d((np.copy(shift))[:2], self.angles[r])
            spatial_shifts[r] = np.array([shift_rotated * i for i in range(base)])
        self.spatial_shifts = spatial_shifts

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
            phase = self._phase_matrix[(r, n, indices[i][1])]
            amplitude = self.harmonics[indices[i]].amplitude
            wavevector = wavevectors[i]
            illumination_density += Sources.IntensityHarmonic3D(amplitude, phase, wavevector).get_intensity(grid)

        return illumination_density.real

    def get_elementary_cell(self):
        ...


class IlluminationPlaneWaves2D(PlaneWavesSIM):
    dimensionality=2
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...],Sources.IntensityHarmonic2D],
                 dimensions: tuple[int, int] = (1, 1),
                 Mr: int = 1,
                 spatial_shifts=None, 
                 angles=None):
        super().__init__(intensity_harmonics_dict, dimensions, Mr, spatial_shifts, angles)

    @classmethod
    def init_from_3D(cls, illumination_3d: IlluminationPlaneWaves3D, dimensions: tuple[int, int] = (1, 1), force=False):
        
        warning_raised = False
        for harmonic in illumination_3d.harmonics:
            if harmonic[1][2] != 0 and not force:
                raise ValueError("The 3D illumination pattern cannot be converted to 2D unless it's not changing in z direction!")
            elif harmonic[1][2] != 0 and not warning_raised:
                warning_raised = True
                warn(f"3D illumination with non-zero axial wavevectors is forcefully converted to 2D! The kz-component of the wavevector is ignored!")
        
        intensity_harmonics_dict = {(harmonic[0], tuple(harmonic[1][:2])): Sources.IntensityHarmonic2D.init_from_3D(illumination_3d.harmonics[harmonic])
                                    for harmonic in illumination_3d.harmonics}
        spatial_shifts = illumination_3d.spatial_shifts[:, :, :2]
        illumination = cls(intensity_harmonics_dict, dimensions, illumination_3d.Mr, spatial_shifts, angles=illumination_3d.angles)
        if illumination_3d.electric_field_plane_waves:
            illumination.electric_field_plane_waves = illumination_3d.electric_field_plane_waves
        return illumination
    
    @staticmethod
    def index_harmonics(waves_list: list[Sources.IntensityHarmonic2D], base_vector_lengths: tuple[float, float]) -> dict[tuple[int, int], Sources.IntensityHarmonic2D]:
        intensity_harmonics_dict = {}
        for wave in waves_list:
            wavevector = wave.wavevector
            m1, m2 = int(round(wavevector[0] / base_vector_lengths[0])), int(round(wavevector[1] / base_vector_lengths[1]))
            if not (m1, m2) in intensity_harmonics_dict.keys():
                intensity_harmonics_dict[(m1, m2)] = wave
            else:
                intensity_harmonics_dict[(m1, m2)].amplitude += wave.amplitude
        return intensity_harmonics_dict

    def set_spatial_shifts_diagonally(self, number: int = 0):
        expanded_lattice = self.compute_expanded_lattice()
        shift_ratios = ShiftsFinder2d.get_shift_ratios(expanded_lattice, len(expanded_lattice) + 5)
        if len(shift_ratios) == 0:
            raise ValueError("No shift ratios found!")
        bases = sorted(list(shift_ratios.keys()))
        base, ratios = bases[number], list(shift_ratios[bases[number]])[0]
        spatial_shifts = np.zeros((self.Mr, base, 2))
        for r in range(self.Mr):
            base_vectors = self.get_base_vectors(r)
            shift = 2 * np.pi / np.where(base_vectors, base_vectors, np.inf) * ratios / base
            shift_rotated = VectorOperations.rotate_vector2d((np.copy(shift))[:2], self.angles[r])
            spatial_shifts[r] = np.array([shift_rotated * i for i in range(base)])
        self.spatial_shifts = spatial_shifts

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
            phase = self._phase_matrix[(r, n, indices[i][1])]
            amplitude = self.harmonics[indices[i]].amplitude
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
                 spatial_shifts=None, 
                 angles=None):
        
        key = next(iter(intensity_harmonics_dict.keys()))
        harmonics_filtered = {key: intensity_harmonics_dict[key] for key in intensity_harmonics_dict.keys() if not np.isclose(intensity_harmonics_dict[key].amplitude, 0)}
        harmonics_nonlinear = {}
        if type(key[0]) is int and  type(key[1]) is tuple:
            for r in range(Mr):
                harmonics_one_rotation ={key[1]: harmonics_filtered[key] for key in harmonics_filtered.keys() if key[0] == r}
                harmonics_nonlinear_one_rotation = self.get_intensity_harmonics_nonlinear(harmonics_one_rotation, nonlinear_expansion_coefficients)
                harmonics_nonlinear.update({(r, key): harmonics_nonlinear_one_rotation[key] for key in harmonics_nonlinear_one_rotation.keys()})
        elif type(key) is tuple:
            harmonics_nonlinear = self.get_intensity_harmonics_nonlinear(harmonics_filtered, nonlinear_expansion_coefficients)
        else:
            raise ValueError("Urecognizable indexing of the intensity harmonics!")


        
        super().__init__(harmonics_nonlinear, dimensions, Mr, spatial_shifts, angles)

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
                       spatial_shifts=None,
                       angles=None,
                       nonlinear_expansion_coefficients: tuple[float, ...] = (0, 1),
                       ):
        intensity_harmonics_dict = cls.index_harmonics(intensity_harmonics_list, base_vector_lengths)
        return cls(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts, angles)

    @classmethod
    def init_from_linear_illumination(cls,
                                      illumination: PlaneWavesSIM,
                                      nonlinear_expansion_coefficients: tuple[float, ...]):
        intensity_harmonics_dict = illumination.harmonics
        return cls(intensity_harmonics_dict, nonlinear_expansion_coefficients, illumination.dimensions, illumination.Mr, illumination.spatial_shifts, illumination.angles)

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
                 spatial_shifts=None, 
                 angles=None
                 ):
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts, angles)


class IlluminationNonLinearSIM3D(PlaneWavesSIMNonlinear, IlluminationPlaneWaves3D):
    dimensionality=3
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nonlinear_expansion_coefficients: tuple[int, ...],
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=None, 
                 angles=None
        ):
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts, angles)


class IlluminationNPhotonSIM2D(IlluminationNonLinearSIM2D):
    dimensionality=2
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nphoton: int,
                 dimensions: tuple[int, int],
                 Mr=1,
                 spatial_shifts=None, 
                 angles=None
                 ):
        nonlinear_expansion_coefficients = tuple([0] * (nphoton - 1) + [1])
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts, angles)


class IlluminationNPhotonSIM3D(IlluminationNonLinearSIM3D):
    dimensionality=3
    def __init__(self,
                 intensity_harmonics_dict: dict[tuple[int, ...], Sources.IntensityHarmonic3D],
                 nphoton: int,
                 dimensions: tuple[int, ...],
                 Mr=1,
                 spatial_shifts=None, 
                 angles=None
                 ):
        if nphoton < 1:
            raise ValueError("The number of photons absorbed must be greater than 0!")
        nonlinear_expansion_coefficients = tuple([0] * nphoton + [1])
        super().__init__(intensity_harmonics_dict, nonlinear_expansion_coefficients, dimensions, Mr, spatial_shifts, angles)
