"""
Sources.py

This module contains classes for different types of sources used in simulations.
The sources can provide either electric fields or intensity fields.
"""

import numpy as np
import cmath
from abc import abstractmethod, ABC
from VectorOperations import VectorOperations


class Source(ABC):
    """
    Abstract base class for sources of electric or intensity fields
    in our simulations.
    """

    @abstractmethod
    def get_source_type(self) -> str:
        """
        Returns a type of the source in a human-readable form.
            str: The type of the source.
        """
        pass


class ElectricFieldSource(Source):
    """Abstract base class for sources that provide an electric field."""

    def get_source_type(self) -> str:
        return "ElectricField"

    @abstractmethod
    def get_electric_field(self, coordinates: np.ndarray[tuple[int, int, int], np.float64]) -> np.ndarray[tuple[int, int, int], np.complex128]:
        """Gets the electric field at the given coordinates.

        Args:
            coordinates (numpy.ndarray[np.float64]): The coordinates at which to get the electric field.

        Returns:
            numpy.ndarray[np.complex128]: The electric field at the given coordinates.
        """
        pass


class IntensitySource(Source):
    """Abstract base class for sources that provide intensity."""

    def get_source_type(self) -> str:
        return "Intensity"

    @abstractmethod
    def get_intensity(self, coordinates: np.ndarray[tuple[int, int, int], np.float64]) -> np.ndarray[tuple[int, int, int], np.float64]:
        """Gets the intensity at the given coordinates.

        Args:
            coordinates (numpy.ndarray[np.float64]): The coordinates at which to get the intensity.

        Returns:
            numpy.ndarray[np.float64]: The intensity at the given coordinates.
        """
        pass


class PlaneWave(ElectricFieldSource):
    """Electric field of a plane wave"""

    def __init__(self, electric_field_p: complex, electric_field_s: complex, phase1: float, phase2: float, wavevector: np.ndarray[3, np.float64]):
        """
        Constructs a PlaneWave object.

        Args:
            electric_field_p (float): The p-polarized electric field component.
            electric_field_s (float): The s-polarized electric field component.
            phase1 (float): The phase of the p-polarized component.
            phase2 (float): The phase of the s-polarized component.
            wavevector (numpy.ndarray): The wavevector of the plane wave.
        """
        self.wavevector = np.array(wavevector)
        if not np.linalg.norm(wavevector) == 0:
            theta = np.arccos(wavevector[2] / np.linalg.norm(wavevector))
            phi = cmath.phase(wavevector[0] + 1j * wavevector[1])
        else:
            theta = 0
            phi = 0
        Ep = electric_field_p * np.array((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)))
        Es = electric_field_s * np.array((-np.sin(phi), np.cos(phi), 0))
        self.field_vectors = [Ep, Es]
        self.phases = [phase1, phase2]

    def get_electric_field(self, coordinates):
        shape = list(coordinates.shape)
        electric_field = np.zeros(shape, dtype=np.complex128)
        for p in [0, 1]:
            electric_field += self.field_vectors[p] * np.exp(
                1j * (np.einsum('ijkl,l ->ijk', coordinates, self.wavevector)
                      + self.phases[p]))[:, :, :, None]
        return electric_field


class PointSource(ElectricFieldSource):
    """Electric field of a point source"""

    def __init__(self, coordinates: np.ndarray[3, np.float64], brightness: float):
        """Constructs a PointSource object.

        Args:
            coordinates (numpy.ndarray): The coordinates of the point source.
            brightness (float): The brightness of the point source.
        """
        self.coordinates = np.array(coordinates)
        self.brightness = brightness

    def get_electric_field(self, grid: np.ndarray[tuple[int, int, int, 3], np.float64]) -> np.ndarray[tuple[int, int, int], np.complex128]:
        rvectors = np.array(grid - self.coordinates)
        rnorms = np.einsum('ijkl, ijkl->ijk', rvectors, rvectors) ** 0.5
        upper_limit = 1000
        electric_field = np.zeros(grid.shape)
        electric_field[rnorms == 0] = np.array((1, 1, 1)) * upper_limit * np.sign(self.brightness)
        electric_field[rnorms != 0] = self.brightness / (rnorms[rnorms > 0] ** 3)[:, None] * rvectors[rnorms != 0]
        electric_field_norms = np.einsum('ijkl, ijkl->ijk', electric_field, electric_field.conjugate()).real ** 0.5
        electric_field[electric_field_norms > upper_limit] = (upper_limit * np.sign(self.brightness))
        return electric_field


class IntensityHarmonic(IntensitySource):
    """
    Intensity plane wave is a component of the Fourier
    transform of the energy density distribution in a given volume
    (e.g., standing waves)
    """

    def __init__(self, amplitude: complex = 0., phase: float = 0., wavevector=np.array(())):
        """
        Constructs an IntensityHarmonic3D object.

        Args:
            amplitude (float): The amplitude of the plane wave.
            phase (float): The phase of the plane wave.
            wavevector (numpy.ndarray): The wavevector of the plane wave.
        """
        self.wavevector = np.array(wavevector)
        self.amplitude = amplitude
        self.phase = phase

    def __add__(self, other):
        if not np.allclose(self.wavevector, other.wavevector):
            raise ValueError("Cannot add harmonics with different wavevectors.")
        combined_amplitude = (self.amplitude * np.exp(1j * self.phase) +
                              other.amplitude * np.exp(1j * other.phase))
        return self.__class__(combined_amplitude, 0, self.wavevector)

    def __iadd__(self, other):
        if not np.allclose(self.wavevector, other.wavevector):
            raise ValueError("Cannot add harmonics with different wavevectors.")
        self.amplitude = (self.amplitude * np.exp(1j * self.phase) +
                          other.amplitude * np.exp(1j * other.phase))
        self.phase = 0
        return self

    def __mul__(self, other):
        combined_amplitude = self.amplitude * other.amplitude
        combined_phase = self.phase + other.phase
        combined_wavevector = self.wavevector + other.wavevector
        return self.__class__(combined_amplitude, combined_phase, combined_wavevector)

    def __imul__(self, other):
        self.amplitude = self.amplitude * other.amplitude
        self.phase = self.phase + other.phase
        self.wavevector = self.wavevector + other.wavevector
        return self


class IntensityHarmonic3D(IntensityHarmonic):
    def get_intensity(self, grid: np.float64, rotated_frame_vector=np.array((0, 0, 1)), rotated_angle=0.):
        wavevector = self.wavevector if not rotated_angle else VectorOperations.rotate_vector3d(self.wavevector, rotated_frame_vector, -rotated_angle)
        intensity = self.amplitude * np.exp(1j * (np.einsum('ijkl,l ->ijk', grid, wavevector)
                                                  + self.phase))
        return intensity


class IntensityHarmonic2D(IntensityHarmonic):
    @classmethod
    def init_from_3D(cls, harmonic: IntensityHarmonic3D):
        return cls(harmonic.amplitude, harmonic.phase, harmonic.wavevector[:2])

    def get_intensity(self, grid: np.float64, rotated_angle=0.):
        wavevector = self.wavevector if not rotated_angle else VectorOperations.rotate_vector2d(self.wavevector, -rotated_angle)
        intensity = self.amplitude * np.exp(1j * (np.einsum('ijl,l ->ij', grid, wavevector)
                                                  + self.phase))
        return intensity
