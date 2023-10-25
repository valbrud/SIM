import numpy as np
import cmath
from abc import abstractmethod

# Any source of electric field generates electric field in the whole space.
# So I create an interface with a method get_electric_field(coordinates), which is inherited by all sources.

class Source:
    @abstractmethod
    def get_source_type(self): ...


class ElectricFieldSource(Source):

    def get_source_type(self):
        return "ElectricField"

    @abstractmethod
    def get_electric_field(self, coordinates): ...


class IntensitySource(Source):
    def get_source_type(self):
        return "Intensity"

    @abstractmethod
    def get_intensity(self, coordinates): ...


class PlaneWave(ElectricFieldSource):
    def __init__(self, electric_field_p, electric_field_s, phase1, phase2, wavevector):
        self.wavevector = np.array(wavevector)

        theta = np.arccos(wavevector[2] / np.linalg.norm(wavevector))
        phi = cmath.phase(wavevector[0] + 1j * wavevector[1])
        Ep = electric_field_p * np.array((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)))
        Es = electric_field_s * np.array((-np.sin(phi), np.cos(phi), 0))
        self.field_vectors = [Ep, Es]
        self.phases = [phase1, phase2]

        ####################################################################################################
        # Previously used for X-Y polarization
        # zvector = np.array((0, 0, 1))
        # rot_vector = np.cross(zvector, wavevector)
        # rot_angle = np.arccos(np.dot(zvector, wavevector)/np.linalg.norm(wavevector))
        # E1 = VectorOperations.rotate_vector3d(np.array((electric_field_x, 0, 0)), rot_vector, rot_angle)
        # E2 = VectorOperations.rotate_vector3d(np.array((0, electric_field_y, 0)), rot_vector, rot_angle)
        ####################################################################################################

    def get_electric_field(self, coordinates):
        shape = list(coordinates.shape)
        electric_field = np.zeros(shape, dtype=np.complex128)
        for p in [0, 1]:
            electric_field += self.field_vectors[p] * np.exp(1j * (np.einsum('ijkl,l ->ijk', coordinates, self.wavevector)
                                                                                + self.phases[p]))[:, :, :, None]
        return electric_field


class PointSource(ElectricFieldSource):
    def __init__(self, coordinates, brightness):
        self.coordinates = np.array(coordinates)
        self.brightness = brightness

    def get_electric_field(self, coordinates):
        rvectors = np.array(coordinates - self.coordinates)
        rnorms = np.einsum('ijkl, ijkl->ijk', rvectors, rvectors)**0.5
        upper_limit = 1000
        electric_field = np.zeros(coordinates.shape)
        electric_field[rnorms == 0] = np.array((1, 1, 1)) * upper_limit * np.sign(self.brightness)
        electric_field[rnorms != 0] = self.brightness / (rnorms[rnorms > 0] ** 3)[:, None] * rvectors[rnorms != 0]
        electric_field_norms = np.einsum('ijkl, ijkl->ijk', electric_field, electric_field.conjugate()).real**0.5
        electric_field[electric_field_norms > upper_limit] = (upper_limit * np.sign(self.brightness))
        return electric_field


class IntensityPlaneWave(IntensitySource):
    def __init__(self, amplitude=0, phase=0, wavevector=np.array((0, 0, 0))):
        self.wavevector = np.array(wavevector)
        self.amplitude = amplitude
        self.phase = phase

    def get_intensity(self, coordinates):
        inensity = self.amplitude * np.exp(1j * (np.dot(coordinates, self.wavevector)
                                                 + self.phase))
        return inensity

class IntensityCosineWave(IntensitySource):
    def __init__(self, amplitude, phase, wavevector):
        self.wavevector = wavevector
        self.amplitude = amplitude
        self.phase = phase

    def get_intensity(self, coordinates):
        inensity = self.amplitude * np.exp(1j * (np.dot(coordinates, self.wavevector)
                                                               + self.phase))
        return inensity

class IntensitySineWave(IntensitySource):
    def __init__(self, amplitude, phase, wavevector):
        self.wavevector = wavevector
        self.amplitude = amplitude
        self.phase = phase

    def get_intensity(self, coordinates):
        inensity = self.amplitude * np.exp(1j * (np.dot(coordinates, self.wavevector)
                                                               + self.phase))
        return inensity
