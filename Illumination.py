import numpy as np
import Sources
from VectorOperations import VectorOperations
import matplotlib.pyplot as plt
import stattools
class Illumination:
    def __init__(self, intensity_plane_waves_dict, Mr = 1):
        self.angles = np.arange(0, np.pi, np.pi / Mr)
        self._spacial_shifts = [np.array((0, 0, 0)), ]
        self._Mr = Mr
        self.Mt = len(self.spacial_shifts)
        self.waves = {key : intensity_plane_waves_dict[key] for key in intensity_plane_waves_dict.keys() if not np.isclose(intensity_plane_waves_dict[key].amplitude, 0)}
        self.wavevectors2d, self.indices2d = self.get_wavevectors_projected(0)
        self.wavevectors3d, self.indices3d = self.get_wavevectors(0)
        self.rearranged_indices = self._rearrange_indices()
        self.xy_fourier_peaks = None
        self.phase_matrix = None

    @classmethod
    def init_from_list(cls, intensity_plane_waves_list, base_vector_lengths, Mr = 1):
        intensity_plane_waves_dict = cls.index_frequencies(intensity_plane_waves_list, base_vector_lengths)
        return cls(intensity_plane_waves_dict, Mr = Mr)
    
    @classmethod
    def init_from_numerical_intensity_fourier_domain(cls, numerical_intensity_fourier_domain, axes, Mr=1):
        fourier_peaks, amplitudes = stattools.estimate_localized_peaks(numerical_intensity_fourier_domain, axes)

    @staticmethod
    def index_frequencies(waves_list, base_vector_lengths):
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

    def _rearrange_indices(self):
        indices = self.waves.keys()
        result_dict = {}
        for index in indices:
            key = index[:2]
            value = index[2]
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
        result_dict = {key: tuple(values) for key, values in result_dict.items()}
        # print(result_dict)
        return result_dict

    @staticmethod
    def find_ipw_from_pw(plane_waves):
        intensity_plane_waves = {}
        for projection_index in (0, 1, 2):
            for plane_wave1 in plane_waves:
                for polarization_index1 in (0, 1):
                        amplitude1, wavevector1 = plane_wave1.field_vectors[polarization_index1][projection_index], \
                                                    plane_wave1.wavevector
                        if np.abs(amplitude1) < 10**-3:
                            continue
                        for plane_wave2 in plane_waves:
                            for polarization_index2 in (0, 1):
                                    amplitude2, wavevector2 = plane_wave2.field_vectors[polarization_index2][projection_index], \
                                                                plane_wave2.wavevector
                                    if np.abs(amplitude2) < 10 ** -3:
                                        continue
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
        self.normalize_spacial_waves()

    @property
    def spacial_shifts(self):
        return self._spacial_shifts

    @spacial_shifts.setter
    def spacial_shifts(self, new_spacial_shifts):
        self._spacial_shifts = new_spacial_shifts
        self.Mt = len(new_spacial_shifts)
        self.normalize_spacial_waves()

    def set_spacial_shifts_diagonally(self, number, base_vectors):
        kx, ky = base_vectors[0], base_vectors[1]
        shiftsx = np.arange(0, number)/number/kx
        shiftsy = np.arange(0, number)/number/ky
        self.spacial_shifts = np.array([(shiftsx[i], shiftsy[i], 0) for i in range(number)])

    def get_wavevectors(self, r):
        wavevectors = []
        angle = self.angles[r]
        indices = []
        for spacial_wave in self.waves.values():
            indices.append(spacial_wave)
            wavevector = VectorOperations.rotate_vector3d(
                spacial_wave.wavevector, np.array((0, 0, 1)), angle)
            wavevectors.append(wavevector)
        return wavevectors, indices
    def get_all_wavevectors(self):
        wavevectors = []
        for r in range(self.Mr):
            wavevectors_r, _ = self.get_wavevectors(r)
            wavevectors.extend(wavevectors_r)
        return wavevectors

    def get_wavevectors_projected(self, r):
        wavevectors2d = []
        angle = self.angles[r]
        indices2d = []
        for spacial_wave in self.waves:
            if not spacial_wave[:2] in indices2d:
                indices2d.append(spacial_wave[:2])
                wavevector = VectorOperations.rotate_vector3d(
                    self.waves[spacial_wave].wavevector, np.array((0, 0, 1)), angle)
                wavevectors2d.append(wavevector[:2])
        return wavevectors2d, indices2d

    def get_all_wavevectors_projected(self):
        wavevectors2d = []
        for r in range(self.Mr):
            wavevectors2d_r, _ = self.get_wavevectors_projected(r)
            wavevectors2d.extend(wavevectors2d_r)
        return wavevectors2d

    def normalize_spacial_waves(self):
        if not (0, 0, 0) in self.waves.keys():
            return AttributeError("Zero wavevector is not found! No constant power in the illumination!")
        norm = self.waves[0, 0, 0].amplitude * self.Mt * self.Mr
        for spacial_wave in self.waves.values():
            spacial_wave.amplitude /= norm

    def compute_expanded_lattice2d(self):
        self.xy_fourier_peaks = set((mx, my) for mx, my, mz in self.waves.keys())
        expanded_lattice2d = set()
        for peak1 in self.xy_fourier_peaks:
            for peak2 in self.xy_fourier_peaks:
                expanded_lattice2d.add((peak1[0] - peak2[0], peak1[1] - peak2[1]))
        print(len(expanded_lattice2d))
        return expanded_lattice2d

    def compute_expanded_lattice3d(self):
        fourier_peaks = set(self.waves.keys())
        expanded_lattice3d = set()
        for peak1 in fourier_peaks:
            for peak2 in fourier_peaks:
                expanded_lattice3d.add((peak1[0] - peak2[0], peak1[1] - peak2[1], peak1[2] - peak2[2]))
        print(len(expanded_lattice3d))
        return expanded_lattice3d

    def compute_phase_matrix(self):
        self.phase_matrix = np.zeros((self.Mr, self.Mt, len(self.get_wavevectors_projected(0)[0])))
        for r in range(self.Mr):
            for n in range(self.Mt):
                wavevectors2d, _ = self.get_wavevectors_projected(r)
                for w in range(len(wavevectors2d)):
                    urn = VectorOperations.rotate_vector2d(self.spacial_shifts[n][:2], self.angles[r])
                    self.phase_matrix[r, n, w] = np.exp(-1j * np.dot(urn, wavevectors2d[w]))