import numpy as np
import VectorOperations
import matplotlib.pyplot as plt
import stattools
class Illumination:
    def __init__(self, intensity_plane_waves_dict,  Mr = 1):
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / Mr)
        self._spacial_shifts = np.array((0, 0, 0))
        self._Mr = Mr
        self.Mt = len(self.spacial_shifts)
        self.waves = intensity_plane_waves_dict

    @classmethod
    def init_from_list(cls, intensity_plane_waves_list, base_vector_lengths, Mr = 1):
        intensity_plane_waves_dict = cls.index_frequencies(intensity_plane_waves_list, base_vector_lengths)
        return cls(intensity_plane_waves_dict, Mr = Mr)
    
    @classmethod
    def init_from_numerical_intensity_fourier_domain(cls, numerical_intensity_fourier_domain, axes, Mr = 1):
        fourier_peaks, amplitudes = stattools.estimate_localized_peaks(numerical_intensity_fourier_domain, axes)


    @staticmethod
    def index_frequencies(waves_list, base_vector_lengths):
        intensity_plane_waves_dict = {}
        for wave in waves_list:
            wavevector = wave.wavevector
            m1, m2, m3 = int(wavevector[0]/base_vector_lengths[0]), int(wavevector[1]/base_vector_lengths[1]), \
                int(wavevector[2]/base_vector_lengths[2])
            intensity_plane_waves_dict[(m1, m2, m3)] = wave
        return intensity_plane_waves_dict

    @property
    def Mr(self):
        return self._Mr

    @Mr.setter
    def Mr(self, new_Mr):
        self.Mr = new_Mr
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / new_Mr)

    @property
    def spacial_shifts(self):
        return self._spacial_shifts

    @spacial_shifts.setter
    def spacial_shifts(self, new_spacial_shifts):
        self._spacial_shifts = new_spacial_shifts
        self.Mt = len(new_spacial_shifts)

    def get_wavevectors(self):
        wavevectors = []
        for angle in self.angles:
            for spacial_wave in self.waves.values():
                wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                    spacial_wave.wavevector, np.array((0, 0, 1)), angle)
                wavevectors.append(wavevector)
        return wavevectors
    
