import numpy as np
import VectorOperations
class Illumination:
    def __init__(self, intensity_plane_waves_dict, spacial_shifts=(np.array((0, 0, 0)), ), M_r = 1):
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / M_r)
        self._spacial_shifts = spacial_shifts
        self._M_r = M_r
        self.M_t = len(self.spacial_shifts)
        self.waves = intensity_plane_waves_dict

    @classmethod
    def init_from_list(cls, intensity_plane_waves_list, base_vector_lengths, spacial_shifts=(np.array((0, 0, 0))), M_r = 1):
        intensity_plane_waves_dict = cls.index_frequencies(intensity_plane_waves_list, base_vector_lengths)
        return cls(intensity_plane_waves_dict, spacial_shifts = spacial_shifts, M_r = M_r)
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
    def M_r(self):
        return self._M_r

    @M_r.setter
    def M_r(self, new_M_r):
        self.M_r = new_M_r
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / new_M_r)

    @property
    def spacial_shifts(self):
        return self._spacial_shifts

    @spacial_shifts.setter
    def spacial_shifts(self, new_spacial_shifts):
        self._spacial_shifts = new_spacial_shifts
        self.M_t = len(new_spacial_shifts)

    def get_wavevectors(self):
        wavevectors = []
        for angle in self.angles:
            for spacial_wave in self.waves.values():
                wavevector = VectorOperations.VectorOperations.rotate_vector3d(
                    spacial_wave.wavevector, np.array((0, 0, 1)), angle)
                wavevectors.append(wavevector)
        return wavevectors
