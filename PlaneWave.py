import numpy as np
import cmath
from VectorOperations import VectorOperations

class PlaneWave:
    def __init__(self, electric_field_p, electric_field_s, phase1, phase2, wavevector):
        self.wavevector = wavevector

        # Previously used for X-Y polarization
        # zvector = np.array((0, 0, 1))
        # rot_vector = np.cross(zvector, wavevector)
        # rot_angle = np.arccos(np.dot(zvector, wavevector)/np.linalg.norm(wavevector))
        # E1 = VectorOperations.rotate_vector3d(np.array((electric_field_x, 0, 0)), rot_vector, rot_angle)
        # E2 = VectorOperations.rotate_vector3d(np.array((0, electric_field_y, 0)), rot_vector, rot_angle)

        theta = np.arccos(wavevector[2]/np.linalg.norm(wavevector))
        phi = cmath.phase(wavevector[0] + 1j * wavevector[1])
        Ep = electric_field_p * np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.sin(theta)))
        Es = electric_field_s * np.array((-np.sin(phi), np.cos(phi), 0))
        self.field_vectors = [Ep, Es]
        self.phases = [phase1, phase2]



