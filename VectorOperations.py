import numpy as np
import globvar
import cmath

class VectorOperations:
    @staticmethod
    def rotation_matrix(angle):
        return np.array(((np.cos(angle), -np.sin(angle)),
                         (np.sin(angle), np.cos(angle))))

    @staticmethod
    def rotate_vector2d(vector2d, angle):
        rotation_matrix = VectorOperations.rotation_matrix(angle)
        return rotation_matrix @ vector2d

    @staticmethod
    def rotate_vector3d(vector3d, rot_ax_vector, rot_angle):
        length = np.dot(vector3d, vector3d) ** 0.5
        if length == 0:
            return vector3d
        theta = np.arccos(vector3d[-1] / length)
        phi = cmath.phase(vector3d[0] + 1j * vector3d[1])
        bloch_vector = np.array((np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)))
        rot_ax_vector = rot_ax_vector / np.dot(rot_ax_vector, rot_ax_vector) ** 0.5
        rot_matrix = np.cos(rot_angle / 2) * globvar.Pauli.I - 1j * np.sin(rot_angle / 2) * \
                    (rot_ax_vector[0] * globvar.Pauli.X + rot_ax_vector[1] * globvar.Pauli.Y + rot_ax_vector[2] * globvar.Pauli.Z)
        
        bloch_vector = rot_matrix @ bloch_vector
        theta = 2 * np.arctan(abs(bloch_vector[1])/abs(bloch_vector[0]))
        phi = cmath.phase(bloch_vector[1]) - cmath.phase(bloch_vector[0])

        vector3d = length * np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
        return vector3d
        
