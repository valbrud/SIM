import numpy as np
import Sources
from Illumination import Illumination
from VectorOperations import VectorOperations
from fractions import Fraction
class BFPConfiguration:
    def __init__(self, wavelength=1, refraction_index=1):
        self.k = 2 * np.pi / wavelength
        self.n = refraction_index

    def get_4_oblique_s_waves_and_circular_normal(self, angle_oblique, strength_oblique, strength_s_normal=1, Mt=32):

        theta = angle_oblique
        k1 = self.k * np.sin(theta) * self.n
        k2 = self.k * (np.cos(theta) - 1)

        p = strength_s_normal
        b = strength_oblique
        a0 = (2 * p ** 2 + 4 * b ** 2)

        s_polarized_waves = {
            (0, 0, 0)  : Sources.IntensityPlaneWave(a0, 0, np.array((0, 0, 0))),

            (-2, 0, 0) : Sources.IntensityPlaneWave((-b ** 2), 0, np.array((-2 * k1, 0, 0))),
            (2, 0, 0)  : Sources.IntensityPlaneWave((-b ** 2), 0, np.array((2 * k1, 0, 0))),
            (0, 2, 0)  : Sources.IntensityPlaneWave((-b ** 2), 0, np.array((0, 2 * k1, 0))),
            (0, -2, 0) : Sources.IntensityPlaneWave((-b ** 2), 0, np.array((0, -2 * k1, 0))),

            (1, 0, 1)  : Sources.IntensityPlaneWave((-1j * b), 0, np.array((k1, 0, k2))),
            (-1, 0, 1) : Sources.IntensityPlaneWave((1j * b), 0, np.array((-k1, 0, k2))),
            (0, 1, 1)  : Sources.IntensityPlaneWave((-1 * b), 0, np.array((0, k1, k2))),
            (0, -1, 1) : Sources.IntensityPlaneWave((1 * b), 0, np.array((0, -k1, k2))),

            (1, 0, -1) : Sources.IntensityPlaneWave((-1j * b), 0, np.array((k1, 0, -k2))),
            (-1, 0, -1): Sources.IntensityPlaneWave((1j * b), 0, np.array((-k1, 0, -k2))),
            (0, 1, -1) : Sources.IntensityPlaneWave((1 * b), 0, np.array((0, k1, -k2))),
            (0, -1, -1): Sources.IntensityPlaneWave((-1 * b), 0, np.array((0, -k1, -k2)))
        }

        illumination = Illumination(s_polarized_waves)
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination

    def get_4_circular_oblique_waves_and_circular_normal(self, angle_oblique, strength_s_oblique, strength_s_normal, Mt=1):

        theta = angle_oblique
        k1 = self.k * np.sin(theta)
        k2 = self.k * (1 - np.cos(theta))

        p = strength_s_normal
        b = strength_s_oblique
        a0 = (2 * p ** 2 + 8 * b ** 2)

        circular_intensity_waves = {
            (1, 1, 0)  : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2, 0, np.array((k1, k1, 0))),
            (-1, 1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2, 0, np.array((-k1, k1, 0))),
            (1, -1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2, 0, np.array((k1, -k1, 0))),
            (-1, -1, 0): Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2, 0, np.array((-k1, -k1, 0))),

            (0, 2, 0)  : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2, 0, np.array((0, 2 * k1, 0))),
            (0, -2, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2, 0, np.array((0, -2 * k1, 0))),
            (2, 0, 0)  : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2, 0, np.array((2 * k1, 0, 0))),
            (-2, 0, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2, 0, np.array((-2 * k1, 0, 0))),

            (1, 0, -1) : Sources.IntensityPlaneWave(b * p * (1 + np.cos(theta)), 0, np.array((k1, 0, -k2))),
            (-1, 0, 1) : Sources.IntensityPlaneWave(b * p * (1 + np.cos(theta)), 0, np.array((-k1, 0, k2))),
            (1, 0, 1)  : Sources.IntensityPlaneWave(-b * p * (1 + np.cos(theta)), 0, np.array((k1, 0, k2))),
            (-1, 0, -1): Sources.IntensityPlaneWave(-b * p * (1 + np.cos(theta)), 0, np.array((-k1, 0, -k2))),

            (0, 1, -1) : Sources.IntensityPlaneWave(-1j * b * p * (1 + np.cos(theta)), 0, np.array((0, k1, -k2))),
            (0, 1, 1)  : Sources.IntensityPlaneWave(-1j * b * p * (1 + np.cos(theta)), 0, np.array((0, k1, k2))),
            (0, -1, 1) : Sources.IntensityPlaneWave(1j * b * p * (1 + np.cos(theta)), 0, np.array((0, -k1, k2))),
            (0, -1, -1): Sources.IntensityPlaneWave(1j * b * p * (1 + np.cos(theta)), 0, np.array((0, -k1, -k2))),

            (0, 0, 0)  : Sources.IntensityPlaneWave(a0, 0, np.array((0, 0, 0)))
        }

        illumination = Illumination(circular_intensity_waves)
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination

    def get_6_oblique_s_waves_and_circular_normal(self, angle_oblique, strength_oblique, strength_s_normal, Mt=1):

        theta = angle_oblique
        k1 = self.k * np.sin(theta)
        k2 = self.k * (1 - np.cos(theta))

        p = strength_s_normal
        b = strength_oblique

        a0 = 2 * p ** 2 + 6 * b**2

        seven_waves_list = [
            Sources.IntensityPlaneWave(-b**2, 0, np.array((2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((-2 * k1, 0, 0))),
            Sources.IntensityPlaneWave(b**2, 0, np.array((k1, 0, 0))),
            Sources.IntensityPlaneWave(b**2, 0, np.array((-k1, 0, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((-k1, 3**0.5*k1, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((k1, 3**0.5*k1, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((-k1, -3**0.5*k1, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((k1, -3**0.5*k1, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((0, 3**0.5*k1, 0))),
            Sources.IntensityPlaneWave(-b ** 2, 0, np.array((0, -3**0.5*k1, 0))),

            Sources.IntensityPlaneWave(a0, 0, np.array((0, 0, 0))),

            Sources.IntensityPlaneWave(b**2, 0, np.array((k1/2, 3**0.5/2 * k1, 0))),
            Sources.IntensityPlaneWave(b**2, 0, np.array((-k1/2, 3**0.5/2 * k1, 0))),
            Sources.IntensityPlaneWave(b**2, 0, np.array((k1/2, -3**0.5/2 * k1, 0))),
            Sources.IntensityPlaneWave(b**2, 0, np.array((-k1/2, -3**0.5/2 * k1, 0))),

            Sources.IntensityPlaneWave(-b**2, 0, np.array((3/2 * k1, -3**0.5/2 * k1, 0))),
            Sources.IntensityPlaneWave(-b**2, 0, np.array((-3/2 * k1, -3**0.5/2 * k1, 0))),
            Sources.IntensityPlaneWave(-b**2, 0, np.array((3/2 * k1, 3**0.5/2 * k1, 0))),
            Sources.IntensityPlaneWave(-b**2, 0, np.array((-3/2 * k1, 3**0.5/2 * k1, 0))),

            Sources.IntensityPlaneWave(1j * b, 0, np.array((-k1, 0, k2))),
            Sources.IntensityPlaneWave(1j * b, 0, np.array((-k1, 0, -k2))),
            Sources.IntensityPlaneWave(-1j * b, 0, np.array((k1, 0, k2))),
            Sources.IntensityPlaneWave(-1j * b, 0, np.array((k1, 0, -k2))),

            Sources.IntensityPlaneWave((2*3**0.5 - 2j)/4 * b * p, 0, np.array((k1/2, 3**0.5/2 * k1, k2))),
            Sources.IntensityPlaneWave((-2*3**0.5 - 2j)/4 * b * p, 0, np.array((k1/2, -3**0.5/2 * k1, k2))),
            Sources.IntensityPlaneWave((2*3**0.5 + 2j)/4 * b * p, 0, np.array((-k1/2, 3**0.5/2 * k1, k2))),
            Sources.IntensityPlaneWave((-2*3**0.5 + 2j)/4 * b * p, 0, np.array((-k1/2, -3**0.5/2 * k1, k2))),
            Sources.IntensityPlaneWave((-2*3**0.5 - 2j)/4 * b * p, 0, np.array((k1/2, 3**0.5/2 * k1, -k2))),
            Sources.IntensityPlaneWave((2*3**0.5 - 2j)/4 * b * p, 0, np.array((k1/2, -3**0.5/2 * k1, -k2))),
            Sources.IntensityPlaneWave((-2*3**0.5 + 2j)/4 * b * p, 0, np.array((-k1/2, 3**0.5/2 * k1, -k2))),
            Sources.IntensityPlaneWave((2*3**0.5 + 2j)/4 * b * p, 0, np.array((-k1/2, -3**0.5/2 * k1, -k2))),
        ]

        illumination = Illumination.init_from_list(seven_waves_list, (k1/2, 3**0.5/2 * k1, k2))
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination


    def get_2_oblique_s_waves_and_s_normal(self, angle_oblique, strength_oblique, strength_s_normal = 1, Mr=3,  Mt=1):

        theta = angle_oblique
        k1 = self.k * np.sin(theta)
        k2 = self.k * (1 - np.cos(theta))

        p = strength_s_normal
        b = strength_oblique
        a0 = p**2 + 2 * b**2

        three_waves_illumination = {
            (0, 0, 0)  : Sources.IntensityPlaneWave(a0 , 0, np.array((0, 0, 0))),

            (2, 0, 0)  : Sources.IntensityPlaneWave(b**2 , 0, np.array((2 * k1, 0, 0))),
            (-2, 0, 0) : Sources.IntensityPlaneWave(b**2 , 0, np.array((-2 * k1, 0, 0))),

            (1, 0 , 1)  : Sources.IntensityPlaneWave(b * p, 0, np.array((k1, 0, k2))),
            (-1, 0 , 1) :  Sources.IntensityPlaneWave(b * p, 0, np.array((-k1, 0, k2))),
            (1, 0, -1) : Sources.IntensityPlaneWave(b * p, 0, np.array((k1, 0, -k2))),
            (-1, 0, -1): Sources.IntensityPlaneWave(b * p, 0, np.array((-k1, 0, -k2))),
        }

        illumination = Illumination(three_waves_illumination, Mr=Mr)
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination

    def widefield(self):

        widefield = {
            (0, 0, 0) : Sources.IntensityPlaneWave(1, 0, np.array((0, 0, 0)))
        }

        illumination = Illumination(widefield)
        
        return illumination
    
    def get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(self, angle1, k_ratio, strength1, strength2, strength_normal = 1, Mr = 3, Mt = 1):

        angle2 = np.arcsin(k_ratio * np.sin(angle1))
        k_ratio = Fraction(k_ratio).limit_denominator()
        base_vector_kx = self.k * np.sin(angle1) / k_ratio.denominator
        base_vector_ky = 100 * self.k  # All ky indices should be zero for a plane illumination
        base_vector_kz = self.k * (1 - np.cos(angle1)) / 100  # We do not really care about a z index. Hopefully
        base_vectors = (base_vector_kx, base_vector_ky, base_vector_kz)

        p = strength_normal
        b = strength1
        c = strength2

        vec_x = np.array((self.k * np.sin(angle1), 0, self.k * np.cos(angle1)))
        vec_mx = np.array((self.k * np.sin(angle2), 0, self.k * np.cos(angle2)))
        ax_z = np.array((0, 0, 1))

        a0 = p**2 + 2 * b**2 + 2 * c**2

        five_waves_2z_illumination = [
            Sources.PlaneWave(0, b, 0, 0, vec_x),
            Sources.PlaneWave(0, c, 0, 0, vec_mx),
            Sources.PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, np.pi)),
            Sources.PlaneWave(0, c, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, np.pi)),
            Sources.PlaneWave(0, 1 * p, 0, 0, np.array((0, 10**-10, self.k))),
        ]

        five_waves_2z_illumination = Illumination.find_ipw_from_pw(five_waves_2z_illumination)
        illumination = Illumination.init_from_list(five_waves_2z_illumination, base_vectors, Mr)
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination

    def get_two_oblique_triangles_at_different_angles_and_one_normal_wave(self, angle1, k_ratio, strength1, strength2,
                                                                          strength_normal=1, mutually_rotated=True, Mt=1):
        angle2 = np.arcsin(k_ratio * np.sin(angle1))

        k_ratio = Fraction(k_ratio).limit_denominator()
        base_vector_kx = self.k * np.sin(angle1) / k_ratio.denominator / 2
        base_vector_ky = self.k * np.sin(angle1) / k_ratio.denominator * 3**0.5 / 2
        base_vector_kz = (1 - np.cos(angle1)) / 100  # We do not really care about a z index. Hopefully
        base_vectors = (base_vector_kx, base_vector_ky, base_vector_kz)

        p = strength_normal
        b = strength1
        c = strength2
        sign = -1 if mutually_rotated else 1

        vec_x = np.array((self.k * np.sin(angle1), 0, self.k * np.cos(angle1)))
        vec_mx = np.array((sign * self.k * np.sin(angle2), 0, self.k * np.cos(angle2)))
        ax_z = np.array((0, 0, 1))

        ttillum = [
            Sources.PlaneWave(0, b, 0, 0, vec_x),
            Sources.PlaneWave(0, c, 0, 0, vec_mx),
            Sources.PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 2 * np.pi/3)),
            Sources.PlaneWave(0, c, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 2 * np.pi/3)),
            Sources.PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 4 * np.pi/3)),
            Sources.PlaneWave(0, c, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 4 * np.pi/3)),

            Sources.PlaneWave(p, 1j * p, 0, 0, np.array((0, 10**-10, self.k))),
        ]

        two_triangles_illumination = Illumination.find_ipw_from_pw(ttillum)
        illumination = Illumination.init_from_list(two_triangles_illumination, base_vectors)
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination

    def get_two_oblique_squares_at_different_angles_and_one_normal_wave(self, angle1, k_ratio, strength1, strength2,
                                                                          strength_normal=1, mutually_rotated=True, Mt=1):
        angle2 = np.arcsin(k_ratio * np.sin(angle1))

        ratio_modified = Fraction(k_ratio * 2**0.5).limit_denominator()  # Sine between rotated square is sqrt(2)/2, so we need an additional sqrt(2)
        base_vector_kx = self.k * np.sin(angle1) / ratio_modified.denominator / 2
        base_vector_ky = base_vector_kx
        base_vector_kz = (1 - np.cos(angle1)) / 100  # We do not really care about a z index. Hopefully
        base_vectors = (base_vector_kx, base_vector_ky, base_vector_kz)

        p = strength_normal
        b = strength1
        c = strength2
        vec_x = np.array((self.k * np.sin(angle1), 0, self.k * np.cos(angle1)))

        if mutually_rotated:
            vec_mx = np.array((self.k * np.sin(angle2) * np.cos(np.pi/4), -self.k * np.sin(angle2) * np.sin(np.pi/4), self.k * np.cos(angle2)))
        else:
            vec_mx = np.array((self.k * np.sin(angle2), 0, self.k * np.cos(angle2)))

        ax_z = np.array((0, 0, 1))

        ttillum = [
            Sources.PlaneWave(0, b, 0, 0, vec_x),
            Sources.PlaneWave(0, c, 0, 0, vec_mx),
            Sources.PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, np.pi/2)),
            Sources.PlaneWave(0, c, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, np.pi/2)),
            Sources.PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, np.pi)),
            Sources.PlaneWave(0, c, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, np.pi)),
            Sources.PlaneWave(0, b, 0, 0, VectorOperations.rotate_vector3d(vec_x, ax_z, 3 * np.pi/2)),
            Sources.PlaneWave(0, c, 0, 0, VectorOperations.rotate_vector3d(vec_mx, ax_z, 3 * np.pi/2)),
            Sources.PlaneWave(1 * p, 1j * p, 0, 0, np.array((0, 10**-10, self.k))),
        ]

        two_triangles_illumination = Illumination.find_ipw_from_pw(ttillum)
        illumination = Illumination.init_from_list(two_triangles_illumination, base_vectors)
        illumination.Mt = Mt
        illumination.normalize_spacial_waves()

        return illumination