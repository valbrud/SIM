import numpy as np
import Sources
from Illumination import Illumination

theta = np.pi / 4
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (np.cos(theta) - 1)

NA = np.sin(theta)

b = 1
Mt_s_polarized = 32
a0 = (2 + 4 * b ** 2)

norm = a0 * Mt_s_polarized
s_polarized_waves = {
    (0, 0, 0)  : Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

    (-2, 0, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((-2 * k1, 0, 0))),
    (2, 0, 0)  : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((2 * k1, 0, 0))),
    (0, 2, 0)  : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0) : Sources.IntensityPlaneWave((-b ** 2) / norm, 0, np.array((0, -2 * k1, 0))),

    (1, 0, 1)  : Sources.IntensityPlaneWave((-1j * b) / norm, 0, np.array((k1, 0, k2))),
    (-1, 0, 1) : Sources.IntensityPlaneWave((1j * b) / norm, 0, np.array((-k1, 0, k2))),
    (0, 1, 1)  : Sources.IntensityPlaneWave((-1 * b) / norm, 0, np.array((0, k1, k2))),
    (0, -1, 1) : Sources.IntensityPlaneWave((1 * b) / norm, 0, np.array((0, -k1, k2))),

    (1, 0, -1) : Sources.IntensityPlaneWave((-1j * b) / norm, 0, np.array((k1, 0, -k2))),
    (-1, 0, -1): Sources.IntensityPlaneWave((1j * b) / norm, 0, np.array((-k1, 0, -k2))),
    (0, 1, -1) : Sources.IntensityPlaneWave((1 * b) / norm, 0, np.array((0, k1, -k2))),
    (0, -1, -1): Sources.IntensityPlaneWave((-1 * b) / norm, 0, np.array((0, -k1, -k2)))
}

Mt_circular = 32
b = 1/2**0.5
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (1 - np.cos(theta))
a0 = (2 + 8 * b ** 2)
circular_intensity_waves = {
    (1, 1, 0)  : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((k1, k1, 0))),
    (-1, 1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((-k1, k1, 0))),
    (1, -1, 0) : Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((k1, -k1, 0))),
    (-1, -1, 0): Sources.IntensityPlaneWave(2 * b ** 2 * np.sin(theta) ** 2 / (Mt_circular * a0), 0, np.array((-k1, -k1, 0))),

    (0, 2, 0)  : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((0, 2 * k1, 0))),
    (0, -2, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((0, -2 * k1, 0))),
    (2, 0, 0)  : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((2 * k1, 0, 0))),
    (-2, 0, 0) : Sources.IntensityPlaneWave(-2 * b ** 2 * np.cos(theta) ** 2 / (Mt_circular * a0), 0, np.array((-2 * k1, 0, 0))),

    (1, 0, -1) : Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((k1, 0, -k2))),
    (-1, 0, 1) : Sources.IntensityPlaneWave(b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((-k1, 0, k2))),
    (1, 0, 1)  : Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((k1, 0, k2))),
    (-1, 0, -1): Sources.IntensityPlaneWave(-b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((-k1, 0, -k2))),

    (0, 1, -1) : Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, k1, -k2))),
    (0, 1, 1)  : Sources.IntensityPlaneWave(-1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, k1, k2))),
    (0, -1, 1) : Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, -k1, k2))),
    (0, -1, -1): Sources.IntensityPlaneWave(1j * b * (1 + np.cos(theta)) / (Mt_circular * a0), 0, np.array((0, -k1, -k2))),

    (0, 0, 0)  : Sources.IntensityPlaneWave(a0 / (Mt_circular * a0), 0, np.array((0, 0, 0)))
}

b = 1
a0 = 2 + 6 * b**2
Mt_seven_waves = 128
norm = a0 * Mt_seven_waves
seven_waves_list = [
    Sources.IntensityPlaneWave(-b**2/norm, 0, np.array((2 * k1, 0, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((-2 * k1, 0, 0))),
    Sources.IntensityPlaneWave(b**2/norm, 0, np.array((k1, 0, 0))),
    Sources.IntensityPlaneWave(b**2/norm, 0, np.array((-k1, 0, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((-k1, 3**0.5*k1, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((k1, 3**0.5*k1, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((-k1, -3**0.5*k1, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((k1, -3**0.5*k1, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((0, 3**0.5*k1, 0))),
    Sources.IntensityPlaneWave(-b ** 2/norm, 0, np.array((0, -3**0.5*k1, 0))),

    Sources.IntensityPlaneWave((2 + 6 * b**2)/norm, 0, np.array((0, 0, 0))),

    Sources.IntensityPlaneWave(b**2/norm, 0, np.array((k1/2, 3**0.5/2 * k1, 0))),
    Sources.IntensityPlaneWave(b**2/norm, 0, np.array((-k1/2, 3**0.5/2 * k1, 0))),
    Sources.IntensityPlaneWave(b**2/norm, 0, np.array((k1/2, -3**0.5/2 * k1, 0))),
    Sources.IntensityPlaneWave(b**2/norm, 0, np.array((-k1/2, -3**0.5/2 * k1, 0))),

    Sources.IntensityPlaneWave(-b**2/norm, 0, np.array((3/2 * k1, -3**0.5/2 * k1, 0))),
    Sources.IntensityPlaneWave(-b**2/norm, 0, np.array((-3/2 * k1, -3**0.5/2 * k1, 0))),
    Sources.IntensityPlaneWave(-b**2/norm, 0, np.array((3/2 * k1, 3**0.5/2 * k1, 0))),
    Sources.IntensityPlaneWave(-b**2/norm, 0, np.array((-3/2 * k1, 3**0.5/2 * k1, 0))),

    Sources.IntensityPlaneWave(1j * b/norm, 0, np.array((-k1, 0, k2))),
    Sources.IntensityPlaneWave(1j * b/norm, 0, np.array((-k1, 0, -k2))),
    Sources.IntensityPlaneWave(-1j * b/norm, 0, np.array((k1, 0, k2))),
    Sources.IntensityPlaneWave(-1j * b/norm, 0, np.array((k1, 0, -k2))),

    Sources.IntensityPlaneWave((2*3**0.5 - 2j)/4 * b/norm, 0, np.array((k1/2, 3**0.5/2 * k1, -k2))),
    Sources.IntensityPlaneWave((-2*3**0.5 - 2j)/4 * b/norm, 0, np.array((k1/2, -3**0.5/2 * k1, -k2))),
    Sources.IntensityPlaneWave((2*3**0.5 + 2j)/4 * b/norm, 0, np.array((-k1/2, 3**0.5/2 * k1, -k2))),
    Sources.IntensityPlaneWave((-2*3**0.5 + 2j)/4 * b/norm, 0, np.array((-k1/2, -3**0.5/2 * k1, -k2))),
    Sources.IntensityPlaneWave((-2*3**0.5 - 2j)/4 * b/norm, 0, np.array((k1/2, 3**0.5/2 * k1, k2))),
    Sources.IntensityPlaneWave((2*3**0.5 - 2j)/4 * b/norm, 0, np.array((k1/2, -3**0.5/2 * k1, k2))),
    Sources.IntensityPlaneWave((-2*3**0.5 + 2j)/4 * b/norm, 0, np.array((-k1/2, 3**0.5/2 * k1, k2))),
    Sources.IntensityPlaneWave((2*3**0.5 + 2j)/4 * b/norm, 0, np.array((-k1/2, -3**0.5/2 * k1, k2))),
]
seven_waves_illumination = Illumination.index_frequencies(seven_waves_list, (k1/2, 3**0.5/2 * k1, k2))


b = 1
a0 = 1 + 2 * b**2
Mr_tree_waves = 3
Mt_three_waves = 4
norm = a0 * Mr_tree_waves * Mt_three_waves
three_waves_illumination = {
    (0, 0, 0)  : Sources.IntensityPlaneWave(a0 / norm, 0, np.array((0, 0, 0))),

    (2, 0, 0)  : Sources.IntensityPlaneWave(b**2 / norm, 0, np.array((2 * k1, 0, 0))),
    (-2, 0, 0) : Sources.IntensityPlaneWave(b**2 / norm, 0, np.array((-2 * k1, 0, 0))),

    (1, 0 , 1)  : Sources.IntensityPlaneWave(b / norm, 0, np.array((k1, 0, k2))),
    (-1, 0 , 1) :  Sources.IntensityPlaneWave(b / norm, 0, np.array((-k1, 0, k2))),
    (1, 0, -1) : Sources.IntensityPlaneWave(b / norm, 0, np.array((k1, 0, -k2))),
    (-1, 0, -1): Sources.IntensityPlaneWave(b / norm, 0, np.array((-k1, 0, -k2))),
}


Mt_widefield = 1
widefield = {
    (0, 0, 0) : Sources.IntensityPlaneWave(1/Mt_widefield, 0, np.array((0, 0, 0)))
}
