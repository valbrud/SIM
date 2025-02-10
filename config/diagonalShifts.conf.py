import numpy as np
from Sources import IntensityHarmonic3D, PlaneWave
from Box import Box

theta = np.pi / 4
b = 1
k = 2 * np.pi
k1 = k * np.sin(theta)
k2 = k * (np.cos(theta) - 1)
a0 = 2 + 4 * b**2
sources = [
    IntensityHarmonic3D((2 + 4 * b ** 2)/a0, 0, np.array((0, 0, 0))),
    IntensityHarmonic3D(-b ** 2/a0, 0, np.array((-2 * k1, 0, 0))),
    IntensityHarmonic3D(-b ** 2/a0, 0, np.array((2 * k1, 0, 0))),
    IntensityHarmonic3D(-b ** 2/a0, 0, np.array((0, 2 * k1, 0))),
    IntensityHarmonic3D(-b ** 2/a0, 0, np.array((0, -2 * k1, 0))),
    IntensityHarmonic3D(-1j * b/a0, 0, np.array((k1, 0, k2))),
    IntensityHarmonic3D(1j * b/a0, 0, np.array((-k1, 0, k2))),
    IntensityHarmonic3D(-1 * b/a0, 0, np.array((0, k1, k2))),
    IntensityHarmonic3D(1 * b/a0, 0, np.array((0, -k1, k2))),
    IntensityHarmonic3D(-1j * b/a0, 0, np.array((k1, 0, -k2))),
    IntensityHarmonic3D(1j * b/a0, 0, np.array((-k1, 0, -k2))),
    IntensityHarmonic3D(1 * b/a0, 0, np.array((0, k1, -k2))),
    IntensityHarmonic3D(-1 * b/a0, 0, np.array((0, -k1, -k2))),
]