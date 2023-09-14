import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import wrappers


class Box:
    def __init__(self, sources, box_size, point_number):
        self.point_number = point_number
        self.box_size = box_size
        self.sources = sources
        self.electric_field = np.zeros((self.point_number, self.point_number, self.point_number, 3), dtype=np.complex128)
        self.intensity = np.zeros((self.point_number, self.point_number, self.point_number))
        self.intensity_fourier_space = np.zeros((self.point_number, self.point_number, self.point_number))
        self.analytic_frequencies = []
    def compute_field(self):
        for i in range(self.point_number):
            print(i)
            for j in range(self.point_number):
                for k in range(self.point_number):
                    coordinates = self.box_size * (np.array((i, j, k)) / self.point_number - 1 /2)
                    for source_type in self.sources:
                        if source_type != "SpacialWave":
                            for source in self.sources[source_type]:
                                self.electric_field[i,j,k] = self.electric_field[i,j,k] + source.get_electric_field(coordinates)
                    self.intensity[i, j, k] = 1 / 2 * np.vdot(self.electric_field[i, j, k], self.electric_field[i, j, k]).real
                    assert self.intensity[i, j, k] > 0, "Intensity < 0!"

    def set_intensity_from_spacial_waves(self):
        for i in range(self.point_number):
            print(i)
            for j in range(self.point_number):
                for k in range(self.point_number):
                    coordinates = self.box_size * (np.array((i, j, k)) / self.point_number - 1 / 2)
                    intensity = 0 + 0j
                    for source in self.sources['SpacialWave']:
                        intensity = intensity + source.amplitude * \
                                                   np.exp(1j * np.dot(source.wavevector, coordinates))
                    self.intensity[i,j,k] = intensity
                    # assert self.intensity[i, j, k] > 0, "Intensity < 0!"
    def compute_intensity_fourier_space(self):
        self.intensity_fourier_space = wrappers.wrapped_fftn(self.intensity) * (self.box_size/self.point_number)**3

    def plot_intensity_slices(self):
        self.plot_slices(self.intensity)

    def plot_intensity_fourier_space_slices(self):
        self.plot_slices(abs(self.intensity_fourier_space))


    def plot_slices(self, array3d):
        k_init = self.point_number / 2
        fig, ax = plt.subplots()
        values = (np.arange(self.point_number) / self.point_number - 1 / 2) * self.box_size
        X, Y = np.meshgrid(values, values)
        Z = array3d[:, :, int(k_init)]
        minValue = np.amin(self.intensity)
        maxValue = min(np.amax(self.intensity), 100)
        levels = np.linspace(minValue, maxValue, 30)
        CS = ax.contourf(X, Y, Z, levels)
        contour_axis = plt.gca()

        cbar = fig.colorbar(CS)
        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        jslider = Slider(slider_loc, 'z', 0, self.point_number - 1)  # slider properties

        def update(val):
            contour_axis.clear()
            Z = array3d[:, :, int(jslider.val)]
            contour_axis.contourf(X, Y, Z, levels)

        jslider.on_changed(update)
        plt.show()