import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class Voxel:
    def __init__(self):
        self.electric_field = np.array((0., 0., 0.))
        self.intensity = 0

    def compute_intensity(self):
        self.intensity = 0.5 * self.electric_field ** 2


class Box:
    def __init__(self, plane_waves, box_size, point_number):
        self.voxels = []
        self.point_number = point_number
        self.box_size = box_size
        self.plane_waves = plane_waves
        self.electric_field = np.zeros((self.point_number, self.point_number, self.point_number, 3), dtype=np.complex128)
        self.intensity = np.zeros((self.point_number, self.point_number, self.point_number))

    def compute_field(self):
        for i in range(self.point_number):
            for j in range(self.point_number):
                for k in range(self.point_number):
                    test_value = 0
                    for plane_wave in self.plane_waves:
                        # print(" ", i, " ", j)
                        # print(plane_wave.wavevector.dot(
                        #     self.box_size * (np.array((i, j, k)) / self.point_number)))
                        for p in [0, 1]:
                            test_value += np.exp(1j * plane_wave.wavevector.dot(
                                    self.box_size * (np.array((i, j, k)) / self.point_number - 1 / 2)))
                            self.electric_field[i, j, k] = self.electric_field[i, j, k] + \
                                plane_wave.field_vectors[p] * np.exp(1j * plane_wave.wavevector.dot(
                                    self.box_size * (np.array((i, j, k)) / self.point_number - 1 / 2)) +
                                                                     plane_wave.phases[p])
                    # print(i, j, k)
                    # print(abs(test_value))
                    self.intensity[i, j, k] = 1 / 2 * (np.vdot(self.electric_field[i, j, k], self.electric_field[i, j, k]).real)
                    assert self.intensity[i, j, k] > 0, "Intensity < 0!"
    # def get_intensity_slice(self, j):
    #     voxels = self.voxels[:, j, :]
    #     intensities = np.zeros((self.point_number, 2))
    #     for i in self.point_number:
    #         for k in self.point_number:
    #             intensities[i, k] = voxels[i, k].intensity
    #     return intensities

    def plot_intensity_slice(self):
        k_init = self.point_number / 2
        fig, ax = plt.subplots()
        values = (np.arange(self.point_number) / self.point_number - 1 / 2) * self.box_size
        X, Y = np.meshgrid(values, values)
        Z = self.intensity[:, :, int(k_init)]
        levels = np.linspace(0, np.amax(Z), 30)
        CS = ax.contourf(X, Y, Z, levels)
        contour_axis = plt.gca()

        cbar = fig.colorbar(CS)
        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        jslider = Slider(slider_loc, 'z', 0, self.point_number - 1)  # slider properties

        def update(val):
            contour_axis.clear()
            Z = self.intensity[:, :, int(jslider.val)]
            contour_axis.contourf(X, Y, Z, levels)

        jslider.on_changed(update)

        plt.show()
