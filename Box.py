import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import wrappers


class FieldHolder:
    def __init__(self, source, grid, identifier):
        self.identifier = identifier
        self.field_type = source.get_source_type()
        self.source = source
        if self.field_type == "ElectricField":
            self.field = np.zeros((len(grid[0]), len(grid[1]), len(grid[2]), 3), dtype=np.complex128)
            self.field += source.get_electric_field(grid)

        elif self.field_type == "Intensity":
            self.field = np.zeros((len(grid[0]), len(grid[1]), len(grid[2])), dtype=np.complex128)
            self.field = source.get_intensity(grid)


class Box:
    def __init__(self, sources, box_size, point_number, additional_info=None):
        self.info = additional_info
        self.point_number = point_number
        self.box_size = box_size
        self.fields = []
        self.source_identifier = 0
        self.grid = np.zeros((self.point_number, self.point_number, self.point_number, 3))
        self.compute_grid()
        self.electric_field = np.zeros((self.point_number, self.point_number, self.point_number, 3),
                                       dtype=np.complex128)
        self.intensity = np.zeros((self.point_number, self.point_number, self.point_number))
        self.intensity_fourier_space = np.zeros((self.point_number, self.point_number, self.point_number))
        self.analytic_frequencies = []

        for source in sources:
            self.fields.append(FieldHolder(source, self.grid, self.source_identifier))
            self.source_identifier += 1

    def compute_grid(self):
        indices = np.array(np.meshgrid(np.arange(self.point_number), np.arange(self.point_number),
                                       np.arange(self.point_number))).T.reshape(-1, 3)
        indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
            self.point_number, self.point_number, self.point_number, 3)
        self.grid = self.box_size * (indices / self.point_number - 1 / 2)

    def compute_electric_field(self):
        self.electric_field = np.zeros(self.electric_field.shape, dtype=np.complex128)
        for field in self.fields:
            if field.field_type == "ElectricField":
                self.electric_field += field.field

    def compute_intensity_from_electric_field(self):
        self.intensity = np.einsum('ijkl, ijkl->ijk', self.electric_field, self.electric_field.conjugate()).real

    def compute_intensity_from_spacial_waves(self):
        self.intensity = np.zeros(self.intensity.shape)
        for field in self.fields:
            if field.field_type == "Intensity":
                self.intensity = self.intensity + field.field
        self.intensity = self.intensity.real

    def compute_intensity_fourier_space(self):
        self.intensity_fourier_space = wrappers.wrapped_fftn(self.intensity) * (self.box_size / self.point_number) ** 3

    def add_source(self, source):
        self.fields.append(FieldHolder(source, self.grid, self.source_identifier))
        self.source_identifier += 1

    def plot_intensity_slices(self, ax=None, slider=None):
        self.plot_slices(self.intensity, ax, slider)

    def plot_intensity_fourier_space_slices(self, ax=None, slider=None):
        self.plot_slices(abs(self.intensity_fourier_space), ax, slider)

    def plot_slices(self, array3d, ax=None, slider=None):
        k_init = self.point_number / 2
        if ax is None:
            fig, ax = plt.subplots()
        values = (np.arange(self.point_number) / self.point_number - 1 / 2) * self.box_size
        X, Y = np.meshgrid(values, values)
        Z = array3d[:, :, int(k_init)].T
        minValue = np.amin(self.intensity)
        maxValue = min(np.amax(self.intensity), 100.0)
        levels = np.linspace(minValue, maxValue + 1, 30)
        cf = ax.contourf(X, Y, Z[:, :], levels)
        plt.colorbar(cf)
        contour_axis = ax

        if slider is None or type(slider) == Slider:
            def update(val):
                contour_axis.clear()
                Z = array3d[:, :, int(val)].T
                ax.clear()
                contour_axis.contourf(X, Y, Z, levels)

            slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
            slider = Slider(slider_loc, 'z', 0, self.point_number - 1)  # slider properties
            slider.on_changed(update)

        plt.show()
