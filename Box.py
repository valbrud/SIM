import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import Sources
import wrappers
import stattools

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
        if type(box_size) == float or type(box_size) == int:
            box_size = (box_size, box_size, box_size)
        self.box_size = np.array(box_size)
        self.box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
        self.fields = []
        self.numerically_approximated_intensity_fields = []
        self.source_identifier = 0
        self.axes, self.frequency_axes = self.compute_axes()
        self.grid = np.zeros((self.point_number, self.point_number, self.point_number, 3))
        self.compute_grid()
        self.electric_field = np.zeros((self.point_number, self.point_number, self.point_number, 3),
                                       dtype=np.complex128)
        self.intensity = np.zeros((self.point_number, self.point_number, self.point_number))
        self.numerically_approximated_intensity = np.zeros((self.point_number, self.point_number, self.point_number))
        self.intensity_fourier_space = np.zeros((self.point_number, self.point_number, self.point_number))
        self.numerically_approximated_intensity_fourier_space = np.zeros((self.point_number, self.point_number, self.point_number))
        self.analytic_frequencies = []

        for source in sources:
            self.add_source(source)

    def compute_axes(self):
        N = self.point_number
        dx = self.box_size[0] / N
        dy = self.box_size[1] / N
        dz = self.box_size[2] / N

        x = np.arange(-self.box_size[0]/2, self.box_size[0]/2, dx)
        y = np.arange(-self.box_size[1]/2, self.box_size[1]/2, dy)
        z = np.arange(-self.box_size[2]/2, self.box_size[2]/2, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.box_size[0], N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / self.box_size[1], N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / self.box_size[2], N)
        return (x, y, z), (fx, fy, fz)

    def compute_grid(self):
        indices = np.array(np.meshgrid(np.arange(self.point_number), np.arange(self.point_number),
                                       np.arange(self.point_number))).T.reshape(-1, 3)
        indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
            self.point_number, self.point_number, self.point_number, 3)
        self.grid = self.box_size[None, None, None, :] * (indices / self.point_number - 1 / 2)

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

    def compute_intensity_and_spacial_waves_numerically(self):
        self.compute_electric_field()
        self.compute_intensity_from_electric_field()
        self.compute_intensity_fourier_space()
        fourier_peaks, amplitudes = stattools.estimate_localized_peaks(self.intensity_fourier_space, self.frequency_axes)
        numeric_spacial_waves = []
        for fourier_peak, amplitude in zip(fourier_peaks, amplitudes):
            numeric_spacial_waves.append(Sources.IntensityPlaneWave(amplitude, 0, 2 * np.pi * np.array(fourier_peak)))
        for wave in numeric_spacial_waves:
            self.numerically_approximated_intensity_fields.append(FieldHolder(wave, self.grid, self.source_identifier))
            self.source_identifier += 1
        self.numerically_approximated_intensity = np.zeros(self.intensity.shape, dtype = np.complex128)
        for field in self.numerically_approximated_intensity_fields:
            self.numerically_approximated_intensity += field.field
        self.numerically_approximated_intensity = self.numerically_approximated_intensity.real
        self.numerically_approximated_intensity_fourier_space = (
                wrappers.wrapped_ifftn(self.numerically_approximated_intensity) * self.box_volume)

    def compute_intensity_fourier_space(self):
        self.intensity_fourier_space = (wrappers.wrapped_fftn(self.intensity) * (self.box_volume / self.point_number ** 3))

    def add_source(self, source):
        self.fields.append(FieldHolder(source, self.grid, self.source_identifier))
        self.source_identifier += 1

    def remove_source(self, source_identifier):
        for field in self.fields:
            if field.identifier == source_identifier:
                self.fields.remove(field)
                return
    def get_sources(self):
        return [field.source for field in self.fields]
    def get_plane_waves(self):
        return [field.source for field in self.fields if field.field_type == "ElectricField"]
    def get_spacial_waves(self):
        return [field.source for field in self.fields if field.field_type == "Intensity"]
    def get_approximated_intensity_sources(self):
        return [field.source for field in self.numerically_approximated_intensity_fields if field.field_type == "Intensity"]

    def plot_approximate_intensity_slices(self, ax = None, slider=None):
        self.plot_slices(self.numerically_approximated_intensity, ax, slider)

    def plot_approximate_intensity_fourier_space_slices(self, ax=None, slider=None):
        self.plot_slices(self.numerically_approximated_intensity_fourier_space, ax, slider)

    def plot_intensity_slices(self, ax=None, slider=None):
        self.plot_slices(self.intensity, ax, slider)

    def plot_intensity_fourier_space_slices(self, ax=None, slider=None):
        self.plot_slices(abs(self.intensity_fourier_space), ax, slider)

    def plot_slices(self, array3d, ax=None, slider=None):
        k_init = self.point_number / 2
        if ax is None:
            fig, ax = plt.subplots()
        x, y, z = (np.arange(self.point_number) / self.point_number - 1 / 2) * self.box_size[:, None]
        X, Y = np.meshgrid(x, y)
        slice = array3d[:, :, int(k_init)].T
        minValue = np.amin(self.intensity)
        maxValue = min(np.amax(self.intensity), 100.0)
        levels = np.linspace(minValue, maxValue + 1, 30)
        cf = ax.contourf(X, Y, slice[:, :], levels)
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
