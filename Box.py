import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import Sources
import wrappers
import stattools
import config.IlluminationConfigurations as confs
from VectorOperations import VectorOperations
class Field:
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
    def __init__(self, sources=(), box_size=10, point_number=100, additional_info=None):
        self.info = additional_info
        if type(point_number) == int:
            point_number = (point_number, point_number, point_number)
        self.point_number = np.array(point_number)
        if type(box_size) == float or type(box_size) == int:
            box_size = (box_size, box_size, box_size)
        self.box_size = np.array(box_size)
        self.box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
        self.fields = []
        self.numerically_approximated_intensity_fields = []
        self.source_identifier = 0
        self.axes, self.frequency_axes = self.compute_axes()
        self.grid = np.zeros((self.point_number[0], self.point_number[1], self.point_number[2], 3))
        self.compute_grid()
        self.electric_field = np.zeros((self.point_number[0], self.point_number[1], self.point_number[2], 3),
                                       dtype=np.complex128)
        self.intensity = np.zeros((self.point_number[0], self.point_number[1], self.point_number[2]))
        self.numerically_approximated_intensity = np.zeros((self.point_number[0], self.point_number[1], self.point_number[2]))
        self.intensity_fourier_space = np.zeros((self.point_number[0], self.point_number[1], self.point_number[2]))
        self.numerically_approximated_intensity_fourier_space = np.zeros((self.point_number[0], self.point_number[1], self.point_number[2]))
        self.analytic_frequencies = []

        for source in sources:
            self.add_source(source)


    def compute_axes(self):
        Nx, Ny, Nz = self.point_number
        dx = self.box_size[0] / Nx
        dy = self.box_size[1] / Ny
        dz = self.box_size[2] / Nz

        x = np.arange(-self.box_size[0]/2, self.box_size[0]/2, dx)
        y = np.arange(-self.box_size[1]/2, self.box_size[1]/2, dy)
        z = np.arange(-self.box_size[2]/2, self.box_size[2]/2, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.box_size[0], Nx)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / self.box_size[1], Ny)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / self.box_size[2], Nz)
        return (x, y, z), (fx, fy, fz)

    def compute_grid(self):
        indices = np.array(np.meshgrid(np.arange(self.point_number[0]), np.arange(self.point_number[1]),
                                       np.arange(self.point_number[2]))).T.reshape(-1, 3)
        indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
            self.point_number[0], self.point_number[1], self.point_number[2], 3)
        self.grid = self.box_size[None, None, None, :] * (indices / self.point_number[None, None, None, :] - 1 / 2)

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
            self.numerically_approximated_intensity_fields.append(Field(wave, self.grid, self.source_identifier))
            self.source_identifier += 1
        self.numerically_approximated_intensity = np.zeros(self.intensity.shape, dtype = np.complex128)
        for field in self.numerically_approximated_intensity_fields:
            self.numerically_approximated_intensity += field.field
        self.numerically_approximated_intensity = self.numerically_approximated_intensity.real
        self.numerically_approximated_intensity_fourier_space = (
                wrappers.wrapped_ifftn(self.numerically_approximated_intensity) * self.box_volume)

    def compute_intensity_fourier_space(self):
        self.intensity_fourier_space = (wrappers.wrapped_fftn(self.intensity) *
                                        (self.box_volume /self.point_number[0] / self.point_number[1] / self.point_number[2]))

    def add_source(self, source):
        self.fields.append(Field(source, self.grid, self.source_identifier))
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
        x = (np.arange(self.point_number[0]) / self.point_number[0] - 1 / 2) * self.box_size[0, None]
        y = (np.arange(self.point_number[1]) / self.point_number[1] - 1 / 2) * self.box_size[1, None]
        z = (np.arange(self.point_number[2]) / self.point_number[2] - 1 / 2) * self.box_size[2, None]
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
            slider = Slider(slider_loc, 'z', 0, self.point_number[2] - 1)  # slider properties
            slider.on_changed(update)

        plt.show()

class BoxSIM(Box):
    def __init__(self, illumination=confs.BFPConfiguration().get_widefield(), box_size=10, point_number=100, additional_info=None):
        sources = illumination.waves.values()
        super().__init__(sources, box_size, point_number, additional_info)
        self.illumination = illumination
        self.illuminations_shifted = np.zeros((illumination.Mr, illumination.Mt, *self.point_number))
        self._compute_illumination_at_all_rm()

    def _compute_illumination_at_all_rm(self):
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                self.illuminations_shifted[r, n] = self._compute_illumination_at_given_rm(r, n)

    def _compute_illumination_at_given_rm(self, r, n):
        intensity = np.zeros(self.point_number, dtype=np.complex128)
        urn = VectorOperations.rotate_vector3d(self.illumination.spacial_shifts[n],
                                                        np.array((0, 0, 1)), self.illumination.angles[r])
        for field in self.fields:
            if r == 0:
                k0n = field.source.wavevector
                phase = np.dot(urn, k0n)
                intensity += field.field * np.exp(1j * phase)
            else:
                krm = VectorOperations.rotate_vector3d(field.source.wavevector,
                                                        np.array((0, 0, 1)), self.illumination.angles[r])
                phase = np.dot(urn, krm)
                source = Sources.IntensityPlaneWave(field.source.amplitude, field.source.phase, krm)
                field_rotated = Field(source, self.grid, 0).field
                intensity += field_rotated * np.exp(1j * phase)
        self.illuminations_shifted[r, n] = intensity.real
        return self.illuminations_shifted[r, n]

        #
        # self.intensity = np.zeros(self.intensity.shape, dtype=np.complex128)
        # for field in self.fields:
        #     if field.field_type == "Intensity":
        #         krm = VectorOperations.rotate_vector3d(self.illumination.spacial_shifts[m],
        #                                                         np.array((0, 0, 1)), self.illumination.angles[r])
        #         wavevector = VectorOperations.rotate_vector3d(field.source.wavevector,
        #                                                         np.array((0, 0, 1)), self.illumination.angles[r])
        #         phase = np.dot(krm, wavevector)
        #         debug_phase = round(phase * 57.21) % 360
        #         self.intensity += field.field * np.exp(1j * phase)
        # self.intensity = self.intensity.real

    def get_intensity(self, r, n):
        return self.illuminations_shifted[r, n]

    def compute_total_illumination(self):
        total = np.zeros(self.intensity.shape, dtype=np.complex128)
        for rotation in self.illuminations_shifted:
            for illumination_shifted in rotation:
                total += illumination_shifted
        return total