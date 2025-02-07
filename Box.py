"""
Box.py

This module contains classes for handling simulation volume and containing fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import Sources
import wrappers
import stattools
import config.IlluminationConfigurations as confs
from Illumination import Illumination
from VectorOperations import VectorOperations

class Field:
    """
    This class keeps field values within a given numeric volume.

    Attributes:
        identifier (int): Unique identifier for the field.
        field_type (str): Type of the field (either "ElectricField" or "Intensity").
        source (Source): The source that produces the field.
        field (np.ndarray): The computed field values.
    """

    def __init__(self, source: Sources.Source, grid: np.ndarray[tuple[int, int, int, 3], np.float64], identifier: int):
        """
        Initializes the Field, given its source, grid where it must be computed and its unique identifier.

        Args:
            source (Source): The source that produces the field.
            grid (np.ndarray): The grid of points in the simulation volume.
            identifier (int): Integer number, identifying the field.
        """
        self.identifier = identifier
        self.field_type = source.get_source_type()
        self.source = source
        if self.field_type == "ElectricField":
            self.field = np.zeros(grid.shape, dtype=np.complex128)
            self.field += source.get_electric_field(grid)

        elif self.field_type == "Intensity":
            self.field = np.zeros(grid.shape[:3], dtype=np.complex128)
            self.field = source.get_intensity(grid)


class Box:
    """
    This class represents a simulation volume where fields and intensities are computed.

    Attributes:
        info (dict): Additional information about the box.
        box_size (np.ndarray): Size of the box in each dimension.
        point_number (np.ndarray): Number of points in each dimension.
        box_volume (float): Volume of the box.
        fields (list): List of fields in the box.
        numerically_approximated_intensity_fields (list): List of numerically approximated intensity fields.
        source_identifier (int): Identifier for the sources.
        axes (tuple): Axes for the box.
        frequency_axes (tuple): Frequency axes for the box.
        grid (np.ndarray): Grid of points in the box.
        electric_field (np.ndarray): Electric field in the box.
        intensity (np.ndarray): Intensity in the box.
        numerically_approximated_intensity (np.ndarray): Numerically approximated intensity in the box.
        intensity_fourier_space (np.ndarray): Intensity in the Fourier space.
        numerically_approximated_intensity_fourier_space (np.ndarray): Numerically approximated intensity in the Fourier space.
        analytic_frequencies (list): List of analytic frequencies.
    """

    def __init__(self, sources=(), box_size=10, point_number=100, additional_info=None):
        """
        Initializes the Box with given sources, size, and point number.

        Args:
            sources (tuple): Tuple of sources of the electric field/intensity.
            box_size (int or tuple): Size of the box in each dimension.
            point_number (int or tuple): Number of points in each dimension.
            additional_info (dict, optional): Additional information about the configuration.
        """
        self.info = additional_info

        if type(box_size) == float or type(box_size) == int:
            box_size = (box_size, box_size, box_size)
        if len(box_size) == 2:
            box_size = (box_size[0], box_size[1], 0)
        self.box_size = np.array(box_size)

        if type(point_number) == int:
            point_number = [point_number, point_number, point_number]
        if box_size[2] == 0:
            point_number[2] = 1
        self.point_number = np.array(point_number)

        self.box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2] if box_size[2] != 0 else self.box_size[0] * self.box_size[1]
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
        """
        Computes the axes and frequency axes for the box.

        Returns:
            tuple: Axes and frequency axes for the box.
        """
        Nx, Ny, Nz = self.point_number

        dx = self.box_size[0] / (Nx - 1)
        dy = self.box_size[1] / (Ny - 1)
        dz = self.box_size[2] / (Nz - 1)

        x = np.linspace(-self.box_size[0] / 2, self.box_size[0] / 2, Nx)
        y = np.linspace(-self.box_size[1] / 2, self.box_size[1] / 2, Ny)
        z = np.linspace(-self.box_size[2] / 2, self.box_size[2] / 2 / 10, Nz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), Nx)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), Ny)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), Nz)
        return (x, y, z), (fx, fy, fz)

    def compute_grid(self):
        """
        Computes the grid of points in the box.
        """

        # By default, meshgrid uses 'xy' indexing, so specify 'ij' for (nx,ny,nz)
        X, Y, Z = np.meshgrid(*self.axes, indexing='ij')

        # shape => (nx, ny, nz, 3)
        self.grid = np.stack([X, Y, Z], axis=-1)


    def compute_electric_field(self):
        """
        Computes the electric field in the box.
        """
        self.electric_field = np.zeros(self.electric_field.shape, dtype=np.complex128)
        for field in self.fields:
            if field.field_type == "ElectricField":
                self.electric_field += field.field

    def compute_intensity_from_electric_field(self):
        """
        Computes the intensity from the electric field.
        """
        self.intensity = np.einsum('ijkl, ijkl->ijk', self.electric_field, self.electric_field.conjugate()).real

    def compute_intensity_from_spatial_waves(self):
        """
        Computes the intensity from intensity spatial waves.
        """
        self.intensity = np.zeros(self.intensity.shape)
        for field in self.fields:
            if field.field_type == "Intensity":
                self.intensity = self.intensity + field.field
        self.intensity = self.intensity.real

    def compute_intensity_and_spatial_waves_numerically(self):
        """
        Find approximately spatial waves from intensity in Fourier space and compute from them the approximated intensity in the box.
        """
        self.compute_electric_field()
        self.compute_intensity_from_electric_field()
        self.compute_intensity_fourier_space()
        fourier_peaks, amplitudes = stattools.estimate_localized_peaks(self.intensity_fourier_space, self.frequency_axes)
        numeric_spatial_waves = []
        for fourier_peak, amplitude in zip(fourier_peaks, amplitudes):
            numeric_spatial_waves.append(Sources.IntensityHarmonic(amplitude, 0, 2 * np.pi * np.array(fourier_peak)))
        for wave in numeric_spatial_waves:
            self.numerically_approximated_intensity_fields.append(Field(wave, self.grid, self.source_identifier))
            self.source_identifier += 1
        self.numerically_approximated_intensity = np.zeros(self.intensity.shape, dtype=np.complex128)
        for field in self.numerically_approximated_intensity_fields:
            self.numerically_approximated_intensity += field.field
        self.numerically_approximated_intensity = self.numerically_approximated_intensity.real
        self.numerically_approximated_intensity_fourier_space = (
                wrappers.wrapped_ifftn(self.numerically_approximated_intensity) * self.box_volume)

    def compute_intensity_fourier_space(self):
        self.intensity_fourier_space = (wrappers.wrapped_fftn(self.intensity) *
                                        (self.box_volume / self.point_number[0] / self.point_number[1] / self.point_number[2]))
        self.intensity_fourier_space /= np.sum(self.intensity)

    def add_source(self, source):
        """
        Adds a source to the box. The corresponding field is added automatically.

        Args:
            source: Source to add.
        """
        self.fields.append(Field(source, self.grid, self.source_identifier))
        self.source_identifier += 1

    def remove_source(self, source_identifier):
        """
        Removes a source from the box by its identifier. The corresponding field is removed as well.

        Args:
            source_identifier (int): Identifier of the source to remove.
        """
        for field in self.fields:
            if field.identifier == source_identifier:
                self.fields.remove(field)
                return

    def get_sources(self):
        """
        Returns a list of sources in the box.

        Returns:
            list: List of sources.
        """
        return [field.source for field in self.fields]

    def get_plane_waves(self):
        """
        Returns a list of plane waves in the box.

        Returns:
            list: List of plane waves.
        """
        return [field.source for field in self.fields if field.field_type == "ElectricField"]

    def get_spatial_waves(self):
        """
        Returns a list of spatial waves in the box.

        Returns:
            list: List of spatial waves.
        """
        return [field.source for field in self.fields if field.field_type == "Intensity"]

    def get_approximated_intensity_sources(self):
        """
        Returns a list of numerically estimated intensity sources in the box.

        Returns:
            list: List of approximated intensity sources.
        """
        return [field.source for field in self.numerically_approximated_intensity_fields if field.field_type == "Intensity"]

    def plot_approximate_intensity_slices(self, ax=None, slider=None):
        """
        Plots slices of the intensity in the real space, computed from spatial waves found numerically.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            slider (matplotlib.widgets.Slider, optional): Slider for interactive plotting. Defaults to None.
        """
        self.plot_slices(self.numerically_approximated_intensity, ax, slider)

    def plot_approximate_intensity_fourier_space_slices(self, ax=None, slider=None):
        """
        Plots slices of the intensity in the Fourier space, computed from spatial waves found numerically .

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            slider (matplotlib.widgets.Slider, optional): Slider for interactive plotting. Defaults to None.
        """
        self.plot_slices(self.numerically_approximated_intensity_fourier_space, ax, slider)

    def plot_intensity_slices(self, ax=None, slider=None):
        """
        Plots slices of the intensity in the real space.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            slider (matplotlib.widgets.Slider, optional): Slider for interactive plotting. Defaults to None.
        """
        self.plot_slices(self.intensity, ax, slider)

    def plot_intensity_fourier_space_slices(self, ax=None, slider=None):
        """
        Plots slices of the intensity in the Fourier space.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            slider (matplotlib.widgets.Slider, optional): Slider for interactive plotting. Defaults to None.
        """
        self.plot_slices(abs(self.intensity_fourier_space), ax, slider)

    def plot_slices(self, array3d, ax=None, slider=None):
        """
        Plots slices of a 3D array.

        Args:
            array3d (np.ndarray): 3D array to plot.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            slider (matplotlib.widgets.Slider, optional): Slider for interactive plotting. Defaults to None.
        """
        k_init = self.point_number[2] // 2
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
    """
    This class is an extension of the class Box that supports SIM specific operations,
    such as illumination shifts.

    Attributes:
        illumination (IlluminationConfiguration): The illumination configuration.
        illuminations_shifted (np.ndarray): Array of shifted illuminations for different angles and shifts.
    """

    def __init__(self, illumination: Illumination = confs.BFPConfiguration().get_widefield(), box_size=10, point_number=100, additional_info=None):
        """
        Initializes the BoxSIM with given illumination, size, and point number.

        Args:
            illumination (IlluminationConfiguration): The illumination configuration.
            box_size (int or tuple): Size of the box in each dimension.
            point_number (int or tuple): Number of points in each dimension.
            additional_info (dict, optional): Additional information about the configuration.
        """
        sources = illumination.waves.values()
        super().__init__(sources, box_size, point_number, additional_info)
        self.illumination = illumination
        self.illuminations_shifted = np.zeros((illumination.Mr, illumination.Mt, *self.point_number))
        self._compute_illumination_at_all_rm()

    def _compute_illumination_at_all_rm(self):
        """
        Computes the illumination at all rotation and shift combinations.
        """
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                self.illuminations_shifted[r, n] = self._compute_illumination_at_given_rm(r, n)

    def _compute_illumination_at_given_rm(self, r: int, n: int):
        """
        Computes the illumination at a given rotation and shift combination.

        Args:
            r (int): Rotation index.
            n (int): Shift index.

        Returns:
            np.ndarray: Computed illumination.
        """
        intensity = np.zeros(self.point_number, dtype=np.complex128)
        urn = VectorOperations.rotate_vector3d(self.illumination.spatial_shifts[n],
                                               np.array((0, 0, 1)), self.illumination.angles[r])
        for field in self.fields:
            if r == 0:
                k0n = field.source.wavevector
                phase = np.dot(urn, k0n)
                intensity += field.field * np.exp(-1j * phase)
            else:
                krm = VectorOperations.rotate_vector3d(field.source.wavevector,
                                                       np.array((0, 0, 1)), self.illumination.angles[r])
                phase = np.dot(urn, krm)
                source = Sources.IntensityHarmonic(field.source.amplitude, field.source.phase, krm)
                field_rotated = Field(source, self.grid, 0).field
                intensity += field_rotated * np.exp(-1j * phase)
        self.illuminations_shifted[r, n] = intensity.real

        #
        # self.intensity = np.zeros(self.intensity.shape, dtype=np.complex128)
        # for field in self.fields:
        #     if field.field_type == "Intensity":
        #         krm = VectorOperations.rotate_vector3d(self.illumination.spatial_shifts[m],
        #                                                         np.array((0, 0, 1)), self.illumination.angles[r])
        #         wavevector = VectorOperations.rotate_vector3d(field.source.wavevector,
        #                                                         np.array((0, 0, 1)), self.illumination.angles[r])
        #         phase = np.dot(krm, wavevector)
        #         debug_phase = round(phase * 57.21) % 360
        #         self.intensity += field.field * np.exp(1j * phase)
        # self.intensity = self.intensity.real

    def get_intensity(self, r: int, n: int) -> np.ndarray:
        return self.illuminations_shifted[r, n]

    def compute_total_illumination(self) -> np.ndarray:
        total = np.zeros(self.intensity.shape, dtype=np.complex128)
        for rotation in self.illuminations_shifted:
            for illumination_shifted in rotation:
                total += illumination_shifted
        return total