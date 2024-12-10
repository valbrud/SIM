"""
GUI.py

This module contains the main graphical user interface (GUI) components of the application.

This module and related ones is currently a demo-version of the user-interface, and will
possibly be sufficiently modified or replaced in the future. For this reason, no in-depth
documentation is provided.
"""

import sys
import os
import numpy as np
import Box
import Sources
from Illumination import Illumination
import GUIWidgets
from input_parser import ConfigParser
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QComboBox, QFileDialog, QLabel, QSlider, QScrollArea)


from enum import Enum
class View(Enum):
    XY = 0
    YZ = 1
    XZ = 2

class PlottingMode(Enum):
    linear = 0
    logarithmic = 1
    mixed = 2
class MainWindow(QMainWindow):
    def __init__(self, box=None):
        super().__init__()

        np.set_printoptions(precision=2, suppress=True)

        self.init_ui()
        if not box:
            self.box = Box.Box({}, box_size=10, point_number=40)
        else:
            self.box = box
            for field in box.fields:
                self.add_source(field.source)

        self.slider = None
        self.colorbar = None
        self.view = View.XY
        self.plotting_mode = PlottingMode.linear

    def init_ui(self):
        width = 1200
        height = 800
        self.setMinimumSize(width, height)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)

        # Top Section
        self.options_layout = QHBoxLayout()
        self.options_label = QLabel("Options:")
        # self.options_combo = QComboBox()
        # self.options_combo.addItem("Save Config")
        # self.options_combo.addItem("Load Config")
        # self.options_combo.currentIndexChanged.connect(self.on_option_selected)

        self.load_configuration_button = QPushButton("Load Configuration")
        self.load_configuration_button.clicked.connect(self.load_config)
        self.load_illumination_button = QPushButton("Load Illumination")
        self.load_illumination_button.clicked.connect(self.load_illumination)
        self.options_layout.addWidget(self.options_label, 1)
        self.options_layout.addWidget(self.load_configuration_button, 1)
        self.options_layout.addWidget(self.load_illumination_button, 1)
        self.options_layout.addStretch(5)

        # Second Section
        self.canvas_layout = QVBoxLayout()
        self.config_layout = QHBoxLayout()

        # Left Column (Canvas)
        self.canvas = FigureCanvas(plt.figure())
        self.canvas_layout.addWidget(self.canvas, 8)
        self.config_layout.addLayout(self.canvas_layout, 8)

        # Second Column(Sources)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.sources_layout = QVBoxLayout()
        self.sources_layout.addStretch()
        scroll_widget.setLayout(self.sources_layout)
        scroll_area.setWidget(scroll_widget)
        self.config_layout.addWidget(scroll_area, 2)

        # Third Column(Add sources)
        self.config_layout.addLayout(self.source_buttons_layout, 1)

        add_point_source_button = QPushButton("Add a Point Source")
        add_point_source_button.clicked.connect(self.add_point_source)

        add_plane_wave_button = QPushButton("Add a PlaneWave")
        add_plane_wave_button.clicked.connect(self.add_plane_wave)

        add_spatial_frequency_button = QPushButton("Add a Spatial Frequency")
        add_spatial_frequency_button.clicked.connect(self.add_intensity_plane_wave)

        find_fourier_peaks_numerically_button = QPushButton("Find Fourier peaks numerically")
        find_fourier_peaks_numerically_button.clicked.connect(self.compute_numerically_approximated_intensities)

        find_ipw_from_pw_button = QPushButton("Get spatial waves from plane waves")
        find_ipw_from_pw_button.clicked.connect(self.get_ipw_from_pw)

        compute_total_intensity_button = QPushButton("Compute Total Intensity")
        compute_total_intensity_button.clicked.connect(self.compute_total_intensity)

        self.shift_number = 0
        # self.peak_position = (0., 0.)
        compute_next_shift_button = QPushButton("Next shift")
        compute_next_shift_button.clicked.connect(self.compute_next_shift)


        self.source_buttons_layout.addStretch()
        self.source_buttons_layout.addWidget(add_plane_wave_button)
        self.source_buttons_layout.addWidget(add_spatial_frequency_button)
        self.source_buttons_layout.addWidget(find_fourier_peaks_numerically_button)
        self.source_buttons_layout.addWidget(find_ipw_from_pw_button)
        self.source_buttons_layout.addWidget(compute_total_intensity_button)
        self.source_buttons_layout.addWidget(compute_next_shift_button)
        self.source_buttons_layout.addStretch()

        # Initialization layout
        self.plot_layout = QHBoxLayout()

        plot_from_electric_fields_button = QPushButton("Intensity from electric fields")
        plot_from_electric_fields_button.clicked.connect(self.compute_and_plot_from_electric_field)

        plot_from_frequencies_button = QPushButton("Intensity from spatial frequencies")
        plot_from_frequencies_button.clicked.connect(self.compute_and_plot_from_intensity_sources)

        plot_fourier_space_button = QPushButton("Fourier space")
        plot_fourier_space_button.clicked.connect(self.compute_and_plot_fourier_space)

        plot_approximate_intensity_button = QPushButton("Approximate intensity")
        plot_approximate_intensity_button.clicked.connect(self.plot_numerically_approximated_intensity)

        plot_approximate_intensity_fourier_space_button = QPushButton("Approximate Fourier space")
        plot_approximate_intensity_fourier_space_button.clicked.connect(self.plot_numerically_approximated_intensity_fourier_space)

        self.plot_layout.addWidget(plot_from_electric_fields_button, 1)
        self.plot_layout.addWidget(plot_from_frequencies_button, 1)
        self.plot_layout.addWidget(plot_fourier_space_button, 1)
        self.plot_layout.addWidget(plot_approximate_intensity_button, 1)
        self.plot_layout.addWidget(plot_approximate_intensity_fourier_space_button, 1)

        self.view_layout = QHBoxLayout()
        change_view_button = QPushButton("Change view")
        change_view_button.clicked.connect(self.change_view3d)

        change_plotting_mode_button = QPushButton("Change plotting mode")
        change_plotting_mode_button.clicked.connect(self.change_plotting_mode)

        self.view_layout.addWidget(change_view_button, 1)
        self.view_layout.addWidget(change_plotting_mode_button, 1)

        # Add sections to the main layout
        self.main_layout.addLayout(self.options_layout, 1)
        self.main_layout.addLayout(self.config_layout, 8)
        self.main_layout.addLayout(self.plot_layout, 1)
        self.main_layout.addLayout(self.view_layout, 1)

        self.setWindowTitle("Interference patterns GUI")

        self.show()

    def on_option_selected(self, index):
        option = self.options_combo.currentText()
        if option == "Save Config":
            self.save_config()
        elif option == "Load Config":
            self.load_config()

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "Config Files (*.cfg)")
        if filename:
            # Save the config logic here
            print(f"Saving config to {filename}")

    def load_config(self):
        GUIWidgets.SourceWidget.identifier = 0
        filename, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "Config Files (*.conf)")
        filename = os.path.basename(filename)
        if filename:
            if filename.endswith(".conf"):
                self.clear_layout(self.sources_layout)
                parser = ConfigParser()
                conf = parser.read_configuration(filename)
                self.box = Box.Box(conf.sources, conf.box_size, conf.point_number, filename + conf.info)
                for field in self.box.fields:
                    self.add_source(field.source)
                self.compute_and_plot_from_intensity_sources()
                self.plot_intensity_slices()

        else:
            raise FileExistsError("Not a valid file format")

    def load_illumination(self):
        GUIWidgets.SourceWidget.identifier = 0
        filename, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "Config Files (*.conf)")
        filename = os.path.basename(filename)
        if filename:
            if filename.endswith(".conf"):
                self.clear_layout(self.sources_layout)
                parser = ConfigParser()
                conf = parser.read_configuration(filename)

                self.box = Box.BoxSIM(conf.illumination, conf.box_size, conf.point_number, filename + conf.info)
                for field in self.box.fields:
                    self.add_source(field.source)
                self.compute_and_plot_from_intensity_sources()
                self.plot_intensity_slices()

        else:
            raise FileExistsError("Not a valid file format")

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                widget = layout.takeAt(0).widget()
                if widget is not None:
                    widget.deleteLater()

    def add_to_box(self, initialized, source):
        if initialized:
            self.box.add_source(source)
            print("source_added")

    def add_source(self, source):
        if type(source) == Sources.IntensityPlaneWave:
            self.add_intensity_plane_wave(source)
        if type(source) == Sources.PlaneWave:
            self.add_plane_wave(source)

    def remove_source(self, initializer):
        self.box.remove_source(initializer)
        print('is deleted')
    def add_point_source(self):
        source = GUIWidgets.PointSourceWidget()
        self.sources_layout.addWidget(source)

        def add_to_box(initialized):
            if initialized:
                self.box.add_source(source.point_source)
                print(self.box.fields)

        source.isSet.connect(add_to_box)

    def add_plane_wave(self, ipw=None):
        source = GUIWidgets.PlaneWaveWidget(ipw)
        self.sources_layout.addWidget(source)
        source.isSet.connect(lambda initialized: self.add_to_box(initialized, source.plane_wave))
        source.isDeleted.connect(lambda identifier: self.remove_source(identifier))

    def add_intensity_plane_wave(self, ipw=None):
        source = GUIWidgets.IntensityPlaneWaveWidget(ipw)
        self.sources_layout.addWidget(source)
        source.isSet.connect(lambda initialized: self.add_to_box(initialized, source.intensity_plane_wave))
        source.isDeleted.connect(lambda identifier: self.remove_source(identifier))

    def choose_plotting_mode(self, Z):
        if self.plotting_mode == PlottingMode.linear:
            return Z
        elif self.plotting_mode == PlottingMode.logarithmic:
            return np.log10(Z)
        elif self.plotting_mode == PlottingMode.mixed:
            return np.log10(1 + Z)

    def change_plotting_mode(self):
        modes = list(PlottingMode)
        self.plotting_mode = modes[(self.plotting_mode.value + 1) % len(modes)]

    def choose_view3d(self, array, number):
        if self.view == View.XY:
            Z = array[:, :, number].T
        elif self.view == View.YZ:
            Z = array[number, :, :]
        elif self.view == View.XZ:
            Z = array[:, number, :]
        return Z
    def change_view3d(self):
        views = list(View)
        self.view = views[(self.view.value + 1) % len(views)]

    def plot_intensity_slices(self, intensity = None):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.set_aspect('equal')

        intensity = self.box.intensity.real if intensity is None else intensity

        if not self.slider:
            self.slider = QSlider(Qt.Horizontal)  # Horizontal slider
            self.canvas_layout.addWidget(self.slider)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.box.point_number[2] - 1)
            k_init = self.box.point_number[2] // 2
        else:
            self.slider.setMaximum(self.box.point_number[2] - 1)
            k_init = self.slider.value()

        x = (np.arange(self.box.point_number[0]) / self.box.point_number[0] - 1 / 2) * self.box.box_size[0, None]
        y = (np.arange(self.box.point_number[1]) / self.box.point_number[1] - 1 / 2) * self.box.box_size[1, None]
        z = (np.arange(self.box.point_number[2]) / self.box.point_number[2] - 1 / 2) * self.box.box_size[2, None]

        X, Y = np.meshgrid(x, y)
        Z = self.choose_view3d(intensity, k_init)
        # self.choose_labels(ax)
        Z = self.choose_plotting_mode(Z)

        minValue = min(np.amin(intensity), 0.0)
        maxValue = min(np.amax(intensity), 100.0)
        print(maxValue)
        levels = np.linspace(minValue, maxValue + 1, 30)
        cf = ax.contourf(X, Y, Z, levels)

        self.colorbar = self.canvas.figure.colorbar(cf)
        z_val = z[k_init]
        ax.set_title("Intensity, z = {:.2f}".format(z_val))
        ax.set_xlabel("X, $\lambda$")
        ax.set_ylabel("Y, $\lambda$")
        if self.box.info:
            ax.text(self.box.box_size[0] / 2, 1.05 * self.box.box_size[1] / 2, self.box.info, color='red')

        self.canvas.draw()

        def update(slider_val):
            ax.clear()
            z_val = z[slider_val]
            ax.set_title("Intensity, z = {:.2f}".format(z_val))
            Z = self.choose_view3d(intensity, slider_val)
            Z = self.choose_plotting_mode(Z)
            ax.contourf(X, Y, Z, levels)
            if self.box.info:
                ax.text(self.box.box_size[0] / 2, 1.05 * self.box.box_size[1] / 2, self.box.info, color='red')
            self.canvas.draw()

        self.slider.valueChanged.connect(update)

    def plot_fourier_space_slices(self, intensity = None):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.set_aspect('equal')

        if not self.slider:
            self.slider = QSlider(Qt.Horizontal)  # Horizontal slider
            self.canvas_layout.addWidget(self.slider)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.box.point_number[2] - 1)
            k_init = self.box.point_number // 2
        else:
            k_init = self.slider.value()

        intensity = ((np.abs(self.box.intensity_fourier_space) *
                     self.box.box_size[0] * self.box.box_size[1] * self.box.box_size[2]
                      / self.box.point_number[0] * self.box.point_number[1] * self.box.point_number[2])
                     if intensity is None else intensity)

        fx = np.linspace(- self.box.point_number[0] / self.box.box_size[0] / 2.,
                         (self.box.point_number[0] - 2) / self.box.box_size[0] / 2, self.box.point_number[0])

        fy = np.linspace(- self.box.point_number[1]/ self.box.box_size[1] / 2.,
                         (self.box.point_number[1] - 2) / self.box.box_size[1] / 2, self.box.point_number[1])

        fz = np.linspace(- self.box.point_number[2] / self.box.box_size[2] / 2.,
                         (self.box.point_number[2] - 2) / self.box.box_size[2] / 2, self.box.point_number[2])

        Fx, Fy = np.meshgrid(fx, fy)

        Z = self.choose_view3d(intensity, k_init)
        Z = self.choose_plotting_mode(Z)

        minValue = min(np.amin(intensity), 0.0)
        maxValue = min(np.amax(intensity), 100.0)
        print(maxValue)
        levels = np.linspace(minValue, maxValue + 1, 30)
        cf = ax.contourf(Fx, Fy, Z, levels)

        self.colorbar = self.canvas.figure.colorbar(cf)
        fz_val = fz[k_init]

        ax.set_title("Intensity, fz = {:.2f}".format(fz_val))
        ax.set_xlabel("Fx, $\\frac{1}{\lambda}$")
        ax.set_ylabel("Fy, $\\frac{1}{\lambda}$")

        if self.box.info:
            ax.text(np.abs(fx[0]), 1.05 * np.abs(fy[0]), self.box.info, color='red')

        self.canvas.draw()

        def update(slider_val):
            ax.clear()
            fz_val = fz[slider_val]
            ax.set_title("Intensity, fz = {:.2f}".format(fz_val))
            ax.set_xlabel("Fx, $\\frac{1}{\lambda}$")
            ax.set_ylabel("Fy, $\\frac{1}{\lambda}$")
            Z = self.choose_view3d(intensity, slider_val)
            Z = self.choose_plotting_mode(intensity[:, :, int(slider_val)].T)
            ax.contourf(Fx, Fy, Z, levels)
            if self.box.info:
                ax.text(np.abs(fx[0] * 0.9), 1.05 * np.abs(fx[0]), self.box.info, color='red')
            self.canvas.draw()

        self.slider.valueChanged.connect(update)

    def plot_numerically_approximated_intensity(self):
        self.plot_intensity_slices(intensity=self.box.numerically_approximated_intensity)
    def plot_numerically_approximated_intensity_fourier_space(self):
        intensity = (np.abs(self.box.numerically_approximated_intensity_fourier_space)
                     * self.box.box_volume / self.box.point_number[0] * self.box.point_number[1] * self.box.point_number[2])
        self.plot_fourier_space_slices(intensity = intensity)

    def compute_and_plot_from_electric_field(self):
        self.box.compute_electric_field()
        self.box.compute_intensity_from_electric_field()
        self.plot_intensity_slices()

    def compute_and_plot_from_intensity_sources(self):
        self.box.compute_intensity_from_spatial_waves()
        self.plot_intensity_slices()

    def compute_and_plot_fourier_space(self):
        self.box.compute_intensity_fourier_space()
        self.plot_fourier_space_slices()

    def compute_numerically_approximated_intensities(self):
        self.box.compute_intensity_and_spatial_waves_numerically()

    def compute_total_intensity(self):
        self.box.compute_total_illumination()
        self.plot_intensity_slices()

    def compute_next_shift(self):
        self.shift_number +=1
        self.shift_number %= self.box.illumination.spatial_shifts.shape[0]
        self.box.compute_intensity_at_given_shift(self.shift_number)
        self.plot_intensity_slices()
        self.plot_shift_arrow()
        # self.peak_position += self.box.illumination.spatial_shifts[self.shift_number][:2]
    def plot_shift_arrow(self):
        self.canvas.figure.gca().arrow(0., 0., *self.box.illumination.spatial_shifts[self.shift_number][:2], width=0.1, color='red')

    def get_ipw_from_pw(self):
        for source in Illumination.find_ipw_from_pw(self.box.get_plane_waves()):
            self.add_source(source)
            self.box.add_source(source)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
