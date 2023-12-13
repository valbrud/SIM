import sys
import os
import numpy as np
import Box
import Sources
import GUIWidgets
from input_parser import ConfigParser
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QComboBox, QFileDialog, QLabel, QSlider, QScrollArea)


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

        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.load_config)

        self.options_layout.addWidget(self.options_label, 1)
        self.options_layout.addWidget(self.load_button, 1)
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
        self.source_buttons_layout = QVBoxLayout()
        self.config_layout.addLayout(self.source_buttons_layout, 1)

        add_point_source_button = QPushButton("Add a Point Source")
        add_point_source_button.clicked.connect(self.add_point_source)

        add_plane_wave_button = QPushButton("Add a PlaneWave")
        add_plane_wave_button.clicked.connect(self.add_plane_wave)

        add_spacial_frequency_button = QPushButton("Add a Spatial Frequency")
        add_spacial_frequency_button.clicked.connect(self.add_intensity_plane_wave)

        self.source_buttons_layout.addStretch()
        self.source_buttons_layout.addWidget(add_plane_wave_button)
        self.source_buttons_layout.addWidget(add_spacial_frequency_button)
        self.source_buttons_layout.addStretch()

        # Initialization layout
        self.initialization_layout = QHBoxLayout()
        setup_sources_button = QPushButton("Set up from sources")
        setup_sources_button.clicked.connect(self.compute_and_plot_from_electric_field)
        setup_frequencies_button = QPushButton("Set up from spatial frequencies")
        setup_frequencies_button.clicked.connect(self.compute_and_plot_from_intensity_sources)
        setup_fourier_space_button = QPushButton("Fourier space")
        setup_fourier_space_button.clicked.connect(self.compute_and_plot_fourier_space)
        self.initialization_layout.addWidget(setup_sources_button, 1)
        self.initialization_layout.addWidget(setup_frequencies_button, 1)
        self.initialization_layout.addWidget(setup_fourier_space_button, 1)

        # Add sections to the main layout
        self.main_layout.addLayout(self.options_layout, 1)
        self.main_layout.addLayout(self.config_layout, 8)
        self.main_layout.addLayout(self.initialization_layout, 1)

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
                # Show an error message for invalid file format
                print("Not a valid file format")

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

    def plotting_mode(self, Z, mode="linear"):
        if mode == "linear":
            return Z
        elif mode == "logarithmic":
            return np.log10(Z)
        elif mode == "mixed":
            return np.log10(1 + Z)

    def plot_fourier_space_slices(self, mode="linear"):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.set_aspect('equal')

        if not self.slider:
            self.slider = QSlider(Qt.Horizontal)  # Horizontal slider
            self.canvas_layout.addWidget(self.slider)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.box.point_number - 1)
            k_init = self.box.point_number // 2
        elif self.slider:
            k_init = self.slider.value()

        intensity = (np.abs(self.box.intensity_fourier_space) *
                     self.box.box_size[0] * self.box.box_size[1] * self.box.box_size[2] / self.box.point_number ** 3)

        fx = np.linspace(- self.box.point_number / self.box.box_size[0] / 2.,
                         (self.box.point_number - 1) / self.box.box_size[0] / 2, self.box.point_number)

        fy = np.linspace(- self.box.point_number / self.box.box_size[1] / 2.,
                         (self.box.point_number - 1) / self.box.box_size[1] / 2, self.box.point_number)

        fz = np.linspace(- self.box.point_number / self.box.box_size[2] / 2.,
                         (self.box.point_number - 1) / self.box.box_size[2] / 2, self.box.point_number)

        Fx, Fy = np.meshgrid(fx, fy)

        Z = self.plotting_mode(intensity[:, :, k_init].T, mode)

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
            Z = self.plotting_mode(intensity[:, :, int(slider_val)].T, mode)
            ax.contourf(Fx, Fy, Z, levels)
            if self.box.info:
                ax.text(np.abs(fx[0] * 0.9), 1.05 * np.abs(fx[0]), self.box.info, color='red')
            self.canvas.draw()

        self.slider.valueChanged.connect(update)

    def plot_intensity_slices(self):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.set_aspect('equal')

        intensity = self.box.intensity.real

        if not self.slider:
            self.slider = QSlider(Qt.Horizontal)  # Horizontal slider
            self.canvas_layout.addWidget(self.slider)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.box.point_number - 1)
            k_init = self.box.point_number // 2
        else:
            k_init = self.slider.value()

        x, y, z = (np.arange(self.box.point_number) / self.box.point_number - 1 / 2) * self.box.box_size[:, None]
        X, Y = np.meshgrid(x, y)
        Z = intensity[:, :, int(k_init)].T
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
            Z = intensity[:, :, int(slider_val)].T
            ax.contourf(X, Y, Z, levels)
            if self.box.info:
                ax.text(self.box.box_size[0] / 2, 1.05 * self.box.box_size[1] / 2, self.box.info, color='red')
            self.canvas.draw()

        self.slider.valueChanged.connect(update)

    def compute_and_plot_from_electric_field(self):
        self.box.compute_electric_field()
        self.box.compute_intensity_from_electric_field()
        self.plot_intensity_slices()

    def compute_and_plot_from_intensity_sources(self):
        self.box.compute_intensity_from_spacial_waves()
        self.plot_intensity_slices()

    def compute_and_plot_fourier_space(self):
        self.box.compute_intensity_fourier_space()
        self.plot_fourier_space_slices()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
