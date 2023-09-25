# main_window.py
import sys
import os
import numpy as np

import Box
import Sources
import GUIWidgets

os.path.dirname(os.path.abspath(__file__))
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QComboBox, QFileDialog, QLabel, QSlider)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class MainWindow(QMainWindow):
    def __init__(self, box = None):
        super().__init__()

        self.slider = None
        if not box:
            self.box = Box.Box({}, box_size=10, point_number=40)

        self.init_ui()

        if box:
            self.box = box
            for field in box.fields:
                self.add_source(field.source)

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)

        # Top Section
        self.options_layout = QHBoxLayout()
        self.options_label = QLabel("Options:")
        self.options_combo = QComboBox()
        self.options_combo.addItem("Save Config")
        self.options_combo.addItem("Load Config")
        self.options_combo.currentIndexChanged.connect(self.on_option_selected)
        self.options_layout.addWidget(self.options_label,1)
        self.options_layout.addWidget(self.options_combo, 1)
        self.options_layout.addStretch(5)

        # Second Section
        self.canvas_layout = QVBoxLayout()
        self.config_layout = QHBoxLayout()

        # Left Column (Canvas)
        self.canvas = FigureCanvas(Figure())
        self.canvas_layout.addWidget(self.canvas, 5)
        self.config_layout.addLayout(self.canvas_layout, 3)

        # Second Column(Sources)
        self.sources_layout = QVBoxLayout()
        self.sources_layout.addStretch()
        self.config_layout.addLayout(self.sources_layout, 1)

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
        self.source_buttons_layout.addWidget(add_point_source_button)
        self.source_buttons_layout.addWidget(add_plane_wave_button)
        self.source_buttons_layout.addWidget(add_spacial_frequency_button)
        self.source_buttons_layout.addStretch()

        #Initialization layout
        self.initialization_layout = QHBoxLayout()
        setup_sources_button = QPushButton("Set up from sources")
        setup_sources_button.clicked.connect(self.compute_and_plot_from_electric_field)
        setup_frequencies_button = QPushButton("Set up from spatial frequencies")
        setup_frequencies_button.clicked.connect(self.compute_and_plot_from_intensity_sources)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_graph)
        self.initialization_layout.addWidget(setup_sources_button, 1)
        self.initialization_layout.addWidget(setup_frequencies_button, 1)
        self.initialization_layout.addWidget(reset_button, 1)

        # Add sections to the main layout
        self.main_layout.addLayout(self.options_layout, 1)
        self.main_layout.addLayout(self.config_layout, 8)
        self.main_layout.addLayout(self.initialization_layout, 1)

        self.setWindowTitle("PyQt5 Window")
        self.setGeometry(100, 100, 800, 600)

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
        filename, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "Config Files (*.cfg)")
        if filename:
            if filename.endswith(".cfg"):
                # Load the config logic here
                print(f"Loading config from {filename}")
            else:
                # Show an error message for invalid file format
                print("Not a valid file format")

    def add_to_box(self, initialized, source):
        if initialized:
            self.box.add_source(source)
            print("source_added")

    def add_source(self, source):
        if type(source) == Sources.IntensityPlaneWave:
            self.add_intensity_plane_wave(source)
        if type(source) == Sources.PlaneWave:
            self.add_plane_wave(source)

    def add_point_source(self):
        source = UIWrappers.PointSourceWidget()
        self.sources_layout.addWidget(source)

        def add_to_box(initialized):
            if initialized:
                self.box.add_source(source.point_source)
                print(self.box.fields)

        source.isSet.connect(add_to_box)

    def add_plane_wave(self, ipw = None):
        source = UIWrappers.PlaneWaveWidget(ipw)
        self.sources_layout.addWidget(source)
        source.isSet.connect(lambda initialized: self.add_to_box(initialized, source.plane_wave))

    def add_intensity_plane_wave(self, ipw = None):
        source = UIWrappers.IntensityPlaneWaveWidget(ipw)
        self.sources_layout.addWidget(source)
        source.isSet.connect(lambda initialized: self.add_to_box(initialized, source.intensity_plane_wave))

    def plot_intensity_slices(self):
        ax = self.canvas.figure.add_subplot(111)
        ax.set_title("Intensity")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        if not self.slider:
            self.slider = QSlider(Qt.Horizontal)  # Horizontal slider
            self.canvas_layout.addWidget(self.slider)
            self.slider.setMinimum(0)
            self.slider.setMaximum(self.box.point_number - 1)

        k_init = self.box.point_number / 2
        values = (np.arange(self.box.point_number) / self.box.point_number - 1 / 2) * self.box.box_size
        X, Y = np.meshgrid(values, values)
        Z = self.box.intensity[:, :, int(k_init)]
        minValue = min(np.amin(self.box.intensity), 0.0)
        maxValue = min(np.amax(self.box.intensity), 100.0)
        print(maxValue)
        levels = np.linspace(minValue, maxValue + 1, 30)
        cf = ax.contourf(X, Y, Z, levels)
        contour_axis = ax

        def update(slider_val):
            contour_axis.clear()
            Z = self.box.intensity[:, :, int(slider_val)]
            ax.contourf(X, Y, Z, levels)
            self.canvas.draw()
            contour_axis.contourf(X, Y, Z, levels)

        self.slider.valueChanged.connect(update)

    def compute_and_plot_from_electric_field(self):
        self.box.compute_electric_field()
        self.box.compute_intensity_from_electric_field()
        self.plot_intensity_slices()

    def compute_and_plot_from_intensity_sources(self):
        print(self.box.fields)
        self.box.compute_intensity_from_spacial_waves()
        self.plot_intensity_slices()

    def reset_graph(self):
        # Implement logic to reset the graph on the canvas
        print("Resetting graph")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
