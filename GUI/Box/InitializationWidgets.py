"""
InitializationWidgets.py

This module contains widget classes for initializing parameters of different source types
in the GUI.

These widgets provide input fields for users to enter parameters such as amplitudes,
phases, and wavevectors for sources like intensity harmonics and plane waves.

Classes:
    InitializationWidget: Abstract base class for initialization widgets.
    IntensityHarmonic3DInitializationWidget: Widget for IntensityHarmonic3D parameters.
    PlaneWaveInitializationWidget: Widget for PlaneWave parameters.
    PointSourceInitializationWidget: Widget for PointSource parameters.
"""


import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QApplication, QMenu, QAction)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import pyqtSignal

from abc import abstractmethod


class InitializationWidget(QWidget):
    """
    Abstract base class for source initialization widgets.

    Provides a common structure for widgets that collect user input for source parameters.
    """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

    @abstractmethod
    def request_data(self):
        """
        Set up the input fields for the widget.
        """
        ...

    @abstractmethod
    def on_click_ok(self):
        """
        Handle the OK button click to process and emit the input data.
        """
        ...


class PlaneWaveInitializationWidget(InitializationWidget):
    """
    Widget for initializing PlaneWave source parameters.

    Provides input fields for electric field components, phases, and wavevector.

    Attributes:
        sendInfo (pyqtSignal): Signal emitted with the collected parameters.
        Ep, Es (QLineEdit): Input fields for electric field components.
        phasep, phases (QLineEdit): Input fields for phases.
        kx, ky, kz (QLineEdit): Input fields for wavevector components.
    """
    sendInfo = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.request_data()

    def request_data(self):
        self.layout.addWidget(QLabel("Plane Wave"))

        self.Ep = QLineEdit()
        self.Es = QLineEdit()

        self.phasep = QLineEdit()
        self.phases = QLineEdit()

        self.kx = QLineEdit()
        self.ky = QLineEdit()
        self.kz = QLineEdit()

        self.Ep.setValidator(QDoubleValidator())
        self.Es.setValidator(QDoubleValidator())

        self.phasep.setValidator(QDoubleValidator())
        self.phases.setValidator(QDoubleValidator())

        self.kx.setValidator(QDoubleValidator())
        self.ky.setValidator(QDoubleValidator())
        self.kz.setValidator(QDoubleValidator())

        self.Ep.setText('0')
        self.Es.setText('0')

        self.phasep.setText('0')
        self.phases.setText('0')

        self.kx.setText('0')
        self.ky.setText('0')
        self.kz.setText('0')

        layout = QHBoxLayout()
        layout.addWidget(QLabel('Ep'))
        layout.addWidget(self.Ep)
        layout.addWidget(QLabel('Es'))
        layout.addWidget(self.Es)

        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel('Phase p'))
        layout.addWidget(self.phasep)
        layout.addWidget(QLabel('Phase s'))
        layout.addWidget(self.phases)

        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel('Wavevector'))
        layout.addWidget(self.kx)
        layout.addWidget(self.ky)
        layout.addWidget(self.kz)

        self.layout.addLayout(layout)

        self.button_ok = QPushButton("Ok")
        self.button_ok.clicked.connect(self.on_click_ok)

        self.layout.addWidget(self.button_ok)

        self.setLayout(self.layout)

    def on_click_ok(self):
        info_fields = [self.Ep, self.Es, self.phasep, self.phases, self.kx, self.ky, self.kz]
        info = [field.text().strip().strip('()').replace(' ', '') for field in info_fields]
        for value in range(len(info)):
            if info[value] == '':
                info[value] = '0'
        self.sendInfo.emit(info)


class IntensityHarmonic3DInitializationWidget(InitializationWidget):
    """
    Widget for initializing IntensityHarmonic3D source parameters.

    Provides input fields for amplitude, phase, and wavevector components.

    Attributes:
        sendInfo (pyqtSignal): Signal emitted with the collected parameters.
        A (QLineEdit): Input field for amplitude.
        phase (QLineEdit): Input field for phase.
        kx, ky, kz (QLineEdit): Input fields for wavevector components.    """
    sendInfo = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.request_data()

    def request_data(self):
        self.layout.addWidget(QLabel("Intensity Plane Wave"))

        self.A = QLineEdit()

        self.phase = QLineEdit()

        self.kx = QLineEdit()
        self.ky = QLineEdit()
        self.kz = QLineEdit()

        self.A.setValidator(QDoubleValidator())

        self.phase.setValidator(QDoubleValidator())

        self.kx.setValidator(QDoubleValidator())
        self.ky.setValidator(QDoubleValidator())
        self.kz.setValidator(QDoubleValidator())

        self.A.setText('0')

        self.phase.setText('0')

        self.kx.setText('0')
        self.ky.setText('0')
        self.kz.setText('0')

        layout = QHBoxLayout()
        layout.addWidget(QLabel('A'))
        layout.addWidget(self.A)

        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel('Phase'))
        layout.addWidget(self.phase)

        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel('Wavevector'))
        layout.addWidget(self.kx)
        layout.addWidget(self.ky)
        layout.addWidget(self.kz)

        self.layout.addLayout(layout)

        self.button_ok = QPushButton("Ok")
        self.button_ok.clicked.connect(self.on_click_ok)

        self.layout.addWidget(self.button_ok)

        self.setLayout(self.layout)

    def on_click_ok(self):
        info_fields = [self.A, self.phase, self.kx, self.ky, self.kz]
        info = [field.text().strip().strip('()').replace(' ', '') for field in info_fields]
        for value in range(len(info)):
            if info[value] == '':
                info[value] = '0'
        self.sendInfo.emit(info)


class PointSourceInitializationWidget(InitializationWidget):
    """
    Widget for initializing PointSource parameters.

    Provides input fields for coordinates and brightness.

    Attributes:
        sendCoordinates (pyqtSignal): Signal emitted with the coordinates.
        sendBrightness (pyqtSignal): Signal emitted with the brightness.
        x_input, y_input, z_input (QLineEdit): Input fields for coordinates.
        brightess_input (QLineEdit): Input field for brightness.
    """
    sendCoordinates = pyqtSignal(tuple)
    sendBrightness = pyqtSignal(float)
    non_numbers = ['', '-']

    def __init__(self):
        super().__init__()
        self.request_data()

    def request_data(self):
        self.layout.addWidget(QLabel("Point Source"))

        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.z_input = QLineEdit()
        self.brightess_input = QLineEdit()
        self.x_input.setValidator(QDoubleValidator())
        self.y_input.setValidator(QDoubleValidator())
        self.z_input.setValidator(QDoubleValidator())
        self.brightess_input.setValidator(QDoubleValidator())
        self.x_input.setText('0')
        self.y_input.setText('0')
        self.z_input.setText('0')
        self.brightess_input.setText('1.0')

        layout = QVBoxLayout()

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Brightness"))
        hlayout.addWidget(self.brightess_input)
        layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Coordinates"))
        hlayout.addWidget(self.x_input)
        hlayout.addWidget(self.y_input)
        hlayout.addWidget(self.z_input)
        layout.addLayout(hlayout)

        self.layout.addLayout(layout)
        self.setLayout(layout)

        self.x_input.textChanged.connect(self.send_coordinates)
        self.y_input.textChanged.connect(self.send_coordinates)
        self.z_input.textChanged.connect(self.send_coordinates)
        self.brightess_input.textChanged.connect(self.send_brightness)

    def send_coordinates(self):
        if (self.x_input.text() in PointSourceInitializationWidget.non_numbers or
                self.y_input.text() in PointSourceInitializationWidget.non_numbers or
                self.z_input.text() in PointSourceInitializationWidget.non_numbers):
            return
        coordinates = (float(self.x_input.text()), float(self.y_input.text()), float(self.z_input.text()))
        self.sendCoordinates.emit(coordinates)

    def send_brightness(self):
        if self.brightess_input.text() in PointSourceInitializationWidget.non_numbers:
            return
        self.sendBrightness.emit(float(self.brightess_input.text()))
