"""
GUIInitializationWidgets.py

This module contains classes and functions for initializing GUI widgets.

This module and related ones is currently a demo-version of the user-interface, and will
possibly be sufficiently modified or replaced in the future. For this reason, no in-depth
documentation is provided.
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
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

    @abstractmethod
    def request_data(self): ...

    @abstractmethod
    def on_click_ok(self): ...


class PlaneWaveInitializationWidget(InitializationWidget):
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
