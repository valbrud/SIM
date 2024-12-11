"""
GUIInitializationWidgets.py

This module contains classes and functions for initializing GUI widgets.

This module and related ones is currently a demo-version of the user-interface, and will
possibly be sufficiently modified or replaced in the future. For this reason, no in-depth
documentation is provided.
"""

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
        info = [field.text() for field in info_fields]
        for value in range(len(info)):
            if info[value] == '':
                info[value] = '0'
        self.sendInfo.emit(info)


class IntensityPlaneWaveInitializationWidget(InitializationWidget):
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
        info = [field.text() for field in info_fields]
        for value in range(len(info)):
            if info[value] == '':
                info[value] = '0'
        self.sendInfo.emit(info)
