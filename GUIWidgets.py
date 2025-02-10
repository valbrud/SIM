"""
GUIWidgets.py

This module contains utility widgets for the GUI components.

This module and related ones is currently a demo-version of the user-interface, and will
possibly be sufficiently modified or replaced in the future. For this reason, no in-depth
documentation is provided.
"""

import Sources
import numpy as np
from GUIInitializationWidgets import *


class SourceWidget(QWidget):
    identifier = 0
    isDeleted = pyqtSignal(int)
    @abstractmethod
    def __init__(self, source):
        super().__init__()
        self.identifier = SourceWidget.identifier
        SourceWidget.identifier += 1

    @abstractmethod
    def init_ui(self, source): ...

    @abstractmethod
    def contextMenuEvent(self, event): ...

    def remove_widget(self):
        self.isDeleted.emit(self.identifier)
        self.close()
    @abstractmethod
    def change_widget(self): ...


class IntensityHarmonic3DWidget(SourceWidget):
    isSet = pyqtSignal(bool)
    def __init__(self, ipw):
        super().__init__(ipw)
        self.init_ui(ipw)
        # self.PlaneWave = Sources.PlaneWave(self.coordinates)

    def init_ui(self, ipw):

        if not ipw:
            self.initialization_window = QWidget()
            self.iwlayout = QVBoxLayout()
            self.data_widget = IntensityHarmonic3DInitializationWidget()
            self.iwlayout.addWidget(self.data_widget)
            self.initialization_window.setLayout(self.iwlayout)
            self.initialization_window.show()

            self.data_widget.sendInfo.connect(self.on_receive_info)

        else:
            self.intensity_plane_wave = ipw

            layout = QVBoxLayout()
            layout.addWidget(QLabel('IntensityHarmonic3D'))
            layout.addWidget(QLabel('A = ' + str(ipw.amplitude)))
            layout.addWidget(QLabel('Phase = ' + str(ipw.phase)))
            layout.addWidget(QLabel('Wavevector = ' + str(ipw.wavevector)))

            self.setLayout(layout)
            # self.isSet.emit(True)

    def on_receive_info(self, info):
        A, phase = [complex(value) for value in info[:2]]
        wavevector = [float(value) for value in info[2:]]
        print(A, phase, wavevector)
        self.intensity_plane_wave = Sources.IntensityHarmonic3D(A, phase, wavevector)

        self.initialization_window.close()
        self.isSet.emit(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('IntensityHarmonic3D'))
        layout.addWidget(QLabel('A = ' + str(A)))
        layout.addWidget(QLabel('Phase = ' + str(phase)))
        layout.addWidget(QLabel('Wavevector = ' + str(wavevector)))
        self.setLayout(layout)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        remove_action = QAction("Remove Widget", self)
        change_action = QAction("Change Widget", self)

        remove_action.triggered.connect(self.remove_widget)
        change_action.triggered.connect(self.change_widget)

        context_menu.addAction(remove_action)
        context_menu.addAction(change_action)

        context_menu.exec_(event.globalPos())


    def change_widget(self):
        # Implement the logic to change the widget
        self.label.setText("Widget Changed")


class PlaneWaveWidget(SourceWidget):
    isSet = pyqtSignal(bool)
    isDeleted = pyqtSignal(int)

    def __init__(self, pw=None):
        super().__init__(pw)
        self.init_ui(pw)

    def init_ui(self, pw):
        if not pw:
            self.initialization_window = QWidget()
            self.iwlayout = QVBoxLayout()
            self.data_widget = PlaneWaveInitializationWidget()
            self.iwlayout.addWidget(self.data_widget)
            self.initialization_window.setLayout(self.iwlayout)
            self.initialization_window.show()
            self.data_widget.sendInfo.connect(self.on_receive_info)


        else:
            self.plane_wave = pw

            layout = QVBoxLayout()
            layout.addWidget(QLabel('PlaneWave'))
            layout.addWidget(QLabel('Ep = ' + str(pw.field_vectors[0]) + "Es = " + str(pw.field_vectors[1])))
            layout.addWidget(QLabel('Phase p= ' + str(pw.phases[0]) + "Phase s = " + str(pw.phases[1])))
            layout.addWidget(QLabel('Wavevector = ' + str(pw.wavevector)))

            self.setLayout(layout)
            # self.isSet.emit(True)

    def on_receive_info(self, info):
        Ep, Es, phasep, phases = [complex(value) for value in info[:4]]
        wavevector = [float(value) for value in info[4:]]
        print(Ep, Es, phasep, phases, wavevector)
        self.plane_wave = Sources.PlaneWave(Ep, Es, phasep, phases, wavevector)

        self.initialization_window.close()
        self.isSet.emit(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('PlaneWave'))
        layout.addWidget(QLabel('Ep = ' + str(Ep) + ', Es = ' + str(Es)))
        layout.addWidget(QLabel('Phase p = ' + str(phasep) + ', Phase s = ' + str(phases)))
        layout.addWidget(QLabel('Wavevector = ' + str(wavevector)))

        self.setLayout(layout)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        remove_action = QAction("Remove Widget", self)
        change_action = QAction("Change Widget", self)

        remove_action.triggered.connect(self.remove_widget)
        change_action.triggered.connect(self.change_widget)

        context_menu.addAction(remove_action)
        context_menu.addAction(change_action)

        context_menu.exec_(event.globalPos())


    def change_widget(self):
        # Implement the logic to change the widget
        self.label.setText("Widget Changed")


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


class PointSourceWidget(SourceWidget):
    isSet = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.coordinates = np.zeros(3)
        self.brightnes = 1.0
        self.init_ui()

    def init_ui(self):
        self.initialization_window = QWidget()
        self.iwlayout = QVBoxLayout()
        self.data_widget = PointSourceInitializationWidget()
        self.iwlayout.addWidget(self.data_widget)
        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(self.on_click_ok)
        self.iwlayout.addWidget(ok_button)
        self.initialization_window.setLayout(self.iwlayout)
        self.initialization_window.show()
        self.data_widget.sendCoordinates.connect(self.on_receive_coordinates)
        self.data_widget.sendBrightness.connect(self.on_receive_brightness)

    def on_receive_brightness(self, brightness):
        self.brightnes = float(brightness)

    def on_receive_coordinates(self, coordinates):
        self.coordinates = np.array(coordinates)

    def on_click_ok(self):
        self.initialization_window.close()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('PointSource'))
        layout.addWidget(QLabel(str(self.coordinates)))
        self.setLayout(layout)
        self.point_source = Sources.PointSource(self.coordinates, self.brightnes)
        self.isSet.emit(True)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        remove_action = QAction("Remove Widget", self)
        change_action = QAction("Change Widget", self)

        remove_action.triggered.connect(self.remove_widget)
        change_action.triggered.connect(self.change_widget)

        context_menu.addAction(remove_action)
        context_menu.addAction(change_action)

        context_menu.exec_(event.globalPos())

    def change_widget(self):
        # Implement the logic to change the widget
        self.label.setText("Widget Changed")
