"""
Widgets.py

This module contains GUI widget classes for configuring and displaying different types of sources
in the application.

The widgets allow users to interactively set parameters for sources such as intensity harmonics,
plane waves, and point sources, and display their configurations.

Classes:
    SourceWidget: Abstract base class for source widgets.
    IntensityHarmonic3DWidget: Widget for IntensityHarmonic3D sources.
    PlaneWaveWidget: Widget for PlaneWave sources.
    PointSourceWidget: Widget for PointSource sources.
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import Sources
import numpy as np
from GUI.Box.InitializationWidgets import *


class SourceWidget(QWidget):
    """
    Abstract base class for source configuration widgets.

    This class provides a common interface for widgets that allow users to configure
    different types of sources in the SIM simulation.

    Attributes:
        identifier (int): Unique identifier for the widget instance.
    """
    identifier = 0
    isDeleted = pyqtSignal(int)

    @abstractmethod
    def __init__(self, source=None):
        """
        Initialize the source widget.

        Args:
            source: The source object to configure or display.
        """
        super().__init__()
        self.identifier = SourceWidget.identifier
        SourceWidget.identifier += 1

    @abstractmethod
    def init_ui(self, source):
        """
        Initialize the user interface for the widget.

        Args:
            source: The source object.
        """
        ...

    @abstractmethod
    def contextMenuEvent(self, event):
        """
        Handle right-click context menu events.

        Args:
            event (QContextMenuEvent): The context menu event.
        """
        ...

    def remove_widget(self):
        """
        Remove the widget and emit deletion signal.
        """
        self.isDeleted.emit(self.identifier)
        self.close()

    @abstractmethod
    def change_widget(self):
        """
        Change the widget's configuration.
        """
        ...


class IntensityHarmonic3DWidget(SourceWidget):
    """
    Widget for configuring and displaying IntensityHarmonic3D sources.

    This widget provides an interface to input parameters for an intensity harmonic 3D source,
    including amplitude, phase, and wavevector components. It can display a pre-configured source
    or allow user input for a new one.

    Attributes:
        isSet (pyqtSignal): Signal emitted when the source is configured.
        intensity_plane_wave (IntensityHarmonic3D): The configured source object.
        data_widget (IntensityHarmonic3DInitializationWidget): The input widget for parameters.
    """
    isSet = pyqtSignal(bool)

    def __init__(self, ipw=None):
        super().__init__(ipw)
        self.init_ui(ipw)

    def init_ui(self, ipw):
        if not ipw:
            layout = QVBoxLayout()
            self.data_widget = IntensityHarmonic3DInitializationWidget()
            layout.addWidget(self.data_widget)
            self.setLayout(layout)
            self.data_widget.sendInfo.connect(self.on_receive_info)
        else:
            self.intensity_plane_wave = ipw
            layout = QVBoxLayout()
            layout.addWidget(QLabel('IntensityHarmonic3D'))
            layout.addWidget(QLabel('A = ' + str(ipw.amplitude)))
            layout.addWidget(QLabel('Phase = ' + str(ipw.phase.real)))
            layout.addWidget(QLabel('Wavevector = ' + str(ipw.wavevector)))
            self.setLayout(layout)

    def on_receive_info(self, info):
        """
        Process the input information from the initialization widget.

        Parses the input, creates the source object, updates the display,
        and emits the configuration signal.

        Args:
            info (list): List of input values [amplitude, phase, kx, ky, kz].
        """
        try:
            A, phase = [complex(value) for value in info[:2]]
            wavevector = [float(value) for value in info[2:]]
        except ValueError as e:
            print(f"Invalid input for intensity wave: {e}. Using defaults.")
            A, phase = 0, 0
            wavevector = [0, 0, 0]
        print(A, phase, wavevector)
        self.intensity_plane_wave = Sources.IntensityHarmonic3D(A, phase.real, wavevector)

        self.data_widget.close()
        self.isSet.emit(True)

        # Clear existing layout and add new widgets
        current_layout = self.layout()
        if current_layout:
            while current_layout.count():
                item = current_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.hide()
                    widget.setParent(None)
            current_layout.addWidget(QLabel('IntensityHarmonic3D'))
            current_layout.addWidget(QLabel('A = ' + str(A)))
            current_layout.addWidget(QLabel('Phase = ' + str(phase.real)))
            current_layout.addWidget(QLabel('Wavevector = ' + str(wavevector)))
        self.update()
        self.show()

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
        # Get current values
        current_values = None
        if hasattr(self, 'intensity_plane_wave') and self.intensity_plane_wave:
            current_values = {
                'A': str(self.intensity_plane_wave.amplitude),
                'phase': str(self.intensity_plane_wave.phase.real).strip('()'),
                'kx': str(self.intensity_plane_wave.wavevector[0]),
                'ky': str(self.intensity_plane_wave.wavevector[1]),
                'kz': str(self.intensity_plane_wave.wavevector[2])
            }
        # Emit deleted for old widget
        self.isDeleted.emit(self.identifier)

        
        # Get the position in the parent layout
        parent_layout = self.parent().layout()
        index = parent_layout.indexOf(self)
        # Remove the old widget
        parent_layout.removeWidget(self)
        # Create new widget with form
        new_widget = IntensityHarmonic3DWidget(ipw=None)
        # new_widget.identifier = self.identifier
        # Set current values if available
        # info_fields = [current_values['A'], current_values['phase'], current_values['kx'], current_values['ky'], current_values['kz']]
        # info = [field.text().strip().strip('()').replace(' ', '') for field in info_fields]
        # for value in range(len(info)):
        #     if info[value] == '':
        #         info[value] = '0'
        if current_values:
            new_widget.data_widget.A.setText(current_values['A'])
            new_widget.data_widget.phase.setText(current_values['phase'])
            new_widget.data_widget.kx.setText(current_values['kx'])
            new_widget.data_widget.ky.setText(current_values['ky'])
            new_widget.data_widget.kz.setText(current_values['kz'])
        # Insert at the same position
        parent_layout.insertWidget(index, new_widget)
        # Store parent to avoid accessing self.parent() after self is deleted
        main_window = self.parent().parent().parent().parent().parent()
        # Keep a reference to prevent garbage collection
        if not hasattr(main_window, '_widgets'):
            main_window._widgets = []
        main_window._widgets.append(new_widget)
        new_widget.isSet.connect(lambda initialized: main_window.add_to_box(initialized, new_widget.intensity_plane_wave))
        new_widget.isDeleted.connect(lambda identifier: main_window.remove_source(identifier))
        # new_widget.on_receive_info(current_values)
        self.deleteLater()


class PlaneWaveWidget(SourceWidget):
    """
    Widget for configuring and displaying PlaneWave sources.

    This widget allows users to set parameters for a plane wave source,
    including field vectors, phases, and wavevector. It displays the configured
    source or provides input fields for a new one.

    Attributes:
        isSet (pyqtSignal): Signal emitted when the source is configured.
        isDeleted (pyqtSignal): Signal emitted when the widget is deleted.
        plane_wave (PlaneWave): The configured source object.
        data_widget (PlaneWaveInitializationWidget): The input widget for parameters.
    """
    isSet = pyqtSignal(bool)
    isDeleted = pyqtSignal(int)

    def __init__(self, pw=None):
        super().__init__(pw)
        self.init_ui(pw)

    def init_ui(self, pw):
        if not pw:
            layout = QVBoxLayout()
            self.data_widget = PlaneWaveInitializationWidget()
            layout.addWidget(self.data_widget)
            self.setLayout(layout)
            self.data_widget.sendInfo.connect(self.on_receive_info)
        else:
            self.plane_wave = pw

            layout = QVBoxLayout()
            layout.addWidget(QLabel('PlaneWave'))
            layout.addWidget(QLabel('Ep = ' + str(np.linalg.norm(pw.field_vectors[0])) + ", Es = " + str(np.linalg.norm(pw.field_vectors[1]))))
            phase_p = round((pw.phases[0] + np.max(np.angle(pw.field_vectors[0] + 10**-4))), 2)
            phase_s = round((pw.phases[1] + np.max(np.angle(pw.field_vectors[1] + 10**-4))), 2)
            layout.addWidget(QLabel(f'Phase p = {phase_p}, Phase s = {phase_s}'))
            layout.addWidget(QLabel('Wavevector = ' + str(pw.wavevector)))

            self.setLayout(layout)
            # self.isSet.emit(True)

    def on_receive_info(self, info):
        """
        Process the input information from the initialization widget.

        Parses the input, creates the source object with real phases, updates the display,
        and emits the configuration signal.

        Args:
            info (list): List of input values [Ep, Es, phasep, phases, kx, ky, kz].
        """
        try:
            Ep, Es, phasep, phases = [complex(value) for value in info[:4]]
            wavevector = [float(value) for value in info[4:]]
        except ValueError as e:
            print(f"Invalid input for plane wave: {e}. Using defaults.")
            Ep, Es, phasep, phases = 0, 0, 0, 0
            wavevector = [0, 0, 0]
        print(Ep, Es, phasep, phases, wavevector)
        self.plane_wave = Sources.PlaneWave(Ep, Es, phasep.real, phases.real, wavevector)

        self.data_widget.close()
        self.isSet.emit(True)

        # Clear existing layout and add new widgets
        current_layout = self.layout()
        if current_layout:
            while current_layout.count():
                item = current_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.hide()
                    widget.setParent(None)
            current_layout.addWidget(QLabel('PlaneWave'))
            current_layout.addWidget(QLabel('Ep = ' + str(abs(Ep)) + ', Es = ' + str(abs(Es))))
            current_layout.addWidget(QLabel('Phase p = ' + str(phasep.real) + ', Phase s = ' + str(phases.real)))
            current_layout.addWidget(QLabel('Wavevector = ' + str(wavevector)))
        self.update()

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
        # Get current values
        current_values = None
        if hasattr(self, 'plane_wave') and self.plane_wave:
            current_values = {
                'Ep': str(np.linalg.norm(self.plane_wave.field_vectors[0])),
                'Es': str(np.linalg.norm(self.plane_wave.field_vectors[1])),
                'phasep': str(self.plane_wave.phases[0] + np.max(np.angle(self.plane_wave.field_vectors[0] + 10**-4))).strip('()'),
                'phases': str(self.plane_wave.phases[1] + np.max(np.angle(self.plane_wave.field_vectors[1] + 10**-4))).strip('()'),
                'kx': str(self.plane_wave.wavevector[0]),
                'ky': str(self.plane_wave.wavevector[1]),
                'kz': str(self.plane_wave.wavevector[2])
            }
        # Emit deleted for old widget
        self.isDeleted.emit(self.identifier)
        # Get the position in the parent layout
        parent_layout = self.parent().layout()
        index = parent_layout.indexOf(self)
        # Remove the old widget
        parent_layout.removeWidget(self)
        # Create new widget with form
        new_widget = PlaneWaveWidget(pw=None)
        new_widget.identifier = self.identifier
        # Set current values if available
        if current_values:
            new_widget.data_widget.Ep.setText(current_values['Ep'])
            new_widget.data_widget.Es.setText(current_values['Es'])
            new_widget.data_widget.phasep.setText(current_values['phasep'])
            new_widget.data_widget.phases.setText(current_values['phases'])
            new_widget.data_widget.kx.setText(current_values['kx'])
            new_widget.data_widget.ky.setText(current_values['ky'])
            new_widget.data_widget.kz.setText(current_values['kz'])
        # Insert at the same position
        parent_layout.insertWidget(index, new_widget)
        # Delete old widget
        main_window = self.parent().parent().parent().parent().parent()
        # Keep a reference to prevent garbage collection
        if not hasattr(main_window, '_widgets'):
            main_window._widgets = []
        main_window._widgets.append(new_widget)
        new_widget.isSet.connect(lambda initialized: main_window.add_to_box(initialized, new_widget.plane_wave))
        new_widget.isDeleted.connect(lambda identifier: main_window.remove_source(identifier))
        # new_widget.on_receive_info(current_values)
        self.deleteLater()

class PointSourceWidget(SourceWidget):
    """
    Widget for configuring and displaying PointSource sources.

    This widget allows users to set parameters for a point source,
    including coordinates and brightness. It displays the configured
    source or provides input fields for a new one.

    Attributes:
        isSet (pyqtSignal): Signal emitted when the source is configured.
        coordinates (numpy.ndarray): The 3D coordinates of the point source.
        brightnes (float): The brightness of the point source.
        point_source (PointSource): The configured source object.
        data_widget (PointSourceInitializationWidget): The input widget for parameters.
    """
    isSet = pyqtSignal(bool)

    def __init__(self):
        super().__init__(None)
        self.coordinates = np.zeros(3)
        self.brightnes = 1.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.data_widget = PointSourceInitializationWidget()
        layout.addWidget(self.data_widget)
        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(self.on_click_ok)
        layout.addWidget(ok_button)
        self.setLayout(layout)
        self.data_widget.sendCoordinates.connect(self.on_receive_coordinates)
        self.data_widget.sendBrightness.connect(self.on_receive_brightness)

    def on_receive_brightness(self, brightness):
        """
        Update the brightness value.

        Args:
            brightness (float): The new brightness value.
        """
        self.brightnes = float(brightness)

    def on_receive_coordinates(self, coordinates):
        """
        Update the coordinates.

        Args:
            coordinates (tuple): The new coordinates.
        """
        self.coordinates = np.array(coordinates)

    def on_click_ok(self):
        """
        Handle the OK button click.

        Creates the source object, updates the display, and emits the configuration signal.
        """
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
        # Get current values
        current_values = None
        if hasattr(self, 'point_source') and self.point_source:
            current_values = {
                'x': str(self.coordinates[0]),
                'y': str(self.coordinates[1]),
                'z': str(self.coordinates[2]),
                'brightness': str(self.brightnes)
            }
        # Emit deleted for old widget
        self.isDeleted.emit(self.identifier)
        # Get the position in the parent layout
        parent_layout = self.parent().layout()
        index = parent_layout.indexOf(self)
        # Remove the old widget
        parent_layout.removeWidget(self)
        # Create new widget with form
        new_widget = PointSourceWidget()
        new_widget.identifier = self.identifier
        # Set current values if available
        if current_values:
            new_widget.data_widget.x_input.setText(current_values['x'])
            new_widget.data_widget.y_input.setText(current_values['y'])
            new_widget.data_widget.z_input.setText(current_values['z'])
            new_widget.data_widget.brightess_input.setText(current_values['brightness'])
        # Insert at the same position
        parent_layout.insertWidget(index, new_widget)
        # Delete old widget
        self.deleteLater()
