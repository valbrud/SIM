"""
This module contains a class for parsing command line arguments for the initialization of GUI
"""

import argparse
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import GUI
import Box
import sys
import os


class ConfigParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("input_file", type=str, help="Path to the input config file")
        self.parser.add_argument("-i", "--compute_intensity", action="store_true", help="Compute intensity")
        self.parser.add_argument("--gui", action="store_true", help="Call a gui initializer")
        self.parser.add_argument("-p", "--plot", action="store_true", help="Plot data")
        self.args = self.parser.parse_args()

    @staticmethod
    def read_configuration(file):
        spec = spec_from_loader("config_file",
                                SourceFileLoader("config_file", os.getcwd() + "/config/" + file))
        conf = module_from_spec(spec)
        spec.loader.exec_module(conf)
        return conf


if __name__ == "__main__":
    parser = ConfigParser()
    conf = parser.read_configuration(parser.args.input_file)
    box = Box.BoxSIM(conf.illumination, conf.box_size, conf.point_number, parser.args.input_file)
    if parser.args.gui:
        app = GUI.QApplication(sys.argv)
        window = GUI.MainWindow(box)
        sys.exit(app.exec_())
