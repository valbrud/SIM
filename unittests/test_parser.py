import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader
import sys

class TestParser(unittest.TestCase):
    def test_configurations_import(self):
        spec = spec_from_loader("example_config.conf",
                                SourceFileLoader("example_config.conf", "./config/example_config.conf"))
        conf = module_from_spec(spec)
        spec.loader.exec_module(conf)
        print(len(conf.sources))