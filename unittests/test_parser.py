import unittest
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader

class TestParser(unittest.TestCase):
    def test_configurations_import(self):
        spec = spec_from_loader("example_config.conf",
                                SourceFileLoader("example_config.conf", "./config/example_config.conf"))
        conf = module_from_spec(spec)
        spec.loader.exec_module(conf)
        print(len(conf.sources))