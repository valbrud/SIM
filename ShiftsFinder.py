''' 
shiftsFinder.py

This module contains classes that allow to find optimal diagonal shifts for 2D and 3D
illumination lattices.

Classes:
    ShiftsFinder: Base class for finding shifts in lattices.
    ShiftsFinder2d: Class for finding shifts in 2D lattices.
    ShiftsFinder3d: Class for finding shifts in 3D lattices.
'''

import numpy as np
from math import gcd
from abc import abstractmethod

class ShiftsFinder:

    @staticmethod
    @abstractmethod
    def get_shift_ratios(expanded_lattice): ...

    @staticmethod
    def combine_dict(d1, d2):
        return {
            k: tuple(d[k] for d in (d1, d2) if k in d)
            for k in set(d1.keys()) | set(d2.keys())
        }


class ShiftsFinder2d(ShiftsFinder):

    @staticmethod
    def generate_conditions(peaks2d):
        return [
            lambda p1, p2, k1=Mx, k2=My: k1 * p1 + k2 * p2
            for Mx, My in peaks2d if Mx != 0 or My != 0
        ]

    @staticmethod
    def generate_table(funcs, bases, p1=1):
        table2d = np.zeros((len(funcs), len(bases)))
        for i in range(len(funcs)):
            table2d[i] = funcs[i](p1, bases)
        return np.unique(table2d, axis=0)

    @staticmethod
    def find_pairs(table2d, modulos, power1=1):
        combinations = {}
        for modulo in modulos:
            table_residues = table2d % modulo
            proper_bases = [
                base for base in np.arange(len(table2d[0])) + 1
                if not (table_residues[:, base - 1] == 0).any()
            ]
            if proper_bases:
                combinations[int(modulo)] = {
                    (int(power1) % int(modulo), int(base) % int(modulo))
                    for base in proper_bases
                }
        return combinations

    @staticmethod
    def get_shift_ratios(expanded_lattice, highest_base_number=25):
        if len(expanded_lattice.pop()) != 2:
            raise ValueError("This method is for 2D lattices only.")
        funcs = ShiftsFinder2d.generate_conditions(expanded_lattice)
        bases = np.arange(1, highest_base_number)
        table = ShiftsFinder2d.generate_table(funcs, bases)
        return ShiftsFinder2d.find_pairs(table, np.arange(1, highest_base_number, 1))


class ShiftsFinder3d(ShiftsFinder):

    @staticmethod
    def generate_conditions(peaks3d):
        return [
            lambda p1, p2, p3, k1=Mx, k2=My, k3=Mz: k1 * p1 + k2 * p2 + k3 * p3
            for Mx, My, Mz in peaks3d if Mx != 0 or My != 0 or Mz != 0
        ]

    @staticmethod
    def generate_table(funcs, bases, p1=1):
        table3d = np.zeros((len(funcs), len(bases), len(bases)))
        for i in range(len(funcs)):
            for p2 in bases:
                for p3 in bases:
                    table3d[i, p2 - 1, p3 - 1] = funcs[i](p1, p2, p3)
        return np.unique(table3d, axis=0)

    @staticmethod
    def find_pairs(table3d, modulos, p1=1):
        combinations = {}
        for modulo in modulos:
            table_residues = table3d % modulo
            proper_bases = [
                (p2, p3) for p2 in np.arange(len(table3d[0])) + 1
                for p3 in np.arange(len(table3d[0])) + 1
                if not (table_residues[:, p2 - 1, p3 - 1] == 0).any()
            ]
            if proper_bases:
                combinations[int(modulo)] = {
                    (int(p1) % int(modulo), int(power[0]) % int(modulo), int(power[1]) % int(modulo))
                    for power in proper_bases
                }
        return combinations

    @staticmethod
    def get_shift_ratios(expanded_lattice, highest_base_number=25):
        if len(expanded_lattice.pop()) != 3:
            raise ValueError("This method is for 3D lattices only.")
        funcs = ShiftsFinder3d.generate_conditions(expanded_lattice)
        bases = np.arange(1, highest_base_number)
        table = ShiftsFinder3d.generate_table(funcs, bases)
        return ShiftsFinder3d.find_pairs(table, np.arange(1, highest_base_number, 1))


# Example Usage
if __name__ == "__main__":
    from config.IlluminationConfigurations import *
    illumination = BFPConfiguration().get_4_circular_oblique_waves_and_circular_normal(np.pi / 4, 1)
    expanded_lattice = illumination.compute_expanded_lattice()

    # 2D Shift Finder
    # shift_finder_2d = ShiftsFinder2d(highest_base_number=50)
    # shift_ratios_2d = shift_finder_2d.get_shift_ratios(expanded_lattice)
    # print("2D Shift Ratios:", shift_ratios_2d)

    # 3D Shift Finder
    shift_ratios_3d = ShiftsFinder3d.get_shift_ratios(expanded_lattice)
    print("3D Shift Ratios:", shift_ratios_3d)
