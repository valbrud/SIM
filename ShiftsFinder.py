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

def build_phase_matrix(shift_number, shift_ratios, m_vectors):
    phase_matrix = np.zeros((shift_number, len(m_vectors)), dtype=complex)
    for i in range(shift_number):
        for j, m_vec in enumerate(m_vectors):
            phase = sum((shift_ratios[dim] * m_vec[dim]) for dim in range(len(m_vec)))
            phase_matrix[i, j] = np.exp(2j * np.pi * phase / shift_number)
    return phase_matrix

# Example Usage
if __name__ == "__main__":
    from config.BFPConfigurations import *
    illumination = BFPConfiguration().get_2_oblique_s_waves_and_s_normal(np.pi / 4, 1)
    expanded_lattice = illumination.compute_expanded_lattice(ignore_projected_dimensions=False)

    # 2D Shift Finder
    # shift_finder_2d = ShiftsFinder2d(highest_base_number=50)
    # shift_ratios_2d = shift_finder_2d.get_shift_ratios(expanded_lattice)
    # print("2D Shift Ratios:", shift_ratios_2d)

    # 3D Shift Finder
    shift_ratios_3d = ShiftsFinder3d.get_shift_ratios(expanded_lattice)
    print("3D Shift Ratios:", shift_ratios_3d)

    ratios_selected = shift_ratios_3d[7]
    print("Selected Ratios:", ratios_selected)


    pi = np.pi
    i = 1j

    M1 = np.array([
        [np.exp(0*i), np.exp(0*i), np.exp(0*i), 1, np.exp(0*i), np.exp(0*i), np.exp(0*i)],
        [np.exp(-4*pi*i/5), np.exp(-2*pi*i/5), np.exp(-2*pi*i/5), 1, np.exp(2*pi*i/5), np.exp(2*pi*i/5), np.exp(4*pi*i/5)],
        [np.exp(-8*pi*i/5), np.exp(-4*pi*i/5), np.exp(-4*pi*i/5), 1, np.exp(4*pi*i/5), np.exp(4*pi*i/5), np.exp(8*pi*i/5)],
        [np.exp(-12*pi*i/5), np.exp(-6*pi*i/5), np.exp(-6*pi*i/5), 1, np.exp(6*pi*i/5), np.exp(6*pi*i/5), np.exp(12*pi*i/5)],
        [np.exp(-16*pi*i/5), np.exp(-8*pi*i/5), np.exp(-8*pi*i/5), 1, np.exp(8*pi*i/5), np.exp(8*pi*i/5), np.exp(16*pi*i/5)],
        [np.exp(0*i), np.exp(0*i - 2*pi*i/4), np.exp(0*i + pi*i/2), 1, np.exp(0*i - pi*i/2), np.exp(0*i), np.exp(0*i)],
        [np.exp(-8*pi*i/5), np.exp(-4*pi*i/5 - pi*i/2), np.exp(-4*pi*i/5 + pi*i/2), 1, np.exp(4*pi*i/5 + pi*i/2), np.exp(4*pi*i/5), np.exp(8*pi*i/5)]
    ], dtype=complex)

    cond1 = np.linalg.cond(M1)

    # ---------- Matrix 2 ----------
    phases3d = np.array([-4*np.pi/7, -6*np.pi/7, 2*np.pi/7, 0, -2*np.pi/7, 6*np.pi/7, 4*np.pi/7])
    M2 = np.array([[np.exp(1j * k * phases3d[j]) for j in range(7)] for k in range(7)], dtype=complex)
    
    cond2 = np.linalg.cond(M2)

    phases2d = np.array([-4*np.pi/5, -2*np.pi/5, 0, 2*np.pi/5, 4*np.pi/5])
    M3 = np.array([[np.exp(1j * k * phases2d[j]) for j in range(5)] for k in range(5)], dtype=complex)
    cond3 = np.linalg.cond(M3)
    print("Condition number of Matrix 1:", cond1)
    print("Condition number of Matrix 2:", cond2)
    print("Condition number of Matrix 3:", cond3)