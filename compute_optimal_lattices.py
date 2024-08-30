import numpy as np
from math import gcd
from config.IlluminationConfigurations import *

def generate_conditions2d(peaks2d):
    funcs2d = []
    for peak in peaks2d:
        Mx, My = peak
        if Mx != 0 or My != 0:
            funcs2d.append(lambda p1, p2, k1=Mx, k2=My: k1 * p1 + k2 * p2)
    return funcs2d

def generate_conditions3d(peaks3d):
    funcs3d = []
    for peak in peaks3d:
        Mx, My, Mz = peak
        if Mx != 0 or My != 0 or Mz != 0:
            funcs3d.append(lambda p1, p2, p3, k1=Mx, k2=My, k3=Mz: k1 * p1 + k2 * p2 + k3 * p3)
    return funcs3d

def generate_tables2d(funcs, max_power):
    tables = np.zeros((len(funcs), max_power, max_power))
    for power1 in range(1, max_power+1):
        for power2 in range(1, max_power+1):
            for n in range(len(funcs)):
                tables[n, power1-1, power2-1] = funcs[n](power1, power2)
    return tables

def generate_table2d(funcs, bases, p1 = 1):
    table2d = np.zeros((len(funcs), len(bases)))
    for i in range(len(funcs)):
        table2d[i] = funcs[i](p1, bases)
    table2d = np.unique(table2d, axis=0)
    return table2d

def generate_table3d(funcs, bases, p1 = 1):
    table3d = np.zeros((len(funcs), len(bases), len(bases)))
    for i in range(len(funcs)):
        for p2 in bases:
            for p3 in bases:
                table3d[i, p2-1, p3-1] = funcs[i](p1, p2, p3)
    table3d = np.unique(table3d, axis=0)
    return table3d

def find_pairs2d(table2d, modulos, power1=1):
    combinations = {}
    for modulo in modulos:
        table_residues = table2d % modulo
        proper_bases = []
        for base in np.arange(len(table2d[0])) + 1:
            if (table_residues[:, base-1] == 0).any():
                continue
            proper_bases.append(base)
        if proper_bases:
            combinations[int(modulo)] = set((int(power1) % int(modulo), int(base) % int(modulo)) for base in proper_bases)
    return combinations

def find_pairs3d(table3d, modulos, p1=1):
    combinations = {}
    for modulo in modulos:
        table_residues = table3d % modulo
        proper_bases = []
        for p2 in np.arange(len(table3d[0])) + 1:
            for p3 in np.arange(len(table3d[0])) + 1:
                # print((table_residues[:, p2-1, : p3 - 1]))
                if (table_residues[:, p2-1, p3 - 1] == 0).any():
                    continue
                proper_bases.append((p2, p3))
        if proper_bases:
            combinations[int(modulo)] = set((int(p1) % int(modulo), int(power[0]) % int(modulo), int(power[1]) % int(modulo)) for power in proper_bases)
    return combinations

def combine_dict(d1, d2):
    return {
        k: tuple(d[k] for d in (d1, d2) if k in d)
        for k in set(d1.keys()) | set(d2.keys())
    }
def find_pairs_extended(tables, modulos):
    combinations = {}
    for i in range(len(tables[1])):
        part = find_pairs2d(tables[:,i,:], modulos, i+1)
        combinations = combine_dict(combinations, part)
    # for modulo in modulos:
    #     table_residues = tables % modulo
    #     proper_bases = []
    #     for base1 in np.arange(len(table[0])) + 1:
    #         for base2 in np.arange(len(table[0])) + 1:
    #
    #             if (table_residues[:, base1-1, base2-2] == 0).any():
    #                 continue
    #             proper_bases.append(np.array((base1, base2), dtype = np.int32))
    #     if proper_bases:
    #         combinations[modulo] = set([tuple(base % modulo) for base in proper_bases])
    return combinations

def exponent_sum2d(matrix, Mx, My):
    exps = np.exp(2 * np.pi * 1j * matrix)
    res = np.sum(exps[0]**Mx * exps[1]**My)
    return round(np.abs(res), 10)

def exponent_sum3d(matrix, Mx, My, Mz):
    exps = np.exp(2 * np.pi * 1j * matrix)
    res = np.sum(exps[0] ** Mx * exps[1] ** My * exps[2] ** Mz)
    return round(np.abs(res), 10)
def check_peaks2d(matrix, peaks):
    for peak in peaks:
        print(peak, exponent_sum2d(matrix, peak[0], peak[1]))

def check_peaks3d(matrix, peaks):
    for peak in peaks:
        print(peak, exponent_sum3d(matrix, peak[0], peak[1], peak[2]))

def get_matrix2d(base, powers):
    ux = np.arange(base) * powers[0] % base
    uy = np.arange(base) * powers[1] % base
    matrix2d = np.array((ux, uy)) / base
    print(matrix2d)
    return matrix2d

def get_matrix3d(base, powers):
    ux = np.arange(base) * powers[0] % base
    uy = np.arange(base) * powers[1] % base
    uz = np.arange(base) * powers[2] % base
    matrix3d = np.array((ux, uy, uz)) / base
    print(matrix3d)
    return matrix3d


if __name__ == "__main__":

    # for k in range(1, 3):
    #     funcs.append(lambda x, d=k: 1 + d * x)
    #     funcs.append(lambda x, d=k: 1 - d * x)
    #     funcs.append(lambda x, d=k: d + x)
    #     funcs.append(lambda x, d=k: d - x)
    #
    # for k in range(1, 3):
    #     funcs.append(lambda x, d=k: d + d * x)
    #     funcs.append(lambda x, d=k: d - d * x)
    #
    # for k in range(1, 5):
    #     funcs.append(lambda x, d=k: d * x)

    # for k in range(-3, 5):
    #     funcs.append(lambda x, d=k: 2 * d - 1 + x)
    #     funcs.append(lambda x, d=k: 2 * d - 1 - x)
    # for k in range(-3, 4):
    #     funcs.append(lambda x, d=k: 2 * d + 2 * x)
    #     funcs.append(lambda x, d=k: 2 * d - 2 * x)
    # for k in range(-2, 4):
    #     funcs.append(lambda x, d=k: 2 * d - 1 + 3 * x)
    #     funcs.append(lambda x, d=k: 2 * d - 1 + 3 * x)
    # for k in range(-2, 3):
    #     funcs.append(lambda x, d=k: 2 * d + 4 * x)
    #     funcs.append(lambda x, d=k: 2 * d - 4 * x)

    # for k in range(-3, 4):
    #     funcs.append(lambda x, d=k: d + x)
    #     funcs.append(lambda x, d=k: d - x)
    # for k in range(-2, 3):
    #     funcs.append(lambda x, d=k: d + 2 * x)
    #     funcs.append(lambda x, d=k: d - 2 * x)
    # for k in range(-1, 2):
    #     funcs.append(lambda x, d=k: d + 3 * x)
    #     funcs.append(lambda x, d=k: d + 3 * x)
    # for k in range(0, 1):
    #     funcs.append(lambda x, d=k: 4 * x)
    #     funcs.append(lambda x, d=k: 4 * x)

    illumination = BFPConfiguration().get_4_circular_oblique_waves_and_circular_normal(np.pi/4, 1)
    # illumination = BFPConfiguration().get_2_oblique_s_waves_and_s_normal(np.pi/4, 1)
    expanded_lattice = illumination.compute_expanded_lattice3d()
    funcs = generate_conditions3d(expanded_lattice)
    table = generate_table3d(funcs, np.arange(1, 50))
    # table = tables[:, 0, :]
    pairs = find_pairs3d(table, np.arange(1, 50, 1))
    # print(table)
    print(pairs)
    bases = sorted(list(pairs.keys()))
    matrix3d = get_matrix3d(bases[0], list(pairs[bases[0]])[0])
    # print(matrix3d)
    # print(len(peaks))
    # matrix = np.array([np.arange(1, 14), (1, 8, 15, 22, 29, 36, 43, 50, 57, 64)])/10
    check_peaks3d(matrix3d, expanded_lattice)
