import numpy as np
from math import gcd
from config.IlluminationConfigurations import *
def generate_table(funcs, bases):
    table = np.zeros((len(funcs), len(bases)))
    for i in range(len(funcs)):
        table[i] = funcs[i](bases)
    table = np.unique(table, axis = 0)
    return table

def find_pairs(table, modulos, min_base):
    combinations = {}
    for modulo in modulos:
        table_residues = table % modulo
        proper_bases = []
        for base in np.arange(len(table[0])) + 1:
            if modulo // gcd(base, modulo) < min_base:
                continue
            if (table_residues[:, base-1] == 0).any():
                continue
            proper_bases.append(base)
        if proper_bases:
            combinations[modulo] = np.array(proper_bases)
    return combinations

def exponent_sum(matrix, Mx, My):
    exps = np.exp(2 * np.pi * 1j * matrix)
    res = np.sum(exps[0]**Mx * exps[1]**My)
    return round(np.abs(res), 10)

def check_peaks(matrix, peaks):
    for peak in peaks:
        print(peak, exponent_sum(matrix, peak[0], peak[1]))

def get_matrix(base, power):
    ux = np.arange(base)
    uy = ux * power % base
    matrix = np.array((ux, uy)) / base
    print(matrix)
    return matrix

if __name__ == "__main__":
    funcs = []


    for k in range(1, 3):
        funcs.append(lambda x, d=k: 1 + d * x)
        funcs.append(lambda x, d=k: 1 - d * x)
        funcs.append(lambda x, d=k: d + x)
        funcs.append(lambda x, d=k: d - x)

    for k in range(1, 3):
        funcs.append(lambda x, d=k: d + d * x)
        funcs.append(lambda x, d=k: d - d * x)

    for k in range(1, 5):
        funcs.append(lambda x, d=k: d * x)

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

    illumination = BFPConfiguration().get_4_oblique_s_waves_and_circular_normal(np.pi/4, 1)
    illumination.compute_expanded_lattice()
    peaks = illumination.expanded_lattice

    table = generate_table(funcs, np.arange(1, 20))
    pairs = find_pairs(table, np.arange(1, 20, 1), 10)
    print(table)
    print(pairs)
    bases = sorted(list(pairs.keys()))
    matrix = get_matrix(bases[0], pairs[bases[0]][0])
    print(matrix)
    # print(len(peaks))
    # matrix = np.array([np.arange(1, 14), (1, 8, 15, 22, 29, 36, 43, 50, 57, 64)])/10
    check_peaks(matrix, peaks)
