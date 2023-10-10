import numpy as np

class Pauli:
    I = np.array(((1, 0),
                  (0, 1)))

    X = np.array(((0, 1),
                  (1, 0)))

    Y = np.array(((0, -1j),
                  (1j, 0)))

    Z = np.array(((1, 0),
                  (0, -1)))

class UnitSystem:
    ...

class SystemSI(UnitSystem):
    pass