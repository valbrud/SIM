import numpy as np

class GameOfShifts:
    def __init__(self, N1, N2, gridDimension=2):
        self.pointNumber = pointNumber
        self.gridDimension = gridDimension
        self.solutionMatrix = np.zeros((pointNumber, gridDimension))
        self.solutionMatrix[:, 0] = np.arange(1, pointNumber+1)

    def solve_brute_force(self):
        ...
