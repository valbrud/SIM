import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import numpy as np
from fractions import Fraction
import math
if __name__=="__main__":
    for k in np.arange(10):
        w = (k + 1)**2 / (2 * k + 1)
        print("k = {}, w = {}\n------------------".format(k, Fraction(w).limit_denominator()))
        denum = 2 * k + 1
        shift = 2 * k + 2
        for n in np.arange(11):
            delta = shift / (n**2 / denum + 1)
            A = delta**2 * n**2 / (denum**2)
            if A > 0 and A < w:
                a = Fraction(A**0.5).limit_denominator()
                print("n**2 = {}, delta = {}, A = {}".format(n**2, Fraction(delta).limit_denominator(), Fraction(A).limit_denominator()))
                print("a = {}, g = {}, sin(theta) = {}, theta = {:.2f}".format(a, Fraction(delta - 1).limit_denominator(), (1/w)**0.5, np.arcsin((1/w)**0.5)*57.29))
                c = Fraction((1 - 1/w)**0.5).limit_denominator()
                calpha = Fraction((1 - A/w)**0.5).limit_denominator()
                cdiff = Fraction(c - calpha).limit_denominator()
                print("k modes are {}\t{}\t{}".format(c, calpha, cdiff))
                if c != 0 and calpha != 0 and cdiff != 0:
                    zmode1 = 1/c
                    zmode2 = 1/calpha
                    zmode3 = 1/cdiff
                    pnum1 = zmode1.numerator * zmode2.denominator * zmode3.denominator
                    pnum2 = zmode1.denominator * zmode2.numerator * zmode3.denominator
                    pnum3 = zmode1.denominator * zmode2.denominator * zmode3.numerator
                    # print(pnum1, pnum2, pnum3)
                    common_denominator = zmode1.denominator * zmode2.denominator * zmode3.denominator
                    minimal_period = math.lcm(pnum1, pnum2, pnum3)/common_denominator
                    print("Minimal period in r direction is {} and in z is {}\n".format(a.numerator, minimal_period))