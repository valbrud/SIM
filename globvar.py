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


class SI:
    class Constants:
        dnuCs = 9192631770
        c = 299792458
        h = 6.62607015 * 10 ** (-34)
        e = 1.602176634 * 10 ** (-19)
        k = 1.380649 * 10 ** (-23)
        NAvogadro = 6.02214076 * 10 ** (-23)
        Kcd = 683

    class Length:
        km = 10 ** 3
        m = 1
        mm = 10 ** (-3)
        mcm = 10 ** (-6)
        nm = 10 ** (-9)
        pm = 10 ** (-12)
        fm = 10 ** (-15)

    class Time:
        s = 1
        ms = 10 ** (-3)
        mcs = 10 ** (-6)
        ns = 10 ** (-9)
        ps = 10 ** (-12)
        fs = 10 ** (-15)
        ats = 10 ** (-18)

    class Frequency:
        EHz = 10 ** 18
        PHz = 10 ** 15
        THz = 10 ** 12
        GHz = 10 ** 9
        MHz = 10 ** 6
        kHz = 10 ** 3
        Hz = 1

    class Force:
        GN = 10 ** 9
        MN = 10 ** 6
        kN = 10 ** 3
        N = 1
        mN = 10 ** (-3)
        mcN = 10 ** (-6)
        nN = 10 ** (-9)
        pN = 10 ** (-12)
        fN = 10 ** (-15)

    class Energy:
        GJ = 10 ** 9
        MJ = 10 ** 6
        kJ = 10 ** 3
        J = 1
        mJ = 10 ** (-3)
        mcJ = 10 ** (-6)
        nJ = 10 ** (-9)
        pJ = 10 ** (-12)
        fJ = 10 ** (-15)
        aJ = 10 ** (-18)
        eV = 1.602176634 ** (-19)
