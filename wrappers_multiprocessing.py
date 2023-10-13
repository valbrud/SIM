import ImageProcessing
import numpy as np
import time
def SSNR_multiprocesssing(i, f, noise_estimator):
    SSNR = np.zeros((len(f[1]), len(f[2])))
    for j in range(len(f[1])):
        for k in range(len(f[2])):
            ft = np.array((f[0][i], f[1][j], f[2][k]))
            q = ft * 2 * np.pi
            SSNR[j, k] = np.abs(noise_estimator.SSNR(q, (i, j, k)))
    return (i, SSNR)