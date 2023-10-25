import ImageProcessing
from numba import jit
import numpy as np
import multiprocessing as mp
import time


def SSNR_inner_cycle(i, q_axes, noise_estimator):
    q_axes_yz = (q_axes[0][i], q_axes[1], q_axes[2])
    SSNRpart = np.abs(noise_estimator.SSNR(q_axes_yz))
    return i, SSNRpart


def get_SSNR_multi(noise_estimator, q_axes):
    pool = mp.Pool(mp.cpu_count())
    SSNR = np.zeros((len(q_axes[0]), len(q_axes[1]), len(q_axes[2])))
    SSNR_enum = [pool.apply_async(SSNR_inner_cycle, args=(i, q_axes, noise_estimator)) for i in range(len(q_axes[0]))]
    pool.close()
    pool.join()

    for i in range(len(q_axes[0])):
        j, SSNRj = SSNR_enum[i].get()
        SSNR[j] = SSNRj

    return SSNR