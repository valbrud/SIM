import numpy as np
import ImageProcessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import unittest
import time, timeit
import scipy as sp
from matplotlib.widgets import Slider
import tqdm
import Box
import multiprocessing as mp
import wrappers_multiprocessing
import cupyx.profiler as cpf
import cupy as cp
import numba as nb

class TestMutliProcessing(unittest.TestCase):
    def test_number_of_cores(self):
        print(mp.cpu_count())

class TestBasicCudaOperations(unittest.TestCase):
    def test_convolution_cuda(self):
        from scipy.signal import convolve2d as convolved2d_cpu
        from cupyx.scipy.signal import convolve2d as convolved2d_gpu
        deltas = np.zeros((2048, 2048))
        deltas[8::16, 8::16] = 1
        x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
        dst = np.sqrt(x * x + y * y)
        sigma = 1
        muu = 0.000
        gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))
        begin_cpu = time.time()
        image_cpu = convolved2d_cpu(deltas, gauss)
        end_cpu = time.time()
        print("cpu time is ", end_cpu - begin_cpu)


        deltas_gpu = cp.asarray(deltas)
        gauss_gpu = cp.asarray(gauss)
        begin_gpu = time.time()
        image_gpu = convolved2d_gpu(deltas_gpu, gauss_gpu)
        end_gpu = time.time()
        print("gpu time is ", end_gpu - begin_gpu)

        print(np.allclose(image_gpu, image_cpu))

    def test_vectrization_numba(self):
        numbers = np.arange(0, 1000_000, dtype=np.int32)
        a = np.zeros(1000_000)
        def check_prime(num):
            for i in range(2, int(num**0.5) + 1):
                if (num % i) == 0:
                    return 0
            return num

        begin_cpu = time.time()
        for number in numbers:
            a[number] = check_prime(number)
        print(a[a > 0][-1])
        end_cpu = time.time()
        print("cpu time is ", end_cpu - begin_cpu)

        @nb.vectorize(['int32(int32)'], target = 'cuda')
        def check_prime_gpu(num):
            for i in range(2, int(num**0.5) + 1):
                if (num % i) == 0:
                    return 0
            return num

        begin_gpu = time.time()
        array = check_prime_gpu(numbers)
        print(array[array > 0][-1])
        end_gpu = time.time()
        print("gpu time is ", end_gpu - begin_gpu)

