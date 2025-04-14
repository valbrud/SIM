import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import numpy as np
import OpticalSystems 
import unittest
from config.SIM_N500_NA15_2D import *

from config.IlluminationConfigurations import *
import wrappers
from Sources import PlaneWave, IntensityHarmonic3D, IntensityHarmonic2D
import scipy
import matplotlib.pyplot as plt
from Apodization import AutoconvolutionApodizationSIM
# N = 511
class TestAutoconvolutionSIM2D(unittest.TestCase):
    def setUp(self):
        self.optical_system = OpticalSystems.System4f2D(alpha=alpha, refractive_index=nobject)
        self.optical_system.compute_psf_and_otf(((psf_size[0], psf_size[1]), N), save_pupil_function=True)
    
    def test_extended_space_conventional(self):
        conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)    
        _, effective_otfs = conventional.get_effective_otfs(self.optical_system.otf, self.optical_system.psf)  
        otf_extended_space = np.zeros((N, N, 3), dtype=np.float64)
        otf_extended_space[:, :, 0] = effective_otfs[(0, 0)] 
        otf_extended_space[:, :, 1] = effective_otfs[(0, 1)]
        otf_extended_space[:, :, 2] = effective_otfs[(0, 2)]
        ideal_kernel = np.where(otf_extended_space > 10**-12, 1, 0)

    def test_autoconvolution(self):
        k1 = 2 * np.pi * np.sin(theta)
        electric_ft = IlluminationPlaneWaves2D(
            {(1, 0): IntensityHarmonic2D(wavevector=np.array([k1, 0]), amplitude=1),
             (-1, 0): IntensityHarmonic2D(wavevector=np.array([-k1, 0]), amplitude=1)},
              dimensions=(1, 1), Mr=1, spatial_shifts=np.array(((0., 0.),))
            )
        pupil_function = np.zeros((N, N), dtype=np.float64)
        Fx, Fy = np.meshgrid(fx, fy)
        pupil_function[(Fx**2 + Fy**2) * 2 * np.pi < k1**2] = 1
        ctf = wrappers.wrapped_ifftn(pupil_function)
        plt.imshow(pupil_function, cmap='gray')
        plt.title('Pupil Function')
        plt.show()
        _, effective_otfs = electric_ft.compute_effective_kernels(ctf, self.optical_system.psf_coordinates)  
        sim_ctf = np.zeros((N, N), dtype=np.complex128)
        for ctf in effective_otfs.values():
            sim_ctf += ctf  
        plt.imshow(sim_ctf.real, cmap='gray')
        plt.title('Simulated CTF')
        plt.show()
        ideal_otf = scipy.signal.convolve2d(sim_ctf, sim_ctf, mode='same').real
        plt.imshow(ideal_otf, cmap='gray')
        plt.title('Ideal OTF')
        plt.show()
        ideal_psf = np.real(wrappers.wrapped_ifftn(ideal_otf))
        plt.imshow(ideal_psf, cmap='gray')
        plt.title('Ideal PSF')
        plt.show()

    def test_apodization_sim(self):
        # conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        # conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)    
        # apodization = AutoconvolutionApodizationSIM(self.optical_system, conventional)
        # fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # axes[0].imshow(apodization.ideal_ctf, cmap='gray')
        # axes[0].set_title('Ideal CTF')
        # axes[1].imshow(apodization.ideal_otf, cmap='gray')
        # axes[1].set_title('Ideal OTF')
        # axes[2].imshow(apodization.ideal_psf, cmap='gray')
        # plt.show()

        square = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 0, Mt=1)
        square = IlluminationPlaneWaves2D.init_from_3D(square)    
        apodization = AutoconvolutionApodizationSIM(self.optical_system, square)
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(apodization.ideal_ctf, cmap='gray')
        axes[0].set_title('Ideal CTF')
        axes[1].imshow(apodization.ideal_otf, cmap='gray')
        axes[1].set_title('Ideal OTF')
        axes[2].imshow(apodization.ideal_psf, cmap='gray')
        plt.show()


from config.SIM_N100_NA15 import *
class TestAutoconvolutionSIM3D(unittest.TestCase):
    def setUp(self):
        self.optical_system = OpticalSystems.System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nobject)
        self.optical_system.compute_psf_and_otf((psf_size, N), save_pupil_function=True)

    def test_apodization_sim(self):
        # conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 3, Mt=1)
        # conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)    
        # apodization = AutoconvolutionApodizationSIM(self.optical_system, conventional)
        # fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        # axes[0].imshow(apodization.ideal_ctf, cmap='gray')
        # axes[0].set_title('Ideal CTF')
        # axes[1].imshow(apodization.ideal_otf, cmap='gray')
        # axes[1].set_title('Ideal OTF')
        # axes[2].imshow(apodization.ideal_psf, cmap='gray')
        # plt.show()

        square = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 0, Mt=1)
        apodization = AutoconvolutionApodizationSIM(self.optical_system, square)
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(apodization.ideal_ctf[N//2, :, :], cmap='gray')
        axes[0].set_title('Ideal CTF')
        axes[1].imshow(apodization.ideal_otf[N//2, :, :], cmap='gray')
        axes[1].set_title('Ideal OTF')
        axes[2].imshow(apodization.ideal_psf[N//2, :, :], cmap='gray')
        plt.show()
