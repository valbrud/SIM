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

from config.BFPConfigurations import *
import hpc_utils
from Sources import PlaneWave, IntensityHarmonic3D, IntensityHarmonic2D
import scipy
import matplotlib.pyplot as plt
from Apodization import AutoconvolutuionApodizationSIM2D, AutoconvolutionApodizationSIM3D


class TestAutoconvolutionSIM2D(unittest.TestCase):
    def setUp(self):
        from config.SIM_N500_NA15_2D import alpha, nobject, N, dx, psf_size, fx, fy, NA, configurations
        self.N = N
        self.dx = dx
        self.psf_size = psf_size
        self.fx = fx
        self.fy = fy
        self.theta = np.asin(0.9 * np.sin(alpha))
        self.nobject = nobject
        self.nmedium = 1.5
        self.NA = NA
        self.configurations = configurations
        print(N, dx *  NA)
        self.optical_system = OpticalSystems.System4f2D(alpha=alpha, refractive_index=nobject)
        self.optical_system.compute_psf_and_otf(((psf_size[0], psf_size[1]), N))
    
    def test_extended_space_conventional(self):
        conventional = self.configurations.get_2_oblique_s_waves_and_s_normal(self.theta, 1, 0, 3, Mt=1)
        conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)    
        _, effective_otfs = conventional.get_effective_otfs(self.optical_system.otf, self.optical_system.psf)  
        otf_extended_space = np.zeros((self.N, self.N, 3), dtype=np.float64)
        otf_extended_space[:, :, 0] = effective_otfs[(0, 0)] 
        otf_extended_space[:, :, 1] = effective_otfs[(0, 1)]
        otf_extended_space[:, :, 2] = effective_otfs[(0, 2)]
        ideal_kernel = np.where(otf_extended_space > 10**-12, 1, 0)

    def test_autoconvolution(self):
        k1 = 2 * np.pi * np.sin(self.theta)
        electric_ft = IlluminationPlaneWaves2D(
            {(1, 0): IntensityHarmonic2D(wavevector=np.array([k1, 0]), amplitude=1),
             (-1, 0): IntensityHarmonic2D(wavevector=np.array([-k1, 0]), amplitude=1)},
              dimensions=(1, 1), Mr=1, spatial_shifts=np.array(((0., 0.),))
            )
        pupil_function = np.zeros((self.N, self.N), dtype=np.float64)
        Fx, Fy = np.meshgrid(self.fx, self.fy)
        pupil_function[(Fx**2 + Fy**2) * 2 * np.pi < k1**2] = 1
        ctf = hpc_utils.wrapped_fftn(pupil_function)
        plt.imshow(pupil_function, cmap='gray')
        plt.title('Pupil Function')
        plt.show()
        _, effective_otfs = electric_ft.compute_effective_kernels(ctf, self.optical_system.psf_coordinates)  
        sim_ctf = np.zeros((self.N, self.N), dtype=np.complex128)
        for ctf in effective_otfs.values():
            sim_ctf += ctf  
        plt.imshow(sim_ctf.real, cmap='gray')
        plt.title('Simulated CTF')
        plt.show()
        ideal_otf = scipy.signal.convolve2d(sim_ctf, sim_ctf, mode='same').real
        plt.imshow(ideal_otf, cmap='gray')
        plt.title('Ideal OTF')
        plt.show()
        ideal_psf = np.real(hpc_utils.wrapped_ifftn(ideal_otf))
        plt.imshow(ideal_psf, cmap='gray')
        plt.title('Ideal PSF')
        plt.show()

    def test_apodization_sim(self):
        conventional = self.configurations.get_2_oblique_s_waves_and_s_normal(self.theta, 1, 0, 3, Mt=1, angles=(0 + np.pi/4, np.pi/3 + np.pi/4, 2*np.pi/3 + np.pi/4))
        conventional = IlluminationPlaneWaves2D.init_from_3D(conventional)    
        apodization = AutoconvolutuionApodizationSIM2D(self.optical_system, conventional)
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(apodization.ideal_ctf, cmap='gray')
        axes[0].set_title('Ideal CTF')
        axes[1].imshow(np.log1p(10**8 * apodization.ideal_otf), cmap='gray')
        axes[1].set_title('Ideal OTF')
        axes[2].imshow(apodization.ideal_psf, cmap='gray')
        plt.show()

        square = self.configurations.get_4_oblique_s_waves_and_s_normal_diagonal(self.theta, 1, 0, Mt=1)
        square = IlluminationPlaneWaves2D.init_from_3D(square)    
        apodization = AutoconvolutuionApodizationSIM2D(self.optical_system, square)
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(apodization.ideal_ctf, cmap='gray')
        axes[0].set_title('Ideal CTF')
        axes[1].imshow(np.log(1 + 10 **8 * apodization.ideal_otf), cmap='gray')
        axes[1].set_title('Ideal OTF')
        axes[2].imshow(apodization.ideal_psf, cmap='gray')
        plt.show()
    
    def test_compare_with_non_apodized(self):
        square = self.configurations.get_4_circular_oblique_waves_and_circular_normal(self.theta, 1, 0, Mt=1)
        square = IlluminationPlaneWaves2D.init_from_3D(square) 
        _, effective_otfs = square.compute_effective_kernels(self.optical_system.psf, self.optical_system.psf_coordinates)
        sim_otf = np.zeros((self.N, self.N), dtype=np.complex128)
        for otf in effective_otfs.values():
            sim_otf += np.abs(otf)
        sim_otf /= np.amax(sim_otf)

        sim_psf = np.real(hpc_utils.wrapped_ifftn(sim_otf))
        sim_psf /= np.sum(sim_psf)
        apodization = AutoconvolutuionApodizationSIM2D(self.optical_system, square)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(np.log(1 + 10 **8 * sim_otf.real), cmap='gray')
        axes[1].imshow(np.log(1 + 10 **8 * apodization.ideal_otf.real), cmap='gray')
        plt.show()
        fig, axes = plt.subplots(1, 4, figsize=(10, 5))
        sim_psf = sim_psf * np.amax(apodization.ideal_psf) / np.amax(sim_psf)
        axes[0].plot(apodization.ideal_psf[:, self.N//2], label = 'Ideal PSF')
        axes[0].plot(sim_psf[:, self.N//2], label = 'SIM PSF')
        axes[0].set_title('fy = 0')
        axes[1].plot(np.diagonal(apodization.ideal_psf), label = 'Ideal PSF')
        axes[1].plot(np.diagonal(sim_psf), label = 'SIM PSF')
        axes[1].set_title('fy = 0')
        axes[0].legend()
        axes[1].legend()
        axes[2].imshow(apodization.ideal_psf, cmap='gray')
        axes[2].set_title('Ideal PSF')
        axes[3].imshow(sim_psf, cmap='gray')
        axes[3].set_title('SIM PSF')

        plt.show()


class TestAutoconvolutionSIM3D(unittest.TestCase):
    def setUp(self):
        from config.SIM_N100_NA15 import alpha, dx, nobject, nmedium, N, psf_size, fx, fy, NA, configurations
        self.N = N
        self.dx = dx
        self.psf_size = psf_size
        self.fx = fx
        self.fy = fy
        self.theta = np.asin(0.9 * np.sin(alpha))
        self.nobject = nobject
        self.nmedium = 1.5
        self.NA = NA
        self.configurations = configurations
        self.optical_system = OpticalSystems.System4f3D(alpha=alpha, refractive_index_medium=nmedium, refractive_index_sample=nobject)
        self.optical_system.compute_psf_and_otf((psf_size, N))

    def test_apodization_sim(self):
        conventional = self.configurations.get_2_oblique_s_waves_and_s_normal(self.theta, 3, 0, 1, Mt=1)
        apodization = AutoconvolutionApodizationSIM3D(self.optical_system, conventional)
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(apodization.ideal_ctf[:, :, self.N//2].real, cmap='gray')
        axes[0].set_title('Ideal CTF')
        axes[1].imshow(np.log(1 + 10**8 * apodization.ideal_otf[:, :, self.N//2].real), cmap='gray')
        axes[1].set_title('Ideal OTF')
        axes[2].imshow(apodization.ideal_psf[:, :, self.N//2].real, cmap='gray')
        plt.show()
        
        square = self.configurations.get_4_oblique_s_waves_and_s_normal_diagonal(self.theta, 1, 0, Mt=1)
        apodization = AutoconvolutionApodizationSIM3D(self.optical_system, square)
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(apodization.ideal_ctf[self.N//2, :, :].real, cmap='gray')
        axes[0].set_title('Ideal CTF')
        axes[1].imshow(np.log(1 + 10**8 * apodization.ideal_otf[self.N//2, :, :].real), cmap='gray')
        axes[1].set_title('Ideal OTF')
        axes[2].imshow(apodization.ideal_psf[self.N//2, :, :].real, cmap='gray')

        def update(i):
            axes[0].imshow(apodization.ideal_ctf[:, :, int(i)].real, cmap='gray')
            axes[1].imshow(np.log(1 + 10**8 * apodization.ideal_otf[:, :, int(i)].real), cmap='gray')
            axes[2].imshow(apodization.ideal_psf[:, :, int(i)].real, cmap='gray')
            plt.pause(0.1)

        from matplotlib.widgets import Slider
        slider_loc = plt.axes([0.2, 0.02, 0.65, 0.03])  # slider location and size
        slider = Slider(slider_loc, 'z', 0, self.N - 1)  # slider properties
        slider.on_changed(update)

        plt.show()

if __name__ == '__main__':
    unittest.main(verbosity=2)