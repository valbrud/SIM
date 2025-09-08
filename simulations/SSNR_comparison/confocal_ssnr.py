"""
confocal_ssnr.py

This script contains test computations of the SSNR in confocal microscopy, ISM and Rescan.
"""

import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)


import numpy as np
import matplotlib.pyplot as plt
import OpticalSystems
import wrappers
import unittest
alpha = 2 * np.pi / 5
nmedium = 1.5
nobject = 1.5
NA = nmedium * np.sin(alpha)
theta = np.asin(0.9 * np.sin(alpha))
fz_max_diff = nmedium * (1 - np.cos(alpha))
dx = 1 / (8 * NA)
dy = dx
dz = 1 / (4 * fz_max_diff)
N = 101
max_r = N // 2 * dx
max_z = N // 2 * dz
psf_size = 2 * np.array((max_r, max_r, max_z))
dV = dx * dy * dz
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
z = np.linspace(-max_z, max_z, N)
fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)
fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz), N)

Fx, Fy = np.meshgrid(fx, fy)

class TestConfocalSSNR(unittest.TestCase):
    def test_SSNR2D(self):
        pupil_function = np.where(np.abs(Fx**2 + Fy**2)**0.5 <= NA, 1, 0)
        # plt.imshow(pupil_function, extent=(fx[0]/(2 * NA), fx[-1]/(2 * NA), fy[0]/(2 * NA), fy[-1]/(2 * NA)))
        optical_system = OpticalSystems.System4f2D(alpha=alpha, refractive_index=1.5)
        PSF_em, OTF_em = optical_system.compute_psf_and_otf((psf_size, N))
        SSNR_widefield = OTF_em ** 2
        OTF_confocal = wrappers.wrapped_fftn(PSF_em ** 2)
        OTF_confocal /= np.amax(OTF_confocal)
        SSNR_confocal = OTF_confocal**2
        fig, ax = plt.subplots()
        # ax.plot(fy/(2 * NA), SSNR_confocal[N//2, :], label='confocal')
        # ax.plot(fy/(2 * NA), SSNR_widefield, label='widefield')
        nmax = 10
        obstacles = np.arange(nmax) / nmax * NA
        for obstacle in obstacles:
            pupil_function_annular = pupil_function * np.where(np.abs(Fx**2 + Fy**2)**0.5 >= obstacle, 1, 0)
            PSF_ex, OTF_ex = optical_system.compute_psf_and_otf(pupil_function=pupil_function_annular)
            # plt.imshow(PSF_ex)
            # ax.plot(fy / (2 * NA), OTF_ex[N//2, :], label = f'r_closed = {round(obstacle/NA, 1)} R')
            # plt.show()
            OTF_eff = wrappers.wrapped_fftn(PSF_ex * PSF_em)
            OTF_eff /= np.amax(OTF_eff)
            SSNR_eff = OTF_eff ** 2
            ratio = SSNR_confocal/SSNR_eff
            # ax.plot((fy / (2 * NA)), ratio[N//2, :], label = f'r_closed = {round(obstacle/NA, 1)} R')
            # ax.set_ylim(0, 1.1)
            ax.hlines(y=10,  xmin=-2, xmax=2, color='black')
            ax.plot(fy / (2 * NA), 1 + 10**3* SSNR_eff[N//2, :], label = f'r_closed = {round(obstacle/NA, 1)} R')
            ax.set_yscale('log')
        ax.legend()
        plt.show()

    def test_SSNR3D(self):
        def annular_pupil_function(rho, r_min = 0, r_max = 1):
            return np.where(r_min < rho < r_max, 1, 0)

        def double_annular_pupil_function(rho, rhomax, relwidth, Iratio):
            return Iratio * (np.where(rho >= 0, 1, 0) * np.where(rho <= relwidth * rhomax, 1, 0)) +  \
                     np.where(rho >= rhomax * (1 - relwidth), 1, 0) * np.where(rho <= rhomax, 1, 0)

        optical_system = OpticalSystems.System4f3D(alpha=alpha, refractive_index_sample =1.5, refractive_index_medium = 1.5)
        PSF_em, OTF_em = optical_system.compute_psf_and_otf((psf_size, N))
        # plt.imshow(PSF_em[:, :, N//2])
        # plt.show()
        SSNR_widefield = OTF_em ** 2
        OTF_confocal = wrappers.wrapped_fftn(PSF_em ** 2)
        OTF_confocal /= np.amax(OTF_confocal)
        # plt.imshow(OTF_confocal[:, N // 2, :].real)
        # plt.show()
        SSNR_confocal = OTF_confocal ** 2
        fig, ax = plt.subplots(2)
        ax[0].plot(fy/(2 * NA), OTF_confocal[N//2, :, N//2], label='confocal')
        ax[1].plot(fy/(2 * NA), OTF_confocal[N//2, N//2, :], label='confocal')
        # ax.plot(fy/(2 * NA), SSNR_widefield, label='widefield')
        Irmax = 3
        rhomax = np.sin(alpha)
        relwidth = 0.2
        for Ir in np.arange(Irmax+1):
            pupil_function = lambda rho: double_annular_pupil_function(rho, rhomax=rhomax, relwidth=relwidth, Iratio=Ir)
            PSF_ex, OTF_ex = optical_system.compute_psf_and_otf((psf_size, N), pupil_function=pupil_function)
            # plt.imshow(PSF_ex[:, :, N//2].real)
            # plt.show()
            # ax.plot(fy / (2 * NA), OTF_ex[N//2, :, N//2], label = f'I ratio = {Ir}')
            # plt.show()
            OTF_eff = wrappers.wrapped_fftn(PSF_ex * PSF_em)
            OTF_eff /= np.amax(OTF_eff)
            # plt.imshow(OTF_eff[:, N//2, :].real)
            # plt.show()
            SSNR_eff = OTF_eff ** 2
            ratio = SSNR_confocal / SSNR_eff
            ax[0].plot(fy / (2 * NA), OTF_eff[N//2, :, N//2], label = f'I ratio = {Ir}')
            ax[1].plot(fy / (2 * NA), OTF_eff[N//2, N//2, :], label = f'I ratio = {Ir}')
            # ax.plot((fy / (2 * NA)), ratio[N//2, :], label = f'I ratioj = {Ir}')
            # ax.set_ylim(0, 1.1)
            # ax.hlines(y=10, xmin=-2, xmax=2, color='black')
            # ax.plot(fy / (2 * NA), 1 + 10 ** 3 * SSNR_eff[N // 2, :], label=f'I ratio = {Ir}')
            # ax.set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        plt.show()
