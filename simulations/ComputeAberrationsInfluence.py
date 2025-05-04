import os.path
import sys
import matplotlib.pyplot as plt
import csv

import numpy as np
from config.BFPConfigurations import *
from SSNRCalculator import SSNRSIM3D, SSNRSIM2D, SSNRWidefield
from OpticalSystems import System4f3D, System4f2D
sys.path.append('../')

configurations = BFPConfiguration(refraction_index=1.5)
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

arg = N // 2
# print(fz[arg])

two_NA_fx = fx / (2 * NA)
two_NA_fy = fy / (2 * NA)
scaled_fz = fz / fz_max_diff

multiplier = 10 ** 5
ylim = 10 ** 2

optical_system = System4f3D(alpha=alpha, refractive_index_sample=nobject, refractive_index_medium=nmedium)

optical_system.compute_psf_and_otf((psf_size, N), high_NA=True)

widefield = configurations.get_widefield()
squareL = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 1, Mt=1)
squareC = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.58, 1, Mt=1, phase_shift=0)
hexagonal = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1)
conventional = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)

aberrations = {
    # 'Spherical' : (4, 0),
    'Comma' : (3, 1),
    # 'Astigmatism' : (2, 2)
}
RMS = 0.072
aberration_power = np.linspace(0, 2, 21)

headers = ["Configuration", "Aberration", "Aberration Strength", "Volume", "Volume_a", "Entropy"]

config_methods = {
    "Widefield" : widefield
    # "Conventional": conventional,
    # "SquareL": squareL,
    # "SquareC": squareC,
    # "Hexagonal": hexagonal,
}

with open("Aberrations_widefield.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for aberration in aberrations:
        for configuration in config_methods:
            for power in aberration_power:
                illumination = config_methods[configuration]
                optical_system.compute_psf_and_otf(high_NA=True, integrate_rho=True, zernieke={aberrations[aberration]: power * RMS})
                ssnr_calc = SSNRSIM3D(illumination, optical_system)
                ssnr = ssnr_calc.ssnri
                # plt.imshow(np.log(1 + 10**8 * ssnr[:, N//2, :]))
                # plt.show()
                volume = ssnr_calc.compute_ssnri_volume()
                volume_a = ssnr_calc.compute_analytic_ssnri_volume()
                entropy = ssnr_calc.compute_ssnri_entropy()

                print("Volume ", round(volume))
                print("Volume a", round(volume_a))
                print("Entropy ", round(entropy))
                params = [configuration, aberration, round(power, 2), round(volume), round(volume_a), round(entropy)]
                print(params)

                writer.writerow(params)
