import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from config.BFPConfigurations import BFPConfiguration
import csv
import numpy as np
from SSNRCalculator import SSNRSIM3D
from OpticalSystems import System4f3D


if __name__ == "__main__":
    config = BFPConfiguration()
    max_r = 10
    max_z = 20
    N = 100
    alpha_lens = 2 * np.pi / 5
    psf_size = 2 * np.array((max_r, max_r, max_z))
    dx = 2 * max_r / N
    dy = 2 * max_r / N
    dz = 2 * max_z / N
    dV = dx * dy * dz
    x = np.arange(-max_r, max_r, dx)
    y = np.copy(x)
    z = np.arange(-max_z, max_z, dz)

    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
    fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
    fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_z), N)

    dVf = 1 / (2 * max_r) * 1 / (2 * max_r) * 1 / (2 * max_z)
    arg = N // 2  # - 24

    NA = np.sin(np.pi/4)
    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / (2 * NA)
    factor = 10**5

    q_axes = 2 * np.pi * np.array((fx, fy, fz))

    headers = ["ThetaIncident", "SineRatio", "Power1", "Power2", "volume", "total", "total_a", "entropy"]

    power1, power2 = np.array((0.5, 1, 2)), np.array((0.5, 1, 2))
    theta = np.linspace(1 * np.pi/20, np.pi/2, 10)
    ratios = np.linspace(0.1, 1, 10)

    optical_system = System4f3D(alpha=alpha_lens)
    optical_system.compute_psf_and_otf((psf_size, N), apodization_function="Sine")

    with open("simulations/5waves_new", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        illumination_widefield = config.get_widefield()
        illumination_widefield.normalize_spatial_waves()
        ssnr_widefield = SSNRSIM3D(illumination_widefield, optical_system)
        wssnr = ssnr_widefield.compute_ssnr()
        wvolume = ssnr_widefield.compute_ssnri_volume(factor)
        wtotal = ssnr_widefield.compute_total_signal_to_noise()
        wtotal_a = ssnr_widefield.compute_total_analytic_signal_to_noise()
        wentropy = ssnr_widefield.compute_ssnri_entropy(factor)
        wmeasure, _ = ssnr_widefield.compute_ssnr_waterline_measure(factor)
        print("Volume ", round(wvolume))
        print("Total", round(wtotal))
        print("Total a ", round(wtotal_a))
        print("Entropy ", round(wentropy))
        print("Measure ", round(wmeasure))
        params = [0, 0, 0, 0, round(wvolume), round(wtotal), round(wtotal_a), round(wentropy), round(wmeasure)]
        print(params)
        writer.writerow(params)

        for angle in theta:
            for a in ratios:
                for b in power1:
                    for c in power2:
                        k = 2 * np.pi
                        k1 = k * np.sin(angle)
                        k3 = k * a * np.sin(angle)
                        k2 = k * (np.cos(angle) - 1)
                        k4 = k * (np.cos(np.arcsin(a * np.sin(angle))) - 1)
                        illumination = config.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(angle, a, b, c, 1)
                        ssnr_calc = SSNRSIM3D(illumination, optical_system)
                        ssnr = ssnr_calc.compute_ssnr()
                        volume = ssnr_calc.compute_ssnri_volume(factor)
                        total = ssnr_calc.compute_total_signal_to_noise()
                        total_a = ssnr_calc.compute_total_analytic_signal_to_noise()
                        entropy = ssnr_calc.compute_ssnri_entropy(factor)
                        measure, _ = ssnr_calc.compute_ssnr_waterline_measure(factor)
                        print("Volume ", round(volume))
                        print("Total", round(total))
                        print("Total a ", round(total_a))
                        print("Entropy ", round(entropy))
                        print("Measure ", round(measure))
                        params = [round(angle * 57.29, 1), a, b, c, round(volume), round(total), round(total_a), round(entropy), round(measure)]
                        print(params)
                        writer.writerow(params)

