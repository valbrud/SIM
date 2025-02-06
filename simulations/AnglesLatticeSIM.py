from config.IlluminationConfigurations import BFPConfiguration
import csv
import numpy as np
from SSNRCalculator import SSNRSIM3D, SSNRSIM3D3dShifts
from OpticalSystems import System4f3D
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = BFPConfiguration()
    N = 101
    alpha_lens = 2 * np.pi / 5
    max_r = N / (16 * (np.sin(alpha_lens)))
    max_z = N / (8 * (1 - np.cos(alpha_lens)))
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

    NA = np.sin(alpha_lens)
    factor = 10 ** 5

    q_axes = 2 * np.pi * np.array((fx, fy, fz))

    headers = ["Configuration", "NA_ratio", "Volume", "Volume_a", "Entropy"]
    configurator = BFPConfiguration()
    squareL = lambda angle: configurator.get_4_oblique_s_waves_and_circular_normal(angle, 1, 1, Mt=1)
    squareC = lambda angle: configurator.get_4_circular_oblique_waves_and_circular_normal(angle, 0.55, 1, Mt=1)
    hexagonal = lambda angle: configurator.get_6_oblique_s_waves_and_circular_normal(angle, 1, 1, Mt=1)
    conventional = lambda angle: configurator.get_2_oblique_s_waves_and_s_normal(angle, 1, 1, Mt=1)

    configurations = {
    "SquareL" : squareL,
    "SquareC" : squareC,
    "Hexagonal" : hexagonal,
    "Conventional" : conventional,
    }

    ratios = np.linspace(0.1, 1, 19)
    optical_system = System4f3D(alpha=alpha_lens)
    optical_system.compute_psf_and_otf((psf_size, N), apodization_function="Sine")

    with open("Angles.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        widefield = configurator.get_widefield()
        ssnr_calc = SSNRSIM3D(widefield, optical_system)
        ssnr = ssnr_calc.compute_ssnr()
        volume = ssnr_calc.compute_ssnri_volume()
        volume_a = ssnr_calc.compute_analytic_ssnr_volume()
        entropy = ssnr_calc.compute_ssnri_entropy()
        params = ["Widefield", 1, round(volume), round(volume_a), round(entropy)]
        print(params)
        writer.writerow(params)

        for configuration in configurations:
            for ratio in ratios:
                sin_theta = ratio * NA
                theta = np.arcsin(sin_theta)
                illumination = configurations[configuration](theta)
                ssnr_calc = SSNRSIM3D(illumination, optical_system)
                ssnr = ssnr_calc.compute_ssnr()

                volume = ssnr_calc.compute_ssnri_volume()
                volume_a = ssnr_calc.compute_analytic_ssnr_volume()
                entropy = ssnr_calc.compute_ssnri_entropy()

                print("Volume ", round(volume))
                print("Volume a", round(volume_a))
                print("Entropy ", round(entropy))
                params = [configuration, round(ratio, 2), round(volume), round(volume_a),  round(entropy)]
                print(params)

                writer.writerow(params)
