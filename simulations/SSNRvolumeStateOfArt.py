from config.IlluminationConfigurations import BFPConfiguration
import csv
import numpy as np
from SSNRCalculator import SSNRCalculatorProjective3dSIM
from OpticalSystems import Lens
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = BFPConfiguration()
    max_r = 5
    max_z = 10
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

    NA = np.sin(alpha_lens)
    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / (2 * NA)
    factor = 10 ** 5

    q_axes = 2 * np.pi * np.array((fx, fy, fz))

    headers = ["IncidentAngle", "Mr", "Volume", "Volume_a", "Entropy", "RadialEntropy"]

    size = 8
    theta = np.linspace(alpha_lens/size, alpha_lens, size)
    rotations = np.arange(3, 4)

    optical_system = Lens(alpha=alpha_lens)
    optical_system.compute_psf_and_otf((psf_size, N), apodization_filter=None)

    with open("test.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        illumination_widefield = config.get_widefield()
        illumination_widefield.normalize_spacial_waves()
        ssnr_widefield = SSNRCalculatorProjective3dSIM(illumination_widefield, optical_system)
        wssnr = ssnr_widefield.compute_ssnr()
        wvolume = ssnr_widefield.compute_ssnr_volume()
        wvolume_a = ssnr_widefield.compute_analytic_ssnr_volume()
        wentropy = ssnr_widefield.compute_true_ssnr_entropy()
        wrentropy = ssnr_widefield.compute_radial_ssnr_entropy()
        print("Volume ", round(wvolume))
        print("Volume a", round(wvolume_a))
        print("Entropy ", round(wentropy, 3))
        print("Radial entropy ", round(wrentropy))
        params = [0, 0, round(wvolume), round(wvolume_a),  round(wentropy, 3), round(wrentropy, 3)]
        print(params)
        writer.writerow(params)

        for angle in theta:
            for Mr in rotations:
                k = 2 * np.pi
                k1 = k * np.sin(angle)
                k2 = k * (np.cos(angle) - 1)
                strength = 1
                illumination = config.get_2_oblique_s_waves_and_s_normal(angle, strength, Mr=Mr)
                ssnr_calc = SSNRCalculatorProjective3dSIM(illumination, optical_system)
                ssnr = ssnr_calc.compute_ssnr()
                # plt.imshow(1 + 10**8 * np.log(ssnr[:, :, 50]))
                # plt.show()
                volume = ssnr_calc.compute_ssnr_volume()
                volume_a = ssnr_calc.compute_analytic_ssnr_volume()
                entropy = ssnr_calc.compute_true_ssnr_entropy()
                rentropy = ssnr_calc.compute_radial_ssnr_entropy()
                print("Volume ", round(volume))
                print("Mr", Mr)
                print("Volume a", round(volume_a))
                print("Entropy ", round(entropy, 3))
                print("Radial entropy", round(rentropy, 3))
                params = [round(angle * 57.29, 1), Mr, round(volume), round(volume_a), round(entropy, 3), round(rentropy, 3)]
                print(params)
                writer.writerow(params)
