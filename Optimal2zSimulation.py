from config.IlluminationConfigurations import BFPConfiguration
import csv
import numpy as np
from Sources import IntensityPlaneWave
from Illumination import Illumination
from SNRCalculator import SNRCalculator
from OpticalSystems import Lens


if __name__ == "__main__":
    config = BFPConfiguration()
    max_r = 10
    max_z = 20
    N = 100
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

    NA = np.sin(2 * np.pi/5)
    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / (2 * NA)
    q_axes = 2 * np.pi * np.array((fx, fy, fz))

    headers = ["ThetaIncident", "SineRatio", "Power1", "Power2", "sum(log(1 + 10^8 SSNR))"]

    power1, power2 = np.array((0.5, 1, 2)), np.array((0.5, 1, 2))
    theta = np.linspace(np.pi/20, np.pi/2, 10)
    ratios = np.linspace(0.1, 1, 10)

    optical_system = Lens(alpha=2 * np.pi/5)
    optical_system.compute_psf_and_otf((psf_size, N), apodization_filter=None)

    with open("4waves2z.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        illumination_widefield = config.widefield()
        illumination_widefield.normalize_spacial_waves()
        SSNR_widefield = SNRCalculator(illumination_widefield, optical_system)
        SSNR = SSNR_widefield.SSNR(q_axes)
        # print(SSNR[N//2, N//2, N//2])
        widefield_volume = SSNR_widefield.compute_SSNR_volume(SSNR, 1, 10 ** 8)
        params = ["0", "0", "0", "0", round(widefield_volume)]
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
                        illumination = config.get_4_s_oblique_waves_at_2_angles_and_one_normal_s_wave(angle, a, b, c, 0)
                        SSNR_calc = SNRCalculator(illumination, optical_system)
                        SSNR = SSNR_calc.SSNR(q_axes)
                        # print(SSNR[N // 2, N // 2, N // 2])
                        volume = SSNR_calc.compute_SSNR_volume(SSNR, 1, 10**8)
                        params = [round(angle * 57.29, 1), a, b, c, round(volume)]
                        print(params)
                        writer.writerow(params)

