from config.IlluminationConfigurations import BFPConfiguration
import csv
import numpy as np
from SSNRCalculator import SSNR3dSIM2dShifts, SSNR3dSIM3dShifts
from OpticalSystems import Lens3D

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

    NA = np.sin(np.pi / 4)
    two_NA_fx = fx / (2 * NA)
    two_NA_fy = fy / (2 * NA)
    two_NA_fz = fz / (2 * NA)
    factor = 10 ** 5

    q_axes = 2 * np.pi * np.array((fx, fy, fz))

    headers = ["IncidentAngle", "Power", "Volume", "Volume_a", "Entropy"]

    config_methods = {
        # 'conventional' : config.get_2_oblique_s_waves_and_s_normal
        'square_cp': config.get_4_circular_oblique_waves_and_circular_normal,
        # 'square' : config.get_4_oblique_s_waves_and_circular_normal ,
        # 'hexagonal' : config.get_6_oblique_s_waves_and_circular_normal,
        }

    angles = np.linspace(alpha_lens/8, alpha_lens, 8)
    powers = np.linspace(0., 2, 21)
    optical_system = Lens3D(alpha=alpha_lens)
    optical_system.compute_psf_and_otf((psf_size, N), apodization_filter=None)

    with open("circular_b.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        # illumination_widefield = config.get_widefield()
        # illumination_widefield.normalize_spacial_waves()
        # ssnr_widefield = SSNR3dSIM2dShifts(illumination_widefield, optical_system)
        # wssnr = ssnr_widefield.compute_ssnr()
        # wvolume = ssnr_widefield.compute_ssnr_volume()
        # wvolume_a = ssnr_widefield.compute_analytic_ssnr_volume()
        # wtotal = ssnr_widefield.compute_total_ssnr()
        # wtotal_a = ssnr_widefield.compute_analytic_total_ssnr()
        # wentropy = ssnr_widefield.compute_true_ssnr_entropy()
        # wmeasure, _ = ssnr_widefield.compute_ssnr_waterline_measure()
        # print("Volume ", round(wvolume))
        # print("Volume a", round(wvolume_a))
        # print("Total", round(wtotal))
        # print("Total a ", round(wtotal_a))
        # print("Entropy ", round(wentropy))
        # print("Measure ", round(wmeasure))
        # params = [0, 0, round(wvolume), round(wvolume_a), round(wtotal), round(wtotal_a), round(wentropy), round(wmeasure)]
        # print(params)
        # writer.writerow(params)
        theta = 0.8 * 2 * np.pi/5
        for b in powers:
            for config_method in config_methods:
                k = 2 * np.pi
                k1 = k * np.sin(theta)
                k2 = k * (np.cos(theta) - 1)
                # strength = 1 if config_method != config.get_4_circular_oblique_waves_and_circular_normal else 1 / 2 ** 0.5
                strength = b
                illumination = config_methods[config_method](theta, strength)
                ssnr_calc = SSNR3dSIM2dShifts(illumination, optical_system)
                ssnr = ssnr_calc.compute_ssnr()
                volume = ssnr_calc.compute_ssnr_volume()
                volume_a = ssnr_calc.compute_analytic_ssnr_volume()
                # total = ssnr_calc.compute_total_ssnr()
                # total_a = ssnr_calc.compute_analytic_total_ssnr()
                entropy = ssnr_calc.compute_true_ssnr_entropy()
                # measure, _ = ssnr_calc.compute_ssnr_waterline_measure()

                print("Volume ", round(volume))
                print("Volume a", round(volume_a))
                # print("Total", round(total))
                # print("Total a ", round(total_a))
                print("Entropy ", round(entropy))
                # print("Measure ", round(measure))
                params = [round(theta * 57.29, 1), round(b, 1), round(volume), round(volume_a),  round(entropy)]
                print(params)
                writer.writerow(params)
    #
    # with open("lattice_setups_true3d.csv", 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(headers)
    #     illumination_widefield = config.get_widefield()
    #     illumination_widefield.normalize_spacial_waves()
    #     ssnr_widefield = SSNR3dSIM2dShifts(illumination_widefield, optical_system)
    #     wssnr = ssnr_widefield.compute_ssnr()
    #     wvolume = ssnr_widefield.compute_ssnr_volume()
    #     wvolume_a = ssnr_widefield.compute_analytic_ssnr_volume()
    #     wtotal = ssnr_widefield.compute_total_ssnr()
    #     wtotal_a = ssnr_widefield.compute_analytic_total_ssnr()
    #     wentropy = ssnr_widefield.compute_true_ssnr_entropy()
    #     wmeasure, _ = ssnr_widefield.compute_ssnr_waterline_measure()
    #     print("Volume ", round(wvolume))
    #     print("Volume a", round(wvolume_a))
    #     print("Total", round(wtotal))
    #     print("Total a ", round(wtotal_a))
    #     print("Entropy ", round(wentropy))
    #     print("Measure ", round(wmeasure))
    #     params = [0, 0, round(wvolume), round(wvolume_a), round(wtotal), round(wtotal_a), round(wentropy), round(wmeasure)]
    #     print(params)
    #     writer.writerow(params)
    #
    #     for angle in theta:
    #         for config_method in config_methods:
    #             k = 2 * np.pi
    #             k1 = k * np.sin(angle)
    #             k2 = k * (np.cos(angle) - 1)
    #             strength = 1 if config_method != config.get_4_circular_oblique_waves_and_circular_normal else 1 / 2 ** 0.5
    #             illumination = config_methods[config_method](angle, strength)
    #             ssnr_calc = SSNR3dSIM3dShifts(illumination, optical_system)
    #             ssnr = ssnr_calc.compute_ssnr()
    #             volume = ssnr_calc.compute_ssnr_volume()
    #             volume_a = ssnr_calc.compute_analytic_ssnr_volume()
    #             total = ssnr_calc.compute_total_ssnr()
    #             total_a = ssnr_calc.compute_analytic_total_ssnr()
    #             entropy = ssnr_calc.compute_true_ssnr_entropy()
    #             measure, _ = ssnr_calc.compute_ssnr_waterline_measure()
    #
    #             print("Volume ", round(volume))
    #             print("Volume a", round(volume_a))
    #             print("Total", round(total))
    #             print("Total a ", round(total_a))
    #             print("Entropy ", round(entropy))
    #             print("Measure ", round(measure))
    #             params = [round(angle * 57.29, 1), config_method, round(volume), round(volume_a), round(total), round(total_a), round(entropy), round(measure)]
    #             print(params)
    #             writer.writerow(params)

