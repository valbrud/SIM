import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from config.BFPConfigurations import BFPConfiguration
import numpy as np
from Illumination import IlluminationNonLinearSIM2D, IlluminationPlaneWaves2D
from scipy.special import factorial
import matplotlib.pyplot as plt
from SSNRCalculator import SSNRSIM2D
from OpticalSystems import System4f2D

configurations = BFPConfiguration(refraction_index=1.5)
alpha = 2 * np.pi / 5
nmedium = 1.5
nobject = 1.5
NA = nmedium * np.sin(alpha)
theta = np.asin(0.9 * np.sin(alpha))
fz_max_diff = nmedium * (1 - np.cos(alpha))
dx = 1 / (128 * NA)
dy = dx

N = 255
max_r = N // 2 * dx

psf_size = 2 * np.array((2 * max_r, 2 * max_r))

optical_system = System4f2D(alpha=alpha, refractive_index=nmedium)
optical_system.compute_psf_and_otf((psf_size, N))

x = np.linspace(-max_r, max_r, N)
y = np.copy(x)
dimensions = (1, 1)

fig = plt.figure(figsize=(15, 9), constrained_layout=True)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title("Emission density", fontsize=25, pad=15)
ax1.tick_params(labelsize=20)

ax2.set_title("Ring averaged SSNR", fontsize=25)
ax2.tick_params(labelsize=20)

volume_list= []
entropy_list = []


for Mr in np.linspace(1, 10, 10):
    p = 1
    nonlinear_expansion_coefficients = [0, ]
    n = 1
    while (p ** n / factorial(n)) > 10 ** -2:
        nonlinear_expansion_coefficients.append(p ** n / factorial(n) * (-1) ** (n + 1))
        n += 1
    illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, Mr, Mt=1)
    illumination_3waves2d = IlluminationPlaneWaves2D.init_from_3D(illumination_3waves3d, dimensions)

    illumination_3waves_non_linear = IlluminationNonLinearSIM2D.init_from_linear_illumination(illumination_3waves2d, tuple(nonlinear_expansion_coefficients))
    illumination_3waves_non_linear.normalize_spatial_waves()
    print(Mr, np.round(Mr * np.abs(illumination_3waves_non_linear.get_amplitudes()[0]), 4), illumination_3waves_non_linear.get_amplitudes()[1])


for c in np.linspace(-1, 1, 11):
    p = 10**c
    nonlinear_expansion_coefficients = [0, ]
    n = 1
    while (p ** n / factorial(n)) > 10 ** -2:
        nonlinear_expansion_coefficients.append(p ** n / factorial(n) * (-1) ** (n + 1))
        n += 1

    illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 0, 5, Mt=1)
    illumination_3waves2d = IlluminationPlaneWaves2D.init_from_3D(illumination_3waves3d, dimensions)

    illumination_3waves_non_linear = IlluminationNonLinearSIM2D.init_from_linear_illumination(illumination_3waves2d, tuple(nonlinear_expansion_coefficients))
    illumination_3waves_non_linear.normalize_spatial_waves()
    print(c, np.round(np.abs(illumination_3waves_non_linear.get_amplitudes()[0]), 5), illumination_3waves_non_linear.get_amplitudes()[1])
    # for wave in illumination_3waves_non_linear.waves:
    #     print(wave, illumination_3waves_non_linear.waves[wave].wavevector, illumination_3waves_non_linear.waves[wave].amplitude)
    illumination_density = illumination_3waves2d.get_illumination_density(coordinates=(x, y))
    emission_density = illumination_3waves_non_linear.get_illumination_density(coordinates=(x, y))

    ax1.plot(emission_density[127, :], label=f"p={p}")
    ssnr_linear = SSNRSIM2D(illumination_3waves2d, optical_system)
    ssnr_nonlinear = SSNRSIM2D(illumination_3waves_non_linear, optical_system)
    # ax2.plot(illumination_density[127, :], label=f"p={p}")
    ax2.plot(optical_system.otf_frequencies[0][N//2:]/(2 * NA), np.log10(1 + 10**8 * ssnr_linear.ring_average_ssnri()), label=f"p={p}")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    volume = ssnr_nonlinear.compute_ssnri_volume()
    entropy = ssnr_nonlinear.compute_ssnri_entropy()
    volume_list.append(volume)
    entropy_list.append(entropy)

ax1.legend()
ax2.legend()
plt.show()

# plt.plot(np.array(volume_list), label="Volume")
# plt.legend()
# plt.show()
# plt.plot(np.array(entropy_list), label="Entropy")
# plt.legend()
# plt.show()