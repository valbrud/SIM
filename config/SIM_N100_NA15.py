import os.path
import sys
print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from IlluminationConfigurations import BFPConfiguration
import numpy as np

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
