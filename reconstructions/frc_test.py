import numpy as np
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from ResolutionMeasures import frc
import matplotlib.pyplot as plt

folder = r"c:\Users\mag_k\PycharmProjects\SIM\reconstructions\images"

# List to hold the loaded arrays (or use a dict to keep filenames)
arrays = {}
for filename in os.listdir(folder):
    if filename.endswith('.npy'):
        path = os.path.join(folder, filename)
        arrays[filename] = np.load(path)

# Example: print the names and shapes of the loaded arrays
for name, array in arrays.items():
    print(name, array.shape)

# plt.imshow(arrays['reconstructed_widefield1.npy'], cmap='gray')
# plt.show()

frc_widefield, freq = frc(arrays['reconstructed_widefield1.npy'], arrays['reconstructed_widefield2.npy'], is_fourier=False, num_bins=100)
frc_fourier, _ = frc(arrays['reconstructed_fourier1.npy'], arrays['reconstructed_fourier2.npy'], is_fourier=False, num_bins=100)
frc_spatial, _ = frc(arrays['reconstructed_spatial1.npy'], arrays['reconstructed_spatial2.npy'], is_fourier=False, num_bins=100)
frc_finite, _ = frc(arrays['reconstructed_finite1.npy'], arrays['reconstructed_finite2.npy'], is_fourier=False, num_bins=100)

plt.plot(freq, frc_widefield, label='FRC Widefield')
plt.plot(freq, frc_fourier, label='FRC Fourier')
plt.plot(freq, frc_spatial, label='FRC Spatial')
plt.plot(freq, frc_finite, label='FRC Finite')
plt.legend()
plt.show()