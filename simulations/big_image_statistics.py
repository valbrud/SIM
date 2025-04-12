import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

# --- Parameters ---
filename = project_root + '/Figures/100X_ex_170ms_filter_cy3b_H_SNR_1_MMStack_Default.ome.tif'   # set your large TIFF image filename here
M = 31  # block size (M x M)

# --- (a) Read the image (as a numpy array) ---
img_np = tifffile.imread(filename)[1, ...]  # assume grayscale; shape (N, N)
N = img_np.shape[0]
print(f"Image shape: {img_np.shape}")

# --- (b) Determine number of blocks (ignore remainder) ---
nb_r = N // M  # number of blocks in row direction
nb_c = N // M  # number of blocks in column direction
N_blocks = nb_r * nb_c  # total number of blocks
print(f"Processing {nb_r} x {nb_c} blocks = {N_blocks} blocks, each of size {M}x{M}")

# --- (c) Allocate a 3D array on CuPy to store Fourier data as complex numbers ---
# New shape: (N_blocks, M, M)
FT_all = cp.empty((N_blocks, M, M), dtype=cp.complex64)

# --- (d) Compute the Fourier transform for each block and store as complex numbers ---
block_idx = 0
for i in range(nb_r):
    for j in range(nb_c):
        block = img_np[i*M:(i+1)*M, j*M:(j+1)*M]
        block_cp = cp.array(block, dtype=cp.float32)
        # Apply fftshift before and after fft2 so that the zero-frequency is centered.
        ft_block = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(block_cp)))
        FT_all[block_idx, :, :] = ft_block
        block_idx += 1

# --- (e) Example: Compute block-based scaling and noise variance ---
# (This section is mostly preserved; adjust if needed based on complex storage.)
# Here we compute a representative intensity at the central frequency of each block.
I0 = (cp.abs(FT_all))[:, M//2, M//2]  # shape: (N_blocks,)
I0max = cp.max(I0)
block_scaling = 1 / I0
FT_all *= block_scaling[:, cp.newaxis, cp.newaxis]  # apply scaling to all blocks
# FT_all -= cp.mean(FT_all, axis=0)  # remove mean from each block
print("Max block scaling factor:", cp.asnumpy(cp.amax(block_scaling)))

# Example noise variance definition (adjust as needed)
noise_variance = I0 * block_scaling**2 / 2

# --- (f) Compute average and variance for each frequency pixel across blocks ---
avg = cp.mean(np.abs(FT_all), axis=0)       # shape: (M, M) complex
# var = cp.var(FT_all, axis=0)         # variance of complex numbers (per component) 
std = cp.sqrt( cp.sum(FT_all.real**2 + FT_all.imag**2, axis=0)/(2 * (nb_r*nb_c)))
             # note: you may wish to handle complex variance differently
avg_cpu = avg.get()
std_cpu = std.get()
std_low_noise = cp.sqrt( np.sum(FT_all.real**2 + FT_all.imag**2 - 2 * noise_variance[:, cp.newaxis, cp.newaxis], axis=0)/(2 * (nb_r*nb_c)))
cp.nan_to_num(std_low_noise, copy=False)

import stattools
plt.plot (stattools.average_rings2d(np.mean(FT_all.real, axis=0).get()**2 + np.mean(FT_all.imag, axis=0).get()), label = "Average")
plt.plot(stattools.average_rings2d(std.get()/avg.get()), label = "Estimated ratio")
# plt.plot(stattools.average_rings2d(std_low_noise.get()/avg.get()), label = "Estimated ratio noise corrected")
plt.hlines(np.sqrt(2/np.pi), 0, 15, color='red')
plt.legend()
plt.show()

exit()
# --- (g) Compute the correlation between the real and imaginary parts ---
# For each frequency pixel (m,n) we compute the Pearson correlation coefficient across blocks.
FT_all_cpu = FT_all.get()  # shape: (N_blocks, M, M), complex

# Prepare an array to hold correlation coefficients
corr_map = np.zeros((M, M), dtype=np.float32)

for m in range(M):
    for n in range(M):
        # Extract values from all blocks at frequency (m,n)
        vals = FT_all_cpu[:, m, n]
        re_vals = vals.real
        im_vals = vals.imag
        
        # Compute correlation; avoid division by zero.
        std_re = np.std(re_vals)
        std_im = np.std(im_vals)
        if std_re > 0 and std_im > 0:
            corr = np.corrcoef(re_vals, im_vals)[0, 1]
        else:
            corr = 0.0
        corr_map[m, n] = corr

# --- (h) Plot the radial average of the correlation coefficient ---
# Compute the radial coordinate in one MxM block.
y_indices, x_indices = np.indices((M, M))
center = (M // 2, M // 2)
radii = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2)

n_bins = 20
bins = np.linspace(0, np.max(radii), n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

radial_avg_corr = np.zeros(n_bins)
for i in range(n_bins):
    mask = (radii >= bins[i]) & (radii < bins[i+1])
    if np.sum(mask) > 0:
        radial_avg_corr[i] = np.mean(corr_map[mask])

plt.figure(figsize=(8, 6))
plt.plot(bin_centers, radial_avg_corr, 'o-', label='Radial Avg Correlation')
plt.xlabel('Radial frequency')
plt.ylabel('Correlation coefficient')
plt.title('Radial Average of Correlation (Real vs Imag)')
plt.legend()
plt.show()

# --- (i) Example: Plot distributions for selected frequency positions ---
positions = [(M//2, M//2), (M//2-1, M//2), (M-2, M//2), (M-3, M//2), (M-4, M//2), (M-5, M//2)]
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# for pos, col in zip(positions, colors):
#     # Gather all coefficients at this frequency (over all blocks)
#     vals = FT_all_cpu[:, pos[0], pos[1]]
#     coeffs_real = vals.real
#     coeffs_imag = vals.imag
#     # Normalize each distribution by its maximum absolute value to ease plotting.
#     # norm_real = coeffs_real / (np.max(np.abs(coeffs_real)) if np.max(np.abs(coeffs_real)) != 0 else 1)
#     # norm_imag = coeffs_imag / (np.max(np.abs(coeffs_imag)) if np.max(np.abs(coeffs_imag)) != 0 else 1)
#     axes[0].hist(coeffs_real, bins=30, alpha=0.5, color=col, label=f'Freq {pos} Real')
#     axes[1].hist(coeffs_imag, bins=30, alpha=0.5, color=col, label=f'Freq {pos} Imag')

# axes[0].set_title('Distribution (Real Parts)')
# axes[1].set_title('Distribution (Imaginary Parts)')
# for ax in axes:
#     ax.legend()
# plt.show()

# Select the 1,000 brightest blocks (based on I0 before scaling)
I0_cpu = cp.asnumpy(I0)
n_select = min(100, I0_cpu.size)
brightest_idx = np.argsort(I0_cpu)[::-1][:n_select]

# Extract the brightest Fourier blocks into FT_brightest
FT_brightest = FT_all[brightest_idx, :, :]

# Compute the noise for each selected block:
# For each block, noise = I0max * block_scaling (amplitude), so noise power = (I0max * block_scaling)^2
noise_power_per_block = I0max * block_scaling[brightest_idx]
# Compute the average noise power over the selected blocks
avg_noise_power = cp.mean(noise_power_per_block)
print("Average noise power for the brightest blocks:", cp.asnumpy(avg_noise_power))
# fig, axes = plt.subplots(figsize=(12, 6))
# for pos, col in zip(positions, colors):
#     axes.hist(power[:, pos[0], pos[1]], bins=30, alpha=0.5, color=col, label=f'Freq {pos} power')
# plt.show()

power = cp.abs(FT_brightest)
power_mean = cp.mean(power, axis=0)
power_std = cp.std(power, axis=0)
variance = power_std**2 - avg_noise_power
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes[0, 0].set_title("Mean power spectrum")
axes[0, 0].plot(power_mean[M//2, :].get())
axes[0, 1].set_title("Power variance spectrum")
axes[0, 1].plot(variance[M//2, :].get())
axes[1, 0].set_title("Variance check")
axes[1, 0].plot(power_std[M//2, :].get()/(std_cpu[M//2, :]**2))
axes[1, 1].set_title("Rayleigh ratio")
axes[1, 1].plot((power_std[M//2, :]/power_mean[M//2, :]).get())
plt.show()
plt.errorbar(np.arange(31) - 15, power_mean[:, M//2], yerr = power_std[:, M//2], fmt='o')
plt.show()
