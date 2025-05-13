import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import scipy
import numpy as np
import OpticalSystems
from ShapesGenerator import generate_random_lines, generate_random_spherical_particles, generate_line_grid_2d
import tqdm
import wrappers
import matplotlib.pyplot as plt
import stattools
from scipy.optimize import curve_fit

np.random.seed(1234)

theta = np.pi / 4
alpha = np.pi / 4
dx = 1 / (4 * np.sin(alpha))
dy = dx
N = 51
max_r = N // 2 * dx
NA = np.sin(alpha)
psf_size = 2 * np.array((max_r, max_r))
dx = 2 * max_r / N
dy = 2 * max_r / N
x = np.linspace(-max_r, max_r, N)
y = np.copy(x)

fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), N)
fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), N)

arg = N // 2

two_NA_fx = fx / (2 * NA)
two_NA_fy = fy / (2 * NA)

optical_system = OpticalSystems.System4f2D(alpha=alpha)
optical_system.compute_psf_and_otf((psf_size, N))

image_number = 100

ground_truth = np.zeros((image_number, N, N))
ground_truth_ft = np.zeros((image_number, N, N), dtype=np.complex64)

images = np.zeros((image_number, N, N))
images_ft = np.zeros((image_number, N, N), dtype=np.complex64)

f_avg = np.zeros((N, N), dtype=np.float64)
Sigma = np.zeros((N, N), dtype=np.float64)
gaussian_noise = 1
sigma_ft = gaussian_noise * N ** 2

img_size = (N, N)  # (height, width)
pitch = 10.0
line_width = 0.2
intensity_value = 25.0

for i in tqdm.tqdm(range(image_number)):
    ground_truth[i] = (generate_random_lines(psf_size, N, 0.1, 100, np.random.uniform(10, 100)) 
                      + generate_random_spherical_particles(np.array((*psf_size, 2)), point_number=N, r=0.1, N=40, I=100)[:, :, N // 2]
                    )
    ground_truth_ft[i] = wrappers.wrapped_fftn(ground_truth[i])
    images_ft[i] = ground_truth_ft[i] * optical_system.otf
    images_ft[i] += np.random.normal(0, sigma_ft, size = f_avg.shape)
    images[i] = wrappers.wrapped_ifftn(images_ft[i])

# plt.imshow(images[0], cmap='gray')
# plt.show()

fa = np.mean(np.abs(ground_truth_ft), axis=0)
# fa = stattools.average_rings2d(fa, (fx, fy))
# fa = np.abs(stattools.expand_ring_averages2d(fa, (fx, fy)))
# Sigma = np.sqrt(np.sum((np.abs(ground_truth_ft) - fa)**2, axis=0) / (image_number - 1))
Sigma = np.sqrt(np.sum(ground_truth_ft.real**2 + ground_truth_ft.imag**2, axis=0) / (2 * image_number)) 

ft0 = np.abs(ground_truth_ft[:, N//2, N//2])
I0 = np.abs(images_ft[:, N//2, N//2])

ground_truth_ft_rescaled = np.copy(ground_truth_ft) / ft0[:, None, None]
# Compute correlations between consecutive Fourier coefficients (ignoring the DC coefficient)
num_coeffs = ground_truth_ft_rescaled.shape[1] * ground_truth_ft_rescaled.shape[2]
# Compute complex correlation coefficients for selected row and various neighbor types

center = N // 2

def comp_corr(x, y):
    
    corr_real = np.corrcoef(x.real, y.real)[0, 1]
    corr_imag = np.corrcoef(x.imag, y.imag)[0, 1]
    corr_total = (np.abs(corr_real) + np.abs(corr_imag))/2
    return corr_total
    # x_mean = np.mean(x)
    # y_mean = np.mean(y)
    # cov = np.mean((x - x_mean) * np.conjugate(y - y_mean))
    # var_x = np.mean(np.abs(x - x_mean) ** 2)
    # var_y = np.mean(np.abs(y - y_mean) ** 2)
    # if var_x > 0 and var_y > 0:
    #     return np.abs(cov) / np.sqrt(var_x * var_y)
    # else:
    #     return np.nan
    
fig, axes = plt.subplots(1, 3)
# (1) Direct Neighbors: left, right, up, down
corr_left = []
fx_left = []
for j in range(1, N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center, j - 1]
    corr_left.append(comp_corr(x, y))
    fx_left.append(fx[j])

corr_right = []
fx_right = []
for j in range(0, N - 1):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center, j + 1]
    corr_right.append(comp_corr(x, y))
    fx_right.append(fx[j])

corr_up = []
for j in range(N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center - 1, j]
    corr_up.append(comp_corr(x, y))

corr_down = []
for j in range(N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center + 1, j]
    corr_down.append(comp_corr(x, y))

axes[0].plot(fx_left, corr_left, marker='o', label='Left')
axes[0].plot(fx_right, corr_right, marker='o', label='Right')
axes[0].plot(fx, corr_up, marker='o', label='Up')
axes[0].plot(fx, corr_down, marker='o', label='Down')
axes[0].set_xlabel("fx")
axes[0].set_ylabel("Complex Correlation")
axes[0].set_title("Correlation with Direct Neighbors")
axes[0].legend()

# (2) Diagonal Neighbors: up-left, up-right, down-left, down-right
corr_up_left = []
fx_up_left = []
for j in range(1, N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center - 1, j - 1]
    corr_up_left.append(comp_corr(x, y))
    fx_up_left.append(fx[j])

corr_up_right = []
fx_up_right = []
for j in range(0, N - 1):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center - 1, j + 1]
    corr_up_right.append(comp_corr(x, y))
    fx_up_right.append(fx[j])

corr_down_left = []
fx_down_left = []
for j in range(1, N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center + 1, j - 1]
    corr_down_left.append(comp_corr(x, y))
    fx_down_left.append(fx[j])

corr_down_right = []
fx_down_right = []
for j in range(0, N - 1):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center + 1, j + 1]
    corr_down_right.append(comp_corr(x, y))
    fx_down_right.append(fx[j])

axes[1].plot(fx_up_left, corr_up_left, marker='o', label='Up-Left')
axes[1].plot(fx_up_right, corr_up_right, marker='o', label='Up-Right')
axes[1].plot(fx_down_left, corr_down_left, marker='o', label='Down-Left')
axes[1].plot(fx_down_right, corr_down_right, marker='o', label='Down-Right')
axes[1].set_xlabel("fx")
axes[1].set_ylabel("Complex Correlation")
axes[1].set_title("Correlation with Diagonal Neighbors")
axes[1].legend()
# (3) Neighbors at distance 2: vertical (up 2, down 2) and horizontal (left 2, right 2)
corr_left2 = []
fx_left2 = []
for j in range(2, N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center, j - 2]
    corr_left2.append(comp_corr(x, y))
    fx_left2.append(fx[j])

corr_right2 = []
fx_right2 = []
for j in range(0, N - 2):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center, j + 2]
    corr_right2.append(comp_corr(x, y))
    fx_right2.append(fx[j])

corr_up2 = []
for j in range(N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center - 2, j]
    corr_up2.append(comp_corr(x, y))

corr_down2 = []
for j in range(N):
    x = ground_truth_ft_rescaled[:, center, j]
    y = ground_truth_ft_rescaled[:, center + 2, j]
    corr_down2.append(comp_corr(x, y))

axes[2].plot(fx_left2, corr_left2, marker='o', label='Left 2')
axes[2].plot(fx_right2, corr_right2, marker='o', label='Right 2')
axes[2].plot(fx, corr_up2, marker='o', label='Up 2')
axes[2].plot(fx, corr_down2, marker='o', label='Down 2')
axes[2].set_xlabel("fx")
axes[2].set_ylabel("Complex Correlation")
axes[2].set_title("Correlation with Neighbors at Distance 2")
axes[2].legend()
# plt.show()

# exit() 

data = ground_truth_ft_rescaled.reshape(image_number, num_coeffs)
# Exclude the first coefficient (assumed DC)
data = data[:, 1:]
ncoeff = data.shape[1]

real_corrs = []
imag_corrs = []
complex_corrs = []

for k in range(ncoeff - 1):
    x = data[:, k]
    y = data[:, k + 1]
    # Correlation for real parts
    corr_real = np.corrcoef(x.real, y.real)[0, 1]
    real_corrs.append(corr_real)
    # Correlation for imaginary parts
    corr_imag = np.corrcoef(x.imag, y.imag)[0, 1]
    imag_corrs.append(corr_imag)
    # Complex correlation: compute covariance and normalize by the standard deviations
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.mean((x - x_mean) * np.conjugate(y - y_mean))
    var_x = np.mean(np.abs(x - x_mean) ** 2)
    var_y = np.mean(np.abs(y - y_mean) ** 2)
    if var_x > 0 and var_y > 0:
        corr_complex = np.abs(cov) / np.sqrt(var_x * var_y)
    else:
        corr_complex = np.nan
    complex_corrs.append(corr_complex)

# Create single figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].plot(np.arange(ncoeff - 1), real_corrs, marker='o')
axes[0].set_title("Correlation of Real Parts (Adjacent Coefficients)")
axes[0].set_xlabel("Coefficient index")
axes[0].set_ylabel("Pearson correlation")
axes[0].grid(True)

axes[1].plot(np.arange(ncoeff - 1), imag_corrs, marker='o', color='g')
axes[1].set_title("Correlation of Imaginary Parts (Adjacent Coefficients)")
axes[1].set_xlabel("Coefficient index")
axes[1].set_ylabel("Pearson correlation")
axes[1].grid(True)

axes[2].plot(np.arange(ncoeff - 1), complex_corrs, marker='o', color='r')
axes[2].set_title("Complex Correlation Magnitude (Adjacent Coefficients)")
axes[2].set_xlabel("Coefficient index")
axes[2].set_ylabel("Correlation magnitude")
axes[2].grid(True)

plt.tight_layout()
plt.show()

fa_intensity_corrected = np.mean(np.abs(ground_truth_ft_rescaled), axis=0)
# Sigma_intensity_corrected = np.sqrt(np.sum((np.abs(ground_truth_ft_rescaled) - fa_intensity_corrected)**2, axis=0) / (image_number - 1))
Sigma_intensity_corrected = np.sqrt(np.sum(ground_truth_ft_rescaled.real**2 + ground_truth_ft_rescaled.imag**2, axis=0) / (2 * image_number)) 
# Sigma = stattools.average_rings2d(Sigma, (fx, fy))
# Sigma = stattools.expand_ring_averages2d(Sigma, (fx, fy))

Iavg = np.mean(np.abs(images_ft), axis=0)
scaling_factor = 1 / I0
images_ft_rescaled = np.copy(images_ft) * scaling_factor[:, None, None]
sigmas_ft_rescaled = sigma_ft * scaling_factor[:, None, None]
Iavg_intensity_corrected = np.mean(np.abs(images_ft_rescaled), axis=0)

Sigma_estimated_total = np.sqrt(np.sum(images_ft_rescaled.real**2 + images_ft_rescaled.imag**2, axis=0) / (2 *image_number)) 
# Sigma_estimated = np.sqrt(np.sum(images_ft_rescaled.real**2 + images_ft_rescaled.imag**2, axis=0) / (2 *image_number) - (sigma_ft / np.mean(I0))**2/2)
Sigma_estimated_low_noise_limit = np.sqrt( np.sum(images_ft_rescaled.real**2 + images_ft_rescaled.imag**2 - sigmas_ft_rescaled**2, axis=0)/(2 * image_number))
Sigma_estimated_high_noise_limit = np.sqrt(np.sum(1/sigmas_ft_rescaled**2 * (1 - (images_ft_rescaled.real**2 + images_ft_rescaled.imag**2) / (sigmas_ft_rescaled**2)), axis=0) / 
                                           np.sum(1/sigmas_ft_rescaled**4 * (1 - 2* (images_ft_rescaled.real**2 + images_ft_rescaled.imag**2) / (sigmas_ft_rescaled**2)), axis=0) )

np.nan_to_num(Sigma_estimated_low_noise_limit, copy=False, nan=0)
np.nan_to_num(Sigma_estimated_high_noise_limit, copy=False, nan=0)

Sigma_low_noise_ra = stattools.average_rings2d(Sigma_estimated_low_noise_limit, (fx, fy))
Sigma_high_noise_ra = stattools.average_rings2d(Sigma_estimated_high_noise_limit, (fx, fy))
Sigma0 = np.where(Sigma_low_noise_ra > 0.005, Sigma_low_noise_ra, Sigma_high_noise_ra)

plt.plot (fx[N//2:], stattools.average_rings2d(Sigma_intensity_corrected * optical_system.otf, (fx, fy)), label = "Sigma true")
plt.plot(fx[N//2:], Sigma_low_noise_ra, label = "Sigma low noise limit")
plt.plot(fx[N//2:], Sigma_high_noise_ra, label = "Sigma high noise limit")
plt.plot(fx[N//2:], Sigma0, 'd', label = "Sigma initial guess")
plt.legend()
plt.show()

# Sigma_noise_corrected = np.sqrt(np.where(Sigma_estimated**2 - sigma_ft**2 > 0, Sigma_estimated**2 - sigma_ft**2, sigma_ft**2))
# plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma_estimated/Iavg_intensity_corrected)), label = "Sigma estimated true")
# plt.show()
# print(f"fa true = {fa[:, N//2]}, Sigma true = {Sigma[:, N//2]}")
Iavg_true = fa_intensity_corrected * np.abs(optical_system.otf)
plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma/fa)), label = "True sigma not corrected")
plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma_intensity_corrected/fa_intensity_corrected)), label = "True sigma intensity corrected")
plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma_estimated_total/Iavg_intensity_corrected)), label = "Estimated total variance")
plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma_estimated_low_noise_limit/Iavg_true)), label = "Sigma estimated low noise limit")
plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma_estimated_high_noise_limit/Iavg_true)),  label = "Sigma estimated high noise limit")
# plt.plot(fx[N//2+1:], stattools.average_rings2d((Sigma_estimated_high_noise_limit/Iavg_true)), label = "Sigma estimated noise corrected")
plt.legend()
plt.hlines(np.sqrt(2/np.pi), 0, fx[-1], color='r', label='Expected ratio')
plt.show() 

def fit_sigma(fx, fc, alpha):
    return np.sqrt((4 - np.pi) / np.pi) / (fx/fc) ** alpha

def fit_bandwidth(fx, A, fc, alpha):
    return A / (fx/fc)**alpha
# # Extract the values where Sigma_estimated/Iavg_intensity_corrected is less than 0.53
# mask = np.abs(stattools.average_rings2d(Sigma_estimated_low_noise_limit / Iavg_intensity_corrected) - 0.8) < 0.03
fx_fit = fx[N//2:]
# Sigma_fit = stattools.average_rings2d(Sigma_estimated_low_noise_limit)[mask]   
Sigma_fit = Sigma0

# plt.plot(fx_fit, stattools.average_rings2d(Sigma_estimated_low_noise_limit / optical_system.otf)[mask], label='Sigma fit')
# plt.plot(fx_fit, stattools.average_rings2d(Sigma_intensity_corrected )[mask], label='True variance')

# Perform the curve fitting
popt, _ = curve_fit(fit_bandwidth, fx_fit, Sigma_fit, nan_policy='omit')  
# popt, _ = curve_fit(fit_sigma, fx_fit,  stattools.average_rings2d(fa_intensity_corrected)[mask]*0.52)  
A, fc, alpha = popt
print(f"fc = {fc}, alpha = {alpha}")

fitted_variance = fit_bandwidth(fx_fit, A, fc, alpha)
plt.plot(fx_fit, fitted_variance, label='Fitted function')
plt.plot(fx_fit, Sigma0, label='Initial guess')
plt.plot(fx[N//2+1:], 0.52* stattools.average_rings2d(fa_intensity_corrected) * optical_system.otf[N//2, N//2+1:], label='true average')
# plt.plot(fx_fit, np.sqrt((4 - np.pi) / np.pi) / (fx_fit/fc) / (), label='Common sense fit')
plt.legend()
plt.show()



# exit()

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# positions = [(N//2, N//2), (N//2-1, N//2), (N-2, N//2), (N-3, N//2), (N-4, N//2), (N-5, N//2)]
# colors = ['r', 'g', 'b', 'c', 'm', 'y']
# for pos, col in zip(positions, colors):
#     # Gather all coefficients at this frequency (over all blocks)
#     coeffs_real = images_ft[:, pos[0], pos[1]].real
#     coeffs_imag = images_ft[:, pos[0], pos[1]].imag
#     # Normalize each distribution by its maximum absolute value
#     norm_real = coeffs_real / (np.max(np.abs(coeffs_real)) if np.max(np.abs(coeffs_real)) != 0 else 1)
#     norm_imag = coeffs_imag / (np.max(np.abs(coeffs_imag)) if np.max(np.abs(coeffs_imag))!=0 else 1)
#     axes[0].hist(norm_real, bins=30, alpha=0.5, color=col, label=f'Freq {round(fx[pos[0]], 2), round(fy[pos[1]], 2)} Real')
#     axes[1].hist(norm_imag, bins=30, alpha=0.5, color=col)

# axes[0].legend()
# plt.show()
# # exit()

# fa = np.where(optical_system.otf > 10**-12, np.abs(np.mean(images_ft/np.abs(optical_system.otf), axis=0)), 0)
# fa = stattools.average_rings2d(fa, (fx, fy))
# fa = stattools.expand_ring_averages2d(fa, (fx, fy))
# Sigma = np.where(optical_system.otf > 10**-12, np.sqrt(np.sum((np.abs(images_ft) - fa * np.abs(optical_system.otf))**2, axis=0) / (image_number - 1)
#                                                                                                     - sigma_ft**2), 0)
# Sigma = stattools.average_rings2d(Sigma, (fx, fy))
# Sigma = stattools.expand_ring_averages2d(Sigma, (fx, fy))

# print(f"fa estimated = {fa[:, N//2]}, Sigma = {Sigma[:, N//2]}")
# plt.plot(fx, fa[:, N//2])
# plt.show()
# plt.imshow(np.log(1 + np.abs(fa)), cmap='gray')
# plt.show()
#
# plt.imshow(np.log(Sigma), cmap='gray')
# plt.show()


def dMdf(i, f, fa, Sigma, sigma, g):
    return (
            g * (g ** 2 * (f * g - i) ** 3 * Sigma**4
            + g * (f * g - i) * Sigma**2 * ((f - fa) * (2 * f * g + fa * g - 3 * i) - 2 * g * Sigma**2) * sigma**2
            + ((f - fa)**2 * ((f + fa) * g - 2 * i) + 2 * (i - f * g) * Sigma**2) * sigma**4)
            - sigma**2 * (g ** 2 * Sigma **2 + sigma**2) * (g * (f * g - i) * Sigma**2
            + (f - fa) * sigma**2) * np.log(1 + g**2 * Sigma**2 / sigma**2)
    )

# def dMdf(i, f, fa, Sigma, sigma, g):
#     return (
#             (f - fa) ** 2 * ((f + fa) * g - 2 * i) * sigma ** 4
#             + (f * g - i) * Sigma ** 2 * sigma ** 2 * ((f - fa) * g * (2 * f * g + fa * g - 3 * i) - 2 * sigma ** 2)
#             + g ** 2 * (f * g - i) * Sigma ** 4 * ((-f * g + i) ** 2 - 2 * sigma ** 2)
#     )
#

def d2Mdf2(i, f, fa, Sigma, sigma, g):
    return (
            (g ** 2 * Sigma ** 2 + sigma ** 2)
            * (g * (3 * g * (-f * g + i) ** 2 * Sigma ** 2 +
               ((f - fa) * (3 * f * g + fa * g - 4 * i) - 2 * g * Sigma ** 2) * sigma ** 2) -
            sigma**2 * (g**2 * Sigma ** 2 + sigma**2) * np.log(1 + g**2 * Sigma**2 / sigma**2))
    )

# def d2Mdf2(i, f, fa, Sigma, sigma, g):
#     return (
#             (g ** 2 * Sigma ** 2 + sigma ** 2)
#             * (3 * g * (-f * g + i) ** 2 * Sigma ** 2 +
#                ((f - fa) * (3 * f * g + fa * g - 4 * i) - 2 * g * Sigma ** 2) * sigma ** 2)
#     )



# Define the maximum number of iterations and a tolerance for convergence
max_iters = 250
tolerance = 1e-14

# plt.imshow(i, cmap='gray')
# plt.show()

g = np.abs(optical_system.otf)

# f = np.where(g > 10**-12, np.abs(image_ft)/(g + 10**-10), 0)
# f = np.copy(fa)

fitted_variance = Sigma0
fitted_variance[0] = np.inf
nmin = np.argmin(I0)
nmax = np.argmax(I0)
nrand = np.random.randint(0, image_number)
ns = (nmin, nmax, nrand)
fig, axes = plt.subplots(3, 5)

for i in range(len(ns)):
    image = np.copy(images[ns[i]])
    phases = np.angle(images_ft[ns[i]])
    image_ft = np.copy(np.abs(images_ft[ns[i]]))

    axes[i, 0].imshow(scipy.signal.convolve(ground_truth[ns[i]], optical_system.psf, 'same'))
    axes[i, 0].set_title(f'Convolved, I0 = {I0[ns[i]]:.2f}')
    axes[i, 1].imshow(image)
    axes[i, 1].set_title('Noisy')

    c = 0.1
    wiener_ft = np.abs(g * image_ft)/(g**2 + c)
    wiener_ft = image_ft[N//2, N//2] * wiener_ft
    axes[i, 2].imshow(np.abs(wrappers.wrapped_ifftn(g * wiener_ft * np.exp(1j * phases))))
    axes[i, 2].set_title('Normal Wiener')

    Sigma2 = stattools.expand_ring_averages2d(fitted_variance, (fx, fy))
    wiener_ft_smart = np.abs(g * image_ft)/(g**2 * (1 + sigmas_ft_rescaled[ns[i]]**2/Sigma2))
    wiener_ft_smart = image_ft[N//2, N//2] * wiener_ft_smart
    axes[i, 3].imshow(np.abs(wrappers.wrapped_ifftn(g * wiener_ft_smart * np.exp(1j * phases))))
    axes[i, 3].set_title('Wiener + Bayesian')
    axes[i, 4].plot(sigmas_ft_rescaled[ns[i], 0, 0]*(g**2)[N//2:], label = 'sigma')
    axes[i, 4].plot(fitted_variance[N//2:], label = 'fitted variance')
    axes[i, 4].legend()

    fig.tight_layout()
plt.show()

exit()

    # Newton-Raphson iteration
sigma = sigma_ft * np.ones(f.shape)
Sigma = np.where(Sigma < 10**-12, 10**-18, Sigma)

for _ in range(max_iters):
    # plt.plot(i[:, N//2])
    # plt.plot(f[:, N//2])
    # plt.plot(wiener_ft[:, N//2])
    # plt.show()
    # Compute the function value and its derivative
    dm = dMdf(image_ft, f, fa, Sigma, sigma_ft, g)
    d2m = d2Mdf2(image_ft, f, fa, Sigma, sigma_ft, g)
    print(f'i = {image_ft[N//2, N//2]}, f = {f[N//2, N//2]}, fa = {fa[N//2, N//2]}, Sigma = {Sigma[N//2, N//2]}, sigma = {sigma[N//2, N//2]}, g = {g[N//2, N//2]}')
    # Update k using the Newton-Raphson method
    delta_f = np.where((np.abs(d2m) > 10**-12) * (g > 10**-10), -dm / d2m, 0)
    delta_test_low_sigma = (fa * g - i) * sigma**2 * ((-fa * g + i) ** 2             + (g**2 * Sigma**2 + sigma**2) * np.log(1 + g**2 * Sigma**2 / sigma**2)) / \
       ((g**2 * Sigma**2 + sigma**2) * g * ((-fa * g + i) ** 2 + 2 * g**2 * Sigma**2 + (g**2 * Sigma**2 + sigma**2) * np.log(1 + g**2 * Sigma**2 / sigma**2)))
    # delta_test_low_Sigma = (g * (fa * g - i) * Sigma**2 * (g**2 * (i - fa * g)**2 * Sigma**2) - 2 * g**2 * Sigma**2 * sigma**2 - 2 * sigma**4 - (g**2 * Sigma**2 * sigma**2 + sigma**4) * np.log(1 + g**2 * Sigma**2 / sigma**2)) / \
    #                        ((g ** 2 * Sigma**2 + sigma**2) * (g**2 * Sigma**2 * (3 * (i - fa * g) **2 - 2 * sigma**2) - (g ** 2 * sigma**2 * Sigma**2 + sigma**4) * np.log(1 + g**2 * Sigma**2 / sigma**2)))
    print('f_val = ', dm[N//2, N//2], 'delta_f = ', d2m[N//2, N//2])
    print("zeroth oder = ", (i/g)[N//2, N//2])
    print("first correction = ", delta_f[N//2, N//2])
    print("first correction theory = ", delta_test_low_sigma[N//2, N//2])
    print("low sigma mode = ", ((fa - i/g)*sigma**2/g/(g**2 * Sigma**2 + sigma**2))[N//2, N//2])
    f += delta_f
    # plt.imshow(np.abs(wrappers.wrapped_ifftn(np.abs(f) * np.exp(1j * phases))))
    # plt.show()

    # Check for convergence (element-wise)
    if np.all(np.abs(delta_f) < tolerance):
        break

image_updated_ft = f * np.exp(1j * phases)
axes[3].set_title('Decoupled')
axes[3].imshow(np.abs(wrappers.wrapped_ifftn(g * image_updated_ft)))


def M1(sigma, Sigma, g):
    return 1/2 * np.sum(np.log((sigma ** 2 + Sigma ** 2 * g ** 2)/sigma ** 2))

def D(sigma, Sigma, g, i, f, fa):
    Dp = (i - fa * g)**2 / (2 * (g **2 * Sigma ** 2 + sigma ** 2))
    Dp = np.sum(np.where(g > 10 ** - 12, Dp, 0))
    Dm = np.sum(np.where(g > 10**-12, (i - g * f)**2 / (2 * sigma ** 2), 0))
    return Dp - Dm

def A0(M1, D, Sigma, g):
    return np.where(g > 10**-12, (M1 + D) / (Sigma ** 2 * g ** 2), 0)

def A1(M1, D, sigma):
    return (1 + M1 + D) / (sigma ** 2)

def C0(A0, fa):
    return np.where(g > 10**-12, A0 * fa, 10**-14)

def C1(A1, g, i):
    return np.where(g > 10**-12, A1 * i / g, 10**-14)

max_iters = 250
tolerance = 1e-8

g = np.abs(optical_system.otf)
i = np.abs(images_ft[n])
# fnew = np.abs(wrappers.wrapped_fftn(i)/(g+10**-5))
# fnew = np.where(g > 10**-12, np.abs(image_ft)/(g + 10**-10), 10**-14)
fnew = fa
# plt.plot(fa[:, N//2])
# plt.plot(np.abs(ground_truth_ft[0][:, N//2]), label = 'gt')
# plt.legend()
# plt.show()
# Iterative solution
for _ in range(max_iters):
    fold = fnew
    m1 = M1(sigma, Sigma, g)
    d = D(sigma, Sigma, g, i, fold, fa)
    a0 = A0(m1, d, Sigma, g)
    a1 = A1(m1, d, sigma)
    c0 = C0(a0, fa)
    c1 = C1(a1, g, i)
    fnew = np.where(g > 10**-12, (c1 + c0) / (a1 + a0), 10**-14)
    print(f'fold={fold[N//2, N//2]}, fnew={fnew[N//2, N//2]}')
    delta = fnew - fold
    if np.all(np.abs(delta) < tolerance):
        break

full = np.abs(wrappers.wrapped_ifftn(g * fnew * np.exp(1j * phases)))

axes[4].imshow(full, )
axes[4].set_title('Full')

maxPft = np.where(g > 10**-10, ((i/g) * Sigma**2 * g**2 + sigma**2 * fa) / (sigma**2 + Sigma**2 * g**2), 0)

maxP = np.abs(wrappers.wrapped_ifftn(maxPft * np.exp(1j * phases)))
axes[5].imshow(scipy.signal.convolve(maxP, optical_system.psf, 'same'))
axes[5].set_title('Most probable')
plt.show()

plt.plot(np.abs(ground_truth_ft[n, :, N//2]), label = 'ground truth')
plt.plot(f[:, N//2], label = 'decoupled')
plt.plot(wiener_ft[:, N//2], label = 'wiener')
plt.plot(fnew[:, N//2], label = 'full')
plt.plot(maxPft[:, N//2], label = 'most probable')
# plt.plot(maxPft[:, N//2] - fnew[:, N//2], label = 'difference')
# plt.plot(np.where(g > 10**-10, i/g, 0)[:, N//2], label= 'Simple division')
plt.legend()
plt.show()
plt.imshow(np.abs(maxPft - fnew), vmin = 0)
plt.show()
# If not converged, mark those points with NaN

