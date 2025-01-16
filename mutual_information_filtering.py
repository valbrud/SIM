import os
import sys
import scipy

sys.path.insert(0, os.path.abspath('../'))

import numpy as np
import OpticalSystems
from ShapesGenerator import generate_random_lines, generate_random_spheres
import tqdm
import wrappers
import matplotlib.pyplot as plt

np.random.seed(1234)

theta = np.pi / 4
alpha = np.pi / 4
dx = 1 / (8 * np.sin(alpha))
dy = dx
N = 71
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

image_number = 150

ground_truth = np.zeros((image_number, N, N))
ground_truth_ft = np.zeros((image_number, N, N), dtype=np.complex64)

images = np.zeros((image_number, N, N))
images_ft = np.zeros((image_number, N, N), dtype=np.complex64)

f_avg = np.zeros((N, N), dtype=np.float64)
Sigma = np.zeros((N, N), dtype=np.float64)
gaussian_noise = 0.1
sigma_ft = gaussian_noise * N ** 2

for i in tqdm.tqdm(range(image_number)):
    ground_truth[i] = generate_random_lines(psf_size, N, 0.1, 100, 100)
    # + generate_random_spheres(np.array((*psf_size, 2)), point_number=71, r=0.1, N=30, I=200)[:, :, N//2]
    # print(np.sum(ground_truth[i]))
    ground_truth_ft[i] = wrappers.wrapped_fftn(ground_truth[i])

    images_ft[i] = ground_truth_ft[i] * optical_system.otf
    images_ft[i] += np.random.normal(0, sigma_ft, size = f_avg.shape)
    images[i] = wrappers.wrapped_ifftn(images_ft[i])
    # plt.imshow(images[i], cmap='gray')
    # plt.show()
fa = np.mean(np.abs(ground_truth_ft), axis=0)
Sigma = np.sqrt(np.sum((np.abs(ground_truth_ft) - fa)**2, axis=0) / (image_number - 1))
variability = Sigma/fa
print(variability[N//2, N//2])
# plt.plot(fx, fa[:, N//2])
# plt.show()
# plt.imshow(np.log(1 + np.abs(fa)), cmap='gray')
# plt.show()
#
# plt.imshow(np.log(Sigma), cmap='gray')
# plt.show()

n = 30
image = np.copy(images[n])
image_ft = np.copy(np.abs(images_ft[n]))

fig, axes = plt.subplots(1, 5)
axes[0].imshow(scipy.signal.convolve(ground_truth[n], optical_system.psf, 'same'))
axes[0].set_title('Convolved')
axes[1].imshow(image)
axes[1].set_title('Noisy')


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

f = np.where(g > 10**-12, np.abs(image_ft)/(g + 10**-10), 0)
# f = np.copy(fa)
phases = np.angle(images_ft[n])
w = 0.1
wiener_ft = np.abs(g * image_ft)/(g**2 + w)
wiener_ft = image_ft[N//2, N//2]/wiener_ft[N//2, N//2] * wiener_ft
axes[2].imshow(np.abs(wrappers.wrapped_ifftn(g * wiener_ft * np.exp(1j * phases))))
axes[2].set_title('Wiener')
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
    # plt.plot(np.abs(ground_truth_ft[0, :, N//2]), label = 'ground truth')
    # plt.plot(f[:, N//2], label = 'decoupled')
    # plt.plot(wiener_ft[:, N//2], label = 'wiener')
    # plt.plot(fnew[:, N//2], label = 'full')
    # plt.legend()
    # plt.show()
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

plt.show()
# If not converged, mark those points with NaN

