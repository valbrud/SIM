import os.path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import copy
import kernels
import numpy as np
import Reconstructor
import SSNRCalculator
import hpc_utils
import matplotlib.pyplot as plt
import Apodization
kernels_test = [kernels.psf_kernel2d(3) * (1 + 1j), kernels.combined_low_pass_notch_kernel(9, 9) * (1-1j), kernels.sinc_kernel2d(1)]

def generate_OTF_matrix(illumination, illumination_reconstruction, optical_system, n_slices, dz):
    g_vector = np.zeros((n_slices, optical_system.otf.shape[0], optical_system.otf.shape[1]), dtype=np.complex128)
    for i in range(-(n_slices//2), n_slices//2 + 1):
        #low-NaA approximation for now
        aberration_strength = i * dz * optical_system.NA**2 / (4 * optical_system.nm) 
        print(aberration_strength)
        g_vector[n_slices//2 + i] = optical_system.compute_psf_and_otf(zernieke = {(2, 0): aberration_strength, (3, -1): 1 * aberration_strength, (3, 1): 1 * aberration_strength})[1]
    
    # for i, g_rec in enumerate(g_vector):
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(np.log1p(g_rec.real))
    #     ax[1].imshow(np.log1p(g_rec.imag))
    #     plt.show()

    g_matrix_sim = np.zeros((n_slices, *g_vector.shape), dtype=np.complex128)
    for i, g_em in enumerate(g_vector):
        optical_system.otf = g_em
        for j, g_rec in enumerate(g_vector):
            kernel = hpc_utils.wrapped_ifftn(g_rec)
            # kernel = kernels.combined_low_pass_notch_kernel(kernel_size_low_pass=(j+1)**2 if j%2==0 else (j+1)**2 -1, kernel_size_notch=(i+1)**2 if i%2==0 else (i+1)**2 -1) + 0.1
            # kernel = kernels.sinc_kernel2d((j+1)**3 if j%2==0 else (j+1)**2 -1) * (1 + 1j)
            kernel = kernels_test[j]
            ssnr_calculator = SSNRCalculator.SSNRSIM2D(illumination, optical_system, kernel, illumination_reconstruction=illumination_reconstruction)
            g_matrix_sim[i, j] = ssnr_calculator.dj / np.amax(np.abs(ssnr_calculator.dj))
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(np.log1p(np.imag(g_em)), cmap='gray')
            # ax[1].imshow(np.log1p(np.imag(g_rec)), cmap='gray')
            # ax[2].imshow(np.log1p(g_matrix_sim[i, j].imag), cmap='gray')
            # # plt.title(f'g_matrix_sim[{i}, {j}]')
            # plt.show()
    return g_vector, g_matrix_sim


def generate_image_vector(stack, illumination_reconstruction, optical_system, g_vector):
    reconstructor = Reconstructor.ReconstructorFourierDomain2D(illumination_reconstruction, optical_system, return_ft=True, unitary=False)
    image_vector_ft = np.zeros((len(g_vector), *stack.shape[2:]), dtype=np.complex128)
    for i, g_rec in enumerate(g_vector):
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(np.log1p(g_rec.real))
        # ax[1].imshow(np.log1p(g_rec.imag))
        # plt.show()
        # reconstructor.kernel = hpc_utils.wrapped_ifftn(g_rec)
        # kernel = kernels.sinc_kernel2d((i+1)**3 if i%2==0 else (i+1)**2 -1) * (1 + 1j)
        
        reconstructor.kernel = kernels_test[i]
        image_vector_ft[i] = reconstructor.reconstruct(stack)
    return image_vector_ft


def estimate_background(stack, illumination, optical_system, n_slices, dz):
    illumination_reconstruction = copy.deepcopy(illumination)
    for harmonic in illumination_reconstruction.harmonics:
        if not harmonic[1] ==(0, 0):
            illumination_reconstruction.harmonics[harmonic].amplitude *=10
    g_vector, g_matrix_sim = generate_OTF_matrix(illumination, illumination_reconstruction, optical_system, n_slices, dz)
    image_vector_ft = generate_image_vector(stack, illumination_reconstruction, optical_system, g_vector)
    
    # g_matrix_sim.real = np.where(np.abs(g_matrix_sim.real) < 1e-10, 1e-10, g_matrix_sim.real)
    # g_matrix_sim.imag = np.where(np.abs(g_matrix_sim.imag) < 1e-10, 1e-10, g_matrix_sim.imag)
    G = np.moveaxis(g_matrix_sim,    [0, 1], [-2, -1])  # (3,3,N,N) → (N,N,3,3)
    I = np.moveaxis(image_vector_ft,  0,     -1)        # (3,N,N)   → (N,N,3)

    Ginv = np.linalg.pinv(G, rcond=1e-10)
    Ginv = np.where(np.abs(Ginv) < 10, Ginv, 0)
    # Ginv = np.real(Ginv)
    apodization = Apodization.AutoconvolutionApodizationSIM2D(optical_system, illumination)
    apodization_mask = np.where(apodization.apodization_function > 0.5, 1, 0)
    Ginv *= apodization_mask[..., np.newaxis, np.newaxis]
    Ginv = np.nan_to_num(Ginv)
    # Ginv += 1e-1
    Nk, Ns, Nx, Ny = g_matrix_sim.shape
    g_matrix_expanded = np.zeros((Nk * Nx, Ns * Ny), dtype=np.complex128)
    g_matrix_expanded_inv = np.zeros((Nk * Nx, Ns * Ny), dtype=np.complex128)
    for i in range(Nk):
        for j in range(Ns):
            g_matrix_expanded[i *Nx : (i + 1) * Nx, j * Ny:(j + 1) * Ny] = G[..., i, j]
            g_matrix_expanded_inv[i *Nx : (i + 1) * Nx, j * Ny:(j + 1) * Ny] = Ginv[..., i, j]
    
    image_vector_ft_expaneded = np.zeros((Ns *Nx, Ny), dtype=np.complex128)
    for i in range(Ns):
        image_vector_ft_expaneded[i * Nx:(i + 1) * Nx, :] = image_vector_ft[i]

    fig, ax = plt.subplots(1, 4)
    im0 = ax[0].imshow(np.log1p(np.real(g_matrix_expanded)))
    im1 = ax[1].imshow(np.log1p(np.imag(g_matrix_expanded)))
    im2 = ax[2].imshow((np.log1p(np.real(image_vector_ft_expaneded))), cmap='gray')
    im3 = ax[3].imshow((np.log1p(np.imag(image_vector_ft_expaneded))), cmap='gray')
    plt.colorbar(im0, cmap='gray', ax=ax[0])
    plt.colorbar(im1, cmap='gray', ax=ax[1])
    plt.colorbar(im2, cmap='gray', ax=ax[2])
    plt.colorbar(im3, cmap='gray', ax=ax[3])
    plt.show()
    
    sanity_check = Ginv @ G
    fig, ax = plt.subplots(1, 4)
    im0 = ax[0].imshow(np.real(g_matrix_expanded_inv))
    im1 = ax[1].imshow(np.imag(g_matrix_expanded_inv))
    ax[2].imshow(np.real(sanity_check[..., 1, 1]), cmap='gray')
    ax[3].imshow(np.imag(sanity_check[..., 1, 1]), cmap='gray')
    plt.colorbar(im0, cmap='gray', ax=ax[0])
    plt.colorbar(im1, cmap='gray', ax=ax[1])
    plt.colorbar(im2, cmap='gray', ax=ax[2])
    plt.colorbar(im3, cmap='gray', ax=ax[3])
    plt.show()

    f = (Ginv @ I[..., np.newaxis])[..., 0]
    f = np.moveaxis(f, -1, 0)                            # (3,N,N)
    plt.imshow(np.log1p(np.real(f[1])), cmap='gray')
    plt.show()
    plt.imshow(np.log1p(np.imag(f[1])), cmap='gray')
    plt.show()
    object_vector = np.array([np.real(hpc_utils.wrapped_ifftn(apodization.apodization_function**2 * f[i])) for i in range(len(g_vector))])
    
    return object_vector

    # g_matrix_expanded += 1e-6
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(np.log1p(np.real(g_matrix_expanded)), cmap='gray')
    # ax[1].imshow(np.log1p(np.imag(g_matrix_expanded)), cmap='gray')
    # plt.show()


    # # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(np.log1p(np.real(image_vector_ft_expaneded)), cmap='gray')
    # ax[1].imshow(np.log1p(np.imag(image_vector_ft_expaneded)), cmap='gray')
    # plt.show()

   
    # g_matrix_inverse = np.linalg.pinv(g_matrix_expanded, rcond=None)
    # plt.imshow(np.log1p(np.real(g_matrix_inverse)), cmap='gray')
    # plt.show()
    # object_vector_ft_expanded = g_matrix_inverse @ image_vector_ft_expaneded
    # # object_vector_ft_expanded = np.linalg.lstsq(g_matrix_expanded, image_vector_ft_expaneded, rcond=None)[0]
    # object_vector_ft_expanded = np.linalg.lstsq(g_matrix_expanded, image_vector_ft_expaneded, rcond=None)[0]
    # plt.imshow(np.log1p(np.real(object_vector_ft_expanded)), cmap='gray')
    # plt.show()
    object_vector_ft = object_vector_ft_expanded.reshape((n_slices, Nx, Ny))
    object_vector = np.array([np.real(hpc_utils.wrapped_ifftn(object_vector_ft[i])) for i in range(len(g_vector))])
    return object_vector