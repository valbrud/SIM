import os.path
import sys

print(__file__)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys
import utils
import tifffile
import pickle
# --- Imports from your simulation modules ---
from OpticalSystems import System4f2D, System4f3D
from SIMulator import SIMulator2D, SIMulator3D
from config.BFPConfigurations import BFPConfiguration
from Illumination import IlluminationPlaneWaves2D, IlluminationNonLinearSIM2D, IlluminationPlaneWaves3D
import ShapesGenerator
from Reconstructor import ReconstructorFourierDomain2D, ReconstructorSpatialDomain2D, ReconstructorFourierDomain3D, ReconstructorSpatialDomain3DSliced
from kernels import sinc_kernel2d, psf_kernel2d, angular_notch_kernel
from WienerFiltering import filter_true_wiener_sim, filter_flat_noise_sim, filter_constant, filter_simulated_object_wiener
import hpc_utils
import SSNRCalculator
import kernels 
import windowing
import Apodization  

class TestReconstruction2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        # Set simulation parameters similar to the provided example.
        self.N = 101
        self.alpha = 2 * np.pi / 5
        # self.theta = 2 * np.pi / 12
        self.nmedium = 1.5
        self.theta = np.arcsin(0.9 * np.sin(self.alpha))
        self.dimensions = (1, 1)
        NA = self.nmedium * np.sin(self.alpha)
        self.dx = 1 / (8 * NA)
        self.max_r = self.N // 2 * self.dx

        self.psf_size = np.array((2 * self.max_r, 2 * self.max_r))
        self.x = np.linspace(-self.max_r, self.max_r, self.N)
        y = np.copy(self.x)

        self.image = ShapesGenerator.generate_random_lines(
            image_size=self.psf_size,
            point_number=self.N,
            line_width=0.4,
            num_lines=500,
            intensity=10**2
        )

        # self.image = 100 * ShapesGenerator.make_circle_grid(self.N, radius=1)

        # self.image = ShapesGenerator.generate_random_spherical_particles(
        #     image_size=self.psf_size,
        #     point_number=self.N,
        #     r = 0.1,
        #     N = 10, 
        #     I = 100
        # )
        # self.image = utils.introduce_field_aberrations(self.image)
        # self.image = utils.introduce_field_aberrations(self.image)
        # image = utils.radial_fade(self.image, 0.5, 2)
        # image += 100
        image = self.image
        plt.title("Image")
        plt.imshow(image, vmin=0)
        plt.show()

        self.optical_system = System4f2D(alpha=self.alpha, refractive_index=self.nmedium)
        self.optical_system.compute_psf_and_otf((self.psf_size, self.N))
        plt.imshow(self.optical_system.psf[self.N//2-10:self.N//2+10, self.N//2-10:self.N//2+10])
        # plt.title("PSF")
        plt.show()
        # self.image = 10**5 * np.ones(self.optical_system.psf.shape)

        self.widefield = scipy.signal.convolve(self.image, self.optical_system.psf, mode='same')
        # plt.title("Widefield image")
        # plt.imshow(self.widefield)
        # plt.show()
        
        configurations = BFPConfiguration(refraction_index=1.5)
        illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(
            self.theta, 1, 0, 3, Mt=1
        )
        self.illumination = IlluminationPlaneWaves2D.init_from_3D(
            illumination_3waves3d, self.dimensions
        )

        # spatial_shifts = np.array(((0., 0.), (1, 0), (2, 0)))
        # spatial_shifts /= (3 * 2 * self.nmedium * np.sin(self.theta))
        # self.illumination.spatial_shifts = spatial_shifts
        self.illumination.set_spatial_shifts_diagonally()
        plt.imshow(self.illumination.get_illumination_density(coordinates=(self.x, self.x)))
        plt.show()

        # Create the simulator and generate simulated images.
        self.simulator = SIMulator2D(self.illumination, self.optical_system, readout_noise_variance=1)
        self.sim_images = self.simulator.generate_noiseless_sim_images(self.image)
        self.sim_images_distorted = np.copy(self.sim_images)
        # for r in range(self.illumination.Mr):
        #     for n in range(self.illumination.Mt):
        #         image = self.sim_images[r, n]
                # image_distorted = utils.introduce_field_aberrations(image)
                # image_distorted = utils.introduce_field_aberrations(image_distorted)
                # self.sim_images_distorted[r, n] = image_distorted
                # self.sim_images_distorted[r, n] = utils.radial_fade(image_distorted, 0.5, 2)
                # self.sim_images_distorted[r, n] += 10
                # plt.title(f"Simulated image{r, n}")
                # plt.imshow(image_distorted)
                # plt.show()

        self.noisy_images = self.simulator.add_noise(self.sim_images_distorted)

    def test_widefield_reconstruction(self):
        reconstructor = ReconstructorFourierDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system
        )
        reconstructed_image = reconstructor.get_widefield(self.sim_images_distorted)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield)
        axes[0].set_title("Widefield")
        axes[1].imshow(reconstructed_image)
        axes[1].set_title("Reconstructed Widefield")
        plt.show()
        plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(reconstructed_image))))
        plt.title("Reconstructed Widefield FFT")
        plt.show()

    def test_fourier_reconstruction(self):
        # self.sim_images += np.random.normal(0, 20, self.sim_images.shape)

        fourier_reconstructor = ReconstructorFourierDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(9, (self.dx, self.dx)),
            # regularization_filter=self.optical_system.otf**2 + 0.01
            # apodization_filter =
        )
        # Reconstruct the image.
        reconstructed_image = fourier_reconstructor.reconstruct(self.sim_images_distorted)

        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(self.widefield)
        # axes[1].imshow(reconstructed_image)
        # plt.show()
        calc = SSNRCalculator.SSNRSIM2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            readout_noise_variance=1,
            kernel = psf_kernel2d(9, (self.dx, self.dx))
        )            
        filtered, *_  = filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed_image), calc)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(reconstructed_image))))
        axes[0].set_title("Reconstructed FT")
        axes[1].imshow(np.log1p(np.abs((filtered))))
        axes[1].set_title("Filtered FT")
        plt.show()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(np.abs(reconstructed_image[self.N//2 - 100:self.N//2 + 100, self.N//2 - 100:self.N//2 + 100]))
        axes[0].set_title("Reconstructed")
        axes[1].imshow(np.abs(hpc_utils.wrapped_ifftn(filtered)))
        axes[1].set_title("Filtered")
        plt.show()


    def test_spatial_reconstruction(self):
        spatial_reconstructor = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
        )
        # Reconstruct the image.
        reconstructed_image = spatial_reconstructor.reconstruct(self.sim_images_distorted)
        plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(reconstructed_image))))
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield)
        axes[1].imshow(reconstructed_image)
        plt.show()
        calc = SSNRCalculator.SSNRSIM2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            readout_noise_variance=0.1,
            kernel = psf_kernel2d(1, (self.dx, self.dx))
        )            
        filtered= filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed_image), calc)

    def test_spatial_reconstruction_finite_kernel(self):
        spatial_reconstructor = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(9, (self.dx, self.dx))
        )
        # Reconstruct the image.
        plt.title("Spatial-domain reconstruction with finite kernel")
        reconstructed_image = spatial_reconstructor.reconstruct(self.sim_images_distorted)
        plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(reconstructed_image))))
        plt.show()
        plt.imshow(reconstructed_image[self.N//2 - 100:self.N//2 + 100, self.N//2 - 100:self.N//2 + 100])
        plt.show()
        calc = SSNRCalculator.SSNRSIM2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            readout_noise_variance=0.1,
            kernel = psf_kernel2d(9, (self.dx, self.dx))
        )            
        filtered = filter_true_wiener_sim(hpc_utils.wrapped_fftn(reconstructed_image), calc)
 

    def test_compare_kernel_size_effect(self):
        spatial_reconstructor1 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(1, (self.dx, self.dx))
        )

        spatial_reconstructor3 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(3, (self.dx, self.dx))
        )
        spatial_reconstructor5 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(5, (self.dx, self.dx))
        )
        spatial_reconstructor7 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(7, (self.dx, self.dx))
        )

        # self.sim_images += np.random.normal(0, 2, self.sim_images.shape)
        reconstructed_image1 = spatial_reconstructor1.reconstruct(self.noisy_images)
        reconstructed_image3 = spatial_reconstructor3.reconstruct(self.noisy_images)
        reconstructed_image5 = spatial_reconstructor5.reconstruct(self.noisy_images)
        reconstructed_image7 = spatial_reconstructor7.reconstruct(self.noisy_images)

        fig, axes = plt.subplots(1, 4)
        fig.suptitle("Spatial-domain reconstruction with finite kernels")
        axes[0].set_title("Kernel size 1")
        axes[1].set_title("Kernel size 3")  
        axes[2].set_title("Kernel size 5")
        axes[3].set_title("Kernel size 7")
        axes[0].imshow(reconstructed_image1)
        axes[1].imshow(reconstructed_image3)
        axes[2].imshow(reconstructed_image5)
        axes[3].imshow(reconstructed_image7)
        plt.show()

    def test_notch_kernel_effect(self):
        self.sim_images += np.random.normal(0, 2, self.sim_images.shape)
        spatial_reconstructor7 = ReconstructorSpatialDomain2D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=psf_kernel2d(7, (self.dx, self.dx))
        )

        self.sim_images += np.random.normal(0, 2, self.sim_images.shape)
        reconstructed_image7 = spatial_reconstructor7.reconstruct(self.sim_images)
        plt.imshow(reconstructed_image7)
        plt.show()
        notch_kernel = angular_notch_kernel(11, 6, 0.25, theta0=0)
        notch_filter = hpc_utils.wrapped_fftn(notch_kernel)
        ssnr_calc = SSNRCalculator.SSNRSIM2D(self.illumination, self.optical_system, psf_kernel2d(7, (self.dx, self.dx)))
        plt.imshow(np.log1p(1 + 10**2 * ssnr_calc.dj))
        plt.show()
        plt.imshow(np.log1p(1 + 10**2 * (ssnr_calc.dj*utils.expand_kernel(notch_filter, ssnr_calc.dj.shape)).real))
        plt.show()   

        reconstructed_image7 = scipy.signal.convolve(reconstructed_image7, notch_kernel, mode='same')
        plt.imshow(reconstructed_image7)
        plt.show()

class TesReconstruction3D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)
        # Set simulation parameters similar to the provided example.
        self.N = 101
        self.alpha = 2 * np.pi / 5
        # self.theta = 2 * np.pi / 12
        self.nmedium = 1.5
        self.theta = np.arcsin(0.9 * np.sin(self.alpha))
        self.dimensions = (1, 1, 0)
        NA = self.nmedium * np.sin(self.alpha)
        self.dx = 1 / (8 * NA)
        self.dz = 1 / (4 * self.nmedium * (1 - np.cos(self.alpha)))
        self.max_r = self.N // 2 * self.dx
        self.max_z = self.N // 2 * self.dz
        self.psf_size = np.array((2 * self.max_r, 2 * self.max_r, 2 * self.max_z))

        self.x = np.linspace(-self.max_r, self.max_r, self.N)
        self.z = np.linspace(-self.max_z, self.max_z, self.N)

        y = np.copy(self.x)

        if not os.path.exists(current_dir + '/test_data'):
            os.makedirs(current_dir + '/test_data')
        if not os.path.exists(current_dir + '/test_data/lines3d.pkl'):
            self.image = ShapesGenerator.generate_random_lines(
                image_size=self.psf_size,
                point_number=self.N,
                line_width=0.4,
                num_lines=500,
                intensity=10**4
            )
            pickle.dump(self.image, open(current_dir + '/test_data/lines3d.pkl', 'wb'))
        else:
            self.image = pickle.load(open(current_dir + '/test_data/lines3d.pkl', 'rb'))
            
        # self.image = ShapesGenerator.generate_random_spherical_particles(
        #     image_size=self.psf_size,
        #     point_number=self.N,
        #     radius=0.2,
        #     num_particles= 500, 
        #     intensity= 10**4
        # )
        # print("average photon per voxel:", np.mean(self.image))
        # self.image = tifffile.imread(project_root + '/data/Zeiss_Mito_600nm.tiff')[:self.N, :self.N, :self.N]
        # plt.title("Ground truth")
        # plt.imshow(self.image)
        # plt.show()
        self.optical_system = System4f3D(alpha=self.alpha, refractive_index_medium=self.nmedium, refractive_index_sample=self.nmedium)
        self.optical_system.compute_psf_and_otf((self.psf_size, self.N))
        # plt.imshow(np.log1p(10**8 * self.optical_system.otf[self.N//2, :, :]), cmap='gray')
        # plt.title("PSF")
        # plt.show()
        # self.image = 10**5 * np.ones(self.optical_system.psf.shape)

        self.widefield_noiseless = scipy.signal.convolve(self.image, self.optical_system.psf, mode='same')
        # plt.title("Widefield image")
        # plt.imshow(self.widefield)
        # plt.show()
        
        configurations = BFPConfiguration(refraction_index=1.5)
        illumination_3waves3d = configurations.get_2_oblique_s_waves_and_s_normal(
            self.theta, 1, 1 , 5 , Mt=1, dimensionality=3
        )


        self.illumination = illumination_3waves3d
        self.illumination.set_spatial_shifts_diagonally()
        print(self.illumination.angles)

        # plt.imshow(self.illumination.get_illumination_density(coordinates=(self.x, self.x, self.z))[:, :, self.N//2])
        # plt.show()

        self.simulator = SIMulator3D(self.illumination, self.optical_system, readout_noise_variance=1)
        if not os.path.exists(current_dir + '/test_data/noisy_lines3d.pkl'):
            self.sim_images = self.simulator.generate_noiseless_sim_images(self.image)
            self.noisy_images = self.simulator.add_noise(self.sim_images)
            pickle.dump(self.noisy_images, open(current_dir + '/test_data/clean_lines3d.pkl', 'wb'))
            pickle.dump(self.noisy_images, open(current_dir + '/test_data/noisy_lines3d.pkl', 'wb'))
        else:
            self.sim_images = pickle.load(open(current_dir + '/test_data/clean_lines3d.pkl', 'rb'))
            self.noisy_images = pickle.load(open(current_dir + '/test_data/noisy_lines3d.pkl', 'rb'))
        
        # plt.imshow(np.log1p(10**12 * np.abs(hpc_utils.wrapped_fftn(self.illumination.get_illumination_density(self.optical_system.x_grid))))[self.N//2, :, :], cmap='gray')
        # plt.title("SIM image FFT")
        # plt.show()

    def test_reconstruction_widefield(self):
        reconstructor = ReconstructorFourierDomain3D(
            illumination=self.illumination,
            optical_system=self.optical_system
        )
        reconstructed_image = reconstructor.get_widefield(self.noisy_images)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield[self.N//2, :, :])
        axes[0].set_title("Widefield")
        axes[1].imshow(reconstructed_image[self.N//2, :, :])
        axes[1].set_title("Reconstructed Widefield")
        plt.show()
    
    def test_reconstruction_fourier_domain(self):
        fourier_reconstructor = ReconstructorFourierDomain3D(
            illumination=self.illumination,
            optical_system=self.optical_system,
        )
        reconstructed_image = fourier_reconstructor.reconstruct(self.noisy_images)

        ssnr_calc = SSNRCalculator.SSNRSIM3D(
            illumination=self.illumination,
            optical_system=self.optical_system,
        )

        _, _, filtered_image_ft = filter_true_wiener_sim(
            hpc_utils.wrapped_fftn(reconstructed_image), ssnr_calc
        )
        plt.imshow(np.log1p(np.abs(filtered_image_ft[self.N//2, :, :])))
        plt.show()

        filtered_image = hpc_utils.wrapped_ifftn(filtered_image_ft)
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(self.widefield[self.N//2, :, :])
        axes[1].imshow(reconstructed_image[self.N//2, :, :])
        axes[2].imshow(np.abs(filtered_image[self.N//2, :, :]))
        plt.show()

    def test_reconstruction_finite_kernel(self):
        cut_off_frequency_l = 1 / (2 * self.dx)
        cut_off_frequency_z = 1 / (2 * self.dz)
        kernel = utils.expand_kernel(kernels.psf_kernel2d(pixel_size=(self.dx, self.dx), first_zero_frequency=cut_off_frequency_l)[..., None] * kernels.sinc_kernel1d(pixel_size=self.dz, first_zero_frequency=cut_off_frequency_z), (31, 31, 31))
        finite_kernel_reconstructor = ReconstructorFourierDomain3D(
            illumination=self.illumination,
            optical_system=self.optical_system,
            kernel=kernel
        )
        reconstructed_image = finite_kernel_reconstructor.reconstruct(self.noisy_images)
        plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(reconstructed_image[:, :, self.N//2]))))
        plt.show()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.widefield[self.N//2, :, :])
        axes[1].imshow(reconstructed_image[self.N//2, :, :])
        plt.show()

    def test_reconstruction_spatial_domain_2d_kernel(self):
        cut_off_frequency_l = 1 / (2 * self.dx)
        import copy
        illumination_reconstruction = copy.deepcopy(self.illumination).project_in_quasi_2d()
        # illumination_reconstruction = IlluminationPlaneWaves2D.init_from_3D(illumination_reconstruction)
        m1, m2 = 1, 1
        for r in range(illumination_reconstruction.Mr):
            illumination_reconstruction.harmonics[(r, (1, 0, 0))].amplitude *= m1
            illumination_reconstruction.harmonics[(r, (-1, 0, 0))].amplitude *= m1
            illumination_reconstruction.harmonics[(r, (2, 0, 0))].amplitude *= m2
            illumination_reconstruction.harmonics[(r, (-2, 0, 0))].amplitude *= m2

        kernel=kernels.psf_kernel2d(pixel_size=(self.dx, self.dx), first_zero_frequency=cut_off_frequency_l)
        spatial_reconstructor = ReconstructorSpatialDomain3DSliced(
            # illumination=self.illumination, 
            illumination=illumination_reconstruction,
            optical_system=self.optical_system,
            kernel=kernel, 
        )
        illumination_widefield = BFPConfiguration(refraction_index=1.5).get_widefield(3)

        reconstructor_widefield = ReconstructorFourierDomain3D(
            illumination=illumination_widefield,
            optical_system=self.optical_system,
        )


        # Reconstruct the image.
        # sim_images_slices = self.noisy_images[:, :, :, : self.N//2]
        mask = windowing.make_mask_cosine_edge3d((self.N, self.N, self.N), edge=0)
        self.sim_images *= mask[None, None, :, :, :]
        self.noisy_images *= mask[None, None, :, :, :]
        self.widefield = np.sum(self.noisy_images, axis=(0,1))
        apodization = Apodization.AutoconvolutionApodizationSIM3D(self.optical_system, self.illumination)
        apodization_filter = apodization.ideal_otf

        reconstructed_image = spatial_reconstructor.reconstruct(self.noisy_images, backend='gpu') 
        reconstructed_image_ft = hpc_utils.wrapped_fftn(reconstructed_image)
        plt.imshow(np.log1p(np.abs(reconstructed_image_ft[:, self.N//2, :])))
        plt.show()
        # plt.imshow(np.where(np.abs(apodization_filter) > 10**-6, 1, 0)[:, :, self.N//2])
        # plt.show()

        # plt.imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(reconstructed_image[:, :, self.N//2]))))
        # plt.show()
        ssnr_calc = SSNRCalculator.SSNRSIM3D(   
            illumination = self.illumination, 
            optical_system = self.optical_system,
            kernel=kernel[..., None],
            illumination_reconstruction=illumination_reconstruction
            )

        ssnr_widefield = SSNRCalculator.SSNRSIM3D(
            illumination=illumination_widefield,
            optical_system=self.optical_system,
            )
        
        widefield = reconstructor_widefield.reconstruct(self.widefield[None, None, :, :, :])
        # plt.plot(np.log1p(1 + 10**8 * np.real(ssnr_calc.dj[:, self.N//2, self.N//2].T)))
        # plt.show()
        apodization_widefield = Apodization.AutoconvolutionApodizationSIM3D(self.optical_system, illumination_widefield, plane_wave_wavevectors=[np.array((0, 0, 0))])
        # reconstructed_image_ft *= np.where(np.abs(apodization_filter) > 10**-6, 1, 0)
        widefield_ft = hpc_utils.wrapped_fftn(widefield) * np.where(np.abs(apodization_widefield.ideal_otf) > 10**-6, 1, 0)
        # plt.imshow(np.where(np.abs(apodization_widefield.ideal_otf) > 10**-6, 1, 0)[:, self.N//2, :])
        # plt.show()
        filtered_ft, w, ssnr = filter_true_wiener_sim(
            reconstructed_image_ft, ssnr_calc, hpc_utils.wrapped_fftn(self.image))
        
        plt.imshow(np.log1p(np.abs(ssnr[:, self.N//2, :])))
        plt.show()

        filtered_widefield_ft, _, _ = filter_simulated_object_wiener(
            widefield_ft, ssnr_widefield, hpc_utils.wrapped_fftn(self.image))
        
        # fx, fy, fz = self.optical_system.otf_frequencies
        # FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing='ij')
        # filtered_widefield_ft *= np.where(FX**2 + FY**2 < 1.5 * 1 / (2 * self.dx), 1, 0)
        # filtered_widefield_ft *= np.where(FZ**2 < 1.5 * 1 / (2 * self.dz), 1, 0)
        # filtered, _ = filter_constant(reconstructed_image_ft, ssnr_calc.dj, w=1e-6)
        apodized_image_ft = filtered_ft * apodization_filter
        apodized_widefield_ft = filtered_widefield_ft * apodization_widefield.ideal_otf

        apodized_image = hpc_utils.wrapped_ifftn(apodized_image_ft).real
        apodized_widefield = hpc_utils.wrapped_ifftn(apodized_widefield_ft).real


        image_ft_ra = np.abs(utils.average_rings3d(hpc_utils.wrapped_fftn(self.image), (self.x, self.x, self.z)))
        reconstructed_image_ft_ra =np.abs(utils.average_rings3d(reconstructed_image_ft, (self.x, self.x, self.z)))
        filtered_ft_ra = np.abs(utils.average_rings3d(filtered_ft, (self.x, self.x, self.z)))
        widefield_ft_ra = np.abs(utils.average_rings3d(apodized_widefield_ft, (self.x, self.x, self.z)))

        fig, axes = plt.subplots(2, 4)
        axes[0, 0].imshow(np.log1p(np.abs(hpc_utils.wrapped_fftn(self.image)[:, :, self.N//2])), cmap='gray')
        axes[0, 0].set_title("Ground truth FT")
        axes[0, 1].imshow(np.log1p(np.abs(reconstructed_image_ft[:, self.N//2, :])), cmap='gray')
        axes[0, 1].set_title("Reconstructed FT")
        axes[0, 2].imshow(np.log1p(np.abs(filtered_ft[:, :, self.N//2])), cmap='gray')
        axes[0, 2].set_title("Filtered FT")
        axes[0, 3].imshow(np.log1p(np.abs(filtered_widefield_ft[:, :, self.N//2])), cmap='gray')
        axes[0, 3].set_title("Filtered Widefield FT")
        axes[1, 0].plot(image_ft_ra[:, self.N//2], label='Ground truth')
        axes[1, 1].plot(reconstructed_image_ft_ra[:, self.N//2], label='Reconstructed')
        axes[1, 2].plot(filtered_ft_ra[:, self.N//2], label='Filtered')
        axes[1, 3].plot(widefield_ft_ra[:, self.N//2], label='Filtered Widefield')
        for ax in axes[1, :]:
            ax.set_yscale('log')
            ax.legend()
        plt.show()

        fig, axes = plt.subplots(1, 5)
        axes[0].imshow(self.widefield[:, :, self.N//2], cmap='gray')
        axes[1].imshow(reconstructed_image[:, :, self.N//2], cmap='gray')
        axes[2].imshow(np.abs(apodized_image[:, :, self.N//2]), cmap='gray')
        axes[3].imshow(self.image[:, :, self.N//2], cmap='gray')
        axes[4].imshow(np.abs(apodized_widefield[:, :, self.N//2]), cmap='gray')
        plt.show()

class TestNonlinearReconstruction(unittest.TestCase):
    def setUp(self):
        self.N = 255
        self.alpha = 2 * np.pi / 5
        self.nmedium = 1.5
        self.theta = np.arcsin(0.9 * np.sin(self.alpha))
        self.dimensions = (1, 1)
        NA = self.nmedium * np.sin(self.alpha)
        self.dx = 1 / (64 * NA)
        self.max_r = self.N // 2 * self.dx

        self.psf_size = np.array((2 * self.max_r, 2 * self.max_r))
        self.x = np.linspace(-self.max_r, self.max_r, self.N)
        y = np.copy(self.x)

        self.image = ShapesGenerator.generate_random_lines(
            image_size=self.psf_size,
            point_number=self.N,
            line_width=0.25,
            num_lines=150,
            intensity=100
        )
        # plt.title("Ground truth")
        # plt.imshow(self.image)
        # plt.show()
        self.optical_system = System4f2D(alpha=self.alpha, refractive_index=self.nmedium)
        self.optical_system.compute_psf_and_otf((self.psf_size, self.N))

        # self.image = 10**5 * np.ones(self.optical_system.psf.shape)

        self.widefield = scipy.signal.convolve(self.image, self.optical_system.psf, mode='same')
        # plt.title("Widefield image")
        # plt.imshow(self.widefield)
        # plt.show()

        self.configurations = BFPConfiguration(refraction_index=1.5)
        self.Mr = 3

        illumination_3waves3d = self.configurations.get_2_oblique_s_waves_and_s_normal(
            self.theta, 1, 0, self.Mr, Mt=1
        )
        self.illumination_linear = IlluminationPlaneWaves2D.init_from_3D(
            illumination_3waves3d, self.dimensions
        )

        self.illumination_linear.set_spatial_shifts_diagonally()

        self.simulator_linear = SIMulator2D(self.illumination_linear, self.optical_system)
        self.sim_images = self.simulator_linear.generate_noiseless_sim_images(self.image)

        # self.sim_images += np.random.normal(0, 20, self.sim_images.shape)

        self.linear_reconstructor = ReconstructorSpatialDomain2D(
            illumination=self.illumination_linear,
            optical_system=self.optical_system,
        )
        # Reconstruct the image.
        self.reconstructed_linear = self.linear_reconstructor.reconstruct(self.sim_images)


    def test_exponential_intensity_dependence(self):
        p = 4
        nonlinear_expansion_coefficients = [0, ]
        n = 1
        from scipy.special import factorial
        while (p ** n / factorial(n)) > 10 ** -14:
            nonlinear_expansion_coefficients.append(p ** n / factorial(n) * (-1) ** (n + 1))
            n += 1

        illumination_non_linear = IlluminationNonLinearSIM2D.init_from_linear_illumination(self.illumination_linear, tuple(nonlinear_expansion_coefficients))
        illumination_non_linear.set_spatial_shifts_diagonally()

        plt.imshow(illumination_non_linear.get_illumination_density(coordinates=(self.x, self.x)))
        plt.show()

        simulator_non_linear = SIMulator2D(illumination_non_linear, self.optical_system)
        sim_images_non_linear = simulator_non_linear.generate_noiseless_sim_images(self.image)
        nonlinear_reconstructor = ReconstructorSpatialDomain2D(
            illumination=illumination_non_linear,
            optical_system=self.optical_system
        )

        reconstructed_non_linear = nonlinear_reconstructor.reconstruct(sim_images_non_linear)
        fig, axes = plt.subplots(1, 4)
        axes[0].imshow(self.image)
        axes[1].imshow(self.widefield)
        axes[2].imshow(self.reconstructed_linear)
        axes[3].imshow(reconstructed_non_linear)
        plt.show()

