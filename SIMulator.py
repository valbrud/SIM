import numpy as np
import Illumination as illum
import wrappers
from OpticalSystems import Lens
import scipy
import Sources
import matplotlib.pyplot as plt
from Box import BoxSIM, Field
from mpl_toolkits.mplot3d import axes3d
from VectorOperations import VectorOperations
class SIMulator(BoxSIM):
    def __init__(self, illumination, optical_system, box_size=10, point_number=100, additional_info=None):
        super().__init__(illumination, box_size, point_number, additional_info)
        self.optical_system = optical_system
        self.SDR_coefficients = None
        self.effective_otfs = {}


    def _compute_SDR_coefficients(self):
        wavevectors2d = self.illumination.get_all_wavevectors_projected()
        psf_norm = np.sum(self.optical_system.psf)
        Cnorm = 1 / np.sum((self.illuminations_shifted[0] * self.optical_system.psf))
        normalized_illumination = []
        for wavevector in wavevectors2d:
            source = Sources.IntensityPlaneWave(1, 0, np.array((*wavevector, 0)))
            normalized_illumination.append(Field(source, self.grid, self.source_identifier))

        self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number))
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                SDR_coefficient = np.zeros(self.point_number, dtype=np.complex128)
                urn = VectorOperations.rotate_vector3d(self.illumination.spacial_shifts[n],
                                                                np.array((0, 0, 1)), self.illumination.angles[r])
                for field in normalized_illumination:
                    krm = VectorOperations.rotate_vector3d(field.source.wavevector,
                                                                np.array((0, 0, 1)), self.illumination.angles[r])
                    phase = np.dot(urn, krm)
                    SDR_coefficient += field.field * np.exp(1j * phase)
                self.SDR_coefficients[r, n] = np.real(SDR_coefficient) / Cnorm

    def simulate_sim_images(self, image):
        sim_images = np.zeros((self.illumination.Mr, self.illumination.Mt,  *self.point_number))
        for r in range(self.illumination.Mr):
            for n in range(self.illumination.Mt):
                sim_images[r, n] = scipy.signal.convolve(self.illuminations_shifted[r, n] * image, self.optical_system.psf, mode='same')
                # sim_images[r, m] = self.illuminations_shifted[n] * image
                sim_images[r, n] = np.random.poisson(sim_images[r, n])
        return sim_images

    def generate_widefield(self, sim_images):
        widefield_image = np.zeros(sim_images[0].shape)
        for rotation in sim_images:
            for image in rotation:
                widefield_image += image
        return widefield_image
    def reconstruct_real_space(self, sim_images):
        if not self.SDR_coefficients:
            self.SDR_coefficients = np.zeros((self.illumination.Mr, self.illumination.Mt, *self.point_number), dtype=np.float64)
            self._compute_SDR_coefficients()
        reconstructed_image = np.zeros(sim_images[0].shape)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # x, y = np.arange(50), np.arange(50)
        # X, Y = np.meshgrid(x, y)
        # ax1.imshow(self.illuminations_shifted[0, :, :, 25] * self.optical_system.psf[:, :, 25])
        # ax2.imshow(self.illuminations_shifted[0, :, :, 25])
        # plt.show()
        for r in range(sim_images.shape[0]):
            for m in range(sim_images.shape[1]):
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(self.SDR_coefficients[n, :, :, 25])
                # ax[1].imshow(self.illuminations_shifted[n, :, :, 25])
                # plt.show()
             reconstructed_image += self.SDR_coefficients[r, m] * sim_images[r, m]
        return reconstructed_image

    def _rearrange_indices(self, indices):
        result_dict = {}
        for index in indices:
            key = index[:2]
            value = index[2]
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
        result_dict = {key: tuple(values) for key, values in result_dict.items()}
        # print(result_dict)
        return result_dict

    def _compute_effective_otfs(self):
        waves = self.illumination.waves
        # plt.show()
        for r in range(self.illumination.Mr):
            angle = self.illumination.angles[r]
            indices = self._rearrange_indices(waves.keys())
            for xy_indices in indices.keys():
                effective_otf = 0
                for z_index in indices[xy_indices]:
                    wavevector = VectorOperations.rotate_vector3d(
                        waves[(*xy_indices, z_index)].wavevector, np.array((0, 0, 1)), angle)
                    amplitude = waves[(*xy_indices, z_index)].amplitude
                    # interpolated_otf = self.optical_system.interpolate_otf(wavevector)
                    # print(xy_indices, " ", z_index, " ", np.sum(interpolated_otf), " ", amplitude)
                    effective_otf += amplitude * self.optical_system.interpolate_otf(wavevector)
                # plt.imshow(np.log10(1 + 10 ** 16 * np.abs(effective_otf[50, :, :].T)))
                # plt.show()
                # print(np.abs(np.amin(effective_otf)), " ", np.abs(np.amax(effective_otf)))
                self.effective_otfs[(*xy_indices, r)] = effective_otf
        self.otf_sim = np.zeros(self.optical_system.otf.shape, dtype=np.complex128)
        for m in self.effective_otfs:
            self.otf_sim += self.effective_otfs[m]

    def _compute_shifted_images_ft(self, image_ft, r):
        shifted_images_ft = {}
        waves = self.illumination.waves
        axes = 2 * np.pi * self.optical_system.otf_frquencies
        interpolator = scipy.interpolate.RegularGridInterpolator(axes, image_ft,
                                                                      bounds_error=False,
                                                                      fill_value=0.)

        angle = self.illumination.angles[r]
        indices = self._rearrange_indices(waves.keys())
        for xy_indices in indices.keys():
            z_index = indices[xy_indices][0]
            wavevector2d = VectorOperations.rotate_vector2d(
                waves[(*xy_indices, z_index)].wavevector[:2], angle)
            wv3d = np.array((*wavevector2d, 0))
            interpolation_points = np.array(np.meshgrid(*(axes - wv3d))).T.reshape(-1, 3)
            interpolation_points = interpolation_points[
                np.lexsort((interpolation_points[:, 2], interpolation_points[:, 1],
                            interpolation_points[:, 0]))]
            image_shifted = interpolator(interpolation_points)
            image_shifted = image_shifted.reshape(axes[0].size, axes[1].size, axes[2].size)
            shifted_images_ft[(*xy_indices, r)] = image_shifted
        return shifted_images_ft
    def reconstruct_Fourier_space(self, sim_images):
        if not self.effective_otfs:
            self._compute_effective_otfs()
        sim_images_ft = np.zeros(sim_images.shape)
        reconstructed_image_ft = np.zeros(sim_images[2].shape)
        for r in range(sim_images.shape[0]):
            sum_shifts = np.zeros(sim_images.shape[2:])
            for n in range(sim_images.shape[1]):
                shifted_images = self._compute_shifted_images_ft(sim_images_ft[r, n], r)
                for shifted_image in shifted_images:
                    urn = VectorOperations.rotate_vector3d(self.illumination.spacial_shifts[n], np.array((0, 0, 1)),  self.illumination.angles[r])
                    krm = VectorOperations.rotate_vector3d(self,
                                                           np.array((0, 0, 1)), self.illumination.angles[r])
                    phase = np.dot(urn, krm)

                sim_images_ft[r, n] = wrappers.wrapped_fftn(sim_images[r, n])
                reconstructed_image_ft += self.effective_otfs[r, m] * sim_images_ft[r, m]





