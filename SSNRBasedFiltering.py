import SSNRCalculator
import numpy as np
import skimage
import scipy
import matplotlib.pyplot as plt
from abc import abstractmethod
import wrappers
import stattools
class WienerFilter3d:
    def __init__(self, ssnr_calculator, apodization_filter=1):

        self.ssnr_calc = ssnr_calculator
        self.aj = apodization_filter

    @abstractmethod
    def _compute_ssnr_and_rageularization_filter(self, object, **kwargs): ...

    def filter_object(self, object, real_space=True): ...
class WienerFilter3dModel(WienerFilter3d):
    def __init__(self, ssnr_calculator, apodization_filter=1):
        super().__init__(ssnr_calculator, apodization_filter)

    def _compute_ssnr_and_wj(self, object_ft):
        center = np.array(object_ft.shape)//2
        ssnr = (self.ssnr_calc.dj**2 * np.abs(object_ft)**2 /
                (self.ssnr_calc.vj * object_ft[*center]
                 + object_ft.size * self.ssnr_calc.readout_noise_variance**2 * self.ssnr_calc.dj))
        print("WHAT IT SHOULD BE ORIG", ssnr[*center])
        # plt.imshow(np.abs(np.log10(1 + 10**4 *ssnr[:, :, centerz])))
        # plt.show()
        return ssnr, np.abs(self.ssnr_calc.dj/ssnr)

    def filter_object(self, model_object, real_space=True):
        if real_space:
            object_ft = wrappers.wrapped_fftn(model_object)
        else:
            object_ft = model_object
        shape = model_object.shape
        ssnr, wj = np.abs(self._compute_ssnr_and_wj(object_ft))
        otf_sim = ssnr / (ssnr + 1) * self.aj
        tj = self.aj / (self.ssnr_calc.dj + wj)

        fig, ax = plt.subplots(2,2)
        ax[0, 0].imshow((np.log10(1 + 10**4 * ssnr[:, :, shape[2]//2])))
        ax[0, 0].set_title('SSNR')
        ax[0, 1].imshow(np.log10(1 + 10**4 * wj[:, :, shape[2]//2]))
        ax[0, 1].set_title('Wj')
        ax[1, 0].imshow(np.log(1 + 10**4 * np.abs(self.ssnr_calc.dj[:, :, shape[2]//2])))
        ax[1, 0].set_title('Dj')
        ax[1, 1].imshow(tj[:, :, shape[2]//2])
        ax[1, 1].set_title('Tj')
        plt.show()
        filtered_ft = object_ft * otf_sim
        filtered = wrappers.wrapped_ifftn(filtered_ft).real

        return filtered, ssnr, wj, otf_sim, tj

class WienerFilter3dReconstruction(WienerFilter3d):
    def __init(self, ssnr_calculator, reconstruction, real_space=True):
        super().__init__(ssnr_calculator, apodization_filter=1)

    def _compute_ssnr_and_wj(self, object_ft, average="rings", numeric_noise = 10**-25):
        center = np.array(object_ft.shape)//2
        # plt.plot(np.log(1+ obj2_ra[:, 25]))
        # plt.show()
        N = object_ft.shape[2]
        f0 = object_ft[*center] / self.ssnr_calc.dj[*center]
        bj2 = np.abs(object_ft) ** 2
        if average == "surface_levels_3d":
            mask = stattools.find_decreasing_surface_levels3d(np.copy(self.ssnr_calc.dj), direction=0)
            obj2_a = stattools.average_mask(bj2, mask)
            dj_a = stattools.average_mask(np.copy(self.ssnr_calc.dj), mask)
            vj_a = stattools.average_mask(np.copy(self.ssnr_calc.vj), mask)
            ssnr = ((obj2_a - vj_a * f0 - object_ft.size * self.ssnr_calc.readout_noise_variance**2 * dj_a) /
                        (vj_a * f0 + object_ft.size * self.ssnr_calc.readout_noise_variance**2 * dj_a)).real
            ssnr = np.nan_to_num(ssnr)
            print("COMPUTED APPROXIMATION", ssnr[N//2, N//2, N//2])
            # plt.plot(vj_a[:, N//2, N//2])
            # plt.plot(self.ssnr_calc.vj[:, N//2, N//2])
            # plt.show()
            wj = (dj_a + 10**3 * numeric_noise)/(ssnr + numeric_noise)
        elif average == "rings":
            obj2_ra = stattools.average_rings3d(bj2, self.ssnr_calc.optical_system.otf_frequencies)
            dj_ra = stattools.average_rings3d(np.copy(self.ssnr_calc.dj), self.ssnr_calc.optical_system.otf_frequencies)
            vj_ra = stattools.average_rings3d(np.copy(self.ssnr_calc.vj), self.ssnr_calc.optical_system.otf_frequencies)
            ssnr_ra = ((obj2_ra - vj_ra * f0 - object_ft.size * self.ssnr_calc.readout_noise_variance**2 * dj_ra) /
                    (vj_ra * f0 + object_ft.size * self.ssnr_calc.readout_noise_variance**2 * dj_ra))
            ssnr = stattools.expand_ring_averages3d(ssnr_ra, self.ssnr_calc.optical_system.otf_frequencies)
            dj = stattools.expand_ring_averages3d(dj_ra, self.ssnr_calc.optical_system.otf_frequencies)
            wj = (dj + 10**3 * numeric_noise)/(ssnr + numeric_noise)
        else:
            raise AttributeError("Unknown method of averaging {average}")
        # print(ssnr[:, 25, 25])
        # plt.plot(np.log(1 + ssnr[:, 25, 25]))
        # plt.show()
        return np.abs(ssnr), np.abs(wj)

    def filter_object(self, reconstruction, real_space=True, average="surface_levels_3d"):
        if real_space:
            object_ft = wrappers.wrapped_fftn(reconstruction)
        else:
            object_ft = reconstruction
        shape = reconstruction.shape
        plt.imshow(np.log(1 + 10**4 * np.abs(object_ft[:, :, 25])**2))
        plt.show()
        ssnr, wj= self._compute_ssnr_and_wj(object_ft, average)
        otf_sim = ssnr / (ssnr + 1) * self.aj
        tj = self.aj / (self.ssnr_calc.dj + wj)

        fig, ax = plt.subplots(2,2)
        ax[0, 0].imshow((np.log10(1 + 10**4 * ssnr[:, :, shape[2]//2])))
        ax[0, 0].set_title('SSNR')
        ax[0, 1].imshow(np.log10(1 + 10**4 * wj[:, :, shape[2]//2]))
        ax[0, 1].set_title('Wj')
        ax[1, 0].imshow(np.log(1 + 10**4 * np.abs(self.ssnr_calc.dj[:, :, shape[2]//2])))
        ax[1, 0].set_title('Dj')
        ax[1, 1].imshow(tj[:, :, shape[2]//2])
        ax[1, 1].set_title('Tj')
        plt.show()
        filtered_ft = object_ft * tj
        filtered = wrappers.wrapped_ifftn(filtered_ft).real

        return filtered, ssnr, wj, otf_sim, tj

class WienerFilter3dModelSDR(WienerFilter3dModel):
    def __init__(self, ssnr_calculator, apodization_filter=1):
        super().__init__(ssnr_calculator, apodization_filter)

    def _compute_ssnr_and_wj(self, object_ft):
        center = np.array(object_ft.shape)//2

        ssnr = (self.ssnr_calc.dj**2 * np.abs(object_ft)**2 /
                (self.ssnr_calc.vj * object_ft[*center]
                 + object_ft.size * self.ssnr_calc.readout_noise_variance**2 * self.ssnr_calc.dj))
        print("WHAT IT SHOULD BE ORIG", ssnr[*center])
        # plt.imshow(np.abs(np.log10(1 + 10**4 *ssnr[:, :, centerz])))
        # plt.show()
        return ssnr, np.abs(self.ssnr_calc.dj/ssnr)
    def filter_SDR_reconstruction(self, object, reconstruction):
        shape = object.shape
        # object = np.zeros(shape)
        # object[26, 26, 26] = 10**9
        object_ft = wrappers.wrapped_fftn(object)
        # plt.imshow(np.log(1 + 10 ** 4 * np.abs(object_ft[:, :, shape[2]//2])))
        plt.imshow(reconstruction[:, :, shape[2]//2])
        plt.title("Rec")
        plt.show()
        reconstruction_ft = wrappers.wrapped_fftn(reconstruction)
        plt.imshow(np.log(1 + 10**4 * np.abs(object_ft[:, :, shape[2]//2])))
        plt.show()

        ssnr = np.abs(self._compute_ssnr_and_wj(object_ft))
        geff = self.ssnr_calc.otf_sim

        dj = np.abs(geff * geff.conjugate())
        # dj = self.ssnr_calc.dj
        centerx, centery, centerz = shape[0]//2, shape[1]//2, shape[2]//2
        wj = object_ft[centerx, centery, centerz].real / np.abs(object_ft) ** 2
        uj = np.abs(self.aj * geff.conjugate() / (dj + wj))

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow((np.log10(1 + 10 ** 4 * ssnr[:, :, shape[2] // 2])))
        ax[0, 0].set_title('SSNR')
        ax[0, 1].imshow(np.log10(1 + 10 ** 4 * wj[:, :, shape[2] // 2]))
        ax[0, 1].set_title('Wj')
        ax[1, 0].imshow(np.log(np.abs(1 + 10**4 * dj[:, :, shape[2]//2])))
        ax[1, 0].set_title('Dj')
        ax[1, 1].imshow(np.log(np.abs(1 + 10**4 * uj[:, :, shape[2] // 2])))
        ax[1, 1].set_title('Uj')
        plt.show()
        filtered_ft = reconstruction_ft * uj
        plt.imshow(np.log(np.abs(1 + 10**4 * filtered_ft[:, :, shape[2]//2])))
        plt.show()
        filtered = wrappers.wrapped_ifftn(filtered_ft).real
        plt.imshow(np.abs(filtered[:, :, shape[2]//2]))
        plt.show()
        return filtered, ssnr, wj, geff, uj

############################
############################
############################

class FlatNoiseFilter3d:
    def __init__(self, ssnr_calculator, apodization_filter=1):

        self.ssnr_calc = ssnr_calculator
        self.aj = apodization_filter

    def _compute_regularization_filter(self):
        return self.ssnr_calc.vj**0.5 - self.ssnr_calc.dj

    def filter_object(self, object, real_space=True): ...

class FlatNoiseFilter3dModel(FlatNoiseFilter3d):
    def __init__(self, ssnr_calculator, apodization_filter=1):
        super().__init__(ssnr_calculator, apodization_filter)


    def filter_object(self, model_object, real_space=True):
        if real_space:
            object_ft = wrappers.wrapped_fftn(model_object)
        else:
            object_ft = model_object
        shape = model_object.shape

        wj = np.abs(self._compute_regularization_filter())
        otf_sim = self.ssnr_calc.dj/self.ssnr_calc.vj**0.5
        tj = np.abs(self.aj / (self.ssnr_calc.dj + wj))

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].set_title('SSNR')
        ax[0, 1].imshow(np.log10(1 + 10**4 * wj[:, :, shape[2]//2]))
        ax[0, 1].set_title('Wj')
        ax[1, 0].imshow(otf_sim[:, :, shape[2]//2])
        ax[1, 0].set_title('OTF SIM')
        ax[1, 1].imshow(tj[:, :, shape[2]//2])
        ax[1, 1].set_title('Tj')
        plt.show()
        filtered_ft = object_ft * otf_sim * self.ssnr_calc.otf_sim
        # plt.imshow(np.log(np.abs(1 + 10**4 * filtered_ft)))
        # plt.show()
        filtered = wrappers.wrapped_ifftn(filtered_ft).real
        # plt.imshow(np.log(np.abs(1 + 10**4 * filtered_ft)))
        # plt.show()
        return filtered, wj, otf_sim, tj
