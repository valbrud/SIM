import SNRCalculator
import numpy as np
import skimage
import scipy
import matplotlib.pyplot as plt
from abc import abstractmethod
import wrappers
class WienerFilter3d:
    def __init__(self, ssnr_calculator, apodization_filter=1):

        self.ssnr_calc = ssnr_calculator
        self.aj = apodization_filter

    @abstractmethod
    def _compute_full_object_dependent_ssnr(self, object): ...

    def _compute_regularization_filter(self, ssnr):
        return self.ssnr_calc.dj / ssnr

    def compute_filtered_image(self, object, real_space=True): ...
class WienerFilter3dModel(WienerFilter3d):
    def __init__(self, ssnr_calculator, apodization_filter=1):
        super().__init__(ssnr_calculator, apodization_filter)

    def _compute_full_object_dependent_ssnr(self, object_ft):
        fx, fy, fz = self.ssnr_calc.optical_system.otf_frequencies
        centerx, centery, centerz = (fx.size + 1)//2, (fy.size + 1)//2, (fz.size + 1)//2
        ssnr = (self.ssnr_calc.dj**2 * np.abs(object_ft)**2 /
                (self.ssnr_calc.vj * object_ft[centerx, centery, centerz]
                 + object_ft.size * self.ssnr_calc.readout_noise_variance**2 * self.ssnr_calc.dj))
        # plt.imshow(np.abs(np.log10(1 + 10**4 *ssnr[:, :, centerz])))
        # plt.show()
        return ssnr

    def compute_filtered_image(self, model_object, real_space=True):
        if real_space:
            object_ft = wrappers.wrapped_fftn(model_object)
        else:
            object_ft = model_object
        shape = model_object.shape

        ssnr = np.abs(self._compute_full_object_dependent_ssnr(object_ft))
        wj = np.abs(self._compute_regularization_filter(ssnr))
        otf_sim = ssnr / (ssnr + 1) * self.aj
        tj = np.abs(self.aj / (self.ssnr_calc.dj + wj))

        fig, ax = plt.subplots(2,2)
        ax[0, 0].imshow((np.log10(1 + 10**4 * ssnr[:, :, shape[2]//2])))
        ax[0, 0].set_title('SSNR')
        ax[0, 1].imshow(np.log10(1 + 10**4 * wj[:, :, shape[2]//2]))
        ax[0, 1].set_title('Wj')
        ax[1, 0].imshow(otf_sim[:, :, shape[2]//2])
        ax[1, 0].set_title('OTF SIM')
        ax[1, 1].imshow(tj[:, :, shape[2]//2])
        ax[1, 1].set_title('Tj')

        filtered_ft = object_ft * otf_sim
        filtered = wrappers.wrapped_ifftn(filtered_ft).real

        return filtered, ssnr, wj, otf_sim, tj

class WienerFilter3dImage(WienerFilter3d):
    def __init(self, ssnr_calculator, image, real_space=True):
        super().__init__(ssnr_calculator, apodization_filter=1)

    def compute_full_object_dependent_ssnr(self,): ...


class FlatNoiseFilter3d:
    def __init__(self, ssnr_calculator, apodization_filter=1):

        self.ssnr_calc = ssnr_calculator
        self.aj = apodization_filter

    def _compute_regularization_filter(self):
        return self.ssnr_calc.vj**0.5 - self.ssnr_calc.dj

    def compute_filtered_image(self, object, real_space=True): ...

class FlatNoiseFilter3dModel(FlatNoiseFilter3d):
    def __init__(self, ssnr_calculator, apodization_filter=1):
        super().__init__(ssnr_calculator, apodization_filter)


    def compute_filtered_image(self, model_object, real_space=True):
        if real_space:
            object_ft = wrappers.wrapped_fftn(model_object)
        else:
            object_ft = model_object
        shape = model_object.shape

        wj = np.abs(self._compute_regularization_filter())
        otf_sim = self.ssnr_calc.dj/self.ssnr_calc.vj**0.5
        tj = np.abs(self.aj / (self.ssnr_calc.dj + wj))

        fig, ax = plt.subplots(2,2)
        ax[0, 0].set_title('SSNR')
        ax[0, 1].imshow(np.log10(1 + 10**4 * wj[:, :, shape[2]//2]))
        ax[0, 1].set_title('Wj')
        ax[1, 0].imshow(otf_sim[:, :, shape[2]//2])
        ax[1, 0].set_title('OTF SIM')
        ax[1, 1].imshow(tj[:, :, shape[2]//2])
        ax[1, 1].set_title('Tj')

        filtered_ft = object_ft * otf_sim
        filtered = wrappers.wrapped_ifftn(filtered_ft).real

        return filtered, wj, otf_sim, tj