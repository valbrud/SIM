import unittest
import matplotlib.pyplot as plt
from config.IlluminationConfigurations import *
from OpticalSystems import Lens
class TestOpticalSystems(unittest.TestCase):
    def test_shifted_otf(self):
        max_r = 10
        max_z = 25
        N = 80
        dx = 2 * max_r / N
        dy = 2 * max_r / N
        dz = 2 * max_z / N
        x = np.arange(-max_r, max_r, dx)
        y = np.copy(x)
        z = np.arange(-max_z, max_z, dz)

        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / (2 * max_r), N)
        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy) - 1 / (2 * max_r), N)
        fz = np.linspace(-1 / (2 * dz), 1 / (2 * dz) - 1 / (2 * max_r), N)

        illumination = Illumination(s_polarized_waves)

        # wavevectors = [np.array([ 4.44288294,  0.        , -1.84030237]), np.array([-4.44288294e+00,  5.44096237e-16, -1.84030237e+00]), np.array([ 2.72048118e-16,  4.44288294e+00, -1.84030237e+00]), np.array([ 2.72048118e-16, -4.44288294e+00, -1.84030237e+00]), np.array([4.44288294, 0.        , 1.84030237]), np.array([-4.44288294e+00,  5.44096237e-16,  1.84030237e+00]), np.array([2.72048118e-16, 4.44288294e+00, 1.84030237e+00]), np.array([ 2.72048118e-16, -4.44288294e+00,  1.84030237e+00])]
        wavevectors = [np.array([0.0 * np.pi,  0., 0.3 * np.pi])]
        # print(wavevectors)
        optical_system = Lens()
        optical_system.compute_psf_and_otf((np.array((2 * max_r, 2 * max_r, 2 * max_z)), N))
        optical_system.prepare_Fourier_interpolation(wavevectors)
        otf_sum = np.zeros((len(x), len(y), len(z)), dtype=np.complex128)
        for otf in optical_system._shifted_otfs:
            otf_sum += optical_system._shifted_otfs[otf]
        print(optical_system)
        # print(otf_sum[:, :, 10])
        fig = plt.figure()
        IM = otf_sum[:, int(N/2), :]

        ax = fig.add_subplot(111)
        mp1 = ax.imshow(np.abs(IM))

        plt.colorbar(mp1)
        def update(val):
            ax.clear()
            ax.set_title("otf_sum, fy = {:.2f}, ".format(fy[int(val)]) + "$\\lambda^{-1}$")
            ax.set_xlabel("fx, $\lambda^{-1}$")
            ax.set_ylabel("fy, $\lambda^{-1}$")
            # IM = abs(otf_sum[:, int(val), :])
            IM = abs(optical_system.psf[:, int(val), :])
            mp1.set_clim(vmin=IM.min(), vmax=IM.max())
            ax.imshow(np.abs(IM), extent=(x[0], x[-1], z[0], z[-1]))
            ax.set_aspect(1. / ax.get_data_ratio())


        slider_loc = plt.axes((0.2, 0.1, 0.65, 0.03))  # slider location and size
        slider_ssnr = Slider(slider_loc, 'fz', 0, N - 1)  # slider properties
        slider_ssnr.on_changed(update)
        plt.show()