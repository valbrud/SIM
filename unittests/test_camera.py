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
import warnings

# If your Camera class is defined in camera.py, do:
# from camera import Camera
# Otherwise, adjust the import as needed.
from Camera import Camera
import ShapesGenerator

class TestCamera2D(unittest.TestCase):

    def test_no_rebin(self):
        """
        2D camera: no object pixel size => no rebin; object must match camera size.
        We also check that the noise is applied.c
        """
        # Camera is 2D, Nx_cam=100, Ny_cam=100
        cam = Camera(pixel_number=(100, 100), pixel_size=(1.0, 1.0), mode='2D',
                     readout_noise_variance=0, hot_pixel_fraction=0,
                     exposure_time=0, dark_current_rate=0)

        # Make an object that matches shape (Nx_obj, Ny_obj) = (100, 100)
        object_2d = np.full((100, 100), 100.0)  # uniform intensity 100 e^-

        # No object pixel size => must skip rebin => shapes must match
        camera_image = cam.get_image(object_2d, object_pixel_size=())

        self.assertEqual(camera_image.shape, (100, 100),
                         "2D camera image should remain (100, 100).")

        # Check values: no dark current or readout noise => only shot noise
        # The average should be near 100
        self.assertAlmostEqual(camera_image.mean(), 100, delta=15,
                               msg="Mean camera counts should be near 100 after shot noise.")

        # Display
        plt.figure()
        plt.imshow(camera_image, cmap='viridis', origin='lower')
        plt.title("Test 2D No Rebin (Uniform=100) => Poisson Noise")
        plt.colorbar(label="Counts")
        plt.show()

    def test_rebin_clipping_2d(self):
        """
        2D camera: with object pixel size => coverage might exceed camera => clipping + rebin.
        Also add readout noise, hot pixels.
        """
        cam = Camera(pixel_number=(100, 100), pixel_size=(10.0, 10.0), mode='2D',
                     readout_noise_variance=100,  # variance=4 => sigma=2
                     hot_pixel_fraction=0.01,  # 1% hot pixels
                     exposure_time=1.0, dark_current_rate=100.0,
                     saturation_level=1e3
                     )
        # So dark current ~10 e^ per pixel, plus shot noise

        # Suppose the object is 200 x 200 => bigger than camera
        # with pixel size => (sx_obj, sy_obj) = (1.0, 1.0), coverage = 200 in x,y
        # which exceeds camera=50 => must clip.
        # object_2d = np.random.rand(200, 200) * 500  # max ~500 e^ per pixel
        np.random.seed(1234)
        object_2d = ShapesGenerator.generate_random_lines((500, 500), 500, 1, 30, 100)

        plt.figure()
        plt.imshow(object_2d, cmap='viridis', origin='lower')
        plt.title("Ground truth object")
        plt.colorbar(label="Counts")
        plt.show()
        with warnings.catch_warnings(record=True) as w_list:
            camera_image = cam.get_image(object_2d, object_pixel_size=(4.0, 4.0))

        # Expect shape = (50, 50)
        # self.assertEqual(camera_image.shape, (1000, 1000))

        # Check we got a warning about coverage
        self.assertTrue(any("Clipping" in str(w.message) for w in w_list),
                        "Expected a clipping warning when coverage_x/y > camera size.")

        # Visual check
        plt.figure()
        plt.imshow(camera_image, cmap='viridis', origin='lower')
        plt.title("Test 2D Rebin + Clipping + Noise (Dark current, readout, hot pixels)")
        plt.colorbar(label="Counts")
        plt.show()


class TestCamera3D(unittest.TestCase):

    def test_no_rebin_3d(self):
        """
        3D camera: no pixel size => skip rebin => object must match Nx_cam, Ny_cam in x,y.
        We capture z_shift_number planes from the last dimension, or clamp if out of range.
        """
        # Make a 3D camera => Nx_cam=40, Ny_cam=30, z_shift_number=3
        cam = Camera(pixel_number=(40, 30), pixel_size=(1.0, 1.0),
                     z_shift_number=3, z_shift_size=1.0, mode='3D',
                     readout_noise_variance=1.0, hot_pixel_fraction=0.0,
                     exposure_time=0.5, dark_current_rate=5.0)

        # Object shape => (Nx_obj, Ny_obj, Nz_obj) = (40, 30, 10)
        # => matches camera in x,y exactly => no rebin needed
        object_3d = np.ones((40, 30, 10)) * 100  # uniform 100

        camera_stack = cam.get_image(object_3d, object_pixel_size=())
        # shape => (z_shift_number=3, Nx_cam=40, Ny_cam=30)
        self.assertEqual(camera_stack.shape, (3, 40, 30))

        # Check that some noise was applied
        # Dark current => 5 e^/s * 0.5s = 2.5 e^ offset => ~102.5 average => then shot noise
        avg_plane = camera_stack[0].mean()
        self.assertAlmostEqual(avg_plane, 102.5, delta=15.0,
                               msg="Mean after dark current + shot noise should be near 102.5.")

        # Display each plane
        for i in range(camera_stack.shape[0]):
            plt.figure()
            plt.imshow(camera_stack[i], cmap='viridis', origin='lower')
            plt.title(f"Test 3D No Rebin => Plane {i}")
            plt.colorbar(label="Counts")
            plt.show()

    def test_rebin_3d(self):
        """
        3D camera with rebin in x,y.
        We'll have object pixel_size=(0.5, 0.5, 2.0).
        This will cause coverage_x,y to be half the object's dimension in x,y if camera px size=1.0
        """
        cam = Camera(pixel_number=(50, 50), pixel_size=(1.0, 1.0),
                     z_shift_number=2, z_shift_size=2.0, mode='3D',
                     readout_noise_variance=0.0, hot_pixel_fraction=0.02,
                     exposure_time=2.0, dark_current_rate=2.0)

        # object shape => (Nx_obj, Ny_obj, Nz_obj) = (100, 100, 5).
        # => each plane is 100x100 in x,y, last axis=5 in z
        # object px size => (0.5, 0.5, 2.0).
        # coverage_x => (100 * 0.5)/1.0 = 50 => exactly matches camera Nx=50, no clipping
        # coverage_y => (100 * 0.5)/1.0 = 50 => exactly matches camera Ny=50
        # So no clipping needed, but we do rebin from 100 -> 50
        object_3d = np.full((100, 100, 5), 400.0)  # uniform 400 e^-

        camera_stack = cam.get_image(object_3d, object_pixel_size=(0.5, 0.5, 2.0))
        self.assertEqual(camera_stack.shape, (2, 50, 50),
                         "Should produce 2 planes of shape (50,50).")

        # We'll see 2 planes => i=0 => z_i=0.5*(2.0)=1.0 => z_index= round(1.0/2.0 -0.5)= round(0.5)=1
        # i=1 => z_i=2.5 => z_index= round(2.5/2 -0.5)=round(1.25-0.5)= round(0.75)=1 => same plane
        # => we might see identical results for both planes

        for i in range(camera_stack.shape[0]):
            mean_val = camera_stack[i].mean()
            # Should be ~ 400 + dark_current => 2 e^/s * 2 s=4 e^ => 404 => shot noise
            # But camera shot noise => Poisson(404). Then 2% hot pixels => some saturate
            self.assertGreater(mean_val, 300.0,
                               msg="Should be around 400 + dark ~ 404 minus some shot noise, plus hot pixels.")

            # Display
            plt.figure()
            plt.imshow(camera_stack[i], cmap='viridis', origin='lower')
            plt.title(f"Test 3D Rebin => plane {i} (mean ~ {mean_val:.1f})")
            plt.colorbar(label="Counts")
            plt.show()


if __name__ == '__main__':
    unittest.main()
