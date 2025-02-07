import numpy as np
import warnings

from matplotlib import pyplot as plt
from skimage.transform import resize

class Camera:
    def __init__(
            self,
            pixel_number=(100, 100),   # (Nx_cam, Ny_cam)
            pixel_size=(1.0, 1.0),     # (sx_cam, sy_cam)
            z_shift_number=1,
            z_shift_size=1.0,
            readout_noise_variance=0.0,
            hot_pixel_fraction=0.0,
            saturation_level=1e4,
            exposure_time=0.0,
            dark_current_rate=0.0,
            mode='2D'
    ):
        """
        Parameters
        ----------
        pixel_number : tuple(int)
            (Nx_cam, Ny_cam). Final 2D image shape is (Nx_cam, Ny_cam).
        pixel_size : tuple(float)
            (sx_cam, sy_cam). Physical size of each camera pixel in x, y.
        z_shift_number : int
            If mode='3D', how many z-planes we capture in the output stack.
        z_shift_size : float
            Spacing in z between consecutive camera planes (if mode='3D').
        readout_noise_variance : float
            Gaussian readout noise variance (in e^-).
        hot_pixel_fraction : float
            Fraction [0..1] of pixels that are "hot" (fixed at saturation).
        saturation_level : float
            Maximum electron count per pixel.
        exposure_time : float
            Seconds of exposure (for dark current).
        dark_current_rate : float
            e^- per second per pixel (added to each pixel).
        mode : {"2D", "3D"}
            - "2D": object_image must be 2D => returns shape (Nx_cam, Ny_cam).
            - "3D": object_image must be 3D => returns shape (z_shift_number, Nx_cam, Ny_cam).
        """
        # Camera dimensions
        self.pixel_number = pixel_number   # (Nx_cam, Ny_cam)
        self.pixel_size   = pixel_size     # (sx_cam, sy_cam)

        self.z_shift_number = z_shift_number
        self.z_shift_size   = z_shift_size

        self.readout_noise_variance = readout_noise_variance
        self.hot_pixel_fraction     = hot_pixel_fraction
        self.saturation_level       = saturation_level
        self.exposure_time          = exposure_time
        self.dark_current_rate      = dark_current_rate

        if mode not in ('2D', '3D'):
            raise ValueError("Camera mode must be '2D' or '3D'.")

        self.mode = mode

        # Unpack camera shape and pixel sizes
        self.Nx_cam, self.Ny_cam = self.pixel_number
        self.sx_cam, self.sy_cam = self.pixel_size

    # -------------------------------------------------------------------------
    #   Rebin utilities
    # -------------------------------------------------------------------------
    def _rebin_integer(self, image, new_shape):
        """
        Integer rebin by summing blocks.
        image.shape => (Nx_old, Ny_old)
        new_shape   => (Nx_new, Ny_new)
        Must divide exactly in each dimension.
        """
        Nx_old, Ny_old = image.shape
        Nx_new, Ny_new = new_shape
        factor_x = Nx_old // Nx_new
        factor_y = Ny_old // Ny_new

        # Reshape into (Nx_new, factor_x, Ny_new, factor_y), sum over block axes
        return image.reshape(Nx_new, factor_x, Ny_new, factor_y).sum(axis=(1, 3))

    def _rebin_fractional(self, image, new_shape):
        """
        Fractional rebin using skimage.transform.resize with anti-aliasing.
        image.shape => (Nx_old, Ny_old)
        new_shape   => (Nx_new, Ny_new)
        """
        Nx_new, Ny_new = new_shape
        # skimage.resize expects (rows, cols) => interpret Nx as 'rows'
        # So we'll pass (Nx_new, Ny_new) in that order.
        rebinned = resize(
            image,
            (Nx_new, Ny_new),
            order=1,               # bilinear
            preserve_range=True,
            anti_aliasing=True
        )
        return rebinned

    def _rebin_2d(self, image, new_shape):
        """
        Rebins a 2D array from (Nx_old, Ny_old) to (Nx_new, Ny_new).
        If it divides exactly => integer rebin. Else fractional.
        """
        Nx_new, Ny_new = new_shape
        Nx_old, Ny_old = image.shape

        # Check for integer factor
        if (Nx_old % Nx_new == 0) and (Ny_old % Ny_new == 0):
            return self._rebin_integer(image, new_shape)
        else:
            return self._rebin_fractional(image, new_shape)

    # -------------------------------------------------------------------------
    #   Noise pipeline
    # -------------------------------------------------------------------------
    def apply_dark_current(self, image):
        """
        Add dark current = dark_current_rate * exposure_time to each pixel.
        Then subject to shot noise next.
        """
        if self.exposure_time > 0 and self.dark_current_rate > 0:
            dark_level = self.dark_current_rate * self.exposure_time
            image += dark_level
        return image

    def apply_shot_noise(self, image):
        """
        Poisson (shot) noise.
        """
        image = np.maximum(image, 0)
        return np.random.poisson(image).astype(np.float64)

    def apply_readout_noise(self, image):
        """
        Gaussian readout noise with variance = readout_noise_variance.
        """
        if self.readout_noise_variance > 0:
            sigma = np.sqrt(self.readout_noise_variance)
            image += np.random.normal(loc=0.0, scale=sigma, size=image.shape)
        return image

    def apply_hot_pixels(self, image):
        """
        Random fraction of pixels set to saturation_level.
        """
        if self.hot_pixel_fraction > 0:
            N = image.size
            hot_count = int(self.hot_pixel_fraction * N)
            hot_indices = np.random.choice(N, hot_count, replace=False)
            flat = image.ravel()
            flat[hot_indices] = self.saturation_level
        return image

    # -------------------------------------------------------------------------
    #   2D acquisition
    # -------------------------------------------------------------------------
    def get_image_2d(self, object_image, object_pixel_size=()):
        """
        Produce a single 2D camera image of shape (Nx_cam, Ny_cam) from
        a 2D object_image of shape (Nx_obj, Ny_obj).

        If object_pixel_size=() => no rebin => object_image must match
        (Nx_cam, Ny_cam). Otherwise, we compute coverage => rebin => clip => center.
        Then apply noise pipeline.
        """
        Nx_obj, Ny_obj = object_image.shape

        # If no pixel sizes => skip rebin; shapes must match
        if not object_pixel_size:
            if (Nx_obj != self.Nx_cam) or (Ny_obj != self.Ny_cam):
                raise ValueError(
                    "No object_pixel_size => skip rebin, but shape of 2D object "
                    f"({Nx_obj}, {Ny_obj}) != camera ({self.Nx_cam}, {self.Ny_cam})."
                )
            camera_image = object_image.astype(np.float64)

        else:
            # object_pixel_size => (sx_obj, sy_obj)
            if len(object_pixel_size) != 2:
                raise ValueError("For 2D, object_pixel_size must be (sx_obj, sy_obj).")
            sx_obj, sy_obj = object_pixel_size

            # coverage in x,y (how many camera pixels we want)
            coverage_x = (Nx_obj * sx_obj) / self.sx_cam
            coverage_y = (Ny_obj * sy_obj) / self.sy_cam

            Nx_reb = int(round(coverage_x))
            Ny_reb = int(round(coverage_y))

            # Clip if bigger than camera
            if Nx_reb > self.Nx_cam:
                warnings.warn(f"Coverage in x ({Nx_reb}) > Nx_cam={self.Nx_cam}. Clipping.")
                Nx_reb = self.Nx_cam
            if Ny_reb > self.Ny_cam:
                warnings.warn(f"Coverage in y ({Ny_reb}) > Ny_cam={self.Ny_cam}. Clipping.")
                Ny_reb = self.Ny_cam

            Nx_reb = max(Nx_reb, 1)
            Ny_reb = max(Ny_reb, 1)

            rebinned = self._rebin_2d(object_image, (Nx_reb, Ny_reb))

            # Place it into final (Nx_cam, Ny_cam) array, centered
            camera_image = np.zeros((self.Nx_cam, self.Ny_cam), dtype=np.float64)
            off_x = (self.Nx_cam - Nx_reb)//2
            off_y = (self.Ny_cam - Ny_reb)//2
            camera_image[off_x:off_x + Nx_reb, off_y:off_y + Ny_reb] = rebinned

        # Noise pipeline
        self.apply_dark_current(camera_image)
        camera_image = self.apply_shot_noise(camera_image)
        self.apply_readout_noise(camera_image)
        self.apply_hot_pixels(camera_image)
        np.minimum(camera_image, self.saturation_level, out=camera_image)

        return camera_image

    # -------------------------------------------------------------------------
    #   3D acquisition
    # -------------------------------------------------------------------------
    def get_image_3d(self, object_image, object_pixel_size=()):
        """
        3D mode. object_image has shape (Nx_obj, Ny_obj, Nz_obj).
        We produce a stack of shape (z_shift_number, Nx_cam, Ny_cam).

        For i in [0..z_shift_number-1]:
          - z_i = (i+0.5)*z_shift_size
          - nearest plane index z_index => round(z_i / sz_obj - 0.5) if pixel_size given
            or just z_index = i if no pixel_size (and i < Nz_obj).
          - slice_2d = object_image[..., z_index] => shape (Nx_obj, Ny_obj)
          - rebin in x,y if pixel_size given
          - noise pipeline
        """
        Nx_obj, Ny_obj, Nz_obj = object_image.shape

        # Prepare output stack
        stack = np.zeros((self.z_shift_number, self.Nx_cam, self.Ny_cam), dtype=np.float64)

        no_px_size = (not object_pixel_size)

        if no_px_size:
            # If no pixel size => no rebin => must match Nx_cam, Ny_cam in x,y
            if (Nx_obj != self.Nx_cam) or (Ny_obj != self.Ny_cam):
                raise ValueError(
                    f"No object_pixel_size => skip rebin, but 3D object has (Nx_obj, Ny_obj)=({Nx_obj},{Ny_obj}) "
                    f"!= camera ({self.Nx_cam},{self.Ny_cam})."
                )

        else:
            # object_pixel_size => (sx_obj, sy_obj, sz_obj)
            if len(object_pixel_size) != 3:
                raise ValueError("For 3D, object_pixel_size must be (sx_obj, sy_obj, sz_obj).")
            sx_obj, sy_obj, sz_obj = object_pixel_size

        for i in range(self.z_shift_number):
            z_i = (i + 0.5)*self.z_shift_size
            if no_px_size:
                # interpret i as direct z_index
                z_index = i
            else:
                # nearest plane
                z_index = int(round(z_i / sz_obj - 0.5))

            # clamp z_index
            if z_index < 0:
                z_index = 0
            if z_index >= Nz_obj:
                z_index = Nz_obj - 1

            # Extract plane => shape (Nx_obj, Ny_obj)
            slice_2d = object_image[..., z_index]

            # Possibly rebin in (x,y)
            if no_px_size:
                camera_slice = slice_2d.astype(np.float64)
            else:
                # coverage in x, y
                coverage_x = (Nx_obj * sx_obj)/self.sx_cam
                coverage_y = (Ny_obj * sy_obj)/self.sy_cam
                Nx_reb = int(round(coverage_x))
                Ny_reb = int(round(coverage_y))

                # Clip
                if Nx_reb > self.Nx_cam:
                    warnings.warn(f"Coverage in x ({Nx_reb}) > Nx_cam={self.Nx_cam}. Clipping.")
                    Nx_reb = self.Nx_cam
                if Ny_reb > self.Ny_cam:
                    warnings.warn(f"Coverage in y ({Ny_reb}) > Ny_cam={self.Ny_cam}. Clipping.")
                    Ny_reb = self.Ny_cam

                Nx_reb = max(Nx_reb, 1)
                Ny_reb = max(Ny_reb, 1)

                rebinned = self._rebin_2d(slice_2d, (Nx_reb, Ny_reb))

                # embed in final (Nx_cam, Ny_cam)
                camera_slice = np.zeros((self.Nx_cam, self.Ny_cam), dtype=np.float64)
                off_x = (self.Nx_cam - Nx_reb)//2
                off_y = (self.Ny_cam - Ny_reb)//2
                camera_slice[off_x:off_x + Nx_reb, off_y:off_y + Ny_reb] = rebinned

            # Noise pipeline
            self.apply_dark_current(camera_slice)
            camera_slice = self.apply_shot_noise(camera_slice)
            self.apply_readout_noise(camera_slice)
            self.apply_hot_pixels(camera_slice)
            np.minimum(camera_slice, self.saturation_level, out=camera_slice)

            stack[i] = camera_slice

        return stack

    # -------------------------------------------------------------------------
    #   Main entry
    # -------------------------------------------------------------------------
    def get_image(self, object_image, object_pixel_size=()):
        """
        Public method:
          if mode='2D', object_image must be 2D => returns (Nx_cam, Ny_cam)
          if mode='3D', object_image must be 3D => returns (z_shift_number, Nx_cam, Ny_cam)

        If object_pixel_size=() => no rebin. Otherwise, we rebin in (x,y).
        We do not rebin along z. Instead, we pick planes from the last index.
        """
        if self.mode == '2D':
            if object_image.ndim != 2:
                raise ValueError("Camera mode='2D' but object_image is not 2D.")
            return self.get_image_2d(object_image, object_pixel_size)

        else:  # mode='3D'
            if object_image.ndim != 3:
                raise ValueError("Camera mode='3D' but object_image is not 3D.")
            return self.get_image_3d(object_image, object_pixel_size)
