import numpy as np
def generate_random_spheres(psf_size, point_number, r = 0.1, N=10, I = 1000):
    np.random.seed(1234)
    indices = np.array(np.meshgrid(np.arange(point_number), np.arange(point_number),
                                   np.arange(point_number))).T.reshape(-1, 3)
    indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
        point_number, point_number, point_number, 3)
    grid = (indices / point_number - 1 / 2)
    shape = grid.shape[:3]
    array = np.zeros(shape)
    grid *= psf_size[None, None, None, :]
    sx, sy, sz = psf_size
    cx = np.random.rand(N) * sx - sx//2
    cy = np.random.rand(N) * sy - sy//2
    cz = np.random.rand(N) * sz - sz//2
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i])**2, axis=3)
        array[dist2 < r**2] += I
    return array

def generate_sphere_slices(psf_size, point_number, r = 0.1, N=10, I = 1000):
    np.random.seed(1234)
    indices = np.array(np.meshgrid(np.arange(point_number), np.arange(point_number),
                                   np.arange(point_number))).T.reshape(-1, 3)
    indices = indices[np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))].reshape(
        point_number, point_number, point_number, 3)
    grid = (indices / point_number - 1 / 2)
    shape = grid.shape[:3]
    array = np.zeros(shape)
    grid *= psf_size[None, None, None, :]
    sx, sy, sz = psf_size
    cx = np.random.rand(N) * sx - sx//2
    cy = np.random.rand(N) * sy - sy//2
    cz = (np.random.rand(N) * sz - sz//2)/sz * r
    centers = np.column_stack((cx, cy, cz))

    for i in range(N):
        dist2 = np.sum((grid - centers[None, None, None, i])**2, axis=3)
        array[dist2 < r**2] += I
    return array