from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
import numpy as np
import io
from config.IlluminationConfigurations import BFPConfiguration
from OpticalSystems import System4f3D
from Box import Box

app = Flask(__name__)

# Initialize configurations and optical system
configurations = BFPConfiguration(refraction_index=1.5)
alpha = 2 * np.pi / 5
nmedium = 1.5
nobject = 1.5
NA = nmedium * np.sin(alpha)
theta = np.asin(0.9 * np.sin(alpha))
fz_max_diff = nmedium * (1 - np.cos(alpha))
dx = 1 / (8 * NA)
dy = dx
dz = 1 / (4 * fz_max_diff)
N = 101
max_r = N // 2 * dx
max_z = N // 2 * dz
psf_size = 2 * np.array((max_r, max_r, max_z))

optical_system = System4f3D(alpha=alpha, refractive_index_sample=nobject, refractive_index_medium=nmedium)
optical_system.compute_psf_and_otf((psf_size, N), high_NA=True, apodization_function="Sine")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    config_name = request.form['config']
    if config_name == 'conventional':
        illumination = configurations.get_2_oblique_s_waves_and_s_normal(theta, 1, 1, 3, Mt=1)
    elif config_name == 'squareL':
        illumination = configurations.get_4_oblique_s_waves_and_s_normal_diagonal(theta, 1, 1, Mt=1)
    elif config_name == 'squareC':
        illumination = configurations.get_4_circular_oblique_waves_and_circular_normal(theta, 0.58, 1, Mt=1, phase_shift=0)
    elif config_name == 'hexagonal':
        illumination = configurations.get_6_oblique_s_waves_and_circular_normal(theta, 1, 1, Mt=1)
    else:
        return "Invalid configuration", 400

    box = Box(illumination.waves.values(), box_size=psf_size, point_number=N)
    box.compute_intensity_from_spacial_waves()

    fig, ax = plt.subplots()
    ax.imshow(box.intensity[:, :, N // 2].T, extent=(-max_r, max_r, -max_r, max_r))
    ax.set_title(config_name)
    ax.set_xlabel("x [λ]")
    ax.set_ylabel("y [λ]")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)