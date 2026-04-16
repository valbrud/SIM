import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
filepath1 = current_dir + '/modulation.csv'
# filepath2 = current_dir + '/modulation_reduced_excitation.csv'
files = [
    filepath1,
    # filepath2
]

labels = [
    "Optimal",
    # "Reduced excitation"
]

for (filepath, label) in zip(files, labels):
    data = pd.read_csv(filepath)


    m1 = data['m1'].unique()
    m2 = data['m2'].unique()

    M1, M2 = np.meshgrid(m1, m2, indexing='ij')

    fig1 = plt.figure(figsize=(10, 10), dpi=100)
    axes1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(10, 10), dpi=100)
    axes2 = fig2.add_subplot(111)

    volume = np.array(data['Volume'])
    volume = volume.reshape(M1.shape)
    print('volume max ', np.max(volume), 'm1, m2 at max ', m1[np.unravel_index(np.argmax(volume), volume.shape)[0]], m2[np.unravel_index(np.argmax(volume), volume.shape)[1]])
    entropy = np.array(data['Entropy'])
    entropy = entropy.reshape(M1.shape)
    print('entropy max ', np.max(entropy), 'm1, m2 at max ', m1[np.unravel_index(np.argmax(entropy), entropy.shape)[0]], m2[np.unravel_index(np.argmax(entropy), entropy.shape)[1]])
    print('compbined max ', np.max(entropy * volume), 'm1, m2 at max ', m1[np.unravel_index(np.argmax(entropy * volume), (entropy * volume).shape)[0]], m2[np.unravel_index(np.argmax(entropy * volume), (entropy * volume).shape)[1]])
    im1 = axes1.imshow(volume.T, extent=(m2[0], m2[-1], m1[0], m1[-1]), origin='lower', aspect='auto')
    axes1.plot(m2[np.unravel_index(np.argmax(volume), volume.shape)[1]], m1[np.unravel_index(np.argmax(volume), volume.shape)[0]],  marker="x", linestyle="None")
    axes1.set_xlabel('M1', fontsize=40)
    axes1.set_ylabel('M2', fontsize=40)

    im2 = axes2.imshow(entropy.T, extent=(m2[0], m2[-1], m1[0], m1[-1]), origin='lower', aspect='auto')
    axes2.set_xlabel('M1', fontsize=40)
    axes2.set_ylabel('M2', fontsize=40)
    # axes1.set_title(f"SSNR volume {label}", fontsize=30)
    # axes2.set_title(f"SSNR entropy {label}", fontsize=30)



    axes1.set_aspect(1 / axes1.get_data_ratio())
    axes1.tick_params(labelsize=25)


    axes2.set_aspect(1 / axes2.get_data_ratio())
    axes2.tick_params(labelsize=25)

    plt.colorbar(im1, ax=axes1, fraction=0.046, pad=0.04, label='$SSNR_{K, V}$')
    plt.colorbar(im2, ax=axes2, fraction=0.046, pad=0.04, label='$SSNR_{K, S}$')
    # Increase colorbar label and tick sizes
    im1.colorbar.ax.tick_params(labelsize=25)
    im1.colorbar.ax.yaxis.label.set_size(40)
    im2.colorbar.ax.tick_params(labelsize=25)
    im2.colorbar.ax.yaxis.label.set_size(40)
    fig1.savefig(current_dir + f'/Figures/SSNR/modulation_coefficients_volume_{label.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(current_dir + f'/Figures/SSNR/modulation_coefficients_entropy_{label.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
plt.show()
