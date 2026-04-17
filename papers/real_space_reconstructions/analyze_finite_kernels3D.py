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
filepath1 = current_dir + '/MixedKernels3D.csv'
filepath2 = current_dir + '/SincKernels3D.csv'
files = [
    # filepath1,
    filepath2
]

labels = [
    # "Mixed",
    "Triangular kernel"
]

for (filepath, label) in zip(files, labels):
    data = pd.read_csv(filepath)


    factors_l = data['Lateral factor'].unique()[1:]
    factors_z = data['Axial factor'].unique()[1:]

    FL, FZ = np.meshgrid(factors_l, factors_z, indexing='ij')

    configurations = data['Configuration'].unique()
    for configuration in configurations:
        if configuration != 'Conventional':
            continue
        # if configuration != "Conventional":
        #     continue

        fig1 = plt.figure(figsize=(10, 10), dpi=100)
        axes1 = fig1.add_subplot(111)
        fig2 = plt.figure(figsize=(10, 10), dpi=100)
        axes2 = fig2.add_subplot(111)

        plot_data = data[data['Configuration'] == configuration]
        plot_data = plot_data[plot_data['Lateral factor'] != '0']
        volume = np.array(plot_data['Volume'])[1:]
        volume = volume.reshape(FL.shape)
        entropy = np.array(plot_data['Entropy'])[1:]
        entropy = entropy.reshape(FL.shape)

        vml, vma = np.unravel_index(np.argmax(volume), volume.shape)
        sml, sma = np.unravel_index(np.argmax(entropy), entropy.shape)
        # axes1.plot_wireframe(FL, FZ, volume, label = configuration + " " + label)
        im1 = axes1.imshow(volume.T, extent=(factors_z[0]/2, factors_z[-1]/2, factors_l[0]/2, factors_l[-1]/2), origin='lower', aspect='auto')
        axes1.plot(factors_l[vml]/2, factors_l[vma]/2, marker='x', markersize=20, color='red')
        axes1.set_xlabel('$s_l$', fontsize=40)
        axes1.set_ylabel('$s_a$', fontsize=40)
        cbar1 = plt.colorbar(im1, ax=axes1, fraction=0.046, pad=0.04, label='$SSNR_{K, V}$')
        cbar1.ax.tick_params(labelsize=25)
        cbar1.set_label('$SSNR_{K, V}$', fontsize=40)

        im2 = axes2.imshow(entropy.T, extent=(factors_z[0]/2, factors_z[-1]/2, factors_l[0]/2, factors_l[-1]/2), origin='lower', aspect='auto')
        axes2.plot(factors_l[sml]/2, factors_l[sma]/2, marker='x', markersize=20, color='red')
        axes2.set_xlabel('$s_l$', fontsize=40)
        axes2.set_ylabel('$s_a$', fontsize=40)
        cbar2 = plt.colorbar(im2, ax=axes2, fraction=0.046, pad=0.04, label='$SSNR_{K, S}$')
        cbar2.ax.tick_params(labelsize=25)
        cbar2.set_label('$SSNR_{K, S}$', fontsize=40)

        # axes1.set_title(f"SSNR volume {label}", fontsize=30)
        # axes2.set_title(f"SSNR entropy {label}", fontsize=30)


        # axes1.grid()
        # # axes1.set_xlim(0, 2)
        # # axes1.set_ylim(0, 1)
        axes1.set_aspect(1 / axes1.get_data_ratio())
        axes1.tick_params(labelsize=25)
        # axes1.set_xlabel(r"Size", fontsize=30)
        # axes1.set_ylabel("$SSNR^V$", fontsize=30)
        # axes1.legend(fontsize=20)

        # axes2.grid()
        # # axes2.set_xlim(0, 2)
        # # axes2.set_ylim(0, 0.2)
        axes2.set_aspect(1 / axes2.get_data_ratio())
        axes2.tick_params(labelsize=25)
        # axes2.set_xlabel(r'Size', fontsize=30)
        # axes2.set_ylabel("$SSNR^S$", fontsize=30)

        # axes1.legend(fontsize=20, loc="lower right")
        # fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
        if configuration == "Conventional":
            fig1.savefig(f'{current_dir}/Figures/ssnr_volume_3d_{configuration}_{label.replace(" ", "_")}.png', bbox_inches='tight', pad_inches=0.1)
            fig2.savefig(f'{current_dir}/Figures/ssnr_entropy_3d_{configuration}_{label.replace(" ", "_")}.png', bbox_inches='tight', pad_inches=0.1)

    plt.show()
