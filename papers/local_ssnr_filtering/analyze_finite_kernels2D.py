import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

color_scheme = {"Widefield": "black",
                "Conventional": "red", 
                "Square": "green", 
                "Hexagonal": "blue",}

labels = {"Widefield": "Widefield",
                 "Conventional": "Conventional",  
                 "Square":       "Square",
                 "Hexagonal":    "Hexagonal"}
# Load the data from the provided CSV file
filepath1 = current_dir + '/RadiallySymmetricKernels.csv'
filepath2 = current_dir + '/SincKernels.csv'
files = [
    filepath1,
    filepath2
]

kernel_types = [
    "OTF-like",
    "Triangular"
]

linestyles = {
    "OTF-like": "-",
    "Triangular": "-.",

}

fig1, axes1 = plt.subplots(figsize=(9, 9))
fig2, axes2 = plt.subplots(figsize=(9, 9))


for (filepath, kernel_type) in zip(files, kernel_types):
    data = pd.read_csv(filepath)

    factors = np.round(np.array(data['Factor'].unique()), 2)
    configurations = data['Configuration'].unique()
    for configuration in configurations:
        if configuration == "Widefield":
            continue
        plot_data = data[data['Configuration'] == configuration]
        volume = np.array(plot_data['Volume'])
        entropy = np.array(plot_data['Entropy'])

        if kernel_type == "OTF-like":
            label = labels[configuration]
        else: 
            label = ""
            
        axes1.plot(factors[1:], volume[1:], color=color_scheme[configuration],  linestyle = linestyles[kernel_type], label = label)
        axes2.plot(factors[1:], entropy[1:], color=color_scheme[configuration],  linestyle = linestyles[kernel_type], label = label)
        # axes1.plot(m2[np.unravel_index(np.argmax(volume), volume.shape)[1]], m1[np.unravel_index(np.argmax(volume), volume.shape)[0]],  marker="x", linestyle="None")

        axes1.hlines(y=volume[0], xmin=factors[10], xmax=factors[22], color=color_scheme[configuration], linestyles='dashed')
        axes2.hlines(y=entropy[0], xmin=factors[10], xmax=factors[22], color=color_scheme[configuration], linestyles='dashed')
        # axes1.set_title("SSNR volume ", fontsize=30)
        # axes2.set_title("SSNR entropy ", fontsize=30)


axes1.grid()
axes1.set_xlim(0, 3)
# axes1.set_ylim(0, 1)
axes1.set_aspect(1 / axes1.get_data_ratio())
axes1.tick_params(labelsize=25)
axes1.set_xlabel(r"$s_l$", fontsize=30)
axes1.set_ylabel("$SSNR_{K, V}$", fontsize=30)
axes1.legend(fontsize=20)

axes2.grid()
axes2.set_xlim(0, 3)
axes2.set_ylim(800, 930)
# axes2.set_ylim(0, 0.2)
axes2.set_aspect(1 / axes2.get_data_ratio())
axes2.tick_params(labelsize=25)
axes2.set_xlabel(r'$s_l$', fontsize=30)
axes2.set_ylabel("$SSNR_{K, S}$", fontsize=30)

axes1.legend(fontsize=20, loc="upper right")
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)

# Create custom legend handles for line styles
# Create two separate legends
# First legend for configurations (existing)
axes2.legend(fontsize=20, loc="upper right")

fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig3.legend(loc='center left')
# fig3.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig4.legend(loc='center left')
# fig4.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig1.savefig(f'{current_dir}/Figures/ssnr/ssnr_volume_kernel_size_comparison.png', bbox_inches='tight', pad_inches=0.1)
fig2.savefig(f'{current_dir}/Figures/ssnr/ssnr_entropy_kernel_size_comparison.png', bbox_inches='tight', pad_inches=0.1)
# fig3.savefig(f'{path_to_figures}5wavesSimulationMeasureComparison')
plt.show()
