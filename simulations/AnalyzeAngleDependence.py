import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = '../simulations/Angles.csv'
data = pd.read_csv(file_path)
print(data['Configuration'].unique())
fig1, axes1 = plt.subplots(figsize=(12, 10))
fig2, axes2 = plt.subplots(figsize=(12, 10))

configurations = data['Configuration'].unique()
volume_widefield = int(data[data['Configuration'] == 'Widefield']['Volume'].iloc[0])
entropy_widefield = int(data[data['Configuration'] == 'Widefield']['Entropy'].iloc[0])
print(volume_widefield, entropy_widefield)

for configuration in configurations:
    if configuration== 'Widefield':
        continue

    print(configuration)
    plot_data = data[data['Configuration'] == configuration]
    print(data)
    ratio = np.array(plot_data['NA_ratio'])
    analytical_volume = np.array(plot_data['Volume_a'])
    computed_volume = np.array(plot_data['Volume'])
    entropy = np.array(plot_data['Entropy'])
    axes1.plot(ratio, (analytical_volume - volume_widefield)/volume_widefield, '--')
    color = axes1.lines[-1].get_color()
    axes1.plot(ratio, (computed_volume - volume_widefield)/volume_widefield, label=configuration, color=color)
    axes2.plot(ratio, (entropy - entropy_widefield)/entropy_widefield, label=configuration)

    axes1.set_title("SSNR volume gain \n for different incoming angles", fontsize=30)
    axes2.set_title("SSNR entropy gain \n for different incoming angles", fontsize=30)
    # axes3[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
    # axes4[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
    # axes2[i, j].set_ylim(0, 3000)
    # axes3[i, j].set_ylim(0, 100)
    # axes4[i, j].set_ylim(0, 250)

axes1.grid()
axes1.set_xlim(0, 1)
axes1.set_ylim(0, 1)
axes1.set_aspect(1 / axes1.get_data_ratio())
axes1.tick_params(labelsize=25)
axes1.set_xlabel(r"$sin(\theta)/NA$", fontsize=30)
axes1.set_ylabel(r"$SSNR^V_{GAIN}$", fontsize=30)
axes1.legend(fontsize=20)

axes2.grid()
axes2.set_xlim(0, 1)
axes2.set_ylim(0, 0.2)
axes2.set_aspect(1 / axes2.get_data_ratio())
axes2.tick_params(labelsize=25)
axes2.set_xlabel(r'$sin(\theta)/NA$', fontsize=30)
axes2.set_ylabel("$SSNR^S_{GAIN}$", fontsize=30)

axes1.legend(fontsize=20, loc="lower right")
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig1.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/V_GAIN_b')

axes2.legend(fontsize=20, loc="lower right")
fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig2.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/S_GAIN_b')
plt.show()
