import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = '../simulations/SincKernels2D.csv'
data = pd.read_csv(file_path)
fig1, axes1 = plt.subplots(figsize=(10, 10))
fig2, axes2 = plt.subplots(figsize=(10, 10))
# fig3, axes3 = plt.subplots(3, 3, figsize=(12, 12))
# fig4, axes4 = plt.subplots(3, 3, figsize=(12, 12))
# fig1.suptitle("Dependence of the setup SSNR volume on the incident angle and sine ratio")
# fig2.suptitle("Dependence of the setup SSNR entropy on the incident angle and sine ratio")
# fig3.suptitle("Dependence of the setup SSNR measure on the incident angle and sine ratio")
# fig4.suptitle("Dependence of the setup total SSNR on the incident angle and sine ratio")

    # Filter data for the current combination
# print(data[data['IncidentAngle'] == 0])
# volume_widefield = int(data[data['IncidentAngle'] == 0]['Volume'].iloc[0])
# entropy_widefield = int(data[data['IncidentAngle'] == 0]['Entropy'].iloc[0])
# print(volume_widefield, entropy_widefield)
    # Plotting
sizes = np.array(data['Size'].unique().astype(int))
configurations = data['Configuration'].unique()
for configuration in configurations:
    if configuration == "Widefield":
        continue
    plot_data = data[data['Configuration'] == configuration]
    volume = np.array(plot_data['Volume'])
    entropy = np.array(plot_data['Entropy'])
    axes1.plot(sizes, volume, label = configuration)
    color = axes1.lines[-1].get_color()
    axes2.plot(sizes, entropy, label = configuration)

    axes1.set_title("SSNR volume ", fontsize=30)
    axes2.set_title("SSNR entropy ", fontsize=30)


axes1.grid()
# axes1.set_xlim(0, 2)
# axes1.set_ylim(0, 1)
axes1.set_aspect(1 / axes1.get_data_ratio())
axes1.tick_params(labelsize=25)
axes1.set_xlabel(r"Size", fontsize=30)
axes1.set_ylabel("$SSNR^V$", fontsize=30)
axes1.legend(fontsize=20)

axes2.grid()
# axes2.set_xlim(0, 2)
# axes2.set_ylim(0, 0.2)
axes2.set_aspect(1 / axes2.get_data_ratio())
axes2.tick_params(labelsize=25)
axes2.set_xlabel(r'Size', fontsize=30)
axes2.set_ylabel("$SSNR^S$", fontsize=30)

axes1.legend(fontsize=20, loc="lower right")
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig1.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/comparsison_with_theory_projective')

axes2.legend(fontsize=20, loc="lower right")
fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig3.legend(loc='center left')
# fig3.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig4.legend(loc='center left')
# fig4.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig1.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationVolumeComparison')
# fig2.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationEntropyComparison')
# fig3.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationMeasureComparison')
plt.show()
