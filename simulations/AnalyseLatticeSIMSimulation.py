import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = 'simulations/lattice_setups.csv'
data = pd.read_csv(file_path)
print(data['Configuration'])
fig1, axes1 = plt.subplots(figsize=(12, 12))
fig2, axes2 = plt.subplots(figsize=(12, 12))
# fig3, axes3 = plt.subplots(3, 3, figsize=(12, 12))
# fig4, axes4 = plt.subplots(3, 3, figsize=(12, 12))
# fig1.suptitle("Dependence of the setup SSNR volume on the incident angle and sine ratio")
# fig2.suptitle("Dependence of the setup SSNR entropy on the incident angle and sine ratio")
# fig3.suptitle("Dependence of the setup SSNR measure on the incident angle and sine ratio")
# fig4.suptitle("Dependence of the setup total SSNR on the incident angle and sine ratio")

    # Filter data for the current combination
configurations = data['Configuration'].unique()
    # Plotting
for configuration in configurations:
    print(configuration)
    if configuration == '0' or configuration =="State Of Art":
        continue

    plot_data = data[data['Configuration'] == configuration]
    print(data)
    # Convert analytical volume to rounded integer values
    analytical_volume_rounded = np.array(plot_data['volume_a']).astype(int)
    axes1.plot(np.sin(plot_data['IncidentAngle'].astype(int)/57.29)/np.sin(2 * np.pi /5), plot_data['volume'].astype(float), label=configuration)
    color = axes1.lines[-1].get_color()
    axes1.plot(np.sin(plot_data['IncidentAngle'].astype(int)/57.29)/np.sin(2 * np.pi /5), plot_data['volume_a'].astype(float), '--', color=color)
    # axes2. plot(plot_data['IncidentAngle'], plot_data['entropy'].astype(float), label=configuration)
    # color = axes2.lines[-1].get_color()
    axes2.plot(np.sin(plot_data['IncidentAngle'].astype(int)/57.29)/np.sin(2 * np.pi /5), plot_data['entropy'].astype(float), label = configuration, color=color)


    axes1.set_title("SSNR volume", fontsize=30)
    axes2.set_title("SSNR entropy", fontsize=30)
    # axes3[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
    # axes4[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
    # axes2[i, j].set_ylim(0, 3000)
    # axes3[i, j].set_ylim(0, 100)
    # axes4[i, j].set_ylim(0, 250)

axes1.grid()
axes1.set_aspect(1 / axes1.get_data_ratio())
axes1.set_xlim(0.15, 1)
axes1.tick_params(labelsize=25)
axes1.set_xlabel(r"sin(\theta)/NA", fontsize=30)
axes1.set_ylabel("SSNR volume, a. u. ", fontsize=30)
axes1.legend(fontsize=20)

axes2.grid()
axes2.set_aspect(0.8/axes2.get_data_ratio())
axes2.set_xlim(0.15, 1)
axes2.tick_params(labelsize=25)
axes2.set_xlabel(r'$sin(\theta)/NA$', fontsize=30)
axes2.set_ylabel("SSNR entropy, a. u. ", fontsize=30)

axes1.legend(fontsize=20, loc="upper right")
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
axes2.legend(fontsize=20, loc="upper right")
fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig3.legend(loc='center left')
# fig3.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig4.legend(loc='center left')
# fig4.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig1.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationVolumeComparison')
# fig2.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationEntropyComparison')
# fig3.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationMeasureComparison')
plt.show()
