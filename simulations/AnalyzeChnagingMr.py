import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = '../simulations/varying_mr.csv'
data = pd.read_csv(file_path)
fig1, axes1 = plt.subplots(figsize=(10, 10))
fig2, axes2 = plt.subplots(figsize=(10, 10))
# fig3, axes3 = plt.subplots(3, 3, figsize=(12, 12))
# fig4, axes4 = plt.subplots(3, 3, figsize=(12, 12))
# fig1.suptitle("Dependence of the setup SSNR volume on the incident angle and sine ratio")
# fig2.suptitle("Dependence of the setup SSNR entropy on the incident angle and sine ratio")
# fig3.suptitle("Dependence o   f the setup SSNR measure on the incident angle and sine ratio")
# fig4.suptitle("Dependence of the setup total SSNR on the incident angle and sine ratio")

    # Filter data for the current combination
print(data[data['IncidentAngle'] == 0])
volume_widefield = int(data[data['IncidentAngle'] == 0]['Volume'].iloc[0])
entropy_widefield = int(data[data['IncidentAngle'] == 0]['Entropy'].iloc[0])
print(volume_widefield, entropy_widefield)
    # Plotting
incident_angles = np.array(data['IncidentAngle'].unique().astype(int))
incident_angles = incident_angles[incident_angles > 0]
Mrs = data['Mr'].unique().astype(int)
for Mr in Mrs:
    if Mr == 0:
        continue
    plot_data = data[data['Mr'] == Mr]
    print(Mr)
    # if Mr != 3 and Mr != 5 and Mr != 2:
    #     continue
    analytical_volume = np.array(plot_data['Volume_a'])
    print(analytical_volume)
    computed_volume = np.array(plot_data['Volume'])
    entropy = np.array(plot_data['Entropy'])
    ratios = np.sin(incident_angles/57.29)/np.sin(2 * np.pi / 5)
    print(ratios)
    axes1.plot(ratios, (analytical_volume - volume_widefield)/volume_widefield, '--')
    color = axes1.lines[-1].get_color()
    axes1.plot(ratios, (computed_volume - volume_widefield)/volume_widefield, label=Mr, color=color)
    axes2.plot(ratios, (entropy - entropy_widefield)/entropy_widefield, label=Mr)

    axes1.set_title("SSNR volume gain \n for a different number of rotations", fontsize=30)
    axes2.set_title("SSNR entropy gain \n for a different number of rotations", fontsize=30)


axes1.grid()
# axes1.set_xlim(0, 2)
# axes1.set_ylim(0, 1)
axes1.set_aspect(1 / axes1.get_data_ratio())
axes1.tick_params(labelsize=25)
axes1.set_xlabel(r"$sin(\theta)$/NA", fontsize=30)
axes1.set_ylabel("$SSNR^V_{GAIN}$", fontsize=30)
axes1.legend(fontsize=20)

axes2.grid()
# axes2.set_xlim(0, 2)
# axes2.set_ylim(0, 0.2)
axes2.set_aspect(1 / axes2.get_data_ratio())
axes2.tick_params(labelsize=25)
axes2.set_xlabel(r'$sin(\theta)$/NA', fontsize=30)
axes2.set_ylabel("$SSNR^S_{GAIN}$", fontsize=30)

axes1.legend(fontsize=20, loc="lower right")
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig1.savefig(f'{path_to_figures}comparsison_with_theory_projective')

axes2.legend(fontsize=20, loc="lower right")
fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig3.legend(loc='center left')
# fig3.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig4.legend(loc='center left')
# fig4.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig1.savefig(f'{path_to_figures}5wavesSimulationVolumeComparison')
# fig2.savefig(f'{path_to_figures}5wavesSimulationEntropyComparison')
# fig3.savefig(f'{path_to_figures}5wavesSimulationMeasureComparison')
plt.show()
