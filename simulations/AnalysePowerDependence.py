import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = '../simulations/ALL_b.csv'
data = pd.read_csv(file_path)
print(data['Configuration'].unique())
fig1, axes1 = plt.subplots(figsize=(12, 10))
fig2, axes2 = plt.subplots(figsize=(12, 10))
# fig3, axes3 = plt.subplots(3, 3, figsize=(12, 12))
# fig4, axes4 = plt.subplots(3, 3, figsize=(12, 12))
# fig1.suptitle("Dependence of the setup SSNR volume on the incident angle and sine ratio")
# fig2.suptitle("Dependence of the setup SSNR entropy on the incident angle and sine ratio")
# fig3.suptitle("Dependence of the setup SSNR measure on the incident angle and sine ratio")
# fig4.suptitle("Dependence of the setup total SSNR on the incident angle and sine ratio")

    # Filter data for the current combination
configurations = data['Configuration'].unique()
volume_widefield = int(data[data['Configuration'] == 'Widefield']['Volume'].iloc[0])
entropy_widefield = int(data[data['Configuration'] == 'Widefield']['Entropy'].iloc[0])
print(volume_widefield, entropy_widefield)
    # Plotting
for configuration in configurations:
    # Convert analytical volume to rounded integer values
    if configuration== 'Widefield':
        continue
    print(configuration)
    plot_data = data[data['Configuration'] == configuration]
    print(data)
    bs = np.array(plot_data['Power'])
    analytical_volume = np.array(plot_data['Volume_a'])
    computed_volume = np.array(plot_data['Volume'])
    entropy = np.array(plot_data['Entropy'])
    axes1.plot(bs, (analytical_volume - volume_widefield)/volume_widefield, '--')
    color = axes1.lines[-1].get_color()

    ssnrvinc = (computed_volume - volume_widefield)/volume_widefield
    ssnrsinc = (entropy - entropy_widefield)/entropy_widefield
    axes1.plot(bs, ssnrvinc, label=configuration, color=color)
    axes2.plot(bs, ssnrsinc , label=configuration)
    if configuration == "SquareC":
        axes1.plot(0.55, ssnrvinc[5], 'o',  ms = 10, color=color)
        axes2.plot(0.55, ssnrsinc[5], 'o', ms = 10, color=color)
    else:
        axes1.plot(bs[9], ssnrvinc[9], 'o',  ms = 10,  color=color)
        axes2.plot(bs[9], ssnrsinc[9], 'o',  ms = 10,  color=color)

    axes1.set_title("SSNR volume increase \n for different intensity ratios", fontsize=30)
    axes2.set_title("SSNR entropy increase \n for different intensity ratios", fontsize=30)
    # axes3[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
    # axes4[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
    # axes2[i, j].set_ylim(0, 3000)
    # axes3[i, j].set_ylim(0, 100)
    # axes4[i, j].set_ylim(0, 250)

axes1.grid()
axes1.set_xlim(0, 2)
axes1.set_ylim(0, 0.8)
axes1.set_aspect(1 / axes1.get_data_ratio())
axes1.tick_params(labelsize=25)
axes1.set_xlabel(r"r", fontsize=30)
axes1.set_ylabel("$SSNR^{INC}_V$", fontsize=30)
axes1.legend(fontsize=20)

axes2.grid()
axes2.set_xlim(0, 2)
axes2.set_ylim(0, 0.2)
axes2.set_aspect(1 / axes2.get_data_ratio())
axes2.tick_params(labelsize=25)
axes2.set_xlabel(r'r', fontsize=30)
axes2.set_ylabel("$SSNR^{INC}_S$", fontsize=30)

axes1.legend(fontsize=20, loc="lower right")
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig1.savefig(f'{path_to_figures}V_GAIN_b')

axes2.legend(fontsize=20, loc="lower right")
fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig2.savefig(f'{path_to_figures}S_GAIN_b')
plt.show()
