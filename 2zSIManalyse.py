import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the provided CSV file
file_path = './simulations/5waves_true_entropy.csv'
data = pd.read_csv(file_path).apply(pd.to_numeric, errors = 'coerce')
fig1, axes1 = plt.subplots(3, 3, figsize=(12, 12))
fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))
fig3, axes3 = plt.subplots(3, 3, figsize=(12, 12))
fig1.suptitle("Dependence of the setup SSNR volume on the incident angle and sine ratio")
fig2.suptitle("Dependence of the setup SSNR entropy on the incident angle and sine ratio")
fig3.suptitle("Dependence of the setup SSNR measure on the incident angle and sine ratio")

pow1 = data[['Power1']].drop_duplicates()
pow1 = pow1[(pd.to_numeric(pow1['Power1'], errors ='coerce').notnull())]
pow1 = pow1[pow1['Power1'] > 0]
pow2 = data[['Power2',]].drop_duplicates()
pow2 = pow2[pd.to_numeric(pow2['Power2'], errors ='coerce').notnull()]
pow2 = pow2[pow2['Power2'] > 0]

print(pow2)
i, j = 0, 0
for _, row1 in pow1.iterrows():
    j = 0
    for _, row2 in pow2.iterrows():
        power1 = row1['Power1']
        print('power1 = ', power1)
        power2 = row2['Power2']
        print('power2 = ',  power2)

        # Filter data for the current combination
        data_to_plot = data[(data['Power1'] == power1) & (data['Power2'] == power2)]
        sine_ratios = data_to_plot['SineRatio'].unique()
        # Plotting
        for sine_ratio in sine_ratios:
            # if sine_ratio == 1:
            #     continue
            sine_data = data_to_plot[data_to_plot['SineRatio'] == sine_ratio]
            # Convert analytical volume to rounded integer values
            analytical_volume_rounded = np.array(sine_data['analytical volume'].astype(float).round().astype(int))[0]
            print(analytical_volume_rounded)
            if i == 0 and j == 0:
                # axes1[i, j].plot(sine_data['ThetaIncident'], sine_data['volume'], label=f'{round(sine_ratio, 1)}')
                axes2[i, j].plot(sine_data['ThetaIncident'], sine_data['entropy'], label=f'{round(sine_ratio, 1)}')
                # axes3[i, j].plot(sine_data['ThetaIncident'], sine_data['measure'], label=f'{round(sine_ratio, 1)}')
                # if sine_ratio == 0.1:
                #     axes1[0, 0].hlines(y=267, xmin=0, xmax=90, linewidth=1, color = 'black', label = "Widefield baseline")
                #     axes1[0, 0].hlines(y=analytical_volume_rounded, xmin=0, xmax=90, linewidth=1, color='red', label="Analytic volume")
            else:
                # axes1[i, j].plot(sine_data['ThetaIncident'], sine_data['volume'])
                axes2[i, j].plot(sine_data['ThetaIncident'], sine_data['entropy'])
                # axes3[i, j].plot(sine_data['ThetaIncident'], sine_data['measure'])
                # if sine_ratio == 0.1:
                #     axes1[i, j].hlines(y=267, xmin=0, xmax=90, linewidth=1, color = 'black')
                #     axes1[i, j].hlines(y=analytical_volume_rounded, xmin=0, xmax=90, linewidth=1, color='red')
            axes1[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
            axes2[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
            axes3[i, j].set_title("p1 = {}, p2 = {}".format(power1, power2))
            axes1[i, j].set_ylim(250, 600)
            axes2[i, j].set_ylim(3000, 6200)
            axes3[i, j].set_ylim(0, 100)
        j+=1
    i+=1

fig1.legend(loc='center left')
fig1.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig2.legend(loc='center left')
fig2.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
fig3.legend(loc='center left')
fig3.tight_layout(rect=[0.15, 0, 1, 0.96], pad=1.5)
# fig1.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationVolumeComparison')
fig2.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationEntropyComparison')
# fig3.savefig('/home/valerii/Documents/projects/SIM/SSNR_article_1/Figures/5wavesSimulationMeasureComparison')
plt.show()
