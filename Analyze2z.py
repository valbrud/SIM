import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the provided CSV file
file_path = '/mnt/data/5waves2z.csv'
data = pd.read_csv(file_path)

import seaborn as sns

# Setting the style for the plots
sns.set(style="whitegrid")

# Create a figure and a set of subplots for each combination of Power1 and Power2
fig, axes = plt.subplots(nrows=len(unique_power1), ncols=len(unique_power2), figsize=(15, 15), sharey=True)
fig.suptitle('Dependence of SSNR on Angle Theta for Different Sine Ratios and Powers', fontsize=16)

# Loop through each combination of Power1 and Power2 and plot the data
for i, power1 in enumerate(unique_power1):
    for j, power2 in enumerate(unique_power2):
        # Filter data for the specific power combination
        power_data = data[(data['Power1'] == power1) & (data['Power2'] == power2)]

        # Create a line plot for each sine ratio
        sns.lineplot(ax=axes[i, j], data=power_data, x='ThetaIncident', y='sum(log(1 + 10^8 SSNR))', hue='SineRatio', palette='viridis')

        # Setting plot titles and labels
        axes[i, j].set_title(f'Power1: {power1}, Power2: {power2}')
        axes[i, j].set_xlabel('Theta Incident')
        axes[i, j].set_ylabel('SSNR')

# Adjust layout for better readability
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Show the plot
plt.show()
