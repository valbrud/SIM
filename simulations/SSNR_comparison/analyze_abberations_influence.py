import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)
sys.path.append(current_dir)

import os.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your CSV file (adjust as needed)
file_path = current_dir + "/Aberrations.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
print("Columns in CSV:", df.columns)
print("First few rows:")
print(df.head())

# Get the unique configurations and aberration types (from the entire file)
configurations = (df["Configuration"].unique())
aberration_types = sorted(df["Aberration"].unique())

# For consistent colors among configurations in the second set,
# we build a color mapping for configurations.
# cmap_config = plt.cm.get_cmap("Set1", len(configurations))
# config_color_map = {conf: cmap_config(i) for i, conf in enumerate(configurations)}
# config_color_map = {'Conventional': 'blue', 'SquareL': 'orange', 'SquareC': 'green', 'Hexagonal': 'red'}
# -----------------------------------------------------------
# Second set of plots: one figure for normalized Volume and one for normalized SSNR,
# with one subplot per aberration type (3 subplots) and each subplot showing all configurations.
# -----------------------------------------------------------

# ---- Normalized Volume Plots ----
# fig2_vol.suptitle("Normalized Volume vs Aberration Strength\n(Each subplot: one Aberration type)", fontsize=20)

for idx, ab in enumerate(aberration_types):
    fig2_vol, ax = plt.subplots(figsize=(8, 6))
    # For each configuration, filter rows where Aberration equals the current type.
    for config in configurations:
        # Filter and sort by Aberration Strength
        df_sel = df[(df["Aberration"] == ab) & (df["Configuration"] == config)].sort_values("Aberration Strength")
        if df_sel.empty:
            continue
        strength = df_sel["Aberration Strength"].values
        # Normalize volume by the value at 0 aberration strength.
        try:
            ref_vol = df_sel[df_sel["Aberration Strength"] == 0.0]["Volume"].iloc[0]
        except IndexError:
            # If there is no row exactly at 0, skip this configuration
            continue
        norm_vol = df_sel["Volume"].values / ref_vol
        if config == "Widefield":
            ax.plot(strength, norm_vol, linestyle='-',
                    label=config, color='black')
            # plt.show()
        else:
            ax.plot(strength, norm_vol, linestyle='-',
                    label=config)

    ax.set_title(f"{ab}", fontsize=16)
    ax.set_xlabel("Aberration Strength $[\\frac{RMS}{0.072 \lambda}]$", fontsize=14)
    ax.set_ylabel("$SSNR_V /SSNR_V^{ideal} $", fontsize=14)
    ax.set_xlim(0, 2)
    ax.set_ylim(0.4, 1)
    ax.grid(True)
    ax.legend(fontsize=10)

    fig2_vol.tight_layout(rect=[0, 0, 1, 0.93])
    # fig2_vol.savefig(f"SSNR_volume_with_{ab}.png", bbox_inches='tight', pad_inches=0.1)
# ---- Normalized SSNR (Entropy) Plots ----
# fig2_ssnr.suptitle("Normalized SSNR (Entropy) vs Aberration Strength\n(Each subplot: one Aberration type)", fontsize=20)

for idx, ab in enumerate(aberration_types):
    fig2_ssnr, ax = plt.subplots(figsize=(8, 6))
    for config in configurations:
        df_sel = df[(df["Aberration"] == ab) & (df["Configuration"] == config)].sort_values("Aberration Strength")
        if df_sel.empty:
            continue
        strength = df_sel["Aberration Strength"].values
        try:
            ref_ssnr = df_sel[df_sel["Aberration Strength"] == 0.0]["Entropy"].iloc[0]
        except IndexError:
            continue
        norm_ssnr = df_sel["Entropy"].values / ref_ssnr
        if config == "Widefield":
            ax.plot(strength, norm_ssnr, linestyle='-',
                    label=config, color='black')
        else:
            ax.plot(strength, norm_ssnr, linestyle='-',
                    label=config)

    # ax.set_title(f"{ab}", fontsize=16)
    ax.set_xlabel("Aberration Strength $[\\frac{RMS}{0.072 \lambda}]$", fontsize=25)
    ax.set_ylabel("$SSNR_S /SSNR_S^{ideal} $", fontsize=25)
    ax.set_xlim(0, 2)
    ax.set_ylim(0.82, 1)
    ax.set_aspect(1/ax.get_data_ratio())
    ax.grid(True)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=17)

    fig2_ssnr.tight_layout(rect=[0, 0, 1, 0.93])
    # fig2_ssnr.savefig(f"SSNR_entropy_with_{ab}.png", bbox_inches='tight', pad_inches=0.1)

plt.show()
