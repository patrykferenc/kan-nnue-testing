import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# Adjust this to your actual root
ROOT = pathlib.Path(".")

model_files = {
    "sfnnv9 (referencyjny)": ROOT / "sfnnv9" / "res.csv",
    "kan (3072/15/32) _g=3,k=3_": ROOT / "sfnnvkan_early_3072_15_32" / "res.csv",
    "mlp_kan (3072/15/32) _g=3,k=3_": ROOT / "sfnnvkan_g3_k3_3072_15_32_mlp" / "res.csv",
    "mlp_kan (3072/12/28) _g=3,k=3_": ROOT / "sfnnvkan_g3_k3_3072_12_28_mlp" / "res.csv",
    "mlp_kan (3072/2/4) _g=3,k=3_": ROOT / "sfnnvkan_g3_k3_3072_2_4_mlp" / "res.csv",
    "mlp_kan (3072/2/4) _g=2,k=2_": ROOT / "sfnnvkan_g2_k2_3072_2_4_mlp" / "res.csv",
}

min_depth = 5

# Load models
model_dfs = []
for model_name, path in model_files.items():
    if path.exists():
        df = pd.read_csv(path)
        df["model"] = model_name
        model_dfs.append(df)

model_data = pd.concat(model_dfs, ignore_index=True)
model_data = model_data[model_data["depth"] >= min_depth].sort_values(["model", "depth"])

# Load and compute Stockfish
stock_path = ROOT / "stockfish" / "res.csv"
if stock_path.exists():
    stock_df = pd.read_csv(stock_path)
    stock_df['branching_factor'] = np.power(stock_df['nodes'], 1.0 / stock_df['depth'])
    stock_df = stock_df[stock_df["depth"] >= min_depth].sort_values("depth")
    stock_df["model"] = "Stockfish (real)"  # For consistency, though not used in plot
else:
    stock_df = pd.DataFrame()  # Empty if missing

plt.figure(figsize=(12, 7))  # Slightly wider for legend

# Plot models
ax = sns.lineplot(data=model_data, x="depth", y="branching_factor", hue="model", marker="o")

# Overlay Stockfish standout line
if not stock_df.empty:
    ax.plot(stock_df["depth"], stock_df["branching_factor"],
            marker='s', linewidth=2, markersize=4, linestyle='--',
            color='black', label='Stockfish 17.1', zorder=10)

plt.title('Współczynnik rozgałęzienia $B_f$  w zależności od głębokości wyszukiwania $d$')
plt.xlabel('$d$')
plt.ylabel('$B_f$')
plt.legend(title="Model", loc="lower left")
plt.tight_layout()
plt.savefig('branching_plot.svg', format='svg', bbox_inches='tight', dpi=300)
plt.show()
