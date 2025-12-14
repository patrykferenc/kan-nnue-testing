#!/usr/bin/env python3
"""Plot W/D/L evaluation results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def plot_eval_comparison(csv_path, output_dir=None):
    """Plot model vs engine evaluation comparison."""
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    output_dir = Path(output_dir)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter plot: Engine vs Model evals
    ax = axes[0, 0]
    colors = {'W': 'green', 'D': 'gray', 'L': 'red'}
    for label in ['W', 'D', 'L']:
        mask = df['engine_label'] == label
        ax.scatter(df.loc[mask, 'engine_eval'], df.loc[mask, 'model_eval'],
                   c=colors[label], alpha=0.3, s=10, label=f'{label} (engine)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Engine Eval (cp)')
    ax.set_ylabel('Model Eval')
    ax.set_title('Model vs Engine Evaluation')
    ax.legend()

    # 2. Hexbin density plot
    ax = axes[0, 1]
    hb = ax.hexbin(df['engine_eval'], df['model_eval'], gridsize=40, cmap='YlOrRd', mincnt=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Engine Eval (cp)')
    ax.set_ylabel('Model Eval')
    ax.set_title('Evaluation Density')
    plt.colorbar(hb, ax=ax, label='Count')

    # 3. Error distribution
    ax = axes[1, 0]
    ax.hist(df['eval_difference'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Eval Difference (Model - Engine)')
    ax.set_ylabel('Count')
    ax.set_title(
        f'Eval Difference Distribution\nMean: {df["eval_difference"].mean():.1f}, Std: {df["eval_difference"].std():.1f}')

    # 4. Classification accuracy by engine eval range
    ax = axes[1, 1]
    df['eval_bin'] = pd.cut(df['engine_eval'], bins=20)
    accuracy_by_bin = df.groupby('eval_bin', observed=True)['correct'].mean()
    x_labels = [f"{int(i.left)}" for i in accuracy_by_bin.index]
    ax.bar(range(len(accuracy_by_bin)), accuracy_by_bin.values, edgecolor='black')
    ax.set_xticks(range(0, len(x_labels), 2))
    ax.set_xticklabels(x_labels[::2], rotation=45, ha='right')
    ax.set_xlabel('Engine Eval Range (cp)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Accuracy by Engine Eval Range')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = output_dir / 'wdl_eval_plots.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot W/D/L evaluation results")
    parser.add_argument("csv_path", help="Path to *_details.csv file from wdl_eval.py")
    parser.add_argument("--output-dir", help="Output directory for plots")
    args = parser.parse_args()

    plot_eval_comparison(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()