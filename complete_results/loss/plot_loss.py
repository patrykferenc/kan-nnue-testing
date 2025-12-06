#!/usr/bin/env python3
"""Plot training curves from CSV files"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)


def load_metric_csv(csv_path):
    """Load a metric CSV file"""
    df = pd.DataFrame(pd.read_csv(csv_path))
    return df


def plot_single_comparison(data_dict, output_path, title="Training Loss",
                           xlabel="Step", ylabel="Loss", smoothing=0):
    """Plot all models on a single plot"""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Define a nice color palette
    colors = sns.color_palette("husl", len(data_dict))

    for (label, df), color in zip(data_dict.items(), colors):
        if smoothing > 0:
            # Apply exponential moving average for smoothing
            df['Value_smooth'] = df['Value'].ewm(span=smoothing).mean()
            ax.plot(df['Step'], df['Value_smooth'], label=label,
                    linewidth=2.5, color=color)
        else:
            ax.plot(df['Step'], df['Value'], label=label,
                    linewidth=2.5, color=color)

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(frameon=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")
    plt.close()


def plot_separate(data_dict, output_dir, title_prefix="Training Loss",
                  xlabel="Step", ylabel="Loss", smoothing=0):
    """Create separate plots for each model"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, df in data_dict.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        if smoothing > 0:
            df['Value_smooth'] = df['Value'].ewm(span=smoothing).mean()
            # Plot both raw and smoothed
            ax.plot(df['Step'], df['Value'], alpha=0.3,
                    linewidth=1, label='Raw', color='gray')
            ax.plot(df['Step'], df['Value_smooth'],
                    linewidth=2.5, label='Smoothed', color='#2E86AB')
        else:
            ax.plot(df['Step'], df['Value'], linewidth=2.5, color='#2E86AB')

        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f"{title_prefix} - {label}", fontsize=16,
                     fontweight='bold', pad=20)
        if smoothing > 0:
            ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Sanitize filename
        safe_label = label.replace('/', '_').replace(' ', '_')
        output_path = output_dir / f"{safe_label}_loss.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()


def plot_with_confidence_interval(data_dict, output_path, title="Training Loss",
                                  xlabel="Step", ylabel="Loss", window=100):
    """Plot with rolling mean and confidence interval"""

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("husl", len(data_dict))

    for (label, df), color in zip(data_dict.items(), colors):
        # Calculate rolling statistics
        df['rolling_mean'] = df['Value'].rolling(window=window, center=True).mean()
        df['rolling_std'] = df['Value'].rolling(window=window, center=True).std()

        # Plot mean line
        ax.plot(df['Step'], df['rolling_mean'], label=label,
                linewidth=2.5, color=color)

        # Plot confidence interval
        ax.fill_between(df['Step'],
                        df['rolling_mean'] - df['rolling_std'],
                        df['rolling_mean'] + df['rolling_std'],
                        alpha=0.2, color=color)

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(frameon=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot with confidence intervals to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models on single plot
  python plot_loss.py model1/train_loss.csv model2/train_loss.csv \\
      --labels "Baseline NNUE" "KAN-NNUE" \\
      --output comparison.png

  # Create separate plots
  python plot_loss.py model1/train_loss.csv model2/train_loss.csv \\
      --labels "Model A" "Model B" \\
      --separate --output-dir ./plots

  # With smoothing
  python plot_loss.py model1/train_loss.csv model2/train_loss.csv \\
      --labels "SFNNV9" "SFNNVKAN" \\
      --smoothing 50 --output comparison_smooth.png
        """
    )

    parser.add_argument('csv_files', nargs='+',
                        help='CSV files containing loss data')
    parser.add_argument('--labels', nargs='+',
                        help='Custom labels for each model (default: use folder names)')
    parser.add_argument('--output', default='training_loss_comparison.png',
                        help='Output file path for combined plot')
    parser.add_argument('--output-dir', default='./plots',
                        help='Output directory for separate plots')
    parser.add_argument('--separate', action='store_true',
                        help='Create separate plots for each model')
    parser.add_argument('--both', action='store_true',
                        help='Create both combined and separate plots')
    parser.add_argument('--title', default='Training Loss',
                        help='Plot title')
    parser.add_argument('--xlabel', default='Step',
                        help='X-axis label')
    parser.add_argument('--ylabel', default='Loss',
                        help='Y-axis label')
    parser.add_argument('--smoothing', type=int, default=0,
                        help='Smoothing window size (0=no smoothing)')
    parser.add_argument('--confidence', action='store_true',
                        help='Plot with confidence intervals instead of smoothing')
    parser.add_argument('--window', type=int, default=100,
                        help='Window size for confidence interval calculation')

    args = parser.parse_args()

    # Load data
    data_dict = {}
    for i, csv_file in enumerate(args.csv_files):
        csv_path = Path(csv_file)

        if not csv_path.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue

        # Determine label
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            # Use parent directory name as label
            label = csv_path.parent.name

        print(f"Loading {csv_file} as '{label}'...")
        df = load_metric_csv(csv_path)
        data_dict[label] = df

        print(f"  Loaded {len(df)} data points")
        print(f"  Step range: {df['Step'].min()} - {df['Step'].max()}")
        print(f"  Loss range: {df['Value'].min():.6f} - {df['Value'].max():.6f}")
        print()

    if not data_dict:
        print("Error: No valid CSV files found!")
        return

    # Create plots
    if args.confidence:
        plot_with_confidence_interval(data_dict, args.output,
                                      args.title, args.xlabel, args.ylabel,
                                      window=args.window)
    elif args.separate:
        plot_separate(data_dict, args.output_dir, args.title,
                      args.xlabel, args.ylabel, smoothing=args.smoothing)
    elif args.both:
        plot_single_comparison(data_dict, args.output, args.title,
                               args.xlabel, args.ylabel, smoothing=args.smoothing)
        plot_separate(data_dict, args.output_dir, args.title,
                      args.xlabel, args.ylabel, smoothing=args.smoothing)
    else:
        # Default: single comparison plot
        plot_single_comparison(data_dict, args.output, args.title,
                               args.xlabel, args.ylabel, smoothing=args.smoothing)

    print("\nDone!")


if __name__ == '__main__':
    main()