#!/usr/bin/env python3
"""
Batch W/D/L evaluation for multiple checkpoints.
Tracks classification accuracy evolution during training.
"""

import sys
import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime
import argparse
import re


def find_checkpoints(directory, pattern=None):
    """Find all checkpoint files (.ckpt or .pt) in directory."""
    path = Path(directory)
    ckpts = list(path.glob("**/*.ckpt")) + list(path.glob("**/*.pt"))

    if pattern:
        ckpts = [c for c in ckpts if re.search(pattern, c.name)]

    return sorted(ckpts, key=lambda x: extract_epoch_info(x)['epoch'] or 0)


def extract_epoch_info(checkpoint_path):
    """Extract epoch/step information from checkpoint filename."""
    filename = checkpoint_path.name

    # Pattern 1: epoch=599-step=2929800.ckpt
    match = re.search(r'epoch[=_](\d+)(?:[-_]step[=_](\d+))?', filename, re.IGNORECASE)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2)) if match.group(2) else None
        return {
            'epoch': epoch,
            'step': step,
            'label': f"epoch{epoch:04d}" + (f"_step{step}" if step else "")
        }

    # Pattern 2: nn-epoch459.pt
    match = re.search(r'epoch(\d+)', filename, re.IGNORECASE)
    if match:
        epoch = int(match.group(1))
        return {'epoch': epoch, 'step': None, 'label': f"epoch{epoch:04d}"}

    # Fallback: use file modification time as "epoch" for sorting
    return {
        'epoch': 0,  # Changed from None to 0
        'step': None,
        'label': checkpoint_path.stem
    }


def run_wdl_eval(checkpoint_path, config_path, output_dir,
                 calibrate=True, engine_threshold=150, model_threshold=None):
    """Run W/D/L evaluation for a single checkpoint."""

    epoch_info = extract_epoch_info(checkpoint_path)

    # Create output directory for this checkpoint
    checkpoint_output = Path(output_dir) / epoch_info['label']
    checkpoint_output.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", "wdl_eval.py",
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--output-dir", str(checkpoint_output),
        "--engine-threshold", str(engine_threshold)
    ]

    if calibrate:
        cmd.extend(["--calibrate-threshold", "--calibration-samples", "2000"])
    elif model_threshold is not None:
        cmd.extend(["--model-threshold", str(model_threshold)])

    print(f"\n{'=' * 80}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"  Epoch: {epoch_info['epoch']}, Step: {epoch_info['step']}")
    print(f"  Output: {checkpoint_output}")
    print(f"{'=' * 80}\n")

    start_time = datetime.now()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Load the results
        summary_files = list(checkpoint_output.glob("wdl_*_summary.json"))
        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)
                metrics = summary['metrics']
                thresholds = summary['thresholds']
        else:
            metrics = None
            thresholds = None

        print(f"\n✓ Completed in {elapsed:.1f}s")

        return {
            'success': True,
            'elapsed_seconds': elapsed,
            'epoch_info': epoch_info,
            'metrics': metrics,
            'thresholds': thresholds,
            'output_dir': str(checkpoint_output)
        }

    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✗ Failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")

        return {
            'success': False,
            'elapsed_seconds': elapsed,
            'epoch_info': epoch_info,
            'error': str(e)
        }


def save_batch_summary(results, output_dir, args):
    """Save comprehensive batch summary with metrics evolution."""

    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'checkpoint_directory': str(args.checkpoint_dir),
            'config_path': str(args.config),
            'engine_threshold': args.engine_threshold,
            'calibrate': args.calibrate,
        },
        'total_checkpoints': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'checkpoints': []
    }

    # Extract checkpoint-level data
    for result in results:
        checkpoint_data = {
            'epoch': result['epoch_info']['epoch'],
            'step': result['epoch_info']['step'],
            'label': result['epoch_info']['label'],
            'success': result['success'],
            'elapsed_seconds': result.get('elapsed_seconds', 0)
        }

        if result['success'] and result.get('metrics'):
            checkpoint_data.update({
                'accuracy': result['metrics']['accuracy'],
                'macro_f1': result['metrics']['macro_f1'],
                'win_precision': result['metrics']['class_metrics']['W']['precision'],
                'win_recall': result['metrics']['class_metrics']['W']['recall'],
                'win_f1': result['metrics']['class_metrics']['W']['f1'],
                'draw_precision': result['metrics']['class_metrics']['D']['precision'],
                'draw_recall': result['metrics']['class_metrics']['D']['recall'],
                'draw_f1': result['metrics']['class_metrics']['D']['f1'],
                'loss_precision': result['metrics']['class_metrics']['L']['precision'],
                'loss_recall': result['metrics']['class_metrics']['L']['recall'],
                'loss_f1': result['metrics']['class_metrics']['L']['f1'],
            })

            if result.get('thresholds'):
                checkpoint_data['model_threshold'] = result['thresholds']['model_win_threshold']
                checkpoint_data['threshold_ratio'] = result['thresholds']['threshold_ratio']

        summary['checkpoints'].append(checkpoint_data)

    # Save JSON summary
    summary_path = Path(output_dir) / 'batch_wdl_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nBatch summary saved to: {summary_path}")

    # Save CSV for easy plotting
    csv_path = Path(output_dir) / 'batch_wdl_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'epoch', 'step', 'label', 'accuracy', 'macro_f1',
            'win_precision', 'win_recall', 'win_f1',
            'draw_precision', 'draw_recall', 'draw_f1',
            'loss_precision', 'loss_recall', 'loss_f1',
            'model_threshold', 'threshold_ratio'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ckpt in summary['checkpoints']:
            if ckpt['success'] and 'accuracy' in ckpt:
                writer.writerow({k: ckpt.get(k, '') for k in fieldnames})

    print(f"Metrics CSV saved to: {csv_path}")

    return summary_path


def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("BATCH W/D/L EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Epoch':<8} {'Accuracy':<12} {'Macro F1':<12} {'W F1':<10} {'D F1':<10} {'L F1':<10}")
    print("-" * 80)

    for result in results:
        if result['success'] and result.get('metrics'):
            epoch = result['epoch_info']['epoch']
            metrics = result['metrics']

            print(f"{epoch:<8} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['macro_f1']:<12.4f} "
                  f"{metrics['class_metrics']['W']['f1']:<10.4f} "
                  f"{metrics['class_metrics']['D']['f1']:<10.4f} "
                  f"{metrics['class_metrics']['L']['f1']:<10.4f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch W/D/L evaluation for multiple checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoints with auto-calibrated thresholds
  python batch_wdl_eval.py /path/to/checkpoints config.json

  # Use fixed model threshold
  python batch_wdl_eval.py /path/to/checkpoints config.json \\
      --no-calibrate --model-threshold 400

  # Test only first 3 checkpoints
  python batch_wdl_eval.py /path/to/checkpoints config.json --limit 3

  # Use different engine threshold (2.0 pawns = 200cp)
  python batch_wdl_eval.py /path/to/checkpoints config.json \\
      --engine-threshold 200
        """
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "config",
        help="Configuration JSON file for W/D/L evaluation"
    )
    parser.add_argument(
        "--output-dir",
        default="./batch_wdl_results",
        help="Base directory for results (default: ./batch_wdl_results)"
    )
    parser.add_argument(
        "--engine-threshold",
        type=int,
        default=150,
        help="Engine threshold in centipawns (default: 150 = 1.5 pawns)"
    )
    parser.add_argument(
        "--model-threshold",
        type=int,
        help="Fixed model threshold (if not calibrating)"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=True,
        help="Auto-calibrate model threshold (default: True)"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_false",
        dest="calibrate",
        help="Don't calibrate, use fixed model threshold"
    )
    parser.add_argument(
        "--pattern",
        help="Filter checkpoints by regex pattern"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N checkpoints"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N checkpoints"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )

    args = parser.parse_args()

    # Validate config exists
    if not Path(args.config).exists():
        print(f"✗ Config file not found: {args.config}")
        return 1

    # Find checkpoints
    print(f"Searching for checkpoints in: {args.checkpoint_dir}")
    checkpoints = find_checkpoints(args.checkpoint_dir, args.pattern)

    if not checkpoints:
        print(f"✗ No checkpoint files found")
        return 1

    print(f"Found {len(checkpoints)} checkpoint(s)")

    # Apply skip and limit
    if args.skip > 0:
        checkpoints = checkpoints[args.skip:]
        print(f"Skipped first {args.skip}, {len(checkpoints)} remaining")

    if args.limit:
        checkpoints = checkpoints[:args.limit]
        print(f"Limited to first {args.limit}")

    # Show checkpoint list
    print("\nCheckpoints to evaluate:")
    for i, ckpt in enumerate(checkpoints, 1):
        info = extract_epoch_info(ckpt)
        epoch_str = f"Epoch {info['epoch']:>4}" if info['epoch'] > 0 else "Unknown epoch"
        print(f"  {i:3d}. {epoch_str:20s} - {ckpt.name}")

    if args.dry_run:
        print("\n[DRY RUN MODE]")
        return 0

    # Confirm
    if len(checkpoints) > 3:
        response = input(f"\nProceed with evaluating {len(checkpoints)} checkpoints? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluations
    print(f"\n{'=' * 80}")
    print(f"Starting batch W/D/L evaluation")
    print(f"{'=' * 80}")

    results = []
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}]")

        result = run_wdl_eval(
            checkpoint,
            args.config,
            args.output_dir,
            calibrate=args.calibrate,
            engine_threshold=args.engine_threshold,
            model_threshold=args.model_threshold
        )
        results.append(result)

    # Save summary
    save_batch_summary(results, args.output_dir, args)

    # Print summary table
    print_summary_table(results)

    # Final stats
    success_count = sum(1 for r in results if r['success'])
    total_time = sum(r.get('elapsed_seconds', 0) for r in results)

    print(f"{'=' * 80}")
    print(f"Batch evaluation complete!")
    print(f"  Successful: {success_count}/{len(checkpoints)}")
    print(f"  Failed: {len(checkpoints) - success_count}/{len(checkpoints)}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"  Results directory: {args.output_dir}")
    print(f"{'=' * 80}\n")

    return 0 if success_count == len(checkpoints) else 1


if __name__ == "__main__":
    sys.exit(main())
