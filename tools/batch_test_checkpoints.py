#!/usr/bin/env python3
"""
Batch testing script for multiple neural network checkpoints.
Runs specified test suite on each checkpoint and organizes results.
"""
import os
import sys
import subprocess
import re
import json
from pathlib import Path
from datetime import datetime
import argparse


def find_checkpoints(directory, pattern=None):
    """Find all checkpoint files (.ckpt or .pt) in directory."""
    path = Path(directory)

    # Find all checkpoint files
    ckpts = list(path.glob("**/*.ckpt")) + list(path.glob("**/*.pt"))

    # Optional filtering by pattern
    if pattern:
        ckpts = [c for c in ckpts if re.search(pattern, c.name)]

    # Sort by modification time (or you could sort by epoch number)
    return sorted(ckpts, key=lambda x: x.stat().st_mtime)


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
        return {
            'epoch': epoch,
            'step': None,
            'label': f"epoch{epoch:04d}"
        }

    # Fallback: use filename
    return {
        'epoch': None,
        'step': None,
        'label': checkpoint_path.stem
    }


def run_test_for_checkpoint(checkpoint_path, model_name, test_script,
                            results_base_dir, compile_mode="default",
                            quiet=True, dry_run=False):
    """Run test script for a single checkpoint."""

    epoch_info = extract_epoch_info(checkpoint_path)

    # Create results subdirectory for this checkpoint
    results_dir = Path(results_base_dir) / epoch_info['label']
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build the engine command string that will be passed to the test script
    engine_cmd = f"./sunfish_nnue.py {model_name} {checkpoint_path} {compile_mode}"

    # Build the full command
    # Test scripts expect: script.sh "engine_cmd" "--quiet" "results_dir"
    quiet_flag = "--quiet" if quiet else ""
    cmd = ["bash", test_script, engine_cmd, quiet_flag, str(results_dir)]

    # Print info
    print(f"\n{'=' * 80}")
    print(f"Testing checkpoint: {checkpoint_path.name}")
    print(f"  Epoch: {epoch_info['epoch']}, Step: {epoch_info['step']}")
    print(f"  Label: {epoch_info['label']}")
    print(f"  Results: {results_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    if dry_run:
        print("DRY RUN - Not executing\n")
        return {'success': True, 'dry_run': True}

    # Run the test
    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n✓ Completed {checkpoint_path.name} in {elapsed:.1f}s")

        return {
            'success': True,
            'elapsed_seconds': elapsed,
            'epoch_info': epoch_info
        }

    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✗ Failed {checkpoint_path.name} after {elapsed:.1f}s: {e}")

        return {
            'success': False,
            'elapsed_seconds': elapsed,
            'epoch_info': epoch_info,
            'error': str(e)
        }


def save_batch_summary(results_base_dir, checkpoints_tested, test_results, args):
    """Save a summary JSON of the batch test run."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_directory': str(args.checkpoint_dir),
        'model_name': args.model_name,
        'test_script': args.test_script,
        'total_checkpoints': len(checkpoints_tested),
        'successful': sum(1 for r in test_results if r['success']),
        'failed': sum(1 for r in test_results if not r['success']),
        'results': []
    }

    for ckpt, result in zip(checkpoints_tested, test_results):
        summary['results'].append({
            'checkpoint': str(ckpt),
            'epoch': result['epoch_info']['epoch'],
            'step': result['epoch_info']['step'],
            'label': result['epoch_info']['label'],
            'success': result['success'],
            'elapsed_seconds': result.get('elapsed_seconds', 0)
        })

    summary_path = Path(results_base_dir) / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBatch summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch test multiple neural network checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all checkpoints with test_other.sh
  python batch_test_checkpoints.py /path/to/checkpoints sfnnvkan_early_3072_15_32

  # Test with test_basic.sh instead
  python batch_test_checkpoints.py /path/to/checkpoints sfnnvkan_early_3072_15_32 \\
      --test-script ./tools/test_basic.sh

  # Test only first 3 checkpoints (for debugging)
  python batch_test_checkpoints.py /path/to/checkpoints sfnnvkan_early_3072_15_32 \\
      --limit 3

  # Dry run to see what would be tested
  python batch_test_checkpoints.py /path/to/checkpoints sfnnvkan_early_3072_15_32 \\
      --dry-run
        """
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Directory containing checkpoint files (.ckpt or .pt)"
    )
    parser.add_argument(
        "model_name",
        help="Model name (e.g., sfnnvkan_early_3072_15_32)"
    )
    parser.add_argument(
        "--test-script",
        default="./tools/test_other.sh",
        help="Test script to run (default: ./tools/test_other.sh)"
    )
    parser.add_argument(
        "--results-dir",
        default="./batch_test_results",
        help="Base directory for results (default: ./batch_test_results)"
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        help="Torch compile mode (default: default)"
    )
    parser.add_argument(
        "--pattern",
        help="Filter checkpoints by regex pattern (e.g., 'epoch=5[0-9][0-9]')"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=True,
        help="Pass --quiet flag to test script (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (opposite of --quiet)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to first N checkpoints (useful for testing)"
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

    # Handle quiet/verbose
    quiet = args.quiet and not args.verbose

    # Find all checkpoints
    print(f"Searching for checkpoints in: {args.checkpoint_dir}")
    checkpoints = find_checkpoints(args.checkpoint_dir, args.pattern)

    if not checkpoints:
        print(f"✗ No checkpoint files found in {args.checkpoint_dir}")
        if args.pattern:
            print(f"  (with pattern: {args.pattern})")
        return 1

    print(f"Found {len(checkpoints)} checkpoint(s)")

    # Apply skip
    if args.skip > 0:
        checkpoints = checkpoints[args.skip:]
        print(f"Skipped first {args.skip}, {len(checkpoints)} remaining")

    # Apply limit
    if args.limit:
        checkpoints = checkpoints[:args.limit]
        print(f"Limited to first {args.limit} checkpoint(s)")

    # Show checkpoint list
    print("\nCheckpoints to test:")
    for i, ckpt in enumerate(checkpoints, 1):
        info = extract_epoch_info(ckpt)
        print(f"  {i:3d}. {ckpt.name} → {info['label']}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No tests will be executed]")

    # Confirm
    if not args.dry_run and len(checkpoints) > 5:
        response = input(f"\nProceed with testing {len(checkpoints)} checkpoints? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    # Create base results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Run tests
    print(f"\n{'=' * 80}")
    print(f"Starting batch testing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    test_results = []
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Processing checkpoint")

        result = run_test_for_checkpoint(
            checkpoint,
            args.model_name,
            args.test_script,
            args.results_dir,
            args.compile_mode,
            quiet,
            args.dry_run
        )
        test_results.append(result)

    # Save summary
    if not args.dry_run:
        save_batch_summary(args.results_dir, checkpoints, test_results, args)

    # Final summary
    success_count = sum(1 for r in test_results if r['success'])
    total_time = sum(r.get('elapsed_seconds', 0) for r in test_results)

    print(f"\n{'=' * 80}")
    print(f"Batch testing complete!")
    print(f"{'=' * 80}")
    print(f"  Successful: {success_count}/{len(checkpoints)}")
    print(f"  Failed: {len(checkpoints) - success_count}/{len(checkpoints)}")
    if not args.dry_run:
        print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        print(f"  Results directory: {args.results_dir}")
    print(f"{'=' * 80}\n")

    return 0 if success_count == len(checkpoints) else 1


if __name__ == "__main__":
    sys.exit(main())
