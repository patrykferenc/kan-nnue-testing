#!/usr/bin/env python3
"""
Win/Draw/Loss Classification Evaluation
Evaluates NNUE model's ability to correctly classify positions as winning, drawing, or losing.
"""

import argparse
import json
import csv
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import chess

import models
import commons.features as features
import commons.nnue_dataset as nnue_dataset


class WDLThresholds:
    """Manages thresholds for W/D/L classification with scaling."""

    def __init__(self, engine_win_threshold=150, model_win_threshold=None,
                 engine_scale=208, model_scale=600):
        """
        Args:
            engine_win_threshold: Threshold in centipawns for engine (default: 150 = 1.5 pawns)
            model_win_threshold: Threshold for model (if None, will be auto-calibrated)
            engine_scale: Scaling factor for engine NNUE output (default: 208)
            model_scale: Scaling factor for model output (default: 600)
        """
        self.engine_win_threshold = engine_win_threshold
        self.engine_scale = engine_scale
        self.model_scale = model_scale

        if model_win_threshold is None:
            # Auto-calibrate: scale the engine threshold to model space
            # Engine: NNUE output * 208 = centipawns
            # Model: output * 600 = model units
            # Ratio: (600/208) ≈ 2.88
            scale_ratio = model_scale / engine_scale
            self.model_win_threshold = int(engine_win_threshold * scale_ratio)
        else:
            self.model_win_threshold = model_win_threshold

    def classify(self, score, is_engine=False):
        """Classify a score as 'W', 'D', or 'L'."""
        threshold = self.engine_win_threshold if is_engine else self.model_win_threshold

        if score > threshold:
            return 'W'
        elif score < -threshold:
            return 'L'
        else:
            return 'D'

    def __repr__(self):
        return (f"WDLThresholds(engine={self.engine_win_threshold}cp, "
                f"model={self.model_win_threshold}, ratio={self.model_win_threshold / self.engine_win_threshold:.2f})")


def compute_confusion_matrix(true_labels, pred_labels):
    """Compute confusion matrix for W/D/L classification."""
    labels = ['W', 'D', 'L']
    matrix = {true_label: {pred_label: 0 for pred_label in labels}
              for true_label in labels}

    for true, pred in zip(true_labels, pred_labels):
        matrix[true][pred] += 1

    return matrix


def compute_classification_metrics(confusion_matrix):
    """Compute precision, recall, F1 for each class."""
    labels = ['W', 'D', 'L']
    metrics = {}

    for label in labels:
        # True positives: correctly classified as this label
        tp = confusion_matrix[label][label]

        # False positives: incorrectly classified as this label
        fp = sum(confusion_matrix[other][label] for other in labels if other != label)

        # False negatives: should be this label but classified as something else
        fn = sum(confusion_matrix[label][other] for other in labels if other != label)

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(confusion_matrix[label].values())
        }

    # Overall accuracy
    total = sum(sum(confusion_matrix[label].values()) for label in labels)
    correct = sum(confusion_matrix[label][label] for label in labels)
    metrics['accuracy'] = correct / total if total > 0 else 0

    # Macro-averaged F1
    metrics['macro_f1'] = np.mean([metrics[label]['f1'] for label in labels])

    return metrics


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def filter_fens(fens):
    """Filter out positions where king is in check."""
    filtered_fens = []
    for fen in fens:
        board = chess.Board(fen=fen)
        if not board.is_check():
            filtered_fens.append(fen)
    return filtered_fens


def make_fen_batch_provider(data_path, batch_size):
    return nnue_dataset.FenBatchProvider(
        data_path,
        True,
        1,
        batch_size,
        nnue_dataset.DataloaderSkipConfig(
            random_fen_skipping=10,
        ),
    )


def eval_model_batch(model, batch):
    """Evaluate a batch of positions with the model."""
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch.contents.get_tensors("cuda")

    evals = [
        v.item()
        for v in model.forward(
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        )
                 * 600.0
    ]

    # Flip scores for black to move
    for i in range(len(evals)):
        if them[i] > 0.5:
            evals[i] = -evals[i]

    return evals


def eval_engine_batch(engine_path, net_path, fens):
    """Evaluate positions using Stockfish engine."""
    import re

    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )

    parts = ["uci"]
    if net_path is not None:
        parts.append(f"setoption name EvalFile value {net_path}")

    for fen in fens:
        parts.append(f"position fen {fen}")
        parts.append("eval")
    parts.append("quit")

    query = "\n".join(parts)
    out = engine.communicate(input=query)[0]

    re_nnue_eval = re.compile(r"NNUE evaluation:?\s*?([-+]?\d*?\.\d*)")
    evals = re.findall(re_nnue_eval, out)

    return [int(float(v) * 208) for v in evals]


def get_model_with_fixed_offset(model, batch_size, main_device):
    model.layer_stacks.idx_offset = torch.arange(
        0,
        batch_size * model.layer_stacks.count,
        model.layer_stacks.count,
        device=main_device,
    )
    return model


def calibrate_threshold(engine_evals, model_evals, engine_threshold=150,
                        method='range'):
    """
    Calibrate model threshold to match engine threshold behavior.

    Args:
        engine_evals: List of engine evaluations
        model_evals: List of model evaluations
        engine_threshold: Engine threshold in centipawns
        method: Calibration method - 'range' (default), 'percentile', or 'std'

    Returns:
        Calibrated model threshold and calibration info dict
    """
    if method == 'range':
        # Range mapping method (recommended)
        # Maps the actual evaluation ranges proportionally
        # If engine spans ±4000 and model spans ±6000, ratio is 6000/4000 = 1.5

        engine_max = max(abs(min(engine_evals)), abs(max(engine_evals)))
        model_max = max(abs(min(model_evals)), abs(max(model_evals)))

        if engine_max == 0:
            return engine_threshold, {'method': 'range', 'ratio': 1.0,
                                      'engine_range': 0, 'model_range': 0}

        ratio = model_max / engine_max
        model_threshold = int(engine_threshold * ratio)

        return model_threshold, {
            'method': 'range',
            'ratio': ratio,
            'engine_range': engine_max,
            'model_range': model_max,
            'engine_min': min(engine_evals),
            'engine_max': max(engine_evals),
            'model_min': min(model_evals),
            'model_max': max(model_evals)
        }

    elif method == 'percentile':
        # Percentile matching method
        # Find what percentile the threshold represents in engine space,
        # then find the equivalent percentile in model space

        engine_abs = [abs(e) for e in engine_evals]
        model_abs = [abs(m) for m in model_evals]

        # What percentage of positions have |eval| > threshold?
        engine_pct = sum(1 for e in engine_abs if e > engine_threshold) / len(engine_abs)

        # Find equivalent threshold in model space
        model_sorted = sorted(model_abs)
        target_idx = int((1 - engine_pct) * len(model_sorted))
        model_threshold = model_sorted[target_idx] if target_idx < len(model_sorted) else model_sorted[-1]

        return int(model_threshold), {
            'method': 'percentile',
            'engine_percentile': engine_pct * 100,
            'model_threshold_at_percentile': model_threshold
        }

    elif method == 'std':
        # Standard deviation method
        # Normalize by standard deviation of evaluations

        engine_std = (sum((e - sum(engine_evals) / len(engine_evals)) ** 2
                          for e in engine_evals) / len(engine_evals)) ** 0.5
        model_std = (sum((m - sum(model_evals) / len(model_evals)) ** 2
                         for m in model_evals) / len(model_evals)) ** 0.5

        if engine_std == 0:
            return engine_threshold, {'method': 'std', 'ratio': 1.0}

        ratio = model_std / engine_std
        model_threshold = int(engine_threshold * ratio)

        return model_threshold, {
            'method': 'std',
            'ratio': ratio,
            'engine_std': engine_std,
            'model_std': model_std
        }

    else:
        raise ValueError(f"Unknown calibration method: {method}")


def save_results(engine_evals, model_evals, engine_labels, model_labels,
                 metrics, thresholds, metadata, output_dir, model_name):
    """Save W/D/L evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"wdl_{model_name}_{timestamp}"

    # Save summary with all metrics
    summary_data = {
        "metadata": metadata,
        "thresholds": {
            "engine_win_threshold": thresholds.engine_win_threshold,
            "model_win_threshold": thresholds.model_win_threshold,
            "engine_scale": thresholds.engine_scale,
            "model_scale": thresholds.model_scale,
            "threshold_ratio": thresholds.model_win_threshold / thresholds.engine_win_threshold
        },
        "metrics": metrics,
        "timestamp": timestamp
    }

    json_path = output_path / f"{base_filename}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {json_path}")

    # Save detailed CSV with per-position results
    csv_path = output_path / f"{base_filename}_details.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'engine_eval', 'model_eval',
                         'engine_label', 'model_label', 'correct',
                         'eval_difference', 'abs_eval_difference'])

        for i, (eng, mod, eng_lbl, mod_lbl) in enumerate(
                zip(engine_evals, model_evals, engine_labels, model_labels)):
            correct = eng_lbl == mod_lbl
            diff = mod - eng
            writer.writerow([i, eng, mod, eng_lbl, mod_lbl, correct, diff, abs(diff)])

    print(f"Details saved to: {csv_path}")

    # Save confusion matrix as CSV
    confusion_path = output_path / f"{base_filename}_confusion.csv"
    with open(confusion_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True/Predicted', 'W', 'D', 'L'])
        for true_label in ['W', 'D', 'L']:
            row = [true_label] + [metrics['confusion_matrix'][true_label][pred]
                                  for pred in ['W', 'D', 'L']]
            writer.writerow(row)

    print(f"Confusion matrix saved to: {confusion_path}")


def print_results(metrics, thresholds):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("W/D/L CLASSIFICATION RESULTS")
    print("=" * 80)

    print(f"\nThresholds: {thresholds}")

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")

    print("\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 60)
    for label in ['W', 'D', 'L']:
        m = metrics['class_metrics'][label]
        print(f"{label:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['support']:<10d}")

    print("\nConfusion Matrix:")
    print("(Rows = True Label, Columns = Predicted Label)")
    print(f"{'':>12} {'W':>8} {'D':>8} {'L':>8}")
    print("-" * 40)
    for true_label in ['W', 'D', 'L']:
        row = f"{true_label:>12}"
        for pred_label in ['W', 'D', 'L']:
            row += f" {metrics['confusion_matrix'][true_label][pred_label]:>8d}"
        print(row)

    # Label distribution
    print("\nLabel Distribution:")
    total = sum(metrics['class_metrics'][label]['support'] for label in ['W', 'D', 'L'])
    for label in ['W', 'D', 'L']:
        count = metrics['class_metrics'][label]['support']
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count:>6d} ({pct:>5.2f}%)")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate W/D/L classification accuracy of NNUE model"
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--engine", type=str, help="Path to stockfish")
    parser.add_argument("--net", type=str, help="Path to .nnue file (optional)")
    parser.add_argument("--data", type=str, help="Path to .bin or .binpack dataset")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--count", type=int, help="Number of positions to evaluate")
    parser.add_argument("--model", type=str, help="Name of the model to test")
    parser.add_argument("--batch-size", type=int, help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, help="Directory to save results")

    # Threshold configuration
    parser.add_argument(
        "--engine-threshold",
        type=int,
        default=150,
        help="Engine win/loss threshold in centipawns (default: 150 = 1.5 pawns)"
    )
    parser.add_argument(
        "--model-threshold",
        type=int,
        help="Model win/loss threshold (auto-calibrated if not specified)"
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Auto-calibrate model threshold (default: uses range mapping)"
    )
    parser.add_argument(
        "--calibration-method",
        choices=['range', 'percentile', 'std'],
        default='range',
        help="Calibration method: 'range' (map eval ranges, default), 'percentile' (match percentiles), 'std' (normalize by std dev)"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1000,
        help="Number of samples for threshold calibration (default: 1000)"
    )

    features.add_argparse_args(parser)
    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        engine_path = config.get("engine_path")
        net_path = config.get("net_path")
        data_path = config.get("data_path")
        checkpoint_path = config.get("checkpoint_path")
        model_name = config.get("model_name")
        count = config.get("count", 10000)
        batch_size = config.get("batch_size", 1000)
        output_dir = config.get("output_dir", "./wdl_results")
        engine_threshold = config.get("engine_threshold", args.engine_threshold)
        model_threshold = config.get("model_threshold", args.model_threshold)
    else:
        engine_path = args.engine
        net_path = args.net
        data_path = args.data
        checkpoint_path = args.checkpoint
        model_name = args.model
        count = args.count or 10000
        batch_size = args.batch_size or 1000
        output_dir = args.output_dir or "./wdl_results"
        engine_threshold = args.engine_threshold
        model_threshold = args.model_threshold

    if not all([engine_path, data_path, model_name]):
        parser.error("Either provide --config or specify --engine, --data, and --model")

    print(f"\n=== W/D/L Classification Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Engine: {engine_path}")
    print(f"Net: {net_path}")
    print(f"Data: {data_path}")
    print(f"Count: {count}")
    print(f"Batch size: {batch_size}")
    print(f"Output dir: {output_dir}")
    print(f"Engine threshold: {engine_threshold} cp")
    if model_threshold:
        print(f"Model threshold: {model_threshold} (manual)")
    else:
        print(f"Model threshold: auto-calibrate")
    print()

    # Initialize model
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
    model = models.nets[model_name](feature_set)
    model.eval()
    model.cuda()

    fen_batch_provider = make_fen_batch_provider(data_path, batch_size)

    # First pass: collect calibration data if needed
    if args.calibrate_threshold and model_threshold is None:
        cal_method = args.calibration_method if hasattr(args, 'calibration_method') else 'range'
        print(f"Calibrating threshold using sample positions...")
        print(f"Calibration method: {cal_method}")
        cal_engine_evals = []
        cal_model_evals = []
        cal_done = 0

        while cal_done < args.calibration_samples:
            fens = filter_fens(next(fen_batch_provider))
            model_temp = get_model_with_fixed_offset(model, len(fens), 'cuda')

            b = nnue_dataset.make_sparse_batch_from_fens(
                feature_set, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
            )
            cal_model_evals += eval_model_batch(model_temp, b)
            nnue_dataset.destroy_sparse_batch(b)

            cal_engine_evals += eval_engine_batch(engine_path, net_path, fens)

            cal_done += len(fens)

        model_threshold, cal_info = calibrate_threshold(
            cal_engine_evals, cal_model_evals, engine_threshold,
            method=cal_method
        )
        print(f"\nCalibration results:")
        if cal_method == 'range':
            print(f"  Engine range: {cal_info['engine_min']:.0f} to {cal_info['engine_max']:.0f} "
                  f"(span: {cal_info['engine_range']:.0f})")
            print(f"  Model range:  {cal_info['model_min']:.0f} to {cal_info['model_max']:.0f} "
                  f"(span: {cal_info['model_range']:.0f})")
            print(f"  Ratio: {cal_info['ratio']:.3f}")
        print(f"  Engine threshold: {engine_threshold} cp")
        print(f"  Calibrated model threshold: {model_threshold}")
        print()

    # Initialize thresholds
    thresholds = WDLThresholds(
        engine_win_threshold=engine_threshold,
        model_win_threshold=model_threshold
    )
    print(f"Using thresholds: {thresholds}\n")

    # Main evaluation loop
    model_evals = []
    engine_evals = []
    done = 0

    print(f"Processed {done} positions.")

    while done < count:
        fens = filter_fens(next(fen_batch_provider))
        model = get_model_with_fixed_offset(model, len(fens), 'cuda')

        b = nnue_dataset.make_sparse_batch_from_fens(
            feature_set, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
        )
        model_evals += eval_model_batch(model, b)
        nnue_dataset.destroy_sparse_batch(b)

        engine_evals += eval_engine_batch(engine_path, net_path, fens)

        done += len(fens)
        print(f"Processed {done} positions.")

    # Classify all positions
    engine_labels = [thresholds.classify(e, is_engine=True) for e in engine_evals]
    model_labels = [thresholds.classify(m, is_engine=False) for m in model_evals]

    # Compute metrics
    confusion_matrix = compute_confusion_matrix(engine_labels, model_labels)
    class_metrics = compute_classification_metrics(confusion_matrix)

    metrics = {
        'accuracy': class_metrics['accuracy'],
        'macro_f1': class_metrics['macro_f1'],
        'class_metrics': {label: class_metrics[label] for label in ['W', 'D', 'L']},
        'confusion_matrix': confusion_matrix
    }

    # Print results
    print_results(metrics, thresholds)

    # Save results
    metadata = {
        "engine_path": engine_path,
        "net_path": net_path,
        "data_path": data_path,
        "checkpoint_path": checkpoint_path,
        "model_name": model_name,
        "positions_evaluated": len(engine_evals),
        "batch_size": batch_size
    }

    save_results(
        engine_evals, model_evals, engine_labels, model_labels,
        metrics, thresholds, metadata, output_dir, model_name
    )


if __name__ == "__main__":
    main()
