import argparse
import json
import csv
import re
import subprocess
import torch
from pathlib import Path
from datetime import datetime

from scipy.stats import pearsonr, spearmanr

import chess

import models
import commons.features as features
import commons.nnue_dataset as nnue_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def save_results(engine_evals, model_evals, stats, metadata, output_dir, model_name):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model_name}_{timestamp}"

    summary_data = {
        "metadata": metadata,
        "statistics": stats,
        "timestamp": timestamp
    }

    json_path = output_path / f"{base_filename}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"summary saved to: {json_path}")

    csv_path = output_path / f"{base_filename}_evals.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'engine_eval', 'model_eval', 'difference', 'abs_difference'])
        for i, (eng, mod) in enumerate(zip(engine_evals, model_evals)):
            diff = mod - eng
            writer.writerow([i, eng, mod, diff, abs(diff)])
    print(f"evaluations saved to: {csv_path}")

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
    for i in range(len(evals)):
        if them[i] > 0.5:
            evals[i] = -evals[i]
    return evals


re_nnue_eval = re.compile(r"NNUE evaluation:?\s*?([-+]?\d*?\.\d*)")


def compute_basic_eval_stats(evals):
    min_engine_eval = min(evals)
    max_engine_eval = max(evals)
    avg_engine_eval = sum(evals) / len(evals)
    avg_abs_engine_eval = sum(abs(v) for v in evals) / len(evals)

    return min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval


def compute_correlation(engine_evals, model_evals):
    if len(engine_evals) != len(model_evals):
        raise Exception(
            f"Mismatch in eval counts - engine: {len(engine_evals)}, model: {len(model_evals)}"
        )

    min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval = (
        compute_basic_eval_stats(engine_evals)
    )
    min_model_eval, max_model_eval, avg_model_eval, avg_abs_model_eval = (
        compute_basic_eval_stats(model_evals)
    )

    print(f"\nMin engine/model eval: {min_engine_eval:.2f} / {min_model_eval:.2f}")
    print(f"Max engine/model eval: {max_engine_eval:.2f} / {max_model_eval:.2f}")
    print(f"Avg engine/model eval: {avg_engine_eval:.2f} / {avg_model_eval:.2f}")
    print(f"Avg abs engine/model eval: {avg_abs_engine_eval:.2f} / {avg_abs_model_eval:.2f}")

    relative_model_error = sum(
        abs(model - engine) / (abs(engine) + 0.001)
        for model, engine in zip(model_evals, engine_evals)
    ) / len(engine_evals)
    relative_engine_error = sum(
        abs(model - engine) / (abs(model) + 0.001)
        for model, engine in zip(model_evals, engine_evals)
    ) / len(engine_evals)
    avg_abs_diff = sum(
        abs(model - engine) for model, engine in zip(model_evals, engine_evals)
    ) / len(engine_evals)
    min_diff = min(abs(model - engine) for model, engine in zip(model_evals, engine_evals))
    max_diff = max(abs(model - engine) for model, engine in zip(model_evals, engine_evals))

    print(f"Relative engine error: {relative_engine_error:.4f}")
    print(f"Relative model error: {relative_model_error:.4f}")
    print(f"Avg abs difference: {avg_abs_diff:.2f}")
    print(f"Min difference: {min_diff:.2f}")
    print(f"Max difference: {max_diff:.2f}")

    pearson_corr, pearson_pvalue = pearsonr(engine_evals, model_evals)
    spearman_corr, spearman_pvalue = spearmanr(engine_evals, model_evals)
    r_squared = pearson_corr ** 2

    print(f"Pearson correlation: {pearson_corr:.6f} (p-value: {pearson_pvalue:.2e})")
    print(f"Spearman correlation: {spearman_corr:.6f} (p-value: {spearman_pvalue:.2e})")
    print(f"R² (coefficient of determination): {r_squared:.6f}")

    # Return statistics dictionary
    return {
        "engine_stats": {
            "min": min_engine_eval,
            "max": max_engine_eval,
            "avg": avg_engine_eval,
            "avg_abs": avg_abs_engine_eval
        },
        "model_stats": {
            "min": min_model_eval,
            "max": max_model_eval,
            "avg": avg_model_eval,
            "avg_abs": avg_abs_model_eval
        },
        "error_metrics": {
            "relative_engine_error": relative_engine_error,
            "relative_model_error": relative_model_error,
            "avg_abs_difference": avg_abs_diff,
            "min_difference": min_diff,
            "max_difference": max_diff
        },
        "correlations": {
            "pearson": {
                "coefficient": pearson_corr,
                "p_value": pearson_pvalue
            },
            "spearman": {
                "coefficient": spearman_corr,
                "p_value": spearman_pvalue
            },
            "r_squared": r_squared
        }
    }


def eval_engine_batch(engine_path, net_path, fens):
    """Evaluate positions using Stockfish engine."""
    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    parts = ["uci"] #, f"setoption name EvalFile value {net_path}"]
    if net_path is not None:
        parts.append(f"setoption name EvalFile value {net_path}")
    for fen in fens:
        parts.append(f"position fen {fen}")
        parts.append("eval")
    parts.append("quit")
    query = "\n".join(parts)
    out = engine.communicate(input=query)[0]
    evals = re.findall(re_nnue_eval, out)
    return [int(float(v) * 208) for v in evals]


def filter_fens(fens):
    # We don't want fens where a king is in check, as these cannot be evaluated by the engine.
    filtered_fens = []
    for fen in fens:
        board = chess.Board(fen=fen)
        if not board.is_check():
            filtered_fens.append(fen)
    return filtered_fens


def get_model_with_fixed_offset(model, batch_size, main_device):
    model.layer_stacks.idx_offset = torch.arange(
        0,
        batch_size * model.layer_stacks.count,
        model.layer_stacks.count,
        device=main_device,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check NNUE model evaluation against Stockfish"
    )
    parser.add_argument("--config", type=str, help="path to config JSON file")
    parser.add_argument("--engine", type=str, help="path to stockfish")
    parser.add_argument("--net", type=str, help="path to .nnue file")
    parser.add_argument("--data", type=str, help="path to .bin or .binpack dataset")
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--count", type=int, help="number of positions to evaluate")
    parser.add_argument("--model", type=str, help="name of the model to test")
    parser.add_argument("--batch-size", type=int, help="batch size for evaluation")
    parser.add_argument("--output-dir", type=str, help="directory to save results")
    features.add_argparse_args(parser)
    args = parser.parse_args()

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
        output_dir = config.get("output_dir", "./results")
    else:
        engine_path = args.engine
        net_path = args.net
        data_path = args.data
        checkpoint_path = args.checkpoint
        model_name = args.model
        count = args.count or 10000
        batch_size = args.batch_size or 1000
        output_dir = args.output_dir or "./results"

    if not all([engine_path, data_path, model_name]):
        parser.error("Either provide --config or specify --engine, --data, and --model")

    print(f"\n=== Cross-Check Evaluation ===")
    print(f"Model: {model_name}")
    print(f"Engine: {engine_path}")
    print(f"Net: {net_path}")
    print(f"Data: {data_path}")
    print(f"Count: {count}")
    print(f"Batch size: {batch_size}")
    print(f"Output dir: {output_dir}\n")

    # Initialize model
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
    model = models.nets[model_name](feature_set)
    model.eval()
    model.cuda()

    fen_batch_provider = make_fen_batch_provider(data_path, batch_size)

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

    stats = compute_correlation(engine_evals, model_evals)

    metadata = {
        "engine_path": engine_path,
        "net_path": net_path,
        "data_path": data_path,
        "checkpoint_path": checkpoint_path,
        "model_name": model_name,
        "positions_evaluated": len(engine_evals),
        "batch_size": batch_size
    }

    save_results(engine_evals, model_evals, stats, metadata, output_dir, model_name)


if __name__ == "__main__":
    main()