# Testing of KAN nnue networks

This document explains how to use the scripts to test the KAN nnue networks and allows calculation of ELO points for
them.

## Programs included

| program             | purpose                                          |
|---------------------|--------------------------------------------------|
| cross_check_eval.py | Compare model evaluations against Stockfish NNUE |


## Models and how to test them

You must know the checkpoint files of the models you want to use.
You should add your nets to `nets` in the `models` package.

### Cross-check eval script

Allows checking the model evaluations against stockfish.

#### Quick Start

1. **Copy the config template:**
   ```bash
   cp config_template.json config.json
   ```

2. **Edit `config.json` with your paths:**
   - `engine_path` - Path to Stockfish binary
   - `net_path` - Path to .nnue file for engine evaluation
   - `data_path` - Path to .binpack/.bin dataset
   - `model_name` - Model name (must be in `models/__init__.py`)
   - `checkpoint_path` - Model checkpoint (optional)
   - `count` - Number of positions to evaluate (default: 10000)
   - `batch_size` - Batch size (default: 1000)
   - `output_dir` - Results directory (default: ./results)

3. **Run the evaluation:**
   ```bash
   ./run_cross_check_eval.sh
   ```

   Or specify a custom config:
   ```bash
   ./run_cross_check_eval.sh my_config.json
   ```

   Or use command-line arguments directly:
   ```bash
   python3 cross_check_eval.py \
     --engine /path/to/stockfish \
     --net /path/to/nn.nnue \
     --model sfnnv9 \
     --data /path/to/data.binpack \
     --count 10000
   ```

#### Output Files

Each run generates two timestamped files:

**`{model_name}_{timestamp}_summary.json`** - Complete statistics including:
- Min/max/average evaluations
- Correlation coefficients (Pearson, Spearman, R^2)
- Error metrics (relative errors, differences)
- Run metadata

**`{model_name}_{timestamp}_evals.csv`** - Position-by-position data:
```csv
position,engine_eval,model_eval,difference,abs_difference
0,150,148,-2,2
1,-200,-195,5,5
...
```

## Adding Your Models

To test your own networks:

1. Add your model class to a new file in `models/`
2. Register it in `models/__init__.py`:
   ```python
   from .your_model import model as YourModel
   
   nets = {
       'your_model_name': lambda feature_set: YourModel.NNUE(feature_set),
       # ... existing models
   }
   ```
3. Use `'your_model_name'` in your config file

## Development Files

- `CMakeLists.txt` - Build configuration for the C++ data loader
- `training_data_loader.cpp` - High-performance C++ data loader
- `commons/` - Shared utilities for features, datasets, and transformers

## Notes

- Position filtering excludes king-in-check positions (can't be evaluated by engine)
- Config file values override command-line arguments
- `config_template.json` is tracked in git; `config.json` is ignored
- All paths can be absolute or relative to script location