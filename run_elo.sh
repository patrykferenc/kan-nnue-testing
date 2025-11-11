#!/bin/bash
# run_network_test.sh - Example script for running network tests

# Configuration
ENGINE_PATH="sunfish_nnue.py"
BATCH_SIZE=1
COMPILE_MODE="default" #the only one that works for now xd
NUM_GAMES=1
THREADS=1
DEPTH=5
#NODES=5000  # Nodes per move
BOOK_PATH="./tools/test_files/komodo.bin"

# Network A (reference)
NETWORK_A_NAME="sfnnv9"
NETWORK_A_MODEL="sfnnv9"
NETWORK_A_PATH="/home/patryk/msc/nnue-pytorch/mystuff/experiments/experiment_baseline-20250707_141503/training/run_0/lightning_logs/version_0/checkpoints/last.ckpt"

# Network B (test)
NETWORK_B_NAME="sfnnvkan_early_3072_15_32"
NETWORK_B_MODEL="sfnnvkan_early_3072_15_32"  # or your KAN model name
NETWORK_B_PATH="/home/patryk/msc/nnue-pytorch/mystuff/experiments/experiment_baseline-20250728_132210/training/run_0/lightning_logs/version_0/checkpoints/last.ckpt"

# Output
OUTPUT_DIR="elo_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.json"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the test
python network_tester.py \
    $ENGINE_PATH \
    --network-a "$NETWORK_A_NAME" "$NETWORK_A_MODEL" "$NETWORK_A_PATH" \
    --network-b "$NETWORK_B_NAME" "$NETWORK_B_MODEL" "$NETWORK_B_PATH" \
    --num-games $NUM_GAMES \
    --threads $THREADS \
    --batch-size $BATCH_SIZE \
    --compile-mode $COMPILE_MODE \
    --depth $DEPTH \
    --output "$OUTPUT_FILE" \
    --book-path "$BOOK_PATH" \
    --save-pgn

echo "Test completed! Results saved to $OUTPUT_FILE"

# Optionally combine with previous results
if [ -f "${OUTPUT_DIR}/results_combined.json" ]; then
    echo "Combining with previous results..."
    python results_analyzer.py "${OUTPUT_DIR}/"*.json \
        --summary "${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
fi