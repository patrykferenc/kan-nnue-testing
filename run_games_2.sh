python3 network_tester.py ./sunfish_nnue.py \
  --network-a "sfnnvkan_early_3072_15_32" "sfnnvkan_early_3072_15_32" "/home/patryk/msc/nnue-pytorch/mystuff/experiments/experiment_baseline-20250728_132210/training/run_0/lightning_logs/version_0/checkpoints/epoch=599-step=2929800.ckpt" \
  --network-b "sfnnv9"   "sfnnv9" "/home/patryk/msc/nnue-pytorch/mystuff/experiments/experiment_baseline-20250707_141503/training/run_0/lightning_logs/version_0/checkpoints/last.ckpt" \
  --num-games 50 \
  --depth 4 \
  --book-path "/home/patryk/msc/kan-nnue-testing/tools/test_files/komodo.bin" \
  --output match_results_2.json \
  --save-pgn