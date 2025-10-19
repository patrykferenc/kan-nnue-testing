#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

CONFIG_FILE="${1:-config.json}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo ""
    echo "Usage: $0 [config_file]"
    echo "  Default: config.json"
    echo ""
    echo "To get started:"
    echo "  1. Copy the template: cp config_template.json config.json"
    echo "  2. Edit config.json with your paths"
    echo "  3. Run: $0"
    exit 1
fi

echo "Running cross-check evaluation with config: $CONFIG_FILE"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

python3 cross_check_eval.py --config "$CONFIG_FILE"