#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")
echo "$TOOLS"
T="python3 $TOOLS/tester.py"

echo "Self play"
$T "$1" ${2:-"--quiet"} self-play --time 20000 --inc 1000
echo


