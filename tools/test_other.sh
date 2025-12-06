#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")
echo "$TOOLS"
T="python3 $TOOLS/tester.py"
RESULTS_DIR="${3:-./test_results}"
echo "$RESULTS_DIR"

#echo "CCR"
#$T "$1" ${2:-"--quiet"} --results-dir "$RESULTS_DIR" best $TOOLS/test_files/ccr_one_hour_test.epd --depth 2
#echo

#echo "Eigenmann Endgames"
#$T "$1" ${2:-"--quiet"} --results-dir "$RESULTS_DIR" best $TOOLS/test_files/EigenmannEndgames.epd --depth 8
#echo

#echo "Eigenmann Rapid"
#$T "$1" ${2:-"--quiet"} --results-dir "$RESULTS_DIR" best $TOOLS/test_files/EigenmannRapidEngineTest.epd --movetime 15000
#echo

echo "Benchmark"
$T "$1" ${2:-"--quiet"} --results-dir "$RESULTS_DIR" best $TOOLS/test_files/benchmark.epd --movetime 1000 --clean-epd
echo
