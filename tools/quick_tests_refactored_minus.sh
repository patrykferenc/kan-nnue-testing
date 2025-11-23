#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")
echo "$TOOLS"
T="python3 $TOOLS/tester.py"
RESULTS_DIR="${3:-./test_results}"
echo "$RESULTS_DIR"

#echo "Mate in 1..."
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate1.fen --depth 2
#echo

# These mates should be findable at depth=4, but because of null-move
# We need to go to depth=6.
#echo "Mate in 2..."
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate4.fen --depth 8 --limit 10
#echo
#
#echo "Mate in 3..."
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate3.fen --depth 6 --limit 10
#echo
#
#echo "Stalemate in 0..."
#$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate0.fen --depth 2
#echo
#
#echo "Stalemate in 1..."
#$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate1.fen --depth 2
#echo
#
#echo "Stalemate in 2+"
#$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate2.fen --depth 4
#echo
#
#echo "Other puzzles..."
#$T "$1" ${2:-"--quiet"} best $TOOLS/test_files/win_at_chess_test.epd --depth 4
#$T "$1" ${2:-"--quiet"} best $TOOLS/test_files/bug.epd --depth 2
#echo

echo "Branching factor"
$T "$1" ${2:-"--quiet"} --results-dir "$RESULTS_DIR" branching-factor --depth 4
echo

#echo "Perft"
#$T "$1" ${2:-"--quiet"} perft $TOOLS/test_files/perft.epd --depth 2
#echo