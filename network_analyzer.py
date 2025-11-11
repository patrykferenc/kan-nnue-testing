#!/usr/bin/env python3
"""
Combine and analyze results from multiple test runs
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class ResultsAnalyzer:
    def __init__(self):
        self.all_games = []
        self.network_pairs = defaultdict(lambda: {'games': []})

    def load_results(self, filepath: str):
        """Load results from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            games = data.get('games', [])
            self.all_games.extend(games)

            # Group by network pairs
            for game in games:
                key = tuple(sorted([game['white_network'], game['black_network']]))
                self.network_pairs[key]['games'].append(game)

    def analyze(self):
        """Analyze all loaded games"""
        print("\n" + "=" * 60)
        print("COMBINED RESULTS ANALYSIS")
        print("=" * 60)
        print(f"\nTotal games loaded: {len(self.all_games)}")

        # Analyze each network pair
        for networks, data in self.network_pairs.items():
            games = data['games']
            if not games:
                continue

            net_a, net_b = networks

            # Count results
            results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})

            for game in games:
                if game['result'] == '1/2-1/2':
                    results[game['white_network']]['draws'] += 1
                    results[game['black_network']]['draws'] += 1
                elif game['result'] == '1-0':
                    results[game['white_network']]['wins'] += 1
                    results[game['black_network']]['losses'] += 1
                else:  # 0-1
                    results[game['white_network']]['losses'] += 1
                    results[game['black_network']]['wins'] += 1

            # Calculate statistics for network A vs B
            stats_a = results[net_a]
            stats_b = results[net_b]

            total = len(games)
            wins_a = stats_a['wins']
            wins_b = stats_b['wins']
            draws = stats_a['draws']  # Same for both

            score_a = wins_a + draws * 0.5
            score_b = wins_b + draws * 0.5
            win_rate_a = score_a / total if total > 0 else 0

            # Calculate Elo with error bounds
            elo_diff, elo_error = self._calculate_elo_with_error(
                wins_a, draws, wins_b
            )

            # Calculate LOS (Likelihood of Superiority)
            los = self._calculate_los(wins_a, draws, wins_b)

            print(f"\n{net_a} vs {net_b}")
            print("-" * 40)
            print(f"  Games played: {total}")
            print(f"  Score: +{wins_a}={draws}-{wins_b}")
            print(f"  Win rate: {win_rate_a:.1%}")
            print(f"  Elo difference: {elo_diff:+.1f} ± {elo_error:.1f}")
            print(f"  LOS: {los:.1%}")

            # Determine significance
            if abs(elo_diff) > 2 * elo_error:
                if elo_diff > 0:
                    print(f"  → {net_a} is significantly stronger")
                else:
                    print(f"  → {net_b} is significantly stronger")
            else:
                print(f"  → No significant difference")

    def _calculate_elo_with_error(self, wins_a: int, draws: int, wins_b: int):
        """Calculate Elo difference with error bounds"""
        total = wins_a + draws + wins_b
        if total == 0:
            return 0, 0

        score_a = wins_a + draws * 0.5
        win_rate = score_a / total

        # Avoid division by zero or log of zero
        win_rate = max(0.001, min(0.999, win_rate))

        # Elo difference
        elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

        # Error estimate (95% confidence)
        variance = win_rate * (1 - win_rate)
        std_error = math.sqrt(variance / total)
        elo_error = 400 * 1.96 * std_error / (win_rate * (1 - win_rate) * math.log(10))

        return elo_diff, abs(elo_error)

    def _calculate_los(self, wins_a: int, draws: int, wins_b: int):
        """Calculate Likelihood of Superiority"""
        # Simplified LOS calculation
        total = wins_a + draws + wins_b
        if total == 0:
            return 0.5

        # Use trinomial test approximation
        score_a = wins_a + draws * 0.5
        score_b = wins_b + draws * 0.5

        if score_a == score_b:
            return 0.5

        # Approximate using normal distribution
        p = score_a / total
        variance = p * (1 - p) / total

        if variance == 0:
            return 1.0 if score_a > score_b else 0.0

        z = (score_a - total * 0.5) / math.sqrt(total * variance)
        # Approximate normal CDF
        los = 0.5 * (1 + math.erf(z / math.sqrt(2)))

        return los

    def export_for_ordo(self, output_file: str = "games_combined.pgn"):
        """Export games in PGN format for ordo analysis"""
        pgn_count = 0

        with open(output_file, 'w') as f:
            for game in self.all_games:
                if 'pgn' in game and game['pgn']:
                    f.write(game['pgn'] + "\n\n")
                    pgn_count += 1

        if pgn_count > 0:
            print(f"\nExported {pgn_count} games to {output_file} for ordo analysis")
        else:
            print("\nNo PGN data found in results")

    def generate_summary(self, output_file: str = "summary.txt"):
        """Generate a summary report"""
        with open(output_file, 'w') as f:
            f.write("NETWORK TESTING SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total games analyzed: {len(self.all_games)}\n\n")

            # Network statistics
            network_stats = defaultdict(lambda: {
                'games': 0, 'wins': 0, 'losses': 0, 'draws': 0
            })

            for game in self.all_games:
                white = game['white_network']
                black = game['black_network']
                result = game['result']

                network_stats[white]['games'] += 1
                network_stats[black]['games'] += 1

                if result == '1-0':
                    network_stats[white]['wins'] += 1
                    network_stats[black]['losses'] += 1
                elif result == '0-1':
                    network_stats[white]['losses'] += 1
                    network_stats[black]['wins'] += 1
                else:
                    network_stats[white]['draws'] += 1
                    network_stats[black]['draws'] += 1

            f.write("NETWORK PERFORMANCE\n")
            f.write("-" * 40 + "\n")

            for network, stats in sorted(network_stats.items()):
                total = stats['games']
                if total == 0:
                    continue

                score = stats['wins'] + stats['draws'] * 0.5
                win_rate = score / total * 100

                f.write(f"\n{network}:\n")
                f.write(f"  Games: {total}\n")
                f.write(f"  Wins: {stats['wins']}\n")
                f.write(f"  Draws: {stats['draws']}\n")
                f.write(f"  Losses: {stats['losses']}\n")
                f.write(f"  Score: {score:.1f}/{total}\n")
                f.write(f"  Win Rate: {win_rate:.1f}%\n")

        print(f"Summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and combine network testing results"
    )

    parser.add_argument("results", nargs="+",
                        help="Result JSON files to analyze")
    parser.add_argument("--export-pgn", type=str,
                        help="Export combined PGN file for ordo")
    parser.add_argument("--summary", type=str, default="summary.txt",
                        help="Generate summary report")

    args = parser.parse_args()

    # Load and analyze results
    analyzer = ResultsAnalyzer()

    for filepath in args.results:
        if Path(filepath).exists():
            print(f"Loading {filepath}...")
            analyzer.load_results(filepath)
        else:
            print(f"Warning: {filepath} not found, skipping...")

    # Analyze
    analyzer.analyze()

    # Export for ordo if requested
    if args.export_pgn:
        analyzer.export_for_ordo(args.export_pgn)

    # Generate summary
    if args.summary:
        analyzer.generate_summary(args.summary)


if __name__ == "__main__":
    main()
