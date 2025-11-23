#!/usr/bin/env python3
"""
Simple network testing framework for sunfish_nnue
Runs games between two networks and calculates Elo differences
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import random

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

# LOGS
# Generate a small random id once per process/run
import random
import logging

RUN_RID = random.randint(0, 99)

logging.basicConfig(
    filename="test.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [id=%(rid)02d] %(message)s",
)

# Ensure every LogRecord gets the same run id
_old_factory = logging.getLogRecordFactory()


def _attach_run_id_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    if not hasattr(record, "rid"):
        record.rid = RUN_RID
    return record


logging.setLogRecordFactory(_attach_run_id_factory)


# Result tracking
class GameResult:
    def __init__(self, white_network: str, black_network: str, result: str, pgn: str = ""):
        self.white_network = white_network
        self.black_network = black_network
        self.result = result  # "1-0", "0-1", "1/2-1/2"
        self.pgn = pgn
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'white_network': self.white_network,
            'black_network': self.black_network,
            'result': self.result,
            'pgn': self.pgn,
            'timestamp': self.timestamp
        }


class NetworkTester:
    def __init__(self, engine_path: str, args):
        self.engine_path = Path(engine_path).absolute()
        self.args = args
        self.results: List[GameResult] = []

    def make_engine_cmd(self, model_name: str, model_path: str) -> List[str]:
        """Create command to run sunfish_nnue with specific network"""
        cmd = [
            sys.executable,
            str(self.engine_path),
            model_name,
            model_path,
            str(self.args.batch_size),
            self.args.compile_mode
        ]
        if self.args.book_path:
            cmd.append(self.args.book_path)
        return cmd

    async def play_game(self, network_a: Tuple[str, str, str],
                        network_b: Tuple[str, str, str],
                        white_is_a: bool) -> GameResult:
        """Play a single game between two networks

        Args:
            network_a: (name, model_name, model_path)
            network_b: (name, model_name, model_path)
            white_is_a: True if network_a plays white
        """
        white_net = network_a if white_is_a else network_b
        black_net = network_b if white_is_a else network_a

        # Create engines
        white_cmd = self.make_engine_cmd(white_net[1], white_net[2])
        black_cmd = self.make_engine_cmd(black_net[1], black_net[2])

        # Start engines
        _, white_engine = await chess.engine.popen_uci(white_cmd)
        _, black_engine = await chess.engine.popen_uci(black_cmd)

        try:
            board = chess.Board()
            game = chess.pgn.Game()
            node = game

            # Apply opening book moves if specified
            if self.args.opening_moves:
                for move_uci in self.args.opening_moves.split():
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                    node = node.add_variation(move)

            # Play the game
            move_count = 0
            while not board.is_game_over() and move_count < self.args.max_moves:
                engine = white_engine if board.turn == chess.WHITE else black_engine

                # Set time control
                if self.args.nodes:
                    limit = chess.engine.Limit(nodes=self.args.nodes)
                elif self.args.movetime:
                    limit = chess.engine.Limit(time=self.args.movetime / 1000)
                elif self.args.depth:
                    limit = chess.engine.Limit(depth=self.args.depth)
                else:
                    limit = chess.engine.Limit(nodes=10000)

                logging.info(f"Playing move {move_count + 1}, using limit {limit}")

                result = await engine.play(board, limit)
                board.push(result.move)
                node = node.add_variation(result.move)
                move_count += 1

            # Get game result
            if board.is_checkmate():
                result_str = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material():
                result_str = "1/2-1/2"
            elif board.is_seventyfive_moves() or board.is_fivefold_repetition():
                result_str = "1/2-1/2"
            elif move_count >= self.args.max_moves:
                # Adjudicate by material count
                material = self._count_material(board)
                if abs(material) >= self.args.adjudicate_threshold:
                    result_str = "1-0" if material > 0 else "0-1"
                else:
                    result_str = "1/2-1/2"
            else:
                result_str = "1/2-1/2"

            game.headers["White"] = white_net[0]
            game.headers["Black"] = black_net[0]
            game.headers["Result"] = result_str

            pgn_str = str(game) if self.args.save_pgn else ""

            return GameResult(white_net[0], black_net[0], result_str, pgn_str)

        finally:
            await white_engine.quit()
            await black_engine.quit()

    def _count_material(self, board: chess.Board) -> int:
        """Count material difference (positive = white advantage)"""
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}

        white_material = sum(values.get(piece.piece_type, 0)
                             for piece in board.piece_map().values()
                             if piece.color == chess.WHITE)
        black_material = sum(values.get(piece.piece_type, 0)
                             for piece in board.piece_map().values()
                             if piece.color == chess.BLACK)

        return white_material - black_material

    def run_match(self, network_a: Tuple[str, str, str],
                  network_b: Tuple[str, str, str]) -> Dict:
        """Run a match between two networks"""

        print(f"\nRunning match: {network_a[0]} vs {network_b[0]}")
        print(f"Games: {self.args.num_games}, Threads: {self.args.threads}")

        # Create async event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Prepare games (alternating colors)
        games = []
        for i in range(self.args.num_games):
            white_is_a = (i % 2 == 0)
            games.append((network_a, network_b, white_is_a))

        # Run games in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
            futures = []
            for game_args in games:
                future = executor.submit(self._run_game_sync, *game_args)
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Playing games"):
                result = future.result()
                results.append(result)
                self.results.append(result)

        # Calculate statistics
        stats = self._calculate_stats(results, network_a[0], network_b[0])

        # Save results incrementally
        if self.args.output:
            self._save_results()

        return stats

    def _run_game_sync(self, network_a, network_b, white_is_a):
        """Synchronous wrapper for async game playing"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.play_game(network_a, network_b, white_is_a)
            )
        finally:
            loop.close()

    def _calculate_stats(self, results: List[GameResult],
                         name_a: str, name_b: str) -> Dict:
        """Calculate match statistics"""
        wins_a = sum(1 for r in results if
                     (r.white_network == name_a and r.result == "1-0") or
                     (r.black_network == name_a and r.result == "0-1"))
        wins_b = sum(1 for r in results if
                     (r.white_network == name_b and r.result == "1-0") or
                     (r.black_network == name_b and r.result == "0-1"))
        draws = sum(1 for r in results if r.result == "1/2-1/2")

        total = len(results)
        score_a = wins_a + draws * 0.5
        score_b = wins_b + draws * 0.5

        # Calculate Elo difference (simplified)
        if score_a > 0 and score_b > 0:
            win_rate = score_a / total
            elo_diff = 400 * (win_rate - 0.5) / 0.29  # Simplified Elo formula
        else:
            elo_diff = 0

        stats = {
            'network_a': name_a,
            'network_b': name_b,
            'games': total,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'draws': draws,
            'score_a': score_a,
            'score_b': score_b,
            'win_rate_a': score_a / total if total > 0 else 0,
            'elo_diff': elo_diff,
            'timestamp': datetime.now().isoformat()
        }

        # Print results
        print(f"\nResults: {name_a} vs {name_b}")
        print(f"  Games: {total}")
        print(f"  Score: +{wins_a}={draws}-{wins_b}")
        print(f"  Win rate for {name_a}: {stats['win_rate_a']:.1%}")
        print(f"  Elo difference: {elo_diff:+.1f}")

        return stats

    def _save_results(self):
        """Save results to JSON file"""
        output_path = Path(self.args.output)

        # Load existing results if file exists
        all_results = []
        if output_path.exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
                all_results = data.get('games', [])

        # Add new results
        all_results.extend([r.to_dict() for r in self.results])

        # Save
        data = {
            'games': all_results,
            'last_updated': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {output_path}")

    def run_ordo(self, pgn_file: str = None):
        """Run ordo to calculate Elo ratings"""
        if not self.args.ordo_path:
            print("Ordo path not specified, skipping Elo calculation")
            return

        # Create PGN file for ordo
        if not pgn_file:
            pgn_file = "games_for_ordo.pgn"

        with open(pgn_file, 'w') as f:
            for result in self.results:
                if result.pgn:
                    f.write(result.pgn + "\n\n")

        # Run ordo
        cmd = [self.args.ordo_path, "-q", "-p", pgn_file, "-o", "ordo_ratings.txt"]
        try:
            subprocess.run(cmd, check=True)
            print(f"Ordo ratings saved to ordo_ratings.txt")

            # Display ratings
            with open("ordo_ratings.txt", 'r') as f:
                print("\nOrdo Ratings:")
                print(f.read())
        except subprocess.CalledProcessError as e:
            print(f"Error running ordo: {e}")
        except FileNotFoundError:
            print(f"Ordo executable not found at {self.args.ordo_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test NNUE networks against each other",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Engine settings
    parser.add_argument("engine", help="Path to sunfish_nnue.py")
    parser.add_argument("--network-a", required=True, nargs=3,
                        metavar=("NAME", "MODEL_NAME", "MODEL_PATH"),
                        help="First network: display_name model_name model_path")
    parser.add_argument("--network-b", required=True, nargs=3,
                        metavar=("NAME", "MODEL_NAME", "MODEL_PATH"),
                        help="Second network: display_name model_name model_path")

    # Game settings
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games to play")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of parallel games")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for NNUE evaluation")
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="PyTorch compilation mode")

    # Time control
    parser.add_argument("--nodes", type=int,
                        help="Nodes per move (default)")
    parser.add_argument("--movetime", type=int,
                        help="Time per move in milliseconds")
    parser.add_argument("--depth", type=int,
                        help="Search depth limit")

    # Game control
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Maximum moves per game")
    parser.add_argument("--adjudicate-threshold", type=int, default=6,
                        help="Material threshold for adjudication (pawns)")
    parser.add_argument("--opening-moves", type=str,
                        help="Opening moves in UCI format (e.g. 'e2e4 e7e5')")
    parser.add_argument("--book-path", type=str,
                        help="Path to opening book for sunfish")

    # Output settings
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output file for results")
    parser.add_argument("--save-pgn", action="store_true",
                        help="Save games in PGN format")
    parser.add_argument("--ordo-path", type=str,
                        help="Path to ordo executable for Elo calculation")

    args = parser.parse_args()

    # Run tests
    tester = NetworkTester(args.engine, args)

    # Run match
    stats = tester.run_match(
        tuple(args.network_a),
        tuple(args.network_b)
    )

    # Run ordo if available
    if args.ordo_path and args.save_pgn:
        tester.run_ordo()


if __name__ == "__main__":
    main()
