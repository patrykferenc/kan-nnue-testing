#!/usr/bin/env python3
"""
Simple network testing framework for sunfish_nnue
Runs games between two networks and calculates Elo differences
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

import random
import logging

RUN_RID = random.randint(0, 99)

logging.basicConfig(
    filename="test.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [id=%(rid)02d] %(message)s",
)

_old_factory = logging.getLogRecordFactory()


def _attach_run_id_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    if not hasattr(record, "rid"):
        record.rid = RUN_RID
    return record


logging.setLogRecordFactory(_attach_run_id_factory)


def log_problematic_position(fen: str, reason: str):
    """Log positions where engine failed, for later debugging"""
    with open("problematic_positions.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} | {reason} | {fen}\n")


class GameResult:
    def __init__(self, white_network: str, black_network: str, result: str, pgn: str = ""):
        self.white_network = white_network
        self.black_network = black_network
        self.result = result
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
        # Persistent engine instances
        self.engine_a: Optional[chess.engine.UciProtocol] = None
        self.engine_b: Optional[chess.engine.UciProtocol] = None
        self.network_a_info: Optional[Tuple[str, str, str]] = None
        self.network_b_info: Optional[Tuple[str, str, str]] = None

    def make_engine_cmd(self, model_name: str, model_path: str) -> List[str]:
        cmd = [
            sys.executable,
            str(self.engine_path),
            model_name,
            model_path,
            self.args.compile_mode
        ]
        if self.args.book_path:
            cmd.append(self.args.book_path)
        return cmd

    async def start_engines(self, network_a: Tuple[str, str, str],
                            network_b: Tuple[str, str, str]):
        """Start both engines once - they will be reused for all games"""
        print("Starting engines (this may take a while due to model loading)...")

        self.network_a_info = network_a
        self.network_b_info = network_b

        cmd_a = self.make_engine_cmd(network_a[1], network_a[2])
        cmd_b = self.make_engine_cmd(network_b[1], network_b[2])

        start = time.time()
        _, self.engine_a = await chess.engine.popen_uci(cmd_a)
        print(f"  Engine A ({network_a[0]}) started in {time.time() - start:.1f}s")

        start = time.time()
        _, self.engine_b = await chess.engine.popen_uci(cmd_b)
        print(f"  Engine B ({network_b[0]}) started in {time.time() - start:.1f}s")

    async def stop_engines(self):
        """Stop both engines"""
        if self.engine_a:
            await self.engine_a.quit()
            self.engine_a = None
        if self.engine_b:
            await self.engine_b.quit()
            self.engine_b = None

    async def play_game(self, white_is_a: bool) -> GameResult:
        """Play a single game using the persistent engines

        Args:
            white_is_a: True if engine_a (network_a) plays white
        """
        if white_is_a:
            white_engine = self.engine_a
            black_engine = self.engine_b
            white_name = self.network_a_info[0]
            black_name = self.network_b_info[0]
        else:
            white_engine = self.engine_b
            black_engine = self.engine_a
            white_name = self.network_b_info[0]
            black_name = self.network_a_info[0]

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

            if self.args.nodes:
                limit = chess.engine.Limit(nodes=self.args.nodes)
            elif self.args.movetime:
                limit = chess.engine.Limit(time=self.args.movetime / 1000)
            elif self.args.depth:
                limit = chess.engine.Limit(depth=self.args.depth)
            else:
                limit = chess.engine.Limit(nodes=10000)

            result = await engine.play(board, limit)

            # Handle engine returning no move (uhjh)
            if result.move is None:
                fen = board.fen()
                logging.warning(f"Engine returned None move at position: {fen}")
                log_problematic_position(fen, "None move")
                return None  # Abort this game

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
            material = self._count_material(board)
            if abs(material) >= self.args.adjudicate_threshold:
                result_str = "1-0" if material > 0 else "0-1"
            else:
                result_str = "1/2-1/2"
        else:
            result_str = "1/2-1/2"

        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = result_str

        pgn_str = str(game) if self.args.save_pgn else ""

        return GameResult(white_name, black_name, result_str, pgn_str)

    def _count_material(self, board: chess.Board) -> int:
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}
        white_material = sum(values.get(p.piece_type, 0)
                             for p in board.piece_map().values()
                             if p.color == chess.WHITE)
        black_material = sum(values.get(p.piece_type, 0)
                             for p in board.piece_map().values()
                             if p.color == chess.BLACK)
        return white_material - black_material

    async def run_match_async(self, network_a: Tuple[str, str, str],
                              network_b: Tuple[str, str, str]) -> Dict:
        """Run a match between two networks (async version)"""
        print(f"\nRunning match: {network_a[0]} vs {network_b[0]}")
        print(f"Games: {self.args.num_games}")

        # Start engines once
        await self.start_engines(network_a, network_b)

        try:
            # Play all games sequentially, reusing the same engines
            games_played = 0
            retries = 0
            max_retries = 3

            with tqdm(total=self.args.num_games, desc="Playing games") as pbar:
                while games_played < self.args.num_games:
                    white_is_a = (games_played % 2 == 0)  # Alternate colors
                    result = await self.play_game(white_is_a)

                    if result is None:
                        # Game failed (engine crash), retry
                        retries += 1
                        if retries > max_retries:
                            logging.error(f"Too many retries, skipping game {games_played}")
                            retries = 0
                            games_played += 1
                            pbar.update(1)
                        continue

                    retries = 0
                    self.results.append(result)
                    games_played += 1
                    pbar.update(1)

                    # Save results incrementally
                    if self.args.output and games_played % 10 == 0:
                        self._save_results()

        finally:
            # Always stop engines
            await self.stop_engines()

        # Calculate statistics
        stats = self._calculate_stats(self.results, network_a[0], network_b[0])

        # Final save
        if self.args.output:
            self._save_results()

        return stats

    def run_match(self, network_a: Tuple[str, str, str],
                  network_b: Tuple[str, str, str]) -> Dict:
        """Run a match (sync wrapper)"""
        return asyncio.run(self.run_match_async(network_a, network_b))

    def _calculate_stats(self, results: List[GameResult],
                         name_a: str, name_b: str) -> Dict:
        wins_a = sum(1 for r in results if
                     (r.white_network == name_a and r.result == "1-0") or
                     (r.black_network == name_a and r.result == "0-1"))
        wins_b = sum(1 for r in results if
                     (r.white_network == name_b and r.result == "1-0") or
                     (r.black_network == name_b and r.result == "0-1"))
        draws = sum(1 for r in results if r.result == "1/2-1/2")

        total = len(results)
        score_a = wins_a + draws * 0.5

        if total > 0:
            win_rate = score_a / total
            elo_diff = 400 * (win_rate - 0.5) / 0.29
        else:
            win_rate = 0
            elo_diff = 0

        stats = {
            'network_a': name_a, 'network_b': name_b,
            'games': total, 'wins_a': wins_a, 'wins_b': wins_b,
            'draws': draws, 'score_a': score_a,
            'win_rate_a': win_rate, 'elo_diff': elo_diff,
            'timestamp': datetime.now().isoformat()
        }

        print(f"\nResults: {name_a} vs {name_b}")
        print(f"  Games: {total}")
        print(f"  Score: +{wins_a}={draws}-{wins_b}")
        print(f"  Win rate for {name_a}: {win_rate:.1%}")
        print(f"  Elo difference: {elo_diff:+.1f}")

        return stats

    def _save_results(self):
        output_path = Path(self.args.output)
        all_results = []
        if output_path.exists():
            with open(output_path, 'r') as f:
                data = json.load(f)
                all_results = data.get('games', [])

        # Only add results not already saved
        existing_count = len(all_results)
        new_results = self.results[existing_count:]
        all_results.extend([r.to_dict() for r in new_results])

        data = {'games': all_results, 'last_updated': datetime.now().isoformat()}
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Test NNUE networks against each other",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("engine", help="Path to sunfish_nnue.py")
    parser.add_argument("--network-a", required=True, nargs=3,
                        metavar=("NAME", "MODEL_NAME", "MODEL_PATH"),
                        help="First network: display_name model_name model_path")
    parser.add_argument("--network-b", required=True, nargs=3,
                        metavar=("NAME", "MODEL_NAME", "MODEL_PATH"),
                        help="Second network: display_name model_name model_path")

    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--compile-mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])

    parser.add_argument("--nodes", type=int, help="Nodes per move")
    parser.add_argument("--movetime", type=int, help="Time per move (ms)")
    parser.add_argument("--depth", type=int, help="Search depth limit")

    parser.add_argument("--max-moves", type=int, default=110)
    parser.add_argument("--adjudicate-threshold", type=int, default=6)
    parser.add_argument("--opening-moves", type=str)
    parser.add_argument("--book-path", type=str)

    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--save-pgn", action="store_true")

    args = parser.parse_args()

    tester = NetworkTester(args.engine, args)
    tester.run_match(tuple(args.network_a), tuple(args.network_b))


if __name__ == "__main__":
    main()