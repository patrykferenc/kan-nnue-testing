#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import logging
import re
import time
import argparse
import random
import chess
import chess.engine
import tqdm
import asyncio
import collections
import csv
import math
from pathlib import Path

random.seed(42)

import logging

logging.basicConfig(
    filename="test.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)



###############################################################################
# Results saving utilities
###############################################################################

class ResultsWriter:
    """Handles incremental CSV writing with crash safety"""

    def __init__(self, results_dir, command_name, fieldnames):
        if results_dir is None:
            self.file = None
            self.writer = None
            self.filename = None
            return

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = results_path / f"{command_name}_{timestamp}.csv"

        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def write_row(self, row_data):
        """Write a single row and flush immediately"""
        if self.writer:
            self.writer.writerow(row_data)
            self.file.flush()

    def write_rows(self, rows_data):
        """Write multiple rows and flush"""
        if self.writer:
            self.writer.writerows(rows_data)
            self.file.flush()

    def close(self):
        """Close the file and print confirmation"""
        if self.file:
            self.file.close()
            print(f"Results saved to {self.filename}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_results_csv(results_dir, command_name, results_data, fieldnames):
    """Save results to CSV file with timestamp (legacy batch mode)"""
    if results_dir is None or not results_data:
        return

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_path / f"{command_name}_{timestamp}.csv"

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"Results saved to {filename}")


class Command:
    @classmethod
    def add_arguments(cls, parser):
        raise NotImplementedError

    @classmethod
    async def run(cls, engine, args):
        raise NotImplementedError


###############################################################################
# Perft test
###############################################################################

from chess.engine import BaseCommand, UciProtocol


async def uci_perft(engine, depth):
    class UciPerftCommand(BaseCommand[UciProtocol, None]):
        def __init__(self, engine: UciProtocol):
            super().__init__(engine)
            self.moves = []

        def start(self, engine: UciProtocol) -> None:
            engine.send_line(f"go perft {depth}")

        def line_received(self, engine: UciProtocol, line: str) -> None:
            match = re.match(r"(\w+): (\d+)", line)
            if match:
                move = chess.Move.from_uci(match.group(1))
                cnt = int(match.group(2))
                self.moves.append((move, cnt))

            match = re.match(r"Nodes searched: (\d+)", line)
            if match:
                self.result.set_result(self.moves)
                self.set_finished()

    return await engine.communicate(UciPerftCommand)


class Perft(Command):
    name = "perft"
    help = "tests for correctness and speed of move generator."

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/queen.fen."
        )
        parser.add_argument("--depth", type=int, default=3)

    @classmethod
    async def run(cls, engine, args):
        lines = args.file.readlines()

        with ResultsWriter(getattr(args, 'results_dir', None), cls.name,
                           ['depth', 'fen', 'expected_nodes', 'actual_nodes', 'success']) as writer:

            for d in range(1, args.depth + 1):
                if not args.quiet:
                    print(f"Going to depth {d}/{args.depth}")

                for line in tqdm.tqdm(lines):
                    board, opts = chess.Board.from_epd(line)
                    engine._position(board)
                    moves = await uci_perft(engine, d)

                    cnt = sum(c for m, c in moves)
                    opt_cnt = int(opts[f"D{d}"])
                    success = cnt == opt_cnt

                    writer.write_row({
                        'depth': d,
                        'fen': board.fen(),
                        'expected_nodes': opt_cnt,
                        'actual_nodes': cnt,
                        'success': success
                    })

                    if not success:
                        print("=========================================")
                        print(f"ERROR at depth {d}. Gave {cnt} rather than {opt_cnt}")
                        print("=========================================")
                        print(board)
                        for m, c in moves:
                            print(m, c)
                        break


class Bench(Command):
    name = "bench"
    help = """Run through a fen file, search every position to a certain depth."""

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=100,
            help="Maximum plies at which to find the mate",
        )
        parser.add_argument(
            "--limit", type=int, default=10000, help="Maximum positions to analyse"
        )

    @classmethod
    async def run(self, engine, args):
        # Run through
        limit = chess.engine.Limit(depth=args.depth)
        lines = args.file.readlines()
        lines = lines[: args.limit]

        total_nodes = 0
        start = time.time()

        with ResultsWriter(getattr(args, 'results_dir', None), self.name,
                           ['fen', 'depth', 'nodes', 'time_s', 'knps']) as writer:

            pb = tqdm.tqdm(lines)
            for line in pb:
                board, _ = chess.Board.from_epd(line)
                pos_start = time.time()
                with await engine.analysis(board, limit) as analysis:
                    async for info in analysis:
                        pb.set_description(info_to_desc(info))
                pos_time = time.time() - pos_start
                nodes = info.get("nodes", 0)
                total_nodes += nodes

                writer.write_row({
                    'fen': board.fen(),
                    'depth': info.get("depth", 0),
                    'nodes': nodes,
                    'time_s': round(pos_time, 3),
                    'knps': round(nodes / (pos_time * 1000), 2) if pos_time > 0 else 0
                })

        elapsed = time.time() - start
        print(f"Total nodes: {total_nodes}.")
        print(f"Average knps: {round(total_nodes / elapsed / 1000, 2)}.")


###############################################################################
# Self-play
###############################################################################


class SelfPlay(Command):
    name = "self-play"
    help = "make sure the engine can complete a game without crashing."

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--time", type=int, default=4000, help="start time in ms")
        parser.add_argument("--inc", type=int, default=100, help="increment in ms")

    @classmethod
    async def run(cls, engine, args):
        wtime = btime = args.time / 1000
        inc = args.inc / 1000
        board = chess.Board()
        move_num = 0

        # Include 'result' in fieldnames even though it won't be in every row
        with ResultsWriter(getattr(args, 'results_dir', None), cls.name,
                           ['move_number', 'move', 'side', 'time_s', 'wtime_remaining',
                            'btime_remaining', 'result']) as writer:

            with tqdm.tqdm(total=100) as pbar:
                while not board.is_game_over():
                    limit = chess.engine.Limit(white_clock=wtime, black_clock=btime, white_inc=inc, black_inc=inc)

                    start = time.time()
                    result = await engine.play(board, limit)
                    elapsed = time.time() - start

                    if board.turn == chess.WHITE:
                        wtime -= elapsed - inc
                    else:
                        btime -= elapsed - inc

                    move_num += 1
                    writer.write_row({
                        'move_number': move_num,
                        'move': result.move.uci(),
                        'side': 'white' if board.turn == chess.WHITE else 'black',
                        'time_s': round(elapsed, 3),
                        'wtime_remaining': round(wtime, 2),
                        'btime_remaining': round(btime, 2),
                        'result': ''
                    })

                    board.push(result.move)
                    pbar.update(1)
                pbar.update(100 - pbar.n)

            # Write final game result
            writer.write_row({
                'move_number': move_num + 1,
                'move': 'GAME_OVER',
                'side': 'N/A',
                'time_s': 0,
                'wtime_remaining': round(wtime, 2),
                'btime_remaining': round(btime, 2),
                'result': board.result()
            })


###############################################################################
# Branching Factor Test
###############################################################################


class BranchingFactor(Command):
    name = "branching-factor"
    help = "Calculate branching factor from starting position to specified depth"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--depth",
            type=int,
            default=10,
            help="Depth to search to (default: 10)"
        )

    @classmethod
    async def run(cls, engine, args):
        board = chess.Board()  # Starting position
        limit = chess.engine.Limit(depth=args.depth)

        if not args.quiet:
            print(f"Searching starting position to depth {args.depth}...")

        start = time.time()
        with await engine.analysis(board, limit) as analysis:
            async for info in analysis:
                if not args.quiet:
                    desc = info_to_desc(info)
                    if desc:
                        print(f"  {desc}")
        elapsed = time.time() - start

        nodes = info.get("nodes", 0)
        depth = info.get("depth", args.depth)

        if nodes > 0 and depth > 0:
            branching_factor = math.exp(math.log(nodes) / depth)
        else:
            branching_factor = 0

        print(f"\nResults:")
        print(f"  Nodes searched: {nodes:,}")
        print(f"  Depth reached: {depth}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Branching factor: {branching_factor:.4f}")
        if elapsed > 0:
            print(f"  NPS: {int(nodes / elapsed):,}")

        # Save results
        with ResultsWriter(getattr(args, 'results_dir', None), cls.name,
                           ['depth', 'nodes', 'branching_factor', 'time_s', 'nps']) as writer:
            writer.write_row({
                'depth': depth,
                'nodes': nodes,
                'branching_factor': round(branching_factor, 4),
                'time_s': round(elapsed, 3),
                'nps': int(nodes / elapsed) if elapsed > 0 else 0
            })


###############################################################################
# Find mate test
###############################################################################


def info_to_desc(info):
    desc = []
    if "nodes" in info and "time" in info:
        # Add 1 to denominator, since time could be rounded to 0
        nps = info["nodes"] / (info["time"] + 1)
        desc.append(f"knps: {round(nps / 1000, 2)}")
    if "depth" in info:
        desc.append(f"depth: {info['depth']}")
    if "score" in info:
        desc.append(f"score: {info['score'].pov(chess.WHITE).cp / 100:.1f}")
    return ", ".join(desc)


def add_limit_argument(parser):
    parser.add_argument(
        "--depth",
        dest="limit_depth",
        type=int,
        default=0,
        help="Maximum plies at which to find the move",
    )
    parser.add_argument(
        "--mate-depth",
        dest="limit_mate",
        type=int,
        default=0,
        help="Maximum plies at which to find the mate",
    )
    parser.add_argument(
        "--movetime",
        dest="limit_movetime",
        type=int,
        default=100,
        help="Movetime in ms",
    )


def get_limit(args):
    if args.limit_depth:
        return chess.engine.Limit(depth=args.limit_depth)
    elif args.limit_mate:
        return chess.engine.Limit(mate=args.limit_mate)
    elif args.limit_movetime:
        return chess.engine.Limit(time=args.limit_movetime / 1000)


class Mate(Command):
    name = "mate"
    help = "Find the mates"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/mate{1,2,3}.fen."
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10000,
            help="Take only this many lines from the file",
        )
        add_limit_argument(parser)

    @classmethod
    async def run(cls, engine, args):
        limit = get_limit(args)
        total = 0
        success = 0
        lines = args.file.readlines()
        lines = lines[: args.limit]

        with ResultsWriter(getattr(args, 'results_dir', None), cls.name,
                           ['fen', 'success', 'depth', 'nodes', 'time_s', 'score']) as writer:

            pb = tqdm.tqdm(lines)
            for line in pb:
                total += 1
                board, _ = chess.Board.from_epd(line)
                pos_start = time.time()
                found_mate = False

                with await engine.analysis(board, limit) as analysis:
                    async for info in analysis:
                        pb.set_description(info_to_desc(info))
                        if not "score" in info:
                            continue
                        score = info["score"]
                        if score.is_mate() or score.relative.cp > 10000:
                            if "pv" in info and info["pv"]:
                                b = board.copy()
                                for move in info["pv"]:
                                    b.push(move)
                                if not b.is_game_over():
                                    if args.debug:
                                        print("Got mate score, but PV is not mate...")
                                    continue
                            if args.debug:
                                print("Found it!")
                            success += 1
                            found_mate = True
                            break
                    else:
                        if not args.quiet:
                            print("Failed on", line)
                            print("Result:", info)

                pos_time = time.time() - pos_start
                writer.write_row({
                    'fen': board.fen(),
                    'success': found_mate,
                    'depth': info.get("depth", 0),
                    'nodes': info.get("nodes", 0),
                    'time_s': round(pos_time, 3),
                    'score': str(info.get("score", "N/A"))
                })

        print(f"Succeeded in {success}/{len(lines)} cases.")


class Draw(Command):
    name = "draw"
    help = "Find the draws"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as tests/stalemate2.fen."
        )
        add_limit_argument(parser)

    @classmethod
    async def run(cls, engine, args):
        limit = get_limit(args)
        total, success = 0, 0
        cnt = collections.Counter()

        with ResultsWriter(getattr(args, 'results_dir', None), cls.name,
                           ['fen', 'success', 'depth', 'nodes', 'time_s', 'score']) as writer:

            pb = tqdm.tqdm(args.file.readlines())
            for line in pb:
                total += 1
                board, _ = chess.Board.from_epd(line)
                pos_start = time.time()
                found_draw = False

                with await engine.analysis(board, limit) as analysis:
                    last_lower = -10 ** 10
                    last_upper = 10 ** 10
                    async for info in analysis:
                        pb.set_description(info_to_desc(info))
                        if not "score" in info:
                            continue
                        score = info["score"]
                        if score.is_mate():
                            continue
                        if info.get('lowerbound'):
                            last_lower = score.relative.cp
                        elif info.get('upperbound'):
                            last_upper = score.relative.cp
                        elif score.relative.cp == 0:
                            success += 1
                            cnt[info["depth"]] += 1
                            found_draw = True
                            break
                        if -30 < last_lower and last_upper < 30:
                            success += 1
                            cnt[info["depth"]] += 1
                            found_draw = True
                            break
                    else:
                        if not args.quiet:
                            print("Failed on", line.strip())
                            print("Result:", info, 'lower', last_lower, 'upper', last_upper)

                pos_time = time.time() - pos_start
                writer.write_row({
                    'fen': board.fen(),
                    'success': found_draw,
                    'depth': info.get("depth", 0),
                    'nodes': info.get("nodes", 0),
                    'time_s': round(pos_time, 3),
                    'score': str(info.get("score", "N/A"))
                })

        print(f"Succeeded in {success}/{total} cases.")
        if not args.quiet:
            print("Depths:")
            for depth, c in cnt.most_common():
                print(f"{depth}: {c}")


###############################################################################
# Best move test
###############################################################################


class Best(Command):
    name = "best"
    help = "Find the best move"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "file", type=argparse.FileType("r"), help="such as bratko_kopec_test.epd."
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10000,
            help="Take only this many lines from the file",
        )
        add_limit_argument(parser)

    @classmethod
    async def run(cls, engine, args):
        limit = get_limit(args)
        points, total = 0, 0
        lines = args.file.readlines()
        lines = lines[: args.limit]

        with ResultsWriter(getattr(args, 'results_dir', None), cls.name,
                           ['fen', 'id', 'expected_bm', 'expected_am', 'given_move',
                            'bm_success', 'am_success', 'time_s', 'score']) as writer:

            for line in (pb := tqdm.tqdm(lines)):
                board, opts = chess.Board.from_epd(line)
                if "pv" in opts:
                    for move in opts["pv"]:
                        board.push(move)
                if "c0" in opts:
                    for key, val in re.findall(r"(\w+) (\w+)", opts["c0"]):
                        opts[key] = [chess.Move.from_uci(val)]
                if "am" not in opts and "bm" not in opts:
                    if not args.quiet:
                        print("Line didn't have am/bm in opts", line, opts)
                    continue

                pb.set_description(opts.get("id", ""))
                pos_start = time.time()
                result = await engine.play(board, limit, info=chess.engine.INFO_SCORE)
                pos_time = time.time() - pos_start

                errors = []
                bm_success = None
                am_success = None

                if "bm" in opts:
                    total += 1
                    bm_success = result.move in opts["bm"]
                    if bm_success:
                        points += 1
                    else:
                        errors.append(f'Gave move {result.move} rather than {opts["bm"]}')

                if "am" in opts:
                    total += 1
                    am_success = result.move not in opts["am"]
                    if am_success:
                        points += 1
                    else:
                        errors.append(f'Gave move {result.move} which is in {opts["am"]}')

                logging.info(f"result= {result}")

                writer.write_row({
                    'fen': board.fen(),
                    'id': opts.get("id", ""),
                    'expected_bm': str(opts.get("bm", [])),
                    'expected_am': str(opts.get("am", [])),
                    'given_move': result.move.uci(),
                    'bm_success': bm_success if bm_success is not None else '',
                    'am_success': am_success if am_success is not None else '',
                    'time_s': round(pos_time, 3),
                    'score': str(result.info.get("score", "N/A"))
                })

                if not args.quiet and errors:
                    print("Failed on", line.strip())
                    for er in errors:
                        print(er)
                    print("Full result:", result)
                    print()
                pb.set_postfix(acc=points / total if total > 0 else 0)

        print(f"Succeeded in {points}/{total} cases.")


###############################################################################
# Actions
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Run various tests for speed and correctness of sunfish."
    )
    parser.add_argument("args", help="Command and arguments to run")
    parser.add_argument(
        "--debug", action="store_true", help="Write lots of extra stuff"
    )
    parser.add_argument("--quiet", action="store_true", help="Only write pass/fail")
    parser.add_argument(
        "--xboard", action="store_true", help="Use xboard protocol instead of uci"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save test results as CSV files (created if doesn't exist)"
    )
    subparsers = parser.add_subparsers()
    subparsers.required = True

    for cls in Command.__subclasses__():
        sub = subparsers.add_parser(cls.name, help=cls.help)
        cls.add_arguments(sub)
        sub.set_defaults(func=cls.run)

    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    async def run():
        if args.xboard:
            _, engine = await chess.engine.popen_xboard(args.args.split())
        else:
            _transport, engine = await chess.engine.popen_uci(args.args.split())
        try:
            await args.func(engine, args)
        finally:
            await engine.quit()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    start = time.time()
    asyncio.run(run())
    print(f"Took {round(time.time() - start, 2)} seconds.")


if __name__ == "__main__":
    main()