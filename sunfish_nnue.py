#!/usr/bin/env python3

import models
import sys, time
from itertools import count
from collections import namedtuple
import torch
from functools import partial

print = partial(print, flush=True)

import logging

# logging.basicconfig(
#     filename="test.log",
#     level=logging.info,
#     format="%(asctime)s %(levelname)s %(message)s",
# s)

version = 'sunfish nnue'

###############################################################################
# Neural network setup
###############################################################################

# Import nnue-pytorch modules
from commons import features
from commons import nnue_dataset

# Import model modules (add your model imports here)
# import SFNNv9

# import other model modules as needed
#  nets = {
#     'sfnnv9': lambda feature_set: SFNNv9.NNUE(feature_set),
#     # add other models here
# }
#
# Command line: python sunfish_nnue_kan.py <model_name> <model_path.pt>
model_name = sys.argv[1]
model_path = sys.argv[2]

feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
model = models.nets[model_name](feature_set)
checkpoint = torch.load(model_path, map_location='cuda')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()

# Set up idx_offset for batch size 1
model.layer_stacks.idx_offset = torch.arange(
    0, model.layer_stacks.count, model.layer_stacks.count, device='cuda'
)

# Mate value
MATE = 100000
MATE_LOWER = MATE // 2
MATE_UPPER = MATE * 3 // 2


###############################################################################
# Board to FEN conversion
###############################################################################

def position_to_fen(pos):
    """Convert Position to FEN string - CORRECTED VERSION

    After rotate() does board[::-1].swapcase():
    - The entire board string is reversed (index i → index 119-i)
    - All piece cases are swapped (uppercase↔lowercase)
    - Position 21-28 contains rank 1 reversed (h1→a1)
    - Position 91-98 contains rank 8 reversed (h8→a8)
    """
    white_to_move = not pos.board.startswith("\n")

    fen_rows = []
    for rank in range(8):
        row_start = 21 + rank * 10
        row = pos.board[row_start:row_start + 8]

        # After rotation, each row is horizontally reversed
        # Reverse it back to get a→h for FEN format
        if not white_to_move:
            row = row[::-1]

        fen_row = ""
        empty_count = 0
        for c in row:
            if c == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                if white_to_move:
                    fen_row += c
                else:
                    fen_row += c.swapcase()
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    # After rotation, ranks are reversed (rank 0 = rank 1, rank 7 = rank 8)
    # FEN requires rank 8→1, so reverse for rotated boards
    if not white_to_move:
        fen_rows = fen_rows[::-1]

    piece_placement = '/'.join(fen_rows)
    side = 'w' if white_to_move else 'b'

    if white_to_move:
        castling = ''
        if pos.wc[1]: castling += 'K'
        if pos.wc[0]: castling += 'Q'
        if pos.bc[1]: castling += 'k'
        if pos.bc[0]: castling += 'q'
    else:
        castling = ''
        if pos.bc[1]: castling += 'K'
        if pos.bc[0]: castling += 'Q'
        if pos.wc[1]: castling += 'k'
        if pos.wc[0]: castling += 'q'
    if not castling: castling = '-'

    if pos.ep:
        rank, fil = divmod(pos.ep - A1, 10)
        ep_square = chr(fil + ord('a')) + str(-rank + 1)
    else:
        ep_square = '-'

    return f"{piece_placement} {side} {castling} {ep_square} 0 1"


###############################################################################
# Global constants
###############################################################################

A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  # 0 -  9
    "         \n"  # 10 - 19
    " rnbqkbnr\n"  # 20 - 29
    " pppppppp\n"  # 30 - 39
    " ........\n"  # 40 - 49
    " ........\n"  # 50 - 59
    " ........\n"  # 60 - 69
    " ........\n"  # 70 - 79
    " PPPPPPPP\n"  # 80 - 89
    " RNBQKBNR\n"  # 90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N + N, N + W, N + E),
    "N": (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    "B": (N + E, S + E, S + W, N + W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N + E, S + E, S + W, N + W),
    "K": (N, E, S, W, N + E, S + E, S + W, N + W),
}

EVAL_ROUGHNESS = 13

# minifier-hide start
opt_ranges = dict(
    EVAL_ROUGHNESS=(0, 50),
)
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################

Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score wc bc ep kp")):
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        if d in (N, N + N) and q != ".":
                            break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."):
                            break
                        if d in (N + W, N + E) and q == "." and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1):
                            break
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    yield Move(i, j, "")
                    if p in "PNK" or q.islower():
                        break
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        pos = Position(
            self.board[::-1].swapcase(), 0, self.bc, self.wc,
            0 if nullmove or not self.ep else 119 - self.ep,
            0 if nullmove or not self.kp else 119 - self.kp,
        )
        return pos._replace(score=pos.compute_value())

    def move(self, move):
        put = lambda pos, i, p: pos._replace(
            board=pos.board[:i] + p + pos.board[i + 1:]
        )

        i, j, pr = move
        p, q = self.board[i], self.board[j]
        pos = self._replace(ep=0, kp=0)
        pos = put(pos, j, p)
        pos = put(pos, i, ".")

        if i == A1: pos = pos._replace(wc=(False, pos.wc[1]))
        if i == H1: pos = pos._replace(wc=(pos.wc[0], False))
        if j == A8: pos = pos._replace(bc=(pos.bc[0], False))
        if j == H8: pos = pos._replace(bc=(False, pos.bc[1]))
        if abs(j - self.kp) < 2:
            pos = put(pos, self.kp, "K")
        if p == "K":
            pos = pos._replace(wc=(False, False))
            if abs(j - i) == 2:
                pos = pos._replace(kp=(i + j) // 2)
                pos = put(pos, A1 if j < i else H1, ".")
                pos = put(pos, (i + j) // 2, "R")
        if p == "P":
            if A8 <= j <= H8:
                pos = put(pos, j, pr)
            if j - i == 2 * N:
                pos = pos._replace(ep=i + N)
            if j == self.ep:
                pos = put(pos, j + S, ".")

        return pos.rotate()

    def is_capture(self, move):
        return self.board[move.j] != "." or abs(move.j - self.kp) < 2 or move.prom

    def compute_value(self):
        """Evaluate position using PyTorch model

        Sunfish is a king-capture engine, so we need to check if both kings
        are present before trying to evaluate with the NNUE network.
        If a king is missing, return a mate score immediately.
        """
        # Check if both kings are present
        # After moves, uppercase pieces are "our" pieces, lowercase are "their" pieces
        has_our_king = 'K' in self.board
        has_their_king = 'k' in self.board

        if not has_our_king:
            # Our king was captured - we're mated
            #logging.info(f"Our king missing, returning -MATE_UPPER")
            return -MATE_UPPER

        if not has_their_king:
            # Their king was captured - we're delivering mate
            #logging.info(f"Their king missing, returning MATE_UPPER")
            return MATE_UPPER

        # Both kings present - safe to evaluate with NNUE
        fen = position_to_fen(self)
        #logging.info(f"Computing value for {fen}")

        # Create sparse batch from FEN
        b = nnue_dataset.make_sparse_batch_from_fens(
            feature_set, [fen], [0], [1], [0]
        )

        # Get tensors
        (us, them, white_indices, white_values, black_indices, black_values,
         outcome, score, psqt_indices, layer_stack_indices) = b.contents.get_tensors('cuda')

        # Evaluate with model
        with torch.no_grad():
            eval_tensor = model.forward(
                us, them, white_indices, white_values, black_indices, black_values,
                psqt_indices, layer_stack_indices
            ) * 600.0

        eval_score = eval_tensor.item()

        # Flip score if black to move (them > 0.5)
        if them[0].item() > 0.5:
            eval_score = -eval_score

        # Clean up
        nnue_dataset.destroy_sparse_batch(b)
        return int(eval_score)

    def hash(self):
        return hash((self.board, self.wc, self.bc, self.ep, self.kp))


###############################################################################
# Search logic
###############################################################################

Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        self.nodes += 1
        depth = max(depth, 0)

        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        entry = self.tp_score.get(
            (pos.hash(), depth, root), Entry(-MATE_UPPER, MATE_UPPER)
        )
        if entry.lower >= gamma:
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        if not root and pos.hash() in self.history:
            return 0

        def moves():
            if depth > 2 and not root and any(c in pos.board for c in "NBRQ"):
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3, False)
            if depth == 0:
                yield None, pos.score

            killer = self.tp_move.get(pos.hash())
            if killer and (depth > 0 or pos.is_capture(killer)):
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1, False)

            def mvv_lva(move):
                if abs(move.j - pos.kp) < 2:
                    return -MATE
                return 0  # Simplified ordering

            for move in sorted(pos.gen_moves(), key=mvv_lva):
                if depth > 0 or pos.is_capture(move):
                    yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1, False)

        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                if move is not None:
                    self.tp_move[pos.hash()] = move
                break

        if depth > 0 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0
            #logging.info(f"flipped {flipped} in_check {in_check}")

        self.tp_score[pos.hash(), depth, root] = (
            Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        )

        return best

    def search(self, history):
        self.nodes = 0
        pos = history[-1]
        self.history = {pos.hash() for pos in history}
        self.tp_score.clear()
        gamma = 0

        #logging.info(f"Starting search for pos {pos}")

        for depth in range(1, 1000):
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(pos.hash())
                gamma = (lower + upper + 1) // 2
                #logging.info(f"depth {depth} score {score}")


###############################################################################
# UCI interface
###############################################################################

def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)


hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]
searcher = Searcher()

# minifier-hide start
if '--profile' in sys.argv:
    import cProfile


    def go_depth_5():
        for depth, _, _, _ in searcher.search(hist):
            if depth == 5:
                break


    cProfile.run('go_depth_5()')
else:
    import tools.uci

    tools.uci.run(sys.modules[__name__], hist[-1])
sys.exit()
# minifier-hide end


while True:
    args = input().split()
    if args[0] == "uci":
        print(f"id name {version}")
        print("uciok")

    elif args[0] == "isready":
        print("readyok")

    elif args[0] == "quit":
        break

    elif args[:2] == ["position", "startpos"]:
        del hist[1:]
        for ply, move in enumerate(args[3:]):
            i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
            if ply % 2 == 1:
                i, j = 119 - i, 119 - j
            hist.append(hist[-1].move(Move(i, j, prom)))

    elif args[0] == "go":
        # logging.info(f"got args {args}")
        wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
        if len(hist) % 2 == 0:
            wtime, winc = btime, binc
        think = min(wtime / 40 + winc, wtime / 2 - 1)

        start = time.time()
        move_str = None
        for depth, gamma, score, move in Searcher().search(hist):
            # The only way we can be sure to have the real move in tp_move,
            # is if we have just failed high.
            if score >= gamma:
                i, j = move.i, move.j
                if len(hist) % 2 == 0:
                    i, j = 119 - i, 119 - j
                move_str = render(i) + render(j) + move.prom.lower()
                print(f"info depth {depth} score cp {score} pv {move_str}")
            if move_str and time.time() - start > think * 0.8:
                break

        print("bestmove", move_str or '(none)')
