#!/usr/bin/env python3

import models
import sys, time
from itertools import count
from collections import namedtuple
import torch
from functools import partial

print = partial(print, flush=True)

import logging

# LOGS
# logging.basicconfig(
#     filename="test.log",
#     level=logging.info,
#     format="%(asctime)s %(levelname)s %(message)s",
# s)

version = 'sunfish nnue'

###############################################################################
# Neural network setup
###############################################################################

from commons import features
from commons import nnue_dataset

# python sunfish_nnue.py <model_name:sfnnv9> <model_path.ckpt:/my/model/path/model.ckpt>
model_name = sys.argv[1]
model_path = sys.argv[2]

feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
model = models.nets[model_name](feature_set)
checkpoint = torch.load(model_path, map_location='cuda')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()

# Set up idx_offset (for batch size 1)
model.layer_stacks.idx_offset = torch.arange(
    0, model.layer_stacks.count, model.layer_stacks.count, device='cuda'
)

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
MATE = 100000
MATE_LOWER = MATE // 2
MATE_UPPER = MATE * 3 // 2


###############################################################################
# Board to FEN conversion
###############################################################################

def position_to_fen(pos):
    """Convert Position to FEN string

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
    # The state of a chess game
    # board -- a 120 char representation of the board
    # score -- the board evaluation
    # turn
    # wf -- our features
    # bf -- opponent features
    # wc -- the castling rights, [west/queen side, east/king side]
    # bc -- the opponent castling rights, [west/king side, east/queen side]
    # ep - the en passant square
    # kp - the king passant square

    _eval_cache = {}

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        # If the pawn moves forward, it has to not hit anybody
                        if d in (N, N + N) and q != ".":
                            break
                        # If the pawn moves forward twice, it has to be on the first row
                        # and it has to not jump over anybody
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."):
                            break
                        # If the pawn captures, it has to either be a piece, an
                        # enpassant square, or a moving king.
                        if d in (N + W, N + E) and q == "." and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q.islower():
                        break
                    # Castling, by sliding the rook next to the king. This way we don't
                    # need to worry about jumping over pieces while castling.
                    # We don't need to check for being a root, since if the piece starts
                    # at A1 and castling queen side is still allowed, it must be a rook.
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        # Rotates the board, preserving enpassant.
        # A nullmove is nearly a rotate, but it always clear enpassant.
        pos = Position(
            self.board[::-1].swapcase(), 0, self.bc, self.wc,
            0 if nullmove or not self.ep else 119 - self.ep,
            0 if nullmove or not self.kp else 119 - self.kp,
        )
        return pos._replace(score=pos.compute_value())

    def move(self, move):
        # TODO: I could update a zobrist hash here as well...
        # Then we are really becoming a real chess program...
        put = lambda pos, i, p: pos._replace(
            board=pos.board[:i] + p + pos.board[i + 1:]
        )

        i, j, pr = move
        p, q = self.board[i], self.board[j]
        pos = self._replace(ep=0, kp=0)
        pos = put(pos, j, p)
        pos = put(pos, i, ".")

        # Castling rights, we move the rook or capture the opponent's
        if i == A1: pos = pos._replace(wc=(False, pos.wc[1]))
        if i == H1: pos = pos._replace(wc=(pos.wc[0], False))
        if j == A8: pos = pos._replace(bc=(pos.bc[0], False))
        if j == H8: pos = pos._replace(bc=(False, pos.bc[1]))

        # Capture the moving king. Actually we get an extra free king. Same thing.
        if abs(j - self.kp) < 2:
            pos = put(pos, self.kp, "K")

        # Castling
        if p == "K":
            pos = pos._replace(wc=(False, False))
            if abs(j - i) == 2:
                pos = pos._replace(kp=(i + j) // 2)
                pos = put(pos, A1 if j < i else H1, ".")
                pos = put(pos, (i + j) // 2, "R")

        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                pos = put(pos, j, pr)
            if j - i == 2 * N:
                pos = pos._replace(ep=i + N)
            if j == self.ep:
                pos = put(pos, j + S, ".")

        return pos.rotate()

    def is_capture(self, move):
        # The original sunfish just checked that the evaluation of a move
        # was larger than a certain constant. However the current NN version
        # can have too much fluctuation in the evals, which can lead QS-search
        # to last forever (until python stackoverflows.) Thus we need to either
        # dampen the eval function, or like here, reduce QS search to captures
        # only. Well, captures plus promotions.
        return self.board[move.j] != "." or abs(move.j - self.kp) < 2 or move.prom

    def compute_value(self):
        """Evaluate position using PyTorch model

        Sunfish is a king-capture engine, so we need to check if both kings
        are present before trying to evaluate with the NNUE network.
        If a king is missing, return a mate score immediately.
        """
        cache_key = self.hash()
        if cache_key in Position._eval_cache:
            return Position._eval_cache[cache_key]

        # Check if both kings are present
        has_our_king = 'K' in self.board
        has_their_king = 'k' in self.board

        if not has_our_king:
            return -MATE_UPPER

        if not has_their_king:
            return MATE_UPPER

        # Both kings present - safe to evaluate with NNUE
        fen = position_to_fen(self)

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

        # Store in cache
        eval_score_int = int(eval_score)
        Position._eval_cache[cache_key] = eval_score_int

        return eval_score_int

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
        # returns r where
        #    s(pos) <= r < gamma    if gamma > s(pos)
        #    gamma <= r <= s(pos)   if gamma <= s(pos)
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        # I think this line also makes sure we never fail low on king-capture
        # replies, which might hide them and lead to illegal moves.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        # We need to include depth and root, since otherwise the function wouldn't
        # be consistent. By consistent I mean that if the function is called twice
        # with the same parameters, it will always fail in the same direction (hi / low).
        # It might return different soft values though, exactly because the tp tables
        # have changed.
        entry = self.tp_score.get(
            (pos.hash(), depth, root), Entry(-MATE_UPPER, MATE_UPPER)
        )
        if entry.lower >= gamma:
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # We detect 3-fold captures by comparing against previously
        # _actually played_ positions.
        # Note that we need to do this before we look in the table, as the
        # position may have been previously reached with a different score.
        # This is what prevents a search instability.
        # Actually, this is not true, since other positions will be affected by
        # the new values for all the drawn positions.
        # This is why I've decided to just clear tp_score every time history changes.
        if not root and pos.hash() in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            if depth > 2 and not root and any(c in pos.board for c in "NBRQ"):
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3, False)
            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # Then killer move. We search it twice, but the tp will fix things for us.
            # Note, we don't have to check for legality, since we've already done it
            # before. Also note that in QS the killer must be a capture, otherwise we
            # will be non deterministic.
            killer = self.tp_move.get(pos.hash())
            if killer and (depth > 0 or pos.is_capture(killer)):
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1, False)

            def mvv_lva(move):
                """MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                Returns negative score (lower = better for Python sort)
                """
                piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

                # King captures (highest priority)
                if abs(move.j - pos.kp) < 2:
                    return -MATE

                victim = pos.board[move.j]

                # Calculate capture score
                if victim != '.':
                    # Regular capture: prioritize valuable victims, cheap attackers
                    attacker = pos.board[move.i]
                    score = piece_values[victim.upper()] * 10 - piece_values[attacker.upper()]
                elif move.j == pos.ep:
                    # En passant (PxP: 1000 - 100 = 900)
                    score = 900
                else:
                    # Quiet move or promotion without capture
                    score = 0

                # Promotion bonus (makes Q promotions better than minor promotions)
                if move.prom:
                    score += piece_values[move.prom]

                return -score if score > 0 else 0

            for move in sorted(pos.gen_moves(), key=mvv_lva):
                if depth > 0 or pos.is_capture(move):
                    yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1, False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos.hash()] = move
                break

        # Stalemate checking
        if depth > 0 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Table part 2
        self.tp_score[pos.hash(), depth, root] = (
            Entry(best, entry.upper) if best >= gamma else Entry(entry.lower, best)
        )

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0
        pos = history[-1]
        self.history = {pos.hash() for pos in history}
        # Clearing table due to new history. This is because having a new "seen"
        # position alters the score of all other positions, as there may now be
        # a path that leads to a repetition.
        self.tp_score.clear()
        # We save the gamma function between depths, so we can start from the most
        # interesting position at the next level
        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays
            # better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(pos.hash())
                gamma = (lower + upper + 1) // 2


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
