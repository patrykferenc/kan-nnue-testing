#!/usr/bin/env python3

import sys
from collections import namedtuple, OrderedDict
from functools import partial, lru_cache
from itertools import count

import torch

import models

print = partial(print, flush=True)

# LOGS
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

import warnings

warnings.filterwarnings('ignore', message='std\\(\\): degrees of freedom is <= 0')
warnings.filterwarnings('ignore', category=UserWarning, module='kan.MultKAN')

version = 'sunfish nnue'

###############################################################################
# Neural network setup
###############################################################################

from commons import features
from commons import nnue_dataset

# python sunfish_nnue.py <model_name:sfnnv9> <model_path.ckpt:/my/model/path/model.ckpt>
model_name = sys.argv[1]
model_path = sys.argv[2]
COMPILE_MODE = sys.argv[3] if len(sys.argv) > 3 else "default"

torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)
feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
model = models.nets[model_name](feature_set)
checkpoint = torch.load(model_path, map_location='cuda')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()
model.layer_stacks.idx_offset = torch.arange(
    0, 1 * model.layer_stacks.count, model.layer_stacks.count, device='cuda'
)

try:
    model = torch.compile(model, mode=COMPILE_MODE)
    logging.info(f"Compilation complete with mode={COMPILE_MODE}")
except Exception as e:
    logging.info(f"Warning: Compilation with {COMPILE_MODE} failed: {e}")
    sys.exit(1)

MATE = 100000  # Base mate value
MATE_LOWER = MATE - 1000  # Mate detection threshold
MATE_UPPER = MATE + 1000  # Upper bound for mate

# Maximum expected eval from NNUE (for safety checks)
MAX_EVAL = 10000  # Slightly above 8000 for margin


###############################################################################
# Simplified Evaluation (No Batching)
###############################################################################

class LRUCache:
    """Least Recently Used cache with size limit"""

    def __init__(self, capacity=100000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


eval_cache = LRUCache(capacity=200000)


###############################################################################
# Board to FEN conversion
###############################################################################

@lru_cache(maxsize=10000)
def position_to_fen(pos):
    """Convert Position to FEN string"""
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


class Position(namedtuple("Position", "board score wc bc ep kp myhash")):
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
    def __new__(cls, board, score, wc, bc, ep, kp, myhash=None):
        with timer.time("Position.__new__"):
            if myhash is None:
                myhash = hash((board, wc, bc, ep, kp))
        return super().__new__(cls, board, score, wc, bc, ep, kp, myhash)

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

    def compute_value(self):
        with timer.time("compute_value"):
            """Direct single position evaluation"""
            cache_key = self.hash()
            cached_value = eval_cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Check for missing kings
            has_our_king = 'K' in self.board
            has_their_king = 'k' in self.board

            if not has_our_king:
                eval_cache.put(cache_key, -MATE_UPPER)
                return -MATE_UPPER

            if not has_their_king:
                eval_cache.put(cache_key, MATE_UPPER)
                return MATE_UPPER

            # Convert to FEN and evaluate
            with timer.time("fen"):
                fen = position_to_fen(self)

            # Create single-position batch
            with timer.time("sparse_batch"):
                b = nnue_dataset.make_sparse_batch_from_fens(
                    feature_set, [fen], [0], [1], [0]
                )

            with timer.time("tensors"):
                tensors = b.contents.get_tensors('cuda')
                (us, them, white_indices, white_values, black_indices, black_values,
                outcome, score, psqt_indices, layer_stack_indices) = tensors

            # logging.info(f"them {them} score {score} outcome {outcome}")
            with timer.time("eval"):
                with torch.no_grad():
                    eval_tensor = model.forward(
                        us, them, white_indices, white_values, black_indices, black_values,
                        psqt_indices, layer_stack_indices
                    ) * 600.0

            # logging.info(f"eval_tensor {eval_tensor}")
            eval_score = eval_tensor[0].item()
            # logging.info(f"eval_score {eval_score}")

            # Flip score if black to move
            if them[0].item() > 0.5:
                eval_score = -eval_score

            # logging.info(f"eval_score {eval_score}")
            eval_score = int(eval_score)
            # logging.info(f"eval_score {eval_score}")

            with timer.time("nnue_dataset"):
                nnue_dataset.destroy_sparse_batch(b)
            eval_cache.put(cache_key, eval_score)

        return eval_score

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

    def hash(self):
        return self.myhash


###############################################################################
# Search logic
###############################################################################

Entry = namedtuple("Entry", "lower upper")

PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
    '.': 0
}

def is_in_check(pos):
    """
    Proper check detection: flip the board and see if opponent
    can capture our king (king capture is available as a move)
    """
    # Flip the position to opponent's perspective
    flipped = pos.rotate(nullmove=True)

    # Check if opponent has a king capture available
    # Must check BOTH: direct capture of 'k' AND king passant (kp)
    for move in flipped.gen_moves():
        if flipped.board[move.j] == 'k' or abs(move.j - flipped.kp) < 2:
            return True

    return False


def futility_margin(depth, improving=False):
    """
    Calculate futility pruning margin.

    When improving=True: INCREASE margin (be MORE conservative, prune LESS)
    When improving=False: Use base margin
    """
    base_margin = 180 + 150 * depth

    if improving:
        # When position is improving, be MORE conservative
        # (larger margin = harder to trigger pruning)
        base_margin = int(base_margin * 1.15)  # INCREASE by 15%

    return base_margin


import time


class Timer:
    def __init__(self):
        self.timings = {}

    def time(self, name):
        return TimerContext(self, name)

    def report(self):
        total = sum(self.timings.values())
        print("\n=== Profiling Report ===")
        for name, t in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = 100 * t / total if total > 0 else 0
            print(f"{name:30s}: {t:8.3f}s ({pct:5.1f}%)")


class TimerContext:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.timer.timings[self.name] = self.timer.timings.get(self.name, 0) + elapsed


# Create global timer
timer = Timer()


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0
        self.eval_stack = []  # Track evals at each ply

    def quiesce(self, pos, alpha, beta, ply=0):
        """Quiescence search - ONLY captures and promotions"""
        self.nodes += 1

        # Stand pat
        stand_pat = pos.score
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Generate ONLY captures - no sorting needed for most positions
        captures = []
        for move in pos.gen_moves():
            if pos.is_capture(move):
                captures.append(move)

        # Simple MVV-LVA sort (much faster than full sort with function calls)
        # Sort inline without function call overhead
        captures.sort(key=lambda m: -(
                PIECE_VALUES[pos.board[m.j]] * 10 - PIECE_VALUES[pos.board[m.i]]
        ) if abs(m.j - pos.kp) >= 2 else -MATE)

        for move in captures:
            score = -self.quiesce(pos.move(move), -beta, -alpha, ply + 1)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def bound(self, pos, gamma, depth, root=True, ply=0):
        with timer.time("bound_total"):
            # returns r where
            #    s(pos) <= r < gamma    if gamma > s(pos)
            #    gamma <= r <= s(pos)   if gamma <= s(pos)
            self.nodes += 1

            # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
            # calmness, and from this point on there is no difference in behaviour depending on
            # depth, so so there is no reason to keep different depths in the transposition table.
            if depth <= 0:
                return self.quiesce(pos, gamma - 1, gamma, ply)

            #with timer.time("bound_tp"):
            if len(self.eval_stack) <= ply:
                self.eval_stack.extend([None] * (ply - len(self.eval_stack) + 1))
            # Store current eval
            self.eval_stack[ply] = pos.score

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


            # Check detection for LMR
            with timer.time("check_detection"):
                in_check_cached = None  # Lazy evaluation

                def is_pos_in_check():
                    nonlocal in_check_cached
                    if in_check_cached is None:
                        in_check_cached = is_in_check(pos)
                    return in_check_cached
            # Futility Pruning - prune quiet moves at shallow depths when position is hopeless
            # Only apply in non-PV nodes (when not at root)
            # Futility Pruning - improved version
            futility_pruning = False
            reverse_futility_return = None

            if not root and depth <= 4:
                with timer.time("check_safety_in_check"):
                    in_check = is_pos_in_check()
                with timer.time("check_safety_material"):
                    non_pawn_material = sum(1 for c in pos.board if c in 'NBRQ')
                safe_for_pruning = not in_check and non_pawn_material >= 2

                with timer.time("futility_pruning"):
                    if safe_for_pruning:
                        # Calculate improving CORRECTLY
                        improving = False
                        if ply >= 2 and len(self.eval_stack) > ply - 2:
                            if self.eval_stack[ply - 2] is not None:
                                improving = pos.score > self.eval_stack[ply - 2]

                    # Forward Futility Pruning
                        if depth <= 3:
                            margin = futility_margin(depth, improving)
                            if pos.score + margin < gamma:
                                futility_pruning = True

                    # Reverse Futility Pruning
                        if depth <= 3 and abs(pos.score) < 1000:
                        # Use different margin for RFP (can be more aggressive)
                            rfp_margin = futility_margin(depth, improving)
                            if pos.score - rfp_margin >= gamma:
                                reverse_futility_return = pos.score


            def mvv_lva(move):
                with timer.time("mvv_lva"):
                    # Recall mvv_lva gives the _negative_ score (for ascending sort)
                    # Don't capture near the opponent's king (could be illegal)
                    if abs(move.j - pos.kp) < 2:
                        return -MATE

                    i, j = move.i, move.j
                    p, q = pos.board[i], pos.board[j]
                    p2 = move.prom or p  # Piece after promotion

                    # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                    victim_value = PIECE_VALUES[q]
                    attacker_value = PIECE_VALUES[p2]

                    # Return negative score: lower (more negative) = searched first
                    # Multiply victim by 10 to prioritize victim value over attacker value
                    score = -(victim_value * 10 - attacker_value)

                    # Bonus for promotions (already factored into attacker_value via p2)
                    # But add extra incentive for promotion moves
                    if move.prom:
                        score -= 1000  # Promotions are very good

                return score

            # Generator of moves to search in order.
            # This allows us to define the moves, but only calculate them if needed.
            def moves():
                # First try not moving at all. We only do this if there is at least one major
                # piece left on the board, since otherwise zugzwangs are too dangerous.
                with timer.time("moves_order_full"):
                    if depth > 2 and not root and any(c in pos.board for c in "NBRQ"):
                        yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3, False)
                    # early return...
                    if reverse_futility_return is not None:
                        return  # Don't generate moves, return early
                    # For QSearch we have a different kind of null-move, namely we can just stop
                    # and not capture anything else.
                    if depth == 0:
                        yield None, pos.score

                    # Then killer move. We search it twice, but the tp will fix things for us.
                    # Note, we don't have to check for legality, since we've already done it
                    # before. Also note that in QS the killer must be a capture, otherwise we
                    # will be non deterministic.
                    # Killer move - yield without score for LMR handling
                    killer = self.tp_move.get(pos.hash())
                    if killer and (depth > 0 or pos.is_capture(killer)):
                        yield killer, None

                        # Sort by the score after moving. Since that's from the perspective of our
                        # opponent, smaller score means the move is better for us.
                        # print(f'Searching at {depth=}')
                        # TODO: Maybe try MMT/LVA sorting here. Could be cheaper and work better since
                        # the current evaluation based method doesn't take into account that e.g. capturing
                        # with the queen shouldn't usually be our first option...
                        # It could be fun to train a network too, that scores all the from/too target
                        # squares, say, and uses that to sort...
                        # for move, pos1 in sorted(moves, key=lambda move_pos: move_pos[1].score):
                        # All other moves - yield without score for LMR handling
                    with timer.time("generation"):
                        all_moves = pos.gen_moves()
                    with timer.time("sorting"):
                        for move in sorted(all_moves, key=mvv_lva):
                            # Skip this move if it's the killer (already yielded)
                            if killer and move == killer:
                                continue

                            if futility_pruning and not pos.is_capture(move) and not move.prom:
                                continue
                            if depth > 0 or pos.is_capture(move):
                                yield move, None
                    # TODO: We seem to have some issues with our QS search, which eventually
                    # leads to very large jumps in search time. (Maybe we get the classical
                    # "Queen plunders everything" case?) Hence Improving this might solve some
                    # of our timeout issues. It could also be that using a more simple ordering
                    # would speed up the move generation?
                    # See https://home.hccnet.nl/h.g.muller/mvv.html for inspiration
                    # If depth is 0 we only try moves with high intrinsic score (captures and
                    # promotions). Otherwise we do all moves.
                    # if depth > 0 or -pos1.score-pos.score >= QS_LIMIT:
                    # if depth > 0 or pos.is_capture(move):
                    #     yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1, False)

            with timer.time("search_loop"):
                # Run through the moves, shortcutting when possible
                best = -MATE_UPPER
                moves_searched = 0
                for move, pre_score in moves():
                    if pre_score is not None:
                        # Special moves with pre-computed scores (null move, stand pat)
                        score = pre_score
                    else:
                        # Regular move - potentially apply LMR
                        moves_searched += 1

                        # Determine if we should reduce this move
                        reduction = 0

                        # LMR conditions:
                        # - Not at root (PV node)
                        # - Sufficient depth
                        # - Not one of the first few moves
                        # - Not a tactical move (capture/promotion)
                        # - Not in check
                        if (not root and
                                depth >= 3 and
                                moves_searched >= 4 and
                                move is not None and
                                not pos.is_capture(move) and
                                not move.prom and
                                not is_pos_in_check()):

                            # Calculate reduction based on depth and move count
                            # More aggressive reduction for later moves and higher depths
                            if depth >= 6 and moves_searched >= 12:
                                reduction = 3
                            elif depth >= 5 and moves_searched >= 8:
                                reduction = 2
                            elif depth >= 3 and moves_searched >= 4:
                                reduction = 1

                            # Reduce reduction if move gives check (if we could detect this efficiently)
                            # For now, we use the base reduction

                        # Search with reduction
                        if reduction > 0:
                            # Search at reduced depth
                            new_depth = max(0, depth - 1 - reduction)
                            with timer.time("reduced_search"):
                                score = -self.bound(pos.move(move), 1 - gamma, new_depth, False, ply + 1)

                            # If reduced search fails high (move looks good), re-search at full depth
                            if score >= gamma:
                                with timer.time("extended_search"):
                                    score = -self.bound(pos.move(move), 1 - gamma, depth - 1, False, ply + 1)
                        else:
                            # Normal full-depth search
                            with timer.time("normal_search"):
                                score = -self.bound(pos.move(move), 1 - gamma, depth - 1, False, ply + 1)

                    best = max(best, score)
                    if best >= gamma:
                        # Save the move for pv construction and killer heuristic
                        if move is not None:
                            self.tp_move[pos.hash()] = move
                        break

            # Stalemate checking
            if depth > 0 and best == -MATE_UPPER:
                flipped = pos.rotate(nullmove=True)
                in_check = self.bound(flipped, MATE_UPPER, 0, root=False, ply=ply + 1) == MATE_UPPER
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
        try:
            for depth in range(1, 1000):
                # yield depth, None, 0, "cp"
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
        finally:
            timer.report()


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

    book_path = sys.argv[4] if len(sys.argv) > 4 else None
    if book_path:
        logging.info(f"Using book at {book_path}")

    tools.uci.run(sys.modules[__name__], hist[-1], book_path=book_path)

sys.exit()
