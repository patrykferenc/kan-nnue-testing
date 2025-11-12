#!/usr/bin/env python3

import models
import sys, time
from itertools import count
from collections import namedtuple, OrderedDict
import torch
from functools import partial, lru_cache

print = partial(print, flush=True)

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

import warnings

# Filter specific KAN statistics warnings that occur during inference
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
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 1
COMPILE_MODE = sys.argv[4] if len(sys.argv) > 4 else "default"

torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)
feature_set = features.get_feature_set_from_name("HalfKAv2_hm^")
model = models.nets[model_name](feature_set)
checkpoint = torch.load(model_path, map_location='cuda')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()
model.layer_stacks.idx_offset = torch.arange(
    0, BATCH_SIZE * model.layer_stacks.count, model.layer_stacks.count, device='cuda'
)
# TODO: fix compilation? or remove
# Warmup: Run a few forward passes to initialize CUBLAS before CUDA graph recording
logging.info(f"Warming up model with batch_size={BATCH_SIZE}...")
warmup_fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * BATCH_SIZE
warmup_outcomes = [0] * BATCH_SIZE
warmup_scores = [1] * BATCH_SIZE
warmup_plies = [0] * BATCH_SIZE

for _ in range(3):  # Multiple warmup iterations
    warmup_batch = nnue_dataset.make_sparse_batch_from_fens(
        feature_set, warmup_fens, warmup_outcomes, warmup_scores, warmup_plies
    )
    warmup_tensors = warmup_batch.contents.get_tensors('cuda')
    with torch.no_grad():
        _ = model.forward(
            warmup_tensors[0], warmup_tensors[1], warmup_tensors[2],
            warmup_tensors[3], warmup_tensors[4], warmup_tensors[5],
            warmup_tensors[8], warmup_tensors[9]
        )
    nnue_dataset.destroy_sparse_batch(warmup_batch)
    torch.cuda.synchronize()

# Warmup: Run a few forward passes to initialize CUBLAS before CUDA graph recording
logging.info(f"Warming up model with batch_size={BATCH_SIZE}, compile_mode={COMPILE_MODE}...")
warmup_fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * BATCH_SIZE
warmup_outcomes = [0] * BATCH_SIZE
warmup_scores = [1] * BATCH_SIZE
warmup_plies = [0] * BATCH_SIZE

for _ in range(3):  # Multiple warmup iterations
    warmup_batch = nnue_dataset.make_sparse_batch_from_fens(
        feature_set, warmup_fens, warmup_outcomes, warmup_scores, warmup_plies
    )
    warmup_tensors = warmup_batch.contents.get_tensors('cuda')
    with torch.no_grad():
        _ = model.forward(
            warmup_tensors[0], warmup_tensors[1], warmup_tensors[2],
            warmup_tensors[3], warmup_tensors[4], warmup_tensors[5],
            warmup_tensors[8], warmup_tensors[9]
        )
    nnue_dataset.destroy_sparse_batch(warmup_batch)
    torch.cuda.synchronize()

print(f"Warmup complete. Compiling model with mode='{COMPILE_MODE}'...", flush=True)
try:
    model = torch.compile(model, mode=COMPILE_MODE)

    # Run one more pass after compilation to trigger any graph recording
    warmup_batch = nnue_dataset.make_sparse_batch_from_fens(
        feature_set, warmup_fens, warmup_outcomes, warmup_scores, warmup_plies
    )
    warmup_tensors = warmup_batch.contents.get_tensors('cuda')
    with torch.no_grad():
        _ = model.forward(
            warmup_tensors[0], warmup_tensors[1], warmup_tensors[2],
            warmup_tensors[3], warmup_tensors[4], warmup_tensors[5],
            warmup_tensors[8], warmup_tensors[9]
        )
    nnue_dataset.destroy_sparse_batch(warmup_batch)
    torch.cuda.synchronize()
    logging.info("Compilation complete.")
    print("Model ready!", flush=True)
except Exception as e:
    logging.info(f"Warning: Compilation with {COMPILE_MODE} failed: {e}")
    logging.info("Falling back to default mode...")
    model = torch.compile(model, mode="default")
# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
MATE = 100000
MATE_LOWER = MATE // 2
MATE_UPPER = MATE * 3 // 2


class EvaluationBatcher:
    """Collects positions and evaluates them in batches with static batch size for CUDA graphs"""

    def __init__(self, model, feature_set, batch_size):
        self.model = model
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.pending_positions = []
        self.pending_fens = []
        self.results = {}

        # Padding FEN for incomplete batches (standard starting position)
        self.padding_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def add_position(self, pos, cache_key):
        """Add a position to the batch queue"""
        if cache_key not in self.results:
            fen = position_to_fen(pos)
            self.pending_positions.append((pos, cache_key))
            self.pending_fens.append(fen)

            # Process batch if full
            if len(self.pending_fens) >= self.batch_size:
                self.process_batch()

    def process_batch(self):
        """Process all pending positions as a batch with padding to maintain static batch size"""
        if not self.pending_fens:
            return

        num_real = len(self.pending_fens)

        # Pad to full batch size with dummy positions
        padded_fens = self.pending_fens.copy()
        while len(padded_fens) < self.batch_size:
            padded_fens.append(self.padding_fen)

        # Create batch with static batch size
        outcomes = [0] * self.batch_size
        scores = [1] * self.batch_size
        plies = [0] * self.batch_size

        b = nnue_dataset.make_sparse_batch_from_fens(
            self.feature_set, padded_fens, outcomes, scores, plies
        )

        # Get tensors
        (us, them, white_indices, white_values, black_indices, black_values,
         outcome, score, psqt_indices, layer_stack_indices) = b.contents.get_tensors('cuda')

        # Evaluate batch (no idx_offset modification - it's set at startup!)
        with torch.no_grad():
            eval_tensors = self.model.forward(
                us, them, white_indices, white_values, black_indices, black_values,
                psqt_indices, layer_stack_indices
            ) * 600.0

        # Store results (only for real positions, not padding)
        for i, (pos, cache_key) in enumerate(self.pending_positions):
            eval_score = eval_tensors[i].item()

            # Flip score if black to move
            if them[i].item() > 0.5:
                eval_score = -eval_score

            self.results[cache_key] = int(eval_score)

        # Clean up
        nnue_dataset.destroy_sparse_batch(b)
        self.pending_positions.clear()
        self.pending_fens.clear()

    def get_result(self, cache_key):
        """Get evaluation result, processing batch if needed"""
        if cache_key not in self.results:
            self.process_batch()
        return self.results.get(cache_key)

    def clear(self):
        """Clear all pending and cached results"""
        self.pending_positions.clear()
        self.pending_fens.clear()
        self.results.clear()


# Global batcher instance with static batch size
eval_batcher = EvaluationBatcher(model, feature_set, batch_size=BATCH_SIZE)


class LRUCache:
    """Least Recently Used cache with size limit"""

    def __init__(self, capacity=100000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)


# Global evaluation cache
eval_cache = LRUCache(capacity=200000)


###############################################################################
# Board to FEN conversion
###############################################################################

@lru_cache(maxsize=10000)
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
        cached_value = eval_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Check if both kings are present
        has_our_king = 'K' in self.board
        has_their_king = 'k' in self.board

        if not has_our_king:
            eval_cache.put(cache_key, -MATE_UPPER)
            return -MATE_UPPER

        if not has_their_king:
            eval_cache.put(cache_key, MATE_UPPER)
            return MATE_UPPER

        # Add to batch for evaluation
        eval_batcher.add_position(self, cache_key)

        # Get result (will process batch if needed)
        eval_score = eval_batcher.get_result(cache_key)

        # Store in LRU cache
        eval_cache.put(cache_key, eval_score)

        return eval_score

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
        # History heuristic for move ordering
        self.history_scores = {}

    def get_history_score(self, pos, move):
        """Get history heuristic score for move ordering"""
        if not move:
            return 0
        key = (pos.board[move.i], move.i, move.j)
        return self.history_scores.get(key, 0)

    def update_history_score(self, pos, move, depth):
        """Update history heuristic when a move causes a cutoff"""
        if not move:
            return
        key = (pos.board[move.i], move.i, move.j)
        bonus = depth * depth  # Quadratic depth bonus
        self.history_scores[key] = self.history_scores.get(key, 0) + bonus
        # Prevent overflow
        if self.history_scores[key] > 10000:
            # Age all history scores
            for k in self.history_scores:
                self.history_scores[k] //= 2

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

            def move_score(move):
                """Combined scoring for move ordering"""
                piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

                # Start with MVV-LVA for captures
                if abs(move.j - pos.kp) < 2:
                    return -100000  # King capture highest priority

                victim = pos.board[move.j]
                if victim != '.':
                    attacker = pos.board[move.i]
                    mvv_lva = piece_values[victim.upper()] * 10 - piece_values[attacker.upper()]
                    return -10000 - mvv_lva  # Captures before quiet moves
                elif move.j == pos.ep:
                    return -10900  # En passant
                elif move.prom:
                    return -5000 - piece_values[move.prom]  # Promotions
                else:
                    # Quiet moves ordered by history heuristic
                    return -self.get_history_score(pos, move)

            # Generate and sort moves
            all_moves = list(pos.gen_moves())
            all_moves.sort(key=move_score)

            for move in all_moves:
                if depth > 0 or pos.is_capture(move):
                    yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1, False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            if score > best:
                best = score
            if best >= gamma:
                # Save killer move and update history
                # logging.info(f"will save move? {move is not None}")
                if move is not None:
                    # logging.info(f"saving move {move} with score {score}")
                    self.tp_move[pos.hash()] = move
                    if depth > 0 and not pos.is_capture(move):
                        self.update_history_score(pos, move, depth)
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
        eval_batcher.clear()
        # Clearing table due to new history. This is because having a new "seen"
        # position alters the score of all other positions, as there may now be
        # a path that leads to a repetition.
        self.tp_score.clear()
        # We save the gamma function between depths, so we can start from the most
        # interesting position at the next level
        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        # logging.info(f"Starting search with len(history)={len(history)}, pos={pos}")
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays
            # better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            # logging.info(f"starting binary search at depth={depth}")
            while lower < upper - EVAL_ROUGHNESS:
                # logging.info(f"starting bound at depth={depth}, lower={lower}, upper={upper}")
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                # logging.info(
                #     f"finished bound at depth={depth}, lower={lower}, upper={upper}, score={score}, nodes(k)={self.nodes / 1000:.2f},got {self.tp_move.get(pos.hash())}")
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

    book_path = sys.argv[5] if len(sys.argv) > 5 else None

    if book_path:
        logging.info(f"Using book at {book_path}")

    tools.uci.run(sys.modules[__name__], hist[-1], book_path=book_path)
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
        wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
        if len(hist) % 2 == 0:
            wtime, winc = btime, binc
        think = min(wtime / 40 + winc, wtime / 2 - 1)

        start = time.time()
        move_str = None

        # logging.info(f"Starting search at {start} for {think} seconds, curr pos: {hist[-1]}")
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
        # logging.info(f"Finished search in {time.time() - start} seconds, bestmove is {move_str or '(none)'}")
        print("bestmove", move_str or '(none)')
