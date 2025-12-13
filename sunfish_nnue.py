#!/usr/bin/env python3

import sys
from collections import namedtuple, OrderedDict
from functools import partial, lru_cache
from itertools import count
import math

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
# Late Move Pruning (LMP) thresholds - from Stockfish
# Formula: (3 + depth^2) / (2 - improving)
###############################################################################

def lmp_threshold(depth, improving):
    """How many quiet moves to search before pruning the rest."""
    return (3 + depth * depth) // (2 - int(improving))


# Precomputed LMP table for quick lookup
LMP_TABLE = {
    (d, imp): lmp_threshold(d, imp)
    for d in range(1, 10)
    for imp in [True, False]
}

###############################################################################
# SEE (Static Exchange Evaluation) - Piece values for SEE
###############################################################################

# SEE piece values (centipawns) - standard values
SEE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
    '.': 0, ' ': 0, '\n': 0
}

# Piece ordering for SEE (least valuable first)
SEE_PIECE_ORDER = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}


def get_attackers_to_square(board, sq, by_white):
    """
    Find all pieces of one side attacking a square.
    Returns list of (attacker_square, piece_type) sorted by piece value (LVA).
    """
    attackers = []

    if by_white:
        # White pieces attack (uppercase)
        # Pawns attack diagonally from south
        for d in (S + W, S + E):
            src = sq + d
            if 0 <= src < 120 and board[src] == 'P':
                attackers.append((src, 'P'))

        # Knights
        for d in directions['N']:
            src = sq + d
            if 0 <= src < 120 and board[src] == 'N':
                attackers.append((src, 'N'))

        # Bishops/Queens (diagonals)
        for d in directions['B']:
            for dist in range(1, 8):
                src = sq + d * dist
                if src < 0 or src >= 120 or board[src].isspace():
                    break
                if board[src] == 'B' or board[src] == 'Q':
                    attackers.append((src, board[src]))
                    break
                if board[src] != '.':
                    break

        # Rooks/Queens (orthogonals)
        for d in directions['R']:
            for dist in range(1, 8):
                src = sq + d * dist
                if src < 0 or src >= 120 or board[src].isspace():
                    break
                if board[src] == 'R' or board[src] == 'Q':
                    attackers.append((src, board[src]))
                    break
                if board[src] != '.':
                    break

        # King
        for d in directions['K']:
            src = sq + d
            if 0 <= src < 120 and board[src] == 'K':
                attackers.append((src, 'K'))
    else:
        # Black pieces attack (lowercase)
        # Pawns attack diagonally from north
        for d in (N + W, N + E):
            src = sq + d
            if 0 <= src < 120 and board[src] == 'p':
                attackers.append((src, 'p'))

        # Knights
        for d in directions['N']:
            src = sq + d
            if 0 <= src < 120 and board[src] == 'n':
                attackers.append((src, 'n'))

        # Bishops/Queens (diagonals)
        for d in directions['B']:
            for dist in range(1, 8):
                src = sq + d * dist
                if src < 0 or src >= 120 or board[src].isspace():
                    break
                if board[src] == 'b' or board[src] == 'q':
                    attackers.append((src, board[src]))
                    break
                if board[src] != '.':
                    break

        # Rooks/Queens (orthogonals)
        for d in directions['R']:
            for dist in range(1, 8):
                src = sq + d * dist
                if src < 0 or src >= 120 or board[src].isspace():
                    break
                if board[src] == 'r' or board[src] == 'q':
                    attackers.append((src, board[src]))
                    break
                if board[src] != '.':
                    break

        # King
        for d in directions['K']:
            src = sq + d
            if 0 <= src < 120 and board[src] == 'k':
                attackers.append((src, 'k'))

    # Sort by piece value (least valuable first - LVA)
    attackers.sort(key=lambda x: SEE_PIECE_ORDER.get(x[1], 6))
    return attackers


def get_xray_attacker(board, sq, from_sq, by_white):
    """
    After removing piece at from_sq, check if there's an x-ray attacker behind it.
    Returns (attacker_square, piece_type) or None.
    """
    # Determine direction from sq to from_sq
    diff = from_sq - sq

    # Normalize to get direction
    if diff == 0:
        return None

    # Check if it's a valid sliding direction
    direction = None
    for d in (N, S, E, W, N + E, N + W, S + E, S + W):
        if diff % d == 0 and diff // d > 0:
            steps = diff // d
            # Verify it's actually this direction
            if sq + d * steps == from_sq:
                direction = d
                break

    if direction is None:
        return None

    # Check for pieces along this ray beyond from_sq
    for dist in range(1, 8):
        check_sq = from_sq + direction * dist
        if check_sq < 0 or check_sq >= 120 or board[check_sq].isspace():
            break

        piece = board[check_sq]
        if piece == '.':
            continue

        is_white_piece = piece.isupper()
        if is_white_piece != by_white:
            break  # Enemy piece blocks

        # Check if this piece can attack along this direction
        piece_upper = piece.upper()
        is_diagonal = direction in (N + E, N + W, S + E, S + W)
        is_orthogonal = direction in (N, S, E, W)

        if piece_upper == 'Q':
            return (check_sq, piece)
        if piece_upper == 'B' and is_diagonal:
            return (check_sq, piece)
        if piece_upper == 'R' and is_orthogonal:
            return (check_sq, piece)

        break  # Piece blocks but doesn't attack

    return None


def see(pos, move):
    """
    Static Exchange Evaluation for a capture move.
    Returns expected material gain/loss in centipawns from the moving side's perspective.

    Positive = winning exchange, Negative = losing exchange
    """
    i, j, prom = move
    board = list(pos.board)  # Mutable copy

    # Handle special cases
    # En passant
    if pos.board[i].upper() == 'P' and j == pos.ep:
        # Captured pawn is on a different square
        captured_sq = j + S if pos.board[i].isupper() else j + N
        captured_value = SEE_VALUES['P']
        board[captured_sq] = '.'
    # King passant (sunfish special)
    elif abs(j - pos.kp) < 2 and pos.kp != 0:
        captured_value = SEE_VALUES['K']
    else:
        captured_value = SEE_VALUES[board[j]]

    # Promotion bonus
    if prom:
        prom_bonus = SEE_VALUES[prom] - SEE_VALUES['P']
    else:
        prom_bonus = 0

    attacker = board[i]
    attacker_value = SEE_VALUES[attacker]

    # If capturing nothing and no promotion, it's not a capture
    if captured_value == 0 and not prom:
        return 0

    # Simulate the capture
    board[j] = attacker if not prom else prom
    board[i] = '.'

    # Track gains - index 0 is initial capture
    gains = [captured_value + prom_bonus]

    # Current piece on target square
    piece_on_square = attacker_value if not prom else SEE_VALUES[prom]

    # Alternating sides
    side_to_move_is_white = not attacker.isupper()  # Opponent moves next

    # Build attacker lists for both sides
    depth = 0
    max_depth = 32

    while depth < max_depth:
        # Get attackers for current side
        attackers = get_attackers_to_square(''.join(board), j, side_to_move_is_white)

        if not attackers:
            break

        # Pick least valuable attacker
        atk_sq, atk_piece = attackers[0]
        atk_value = SEE_VALUES[atk_piece]

        # Record the gain (capturing what's on the square)
        gains.append(piece_on_square)

        # Check for x-ray attackers behind this piece
        xray = get_xray_attacker(''.join(board), j, atk_sq, side_to_move_is_white)

        # Make the capture
        board[j] = atk_piece
        board[atk_sq] = '.'

        piece_on_square = atk_value
        side_to_move_is_white = not side_to_move_is_white
        depth += 1

    # Negamax the gains array
    while len(gains) > 1:
        gains[-2] = gains[-2] - max(0, gains[-1])
        gains.pop()

    return gains[0]


def see_capture_is_good(pos, move, threshold=0):
    """Quick check if a capture has SEE >= threshold."""
    return see(pos, move) >= threshold


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
                        if d in (N + W, N + E) and q == "." and j not in (self.ep, self.kp, self.kp - 1,
                                                                          self.kp + 1):
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

    def gen_captures(self):
        """Generate only capture moves (for quiescence)."""
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
                        if d in (N + W, N + E):
                            # Only yield if actual capture or en passant
                            if q == "." and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1):
                                break
                            # This is a capture
                            if A8 <= j <= H8:
                                yield from (Move(i, j, prom) for prom in "NBRQ")
                            else:
                                yield Move(i, j, "")
                            break
                        # Non-capture pawn moves - skip in quiescence except promotions
                        if d == N and A8 <= j <= H8:
                            yield from (Move(i, j, prom) for prom in "NBRQ")
                        break
                    # Non-pawn pieces
                    if q.islower() or abs(j - self.kp) < 2:
                        # Capture
                        yield Move(i, j, "")
                        break
                    if p in "NK":
                        break

    def rotate(self, nullmove=False, skip_score=False):
        # Rotates the board, preserving enpassant.
        # A nullmove is nearly a rotate, but it always clear enpassant.
        pos = Position(
            self.board[::-1].swapcase(), 0, self.bc, self.wc,
            0 if nullmove or not self.ep else 119 - self.ep,
            0 if nullmove or not self.kp else 119 - self.kp,
        )
        if skip_score:
            return pos._replace()
        return pos._replace(score=pos.compute_value())

    def compute_value(self):
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
        fen = position_to_fen(self)

        # Create single-position batch
        b = nnue_dataset.make_sparse_batch_from_fens(
            feature_set, [fen], [0], [1], [0]
        )

        tensors = b.contents.get_tensors('cuda')
        (us, them, white_indices, white_values, black_indices, black_values,
         outcome, score, psqt_indices, layer_stack_indices) = tensors

        with torch.no_grad():
            eval_tensor = model.forward(
                us, them, white_indices, white_values, black_indices, black_values,
                psqt_indices, layer_stack_indices
            ) * 600.0

        eval_score = eval_tensor[0].item()

        # Flip score if black to move
        if them[0].item() > 0.5:
            eval_score = -eval_score

        eval_score = int(eval_score)

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

    def is_quiet(self, move):
        return self.board[move.j] == "." and not move.prom and abs(move.j - self.kp) >= 2

    def hash(self):
        return self.myhash


Entry = namedtuple("Entry", "lower upper")


@lru_cache(maxsize=50000)
def is_in_check_cached(board_hash, board, ep, kp):
    """Cached check detection."""
    # Find our king
    king_sq = None
    for i, c in enumerate(board):
        if c == 'K':
            king_sq = i
            break

    if king_sq is None:
        return True  # No king = effectively in check

    # Check if any enemy piece attacks our king
    # Pawns
    for d in (N + W, N + E):
        sq = king_sq + d
        if 0 <= sq < 120 and board[sq] == 'p':
            return True

    # Knights
    for d in directions['N']:
        sq = king_sq + d
        if 0 <= sq < 120 and board[sq] == 'n':
            return True

    # Bishops/Queens (diagonals)
    for d in directions['B']:
        for dist in range(1, 8):
            sq = king_sq + d * dist
            if sq < 0 or sq >= 120:
                break
            c = board[sq]
            if c.isspace():
                break
            if c in ('b', 'q'):
                return True
            if c != '.':
                break

    # Rooks/Queens (orthogonals)
    for d in directions['R']:
        for dist in range(1, 8):
            sq = king_sq + d * dist
            if sq < 0 or sq >= 120:
                break
            c = board[sq]
            if c.isspace():
                break
            if c in ('r', 'q'):
                return True
            if c != '.':
                break

    # King
    for d in directions['K']:
        sq = king_sq + d
        if 0 <= sq < 120 and board[sq] == 'k':
            return True

    return False


def is_in_check(pos):
    """Check if current side to move is in check."""
    return is_in_check_cached(pos.hash(), pos.board, pos.ep, pos.kp)


###############################################################################
# Search logic
###############################################################################

PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 20000,
    '.': 0
}


def futility_margin(depth, improving):
    """
    Calculate futility pruning margins scaled for NNUE evaluation
    NNUE range: ~±8000 (scaled by 600 from ±100000)
    Margins should be ~5-15% of eval range
    """
    base_margins = {
        1: 100,
        2: 250,
        3: 600,
        4: 850,
    }

    margin = base_margins.get(depth, 850 + depth * 100)

    # Adjust for improving: more conservative when declining
    if not improving:
        margin = int(margin * 1.15)

    return margin


import time


class Timer:
    def __init__(self):
        self.timings = {}

    def time(self, name):
        return TimerContext(self, name)

    def reset(self):
        """Clear all recorded timings."""
        self.timings.clear()

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
        self.eval_stack = []

        # History heuristic tables
        # Format: history[from_square][to_square] = score
        self.history_table = [[0 for _ in range(120)] for _ in range(120)]
        self.history_max = 1

    def update_history(self, move, depth, cutoff=True):
        """Update history heuristic for a move"""
        if move is None:
            return

        # Bonus based on depth squared (deeper = more important)
        bonus = depth * depth

        if cutoff:
            # Move caused a beta cutoff - reward it
            self.history_table[move.i][move.j] += bonus
        else:
            # Move failed to cause cutoff - penalize slightly
            self.history_table[move.i][move.j] -= bonus // 4

        # Keep score non-negative and update max
        self.history_table[move.i][move.j] = max(0, self.history_table[move.i][move.j])
        self.history_max = max(self.history_max, self.history_table[move.i][move.j])

    def get_history_score(self, move):
        """Get history score for move ordering"""
        if move is None:
            return 0
        return self.history_table[move.i][move.j]

    def age_history(self):
        """Age history table by dividing all values"""
        for i in range(120):
            for j in range(120):
                self.history_table[i][j] //= 2
        self.history_max = max(self.history_max // 2, 1)

    def quiesce(self, pos, alpha, beta, ply=0):
        self.nodes += 1

        # Stand pat
        stand_pat = pos.score
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Depth limit for quiescence
        if ply >= 12:
            return alpha

        # Generate captures only using dedicated generator
        captures = list(pos.gen_captures())

        # Score captures using SEE
        def capture_score(m):
            see_val = see(pos, m)
            # MVV-LVA as tiebreaker
            if abs(m.j - pos.kp) >= 2:
                mvv_lva = PIECE_VALUES[pos.board[m.j]] * 10 - PIECE_VALUES[pos.board[m.i]]
            else:
                mvv_lva = MATE
            return -(see_val * 1000 + mvv_lva)

        captures.sort(key=capture_score)

        for move in captures:
            # SEE pruning: skip losing captures
            see_val = see(pos, move)
            if see_val < 0:
                continue  # Skip losing captures entirely

            # Delta pruning with SEE value
            if stand_pat + see_val + 50 < alpha:
                continue

            score = -self.quiesce(pos.move(move), -beta, -alpha, ply + 1)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def bound(self, pos, gamma, depth, root=True, ply=0):
        """
        MTD-bi bound search with forward pruning
        Returns r where:
           s(pos) <= r < gamma    if gamma > s(pos)
           gamma <= r <= s(pos)   if gamma <= s(pos)
        """
        self.nodes += 1

        # Track eval at this ply
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

        # Quiescence search at horizon
        if depth <= 0:
            return self.quiesce(pos, gamma - 1, gamma, ply)

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

        # ==== PRUNING SETUP ====
        # Compute these once upfront
        in_check = is_in_check(pos) if not root else False
        eval_score = pos.score

        # Calculate "improving" once
        improving = False
        if ply >= 2 and len(self.eval_stack) > ply - 2 and self.eval_stack[ply - 2] is not None:
            improving = eval_score > self.eval_stack[ply - 2]

        # Avoid pruning near mate scores
        near_mate = abs(eval_score) > MATE_LOWER - 1000 or abs(gamma) > MATE_LOWER - 1000

        # ==== NULL MOVE PRUNING ====
        if (not root and
                depth >= 3 and
                not in_check and
                not near_mate and
                any(c in pos.board for c in "NBRQ")):

            # Null move with R=3
            null_score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3, False, ply + 1)
            if null_score >= gamma:
                return null_score

        # ==== RAZORING ====
        # Try to prune hopeless positions early
        if (not root and
                depth <= 3 and
                not in_check and
                not near_mate):

            razor_margin = 300 + 200 * depth

            if eval_score + razor_margin < gamma:
                # Position looks hopeless, verify with qsearch
                qscore = self.quiesce(pos, gamma - 1, gamma, ply)
                if qscore < gamma:
                    return qscore

        # ==== REVERSE FUTILITY PRUNING (Static Null Move) ====
        # Prune if position is too good (opponent can't save it)
        if (not root and
                depth <= 6 and
                not in_check and
                not near_mate):

            # Check we have enough material (not bare king endgame)
            non_pawn_material = sum(1 for c in pos.board if c in 'NBRQ')
            if non_pawn_material >= 2:
                # RFP margins (scaled for NNUE)
                rfp_margin = futility_margin(depth, improving)

                if eval_score - rfp_margin >= gamma:
                    return eval_score

        # ==== FORWARD FUTILITY PRUNING SETUP ====
        # Prepare to skip quiet moves in hopeless positions
        futility_pruning = False
        if (not root and
                depth <= 8 and
                not in_check and
                not near_mate):

            non_pawn_material = sum(1 for c in pos.board if c in 'NBRQ')
            if non_pawn_material >= 2:
                margin = futility_margin(depth, improving)
                if eval_score + margin < gamma:
                    futility_pruning = True

        # ==== MOVE GENERATION AND ORDERING ====
        # Get killer move from TT
        killer = self.tp_move.get(pos.hash())

        # Generate all moves once
        all_moves = list(pos.gen_moves())

        def move_score(m):
            is_cap = pos.is_capture(m)
            if is_cap:
                # Use SEE for captures
                see_val = see(pos, m)
                # Good captures first (high SEE), then equal, then losing
                if see_val >= 0:
                    return -(10000000 + see_val)  # Good captures
                else:
                    return -see_val  # Losing captures (negative SEE -> positive score -> later)
            else:
                # Quiet moves: use history
                history = self.get_history_score(m)
                history_normalized = (history * 2000) // max(self.history_max, 1)
                return -history_normalized

        # Sort all moves by score
        all_moves.sort(key=move_score)

        # Prioritize killer move if it exists
        if killer and killer in all_moves:
            all_moves.remove(killer)
            all_moves.insert(0, killer)

        # ==== MAIN SEARCH LOOP ====
        best = -MATE_UPPER
        moves_searched = 0
        quiet_moves_searched = 0
        searched_moves = []  # Track all searched moves for history update

        for move in all_moves:
            is_capture = pos.is_capture(move)
            is_quiet = pos.is_quiet(move)

            # Late Move Pruning (LMP)
            if (not root and depth <= 8 and not in_check and is_quiet and
                    quiet_moves_searched >= LMP_TABLE.get((depth, improving), 64)):
                continue

                # Futility pruning for quiet moves
            if futility_pruning and is_quiet and not move.prom:
                quiet_moves_searched += 1
                continue

                # SEE pruning for captures at shallow depth
            if not root and depth <= 5 and is_capture and not move.prom:
                if see(pos, move) < -50 * depth * depth:
                    continue

                    # SEE pruning for quiet moves (is destination safe?)
            if not root and depth <= 4 and is_quiet and not in_check:
                # Check if moving piece to destination loses material
                test_move = Move(move.i, move.j, "")
                if see(pos, test_move) < -100 * depth:
                    quiet_moves_searched += 1
                    continue

            moves_searched += 1
            if is_quiet:
                quiet_moves_searched += 1
            searched_moves.append(move)

            # ==== LATE MOVE REDUCTION (LMR) ====
            reduction = 0

            # LMR conditions
            if (not root and
                    depth >= 3 and
                    moves_searched >= 2 and not in_check and is_quiet):

                # Base reduction using logarithmic formula
                base_reduction = 0.75 + math.log(depth) * math.log(moves_searched) * 0.5
                reduction = int(base_reduction)
                reduction = max(1, min(reduction, depth - 2))

                # === REDUCTION ADJUSTMENTS ===

                # 1. History: reduce LESS for good moves, MORE for bad moves
                history_score = self.get_history_score(move)
                history_threshold = self.history_max // 3
                if history_score > history_threshold:
                    reduction = max(1, reduction - 1)
                elif history_score < self.history_max // 10:
                    reduction += 1

                # 2. Improving: reduce LESS when improving (search harder)
                #    reduce MORE when declining (opponent may have threats)
                if improving:
                    reduction = max(1, reduction - 1)  # FIXED: was += 1
                else:
                    reduction += 1

                # 3. Very late moves: more aggressive reduction
                if moves_searched >= 20:
                    reduction += 2
                elif moves_searched >= 12:
                    reduction += 1

                # 4. Deep searches: allow more reduction
                if depth >= 8:
                    reduction += 1

                # Final bounds
                reduction = max(1, min(reduction, depth - 1))

            # Search the move
            if reduction > 0:
                # Reduced depth search
                reduced_depth = max(0, depth - 1 - reduction)
                score = -self.bound(pos.move(move), 1 - gamma, reduced_depth, False, ply + 1)

                # Re-search if it looks good
                if score >= gamma:
                    score = -self.bound(pos.move(move), 1 - gamma, depth - 1, False, ply + 1)
            else:
                # Full depth search
                score = -self.bound(pos.move(move), 1 - gamma, depth - 1, False, ply + 1)

            best = max(best, score)
            if best >= gamma:
                # Beta cutoff - update history
                if move is not None:
                    # Save killer move
                    self.tp_move[pos.hash()] = move

                    # Update history heuristic
                    self.update_history(move, depth, cutoff=True)
                    # Penalize moves that were searched before the cutoff
                    for prev_move in searched_moves[:-1]:
                        if prev_move is not None:
                            self.update_history(prev_move, depth, cutoff=False)
                break

        # No beta cutoff: penalize searched quiet moves
        if best < gamma:
            for move in searched_moves:
                if move is not None and not pos.is_capture(move):
                    self.update_history(move, depth, cutoff=False)

        # Stalemate checking
        if depth > 0 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            in_check_mate = self.bound(flipped, MATE_UPPER, 0, root=False, ply=ply + 1) == MATE_UPPER
            best = -MATE_LOWER if in_check_mate else 0

        # Store in transposition table
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

        # Age history table periodically
        self.age_history()

        # We save the gamma function between depths, so we can start from the most
        # interesting position at the next level
        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
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
                #logging.info("yielding move self.tp_move.get(pos.hash())")
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

    book_path = sys.argv[4] if len(sys.argv) > 4 else None
    if book_path:
        logging.info(f"Using book at {book_path}")

    tools.uci.run(sys.modules[__name__], hist[-1], book_path=book_path)

sys.exit()
