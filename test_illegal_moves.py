#!/usr/bin/env python3
"""
Standalone test script to debug illegal move generation
Tests the WAC.230 position where Kc6-d6 is generated but is illegal

Key insight: When Black is to move, Sunfish rotates the board so Black's pieces
are uppercase and at the "bottom". This means all coordinates flip!
"""

import re
from collections import namedtuple

# Copy necessary constants and classes from sunfish_nnue
A1, H1, A8, H8 = 91, 98, 21, 28
N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N + N, N + W, N + E),
    "N": (N + N + E, E + N + E, E + S + E, S + S + E, S + S + W, W + S + W, W + N + W, N + N + W),
    "B": (N + E, S + E, S + W, N + W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N + E, S + E, S + W, N + W),
    "K": (N, E, S, W, N + E, S + E, S + W, N + W),
}

Move = namedtuple("Move", "i j prom")


def parse(c):
    """Convert algebraic notation to sunfish board index"""
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank


def render(i):
    """Convert sunfish board index to algebraic notation"""
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)


class Position(namedtuple("Position", "board score wc bc ep kp myhash")):
    def __new__(cls, board, score, wc, bc, ep, kp, myhash=None):
        if myhash is None:
            myhash = hash((board, wc, bc, ep, kp))
        return super().__new__(cls, board, score, wc, bc, ep, kp, myhash)

    def gen_moves(self):
        """Generate all pseudo-legal moves"""
        from itertools import count
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
        """Rotate the board"""
        pos = Position(
            self.board[::-1].swapcase(), 0, self.bc, self.wc,
            0 if nullmove or not self.ep else 119 - self.ep,
            0 if nullmove or not self.kp else 119 - self.kp,
        )
        return pos

    def move(self, move):
        """Make a move"""
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

    def hash(self):
        return self.myhash


def is_legal_move(pos, move):
    """
    Check if a move is legal (doesn't leave our king in check).

    After making a move, the position is rotated so opponent's pieces
    are uppercase. Our king becomes lowercase 'k' from opponent's view.
    We check if opponent can capture our king.
    """
    new_pos = pos.move(move)

    # Check if opponent can capture our king
    for opp_move in new_pos.gen_moves():
        # Check both: direct king capture AND king passant (castling)
        if new_pos.board[opp_move.j] == 'k' or abs(opp_move.j - new_pos.kp) < 2:
            return False

    return True


def from_fen(board, color, castling, enpas):
    """Convert FEN to sunfish Position"""
    board = re.sub(r"\d", (lambda m: "." * int(m.group(0))), board)
    board = list(21 * " " + "  ".join(board.split("/")) + 21 * " ")
    board[9::10] = ["\n"] * 12
    board = "".join(board)
    wc = ("Q" in castling, "K" in castling)
    bc = ("k" in castling, "q" in castling)
    ep = parse(enpas) if enpas != "-" else 0

    pos = Position(board, 0, wc, bc, ep, 0)
    return pos if color == 'w' else pos.rotate()


def test_wac_230():
    """Test the WAC.230 position"""
    print("=" * 70)
    print("Testing WAC.230 Position")
    print("=" * 70)
    print("FEN: 2b5/1r6/2kBp1p1/p2pP1P1/2pP4/1pP3K1/1R3P2/8 b - -")
    print()
    print("In standard chess notation (from White's perspective):")
    print("  - Black king on c6")
    print("  - White bishop on d6")
    print("  - White pawn on e5 (covering d6)")
    print("  - Illegal move: Kc6xd6 (pawn on e5 covers d6)")
    print()

    # First, let's look at the unrotated position
    print("=" * 70)
    print("UNROTATED POSITION (White's perspective)")
    print("=" * 70)
    board = re.sub(r"\d", (lambda m: "." * int(m.group(0))), "2b5/1r6/2kBp1p1/p2pP1P1/2pP4/1pP3K1/1R3P2/8")
    board_list = list(21 * " " + "  ".join(board.split("/")) + 21 * " ")
    board_list[9::10] = ["\n"] * 12
    board_str = "".join(board_list)
    print(board_str)

    # Find key pieces in unrotated position
    c6_idx = parse('c6')
    d6_idx = parse('d6')
    e5_idx = parse('e5')

    print(f"c6 (Black king) = index {c6_idx}, piece = '{board_str[c6_idx]}'")
    print(f"d6 (White bishop) = index {d6_idx}, piece = '{board_str[d6_idx]}'")
    print(f"e5 (White pawn) = index {e5_idx}, piece = '{board_str[e5_idx]}'")
    print()

    # Now rotate for Black to move
    print("=" * 70)
    print("ROTATED POSITION (Black's perspective - actual game state)")
    print("=" * 70)
    pos = from_fen("2b5/1r6/2kBp1p1/p2pP1P1/2pP4/1pP3K1/1R3P2/8", "b", "-", "-")
    print(pos.board)

    # After rotation, coordinates flip
    # Original c6 (idx 43) -> 119-43 = 76
    # Original d6 (idx 44) -> 119-44 = 75
    # Original e5 (idx 55) -> 119-55 = 64

    black_king_idx = 119 - c6_idx
    white_bishop_idx = 119 - d6_idx
    white_pawn_idx = 119 - e5_idx

    print(f"\nAfter rotation (Black to move):")
    print(f"  Black king (now 'K') at index {black_king_idx} = {render(black_king_idx)}")
    print(f"  White bishop (now 'b') at index {white_bishop_idx} = {render(white_bishop_idx)}")
    print(f"  White pawn (now 'p') at index {white_pawn_idx} = {render(white_pawn_idx)}")
    print(f"\nActual pieces at those positions:")
    print(f"  pos.board[{black_king_idx}] = '{pos.board[black_king_idx]}'")
    print(f"  pos.board[{white_bishop_idx}] = '{pos.board[white_bishop_idx]}'")
    print(f"  pos.board[{white_pawn_idx}] = '{pos.board[white_pawn_idx]}'")
    print()

    # Generate all pseudo-legal moves
    print("=" * 70)
    print("STEP 1: Generated moves")
    print("=" * 70)
    all_moves = list(pos.gen_moves())
    print(f"Total moves generated: {len(all_moves)}\n")

    # Find king moves
    king_moves = [m for m in all_moves if pos.board[m.i] == 'K']
    print(f"King moves: {len(king_moves)}")
    for move in king_moves:
        target = pos.board[move.j]
        target_piece = f" x{target}" if target != '.' else ""
        print(f"  K{render(move.i)}-{render(move.j)}{target_piece}")
    print()

    # Look for the illegal king capture of the bishop
    illegal_move = None
    for move in king_moves:
        if move.i == black_king_idx and move.j == white_bishop_idx:
            illegal_move = move
            print(f"⚠️  FOUND ILLEGAL MOVE: K{render(move.i)}-{render(move.j)} (captures bishop)")
            print(f"    This corresponds to Kc6xd6 in the original notation")
            break

    if not illegal_move:
        print(f"✓ Illegal move was NOT generated by gen_moves()")
        print(f"    (This might mean gen_moves already filters it, which would be good!)")
        # Let's still test the legality check

    print()
    print("=" * 70)
    print("STEP 2: Testing is_legal_move()")
    print("=" * 70)

    if illegal_move:
        is_legal = is_legal_move(pos, illegal_move)
        print(f"\nis_legal_move(K{render(illegal_move.i)}-{render(illegal_move.j)}) = {is_legal}")

        if is_legal:
            print("❌ FAIL: Move was marked as LEGAL but it's ILLEGAL!")
        else:
            print("✓ PASS: Move was correctly marked as ILLEGAL")

        # Trace through
        print("\nTracing the legality check:")
        print("-" * 70)
        new_pos = pos.move(illegal_move)
        print(f"After Black plays K{render(illegal_move.i)}-{render(illegal_move.j)}:")
        print(f"  Position rotates to White's perspective")
        print(f"  Black's king is now at kp={new_pos.kp}")

        # Check what can capture it
        capturing_moves = []
        for m in new_pos.gen_moves():
            if abs(m.j - new_pos.kp) < 2:
                capturing_moves.append(m)

        print(f"  White can capture the king: {len(capturing_moves) > 0}")
        if capturing_moves:
            for m in capturing_moves:
                piece = new_pos.board[m.i]
                print(f"    {piece}{render(119 - m.i)}-{render(119 - m.j)} captures king")
                print(f"    (This is the pawn on e5 capturing on d6)")
    else:
        print("Cannot test is_legal_move() as the illegal move wasn't generated.")
        print("Let's check ALL king moves for legality:")
        for move in king_moves:
            is_legal = is_legal_move(pos, move)
            target = pos.board[move.j]
            status = "✓" if is_legal else "❌"
            print(f"  {status} K{render(move.i)}-{render(move.j)}: legal={is_legal}")

    print()
    print("=" * 70)
    print("STEP 3: Testing move filtering")
    print("=" * 70)

    legal_moves = [m for m in all_moves if is_legal_move(pos, m)]
    print(f"\nBefore filtering: {len(all_moves)} moves")
    print(f"After filtering: {len(legal_moves)} moves")
    print(f"Filtered out: {len(all_moves) - len(legal_moves)} illegal moves")

    if illegal_move:
        illegal_still_there = any(m.i == illegal_move.i and m.j == illegal_move.j for m in legal_moves)
        if illegal_still_there:
            print(f"\n❌ FAIL: Illegal move K{render(illegal_move.i)}-{render(illegal_move.j)} is still in legal moves!")
        else:
            print(f"\n✓ PASS: Illegal move K{render(illegal_move.i)}-{render(illegal_move.j)} was filtered out")

    print("\nLegal king moves:")
    legal_king_moves = [m for m in legal_moves if pos.board[m.i] == 'K']
    for move in legal_king_moves:
        target = pos.board[move.j]
        target_str = f" x{target}" if target != '.' else ""
        print(f"  K{render(move.i)}-{render(move.j)}{target_str}")


if __name__ == "__main__":
    test_wac_230()