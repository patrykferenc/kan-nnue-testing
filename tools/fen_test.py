#!/usr/bin/env python3
"""
Unit tests for position_to_fen conversion.
Run with: python test_position_to_fen.py
"""

import sys
from collections import namedtuple

# Mock the Position class
Position = namedtuple("Position", "board score wc bc ep kp")

# Constants from sunfish
A1, H1, A8, H8 = 91, 98, 21, 28
N, E, S, W = -10, 1, 10, -1

initial = (
    "         \n"  # 0 -  9
    "         \n"  # 10 - 19
    " rnbqkbnr\n"  # 20 - 29  (rank 8)
    " pppppppp\n"  # 30 - 39  (rank 7)
    " ........\n"  # 40 - 49  (rank 6)
    " ........\n"  # 50 - 59  (rank 5)
    " ........\n"  # 60 - 69  (rank 4)
    " ........\n"  # 70 - 79  (rank 3)
    " PPPPPPPP\n"  # 80 - 89  (rank 2)
    " RNBQKBNR\n"  # 90 - 99  (rank 1)
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)


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


def rotate_position(pos):
    """Simulate the rotate() function from sunfish"""
    return Position(
        pos.board[::-1].swapcase(),
        -pos.score,
        pos.bc,
        pos.wc,
        119 - pos.ep if pos.ep else 0,
        119 - pos.kp if pos.kp else 0,
    )


def test_initial_position():
    """Test starting position - White to move"""
    pos = Position(initial, 0, (True, True), (True, True), 0, 0)
    fen = position_to_fen(pos)
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"Test: Initial position")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fen}")
    assert fen == expected, f"FAILED: Initial position"
    print("  ✓ PASSED\n")


def test_initial_position_after_rotation():
    """Test starting position after rotation - Black to move"""
    pos = Position(initial, 0, (True, True), (True, True), 0, 0)
    pos_rotated = rotate_position(pos)
    fen = position_to_fen(pos_rotated)
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    print(f"Test: Initial position after rotation")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fen}")
    print(f"  Board starts with newline: {pos_rotated.board.startswith(chr(10))}")
    assert fen == expected, f"FAILED: Rotated initial position"
    print("  ✓ PASSED\n")


def test_simple_endgame():
    """Test simple endgame: King and pawn vs King"""
    # Position: 8/8/8/8/3k4/8/3P4/3K4 w - - 0 1
    board = (
        "         \n"  # 0-9
        "         \n"  # 10-19
        " ........\n"  # 20-29 rank 8
        " ........\n"  # 30-39 rank 7
        " ........\n"  # 40-49 rank 6
        " ........\n"  # 50-59 rank 5
        " ...k....\n"  # 60-69 rank 4
        " ........\n"  # 70-79 rank 3
        " ...P....\n"  # 80-89 rank 2
        " ...K....\n"  # 90-99 rank 1
        "         \n"  # 100-109
        "         \n"  # 110-119
    )
    pos = Position(board, 0, (False, False), (False, False), 0, 0)
    fen = position_to_fen(pos)
    expected = "8/8/8/8/3k4/8/3P4/3K4 w - - 0 1"
    print(f"Test: Simple endgame (White to move)")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fen}")
    assert fen == expected, f"FAILED: Simple endgame"
    print("  ✓ PASSED\n")


def test_simple_endgame_rotated():
    """Test simple endgame after rotation - Black to move"""
    # Original: 8/8/8/8/3k4/8/3P4/3K4 w - - 0 1
    board = (
        "         \n"
        "         \n"
        " ........\n"  # rank 8
        " ........\n"  # rank 7
        " ........\n"  # rank 6
        " ........\n"  # rank 5
        " ...k....\n"  # rank 4
        " ........\n"  # rank 3
        " ...P....\n"  # rank 2
        " ...K....\n"  # rank 1
        "         \n"
        "         \n"
    )
    pos = Position(board, 0, (False, False), (False, False), 0, 0)
    pos_rotated = rotate_position(pos)
    fen = position_to_fen(pos_rotated)
    expected = "8/8/8/8/3k4/8/3P4/3K4 b - - 0 1"
    print(f"Test: Simple endgame (Black to move)")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fen}")

    # Debug: print the rotated board
    print(f"  Rotated board (positions 21-98):")
    for rank in range(8):
        row_start = 21 + rank * 10
        row = pos_rotated.board[row_start:row_start + 8]
        print(f"    rank {rank}: '{row}'")

    assert fen == expected, f"FAILED: Rotated simple endgame"
    print("  ✓ PASSED\n")


def test_problem_position():
    """Test the problematic position from the error"""
    # The position that caused the error: 8/6p1/5pk1/7R/B7/8/8/7K w - - 0 1
    board = (
        "         \n"
        "         \n"
        " ........\n"  # rank 8
        " ......p.\n"  # rank 7
        " .....pk.\n"  # rank 6
        " .......R\n"  # rank 5
        " B.......\n"  # rank 4
        " ........\n"  # rank 3
        " ........\n"  # rank 2
        " .......K\n"  # rank 1
        "         \n"
        "         \n"
    )
    pos = Position(board, 0, (False, False), (False, False), 0, 0)
    fen = position_to_fen(pos)
    expected = "8/6p1/5pk1/7R/B7/8/8/7K w - - 0 1"
    print(f"Test: Problem position (White to move)")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fen}")
    assert fen == expected, f"FAILED: Problem position"
    print("  ✓ PASSED\n")


def test_problem_position_rotated():
    """Test the problematic position after rotation"""
    board = (
        "         \n"
        "         \n"
        " ........\n"  # rank 8
        " ......p.\n"  # rank 7
        " .....pk.\n"  # rank 6
        " .......R\n"  # rank 5
        " B.......\n"  # rank 4
        " ........\n"  # rank 3
        " ........\n"  # rank 2
        " .......K\n"  # rank 1
        "         \n"
        "         \n"
    )
    pos = Position(board, 0, (False, False), (False, False), 0, 0)
    pos_rotated = rotate_position(pos)
    fen = position_to_fen(pos_rotated)
    expected = "8/6p1/5pk1/7R/B7/8/8/7K b - - 0 1"
    print(f"Test: Problem position (Black to move)")
    print(f"  Expected: {expected}")
    print(f"  Got:      {fen}")

    # Debug output
    print(f"  Rotated board (positions 21-98):")
    for rank in range(8):
        row_start = 21 + rank * 10
        row = pos_rotated.board[row_start:row_start + 8]
        print(f"    rank {rank}: '{row}' (swapcase: '{row.swapcase()}')")

    assert fen == expected, f"FAILED: Rotated problem position"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing position_to_fen conversion")
    print("=" * 60 + "\n")

    tests = [
        test_initial_position,
        test_initial_position_after_rotation,
        test_simple_endgame,
        test_simple_endgame_rotated,
        test_problem_position,
        test_problem_position_rotated,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
