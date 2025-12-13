#!/usr/bin/env python3
"""
Generate SVG chess diagrams with move annotations for Typst integration.
Requires: python-chess (pip install chess)
"""

import chess
import chess.svg

# Define positions with their metadata
positions = [
    # {
    #     "fen": "2b5/2N2p2/6p1/4N3/1B1k1p1r/Kp6/n7/4Q3 w - - 0 1",
    #     "name": "Mate in 4",
    #     "best_move": "e1d2",  # Qd2+
    #     "filename": "position_mate_in_4.svg",
    #     "description": "Qd2+ leads to mate in 4"
    # },
    # {
    #     "fen": "5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - 0 1",
    #     "name": "Win at Chess 5 (WAC.005)",
    #     "best_move": "c6c4",  # Qc4+
    #     "filename": "position_wac_005.svg",
    #     "description": "Qc4+ wins material"
    # },
    # {
    #     "fen": "8/1k6/8/Q7/7p/6p1/6pr/6Kb w - - 0 1",
    #     "name": "EET 70b (D vs T&L&B)",
    #     "best_move": "a5c5",  # Qc5+
    #     "filename": "position_eet_70b.svg",
    #     "description": "Qc5+ winning endgame technique"
    # }
{
        # Karpov vs Kasparov (1984), Partia 9 (ok. 54 posunięcia)
        # Biały skoczek dominuje nad "złym" gońcem czarnych.
        "fen": "8/p4p2/1p2k1p1/3pP1P1/3P4/P2N2K1/bP6/8 w - - 9 54",
        "name": "Karpow vs Kasparow - Dominacja Skoczka",
        "best_move": "d3f4",  # Nf4+ - Skoczek zajmuje kluczowe pole
        "filename": "karpov_kasparov_bad_bishop.svg",
        "description": "Skoczek (3) wart wiecej niz Goniec (3)"
    },
    {
        # Pozycja Saavedry (1895) - Start studium
        # Biały pion wygrywa z czarną wieżą.
        "fen": "8/8/1KP5/3r4/8/8/8/k7 w - - 0 1",
        "name": "Pozycja Saavedry - Siła Piona",
        "best_move": "c6c7",  # c7 - Marsz piona, którego wieża nie zatrzyma
        "filename": "saavedra_pawn_power.svg",
        "description": "Pion (1) silniejszy od Wiezy (5)"
    }
]


def generate_svg_diagram(fen, best_move_uci, filename, board_size=400):
    """
    Generate an SVG chess diagram with the best move highlighted.

    Args:
        fen: FEN string of the position
        best_move_uci: UCI notation of the best move (e.g., 'e1d2')
        filename: Output SVG filename
        board_size: Size of the board in pixels (default 400)
    """
    # Create board from FEN
    board = chess.Board(fen)

    # Parse the best move
    move = chess.Move.from_uci(best_move_uci)

    # Generate SVG with arrow showing the best move
    svg = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(move.from_square, move.to_square)],
        size=board_size,
        coordinates=True,
        flipped=False
    )

    # Write to file
    with open(filename, 'w') as f:
        f.write(svg)

    print(f"Generated: {filename}")
    return svg


# Generate all diagrams
print("Generating SVG chess diagrams...")

for pos in positions:
    print(f"Position: {pos['name']}")
    print(f"  FEN: {pos['fen']}")
    print(f"  Best move: {pos['best_move']} - {pos['description']}")

    generate_svg_diagram(
        fen=pos['fen'],
        best_move_uci=pos['best_move'],
        filename=pos['filename']
    )
    print()

print("All diagrams generated successfully!")
print("\nYou can include these in Typst using:")
print('#image("position_mate_in_4.svg")')
print('#image("position_wac_005.svg")')
print('#image("position_eet_70b.svg")')
