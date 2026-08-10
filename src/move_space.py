"""
move_space.py

The theoretical UCI move space: every string python-chess's Move.uci() can
ever produce, for any piece/square/promotion combination on an otherwise
empty board (1,968 total for standard chess).

Shared by:
    - dataset.py's MoveTokenizer.build_closed_form(), which uses the full
      set to build the fixed, corpus-independent vocab.
    - analyze_vocab_gap.py, which uses the promo/underpromo breakdown to
      check how much of the theoretical space a given real corpus actually
      covers.

Previously these lived as two independent, copy-pasted implementations --
consolidated here so they can't silently drift apart from each other.
"""

import chess


def build_theoretical_uci_space():
    """
    Returns (all_uci, promo_uci, underpromo_uci) sets: every UCI string
    producible by python-chess's Move.uci() for any piece/square/promotion
    combination on an otherwise empty board.
    """
    all_uci = set()
    promo_uci = set()
    underpromo_uci = set()

    def add(frm, to, promo=None):
        move = chess.Move(frm, to, promotion=promo)
        uci = move.uci()
        all_uci.add(uci)
        if promo is not None:
            promo_uci.add(uci)
            if promo != chess.QUEEN:
                underpromo_uci.add(uci)

    for from_sq in chess.SQUARES:
        f_file, f_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
        for df, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for dist in range(1, 8):
                tf, tr = f_file + df * dist, f_rank + dr * dist
                if 0 <= tf < 8 and 0 <= tr < 8:
                    add(from_sq, chess.square(tf, tr))
        for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
            tf, tr = f_file + df, f_rank + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                add(from_sq, chess.square(tf, tr))

    promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    for f_file in range(8):
        from_sq = chess.square(f_file, 6)  # white: rank 7 -> 8
        for df in (-1, 0, 1):
            tf = f_file + df
            if 0 <= tf < 8:
                for p in promo_pieces:
                    add(from_sq, chess.square(tf, 7), p)
        from_sq = chess.square(f_file, 1)  # black: rank 2 -> 1
        for df in (-1, 0, 1):
            tf = f_file + df
            if 0 <= tf < 8:
                for p in promo_pieces:
                    add(from_sq, chess.square(tf, 0), p)

    return all_uci, promo_uci, underpromo_uci
