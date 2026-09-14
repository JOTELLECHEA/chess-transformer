"""
Tests for classify_notable() in sweep_games.py.

This function decides which moves count as evidence of learned behaviour,
so a bug here silently corrupts the sweep results. Two of these tests
guard against real bugs that were found and fixed:

  - a rook sitting on e1/e8 can produce the exact UCI string as castling
    (e8c8), so a naive string match reports false positives
  - Stockfish's own moves were being counted as the model's, inflating
    the notable-event count by roughly 40%
"""
import chess
import pytest

from src.sweep_games import classify_notable, MODEL_PLAYER_NAME


class FakeEvent:
    """Minimal stand-in for MoveEvent -- only the fields classify_notable reads."""
    def __init__(self, uci, san, board_fen_before,
                 player_name=MODEL_PLAYER_NAME, is_legal=True):
        self.uci = uci
        self.san = san
        self.board_fen_before = board_fen_before
        self.player_name = player_name
        self.is_legal = is_legal


def _board_after(moves):
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


def test_en_passant_is_detected():
    board = _board_after(["e2e4", "a7a6", "e4e5", "d7d5"])
    event = FakeEvent("e5d6", "exd6", board.fen())
    assert "en passant" in classify_notable(event)


def test_underpromotion_to_knight_is_detected():
    event = FakeEvent("a7a8n", "a8=N", chess.Board().fen())
    assert "underpromotion" in classify_notable(event)


@pytest.mark.parametrize("uci,san", [
    ("a7a8n", "a8=N"),
    ("a7a8b", "a8=B"),
    ("a7a8r", "a8=R"),
])
def test_all_underpromotion_types_are_detected(uci, san):
    event = FakeEvent(uci, san, chess.Board().fen())
    assert "underpromotion" in classify_notable(event)


def test_queen_promotion_is_not_flagged():
    """Queen is the overwhelmingly common choice -- not notable."""
    event = FakeEvent("a7a8q", "a8=Q", chess.Board().fen())
    assert classify_notable(event) == ""


def test_castling_is_detected():
    board = _board_after(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"])
    event = FakeEvent("e1g1", "O-O", board.fen())
    assert "castled" in classify_notable(event)


def test_rook_move_matching_a_castling_uci_string_is_not_castling():
    """Regression: a rook going e8->c8 produces uci 'e8c8', identical to
    black queenside castling. A string match would report a false positive."""
    board = chess.Board("4r1k1/8/8/8/8/8/8/4K3 b - - 0 1")
    event = FakeEvent("e8c8", "Rc8", board.fen())
    assert classify_notable(event) == ""


def test_stockfish_moves_are_never_flagged():
    """Regression: Stockfish already knows every rule, so its moves say
    nothing about what the model learned."""
    board = _board_after(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"])
    event = FakeEvent("e1g1", "O-O", board.fen(), player_name="stockfish")
    assert classify_notable(event) == ""


def test_illegal_moves_are_not_flagged():
    event = FakeEvent("e1g1", None, chess.Board().fen(), is_legal=False)
    assert classify_notable(event) == ""


def test_result_tokens_are_not_flagged():
    """A result token is legal but has no SAN -- it must not reach
    chess.Move.from_uci(), which would raise on '<|0-1|>'."""
    event = FakeEvent("<|0-1|>", None, chess.Board().fen())
    assert classify_notable(event) == ""


def test_ordinary_move_is_not_flagged():
    event = FakeEvent("e2e4", "e4", chess.Board().fen())
    assert classify_notable(event) == ""
