"""
Tests for engine configuration, probe labelling, and result normalization.
"""
import chess
import pytest

from src.self_play_engine import PlayerConfig
from src.probe_board_state import board_to_labels, PIECE_TO_CLASS, N_CLASSES


# --- PlayerConfig validation -------------------------------------------

def test_valid_player_types_are_accepted():
    PlayerConfig(name="m", player_type="model")
    PlayerConfig(name="s", player_type="stockfish", stockfish_path="/usr/games/stockfish")
    PlayerConfig(name="h", player_type="human")


def test_unknown_player_type_is_rejected():
    with pytest.raises(ValueError):
        PlayerConfig(name="x", player_type="octopus")


def test_stockfish_without_a_path_is_rejected():
    with pytest.raises(ValueError):
        PlayerConfig(name="s", player_type="stockfish")


def test_model_without_a_checkpoint_is_allowed():
    """checkpoint_dir=None means fresh random init, which is the probe baseline."""
    config = PlayerConfig(name="random", player_type="model")
    assert config.checkpoint_dir is None


# --- Probe labelling ---------------------------------------------------

def test_class_count_is_thirteen():
    """Empty, plus six piece types times two colors."""
    assert N_CLASSES == 13


def test_empty_square_is_class_zero():
    assert PIECE_TO_CLASS[None] == 0


def test_labels_cover_all_sixty_four_squares():
    labels = board_to_labels(chess.Board())
    assert labels.shape == (64,)


def test_starting_position_has_thirty_two_occupied_squares():
    labels = board_to_labels(chess.Board())
    assert int((labels != 0).sum()) == 32


def test_empty_board_labels_are_all_zero():
    board = chess.Board(None)  # completely empty
    labels = board_to_labels(board)
    assert int(labels.sum()) == 0


def test_white_and_black_pawns_get_different_classes():
    labels = board_to_labels(chess.Board())
    white_pawn = labels[chess.E2].item()
    black_pawn = labels[chess.E7].item()
    assert white_pawn != black_pawn


def test_labels_stay_within_class_range():
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
        board.push(chess.Move.from_uci(uci))
    labels = board_to_labels(board)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < N_CLASSES