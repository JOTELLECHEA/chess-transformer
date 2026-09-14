"""
Tests for the closed-form vocabulary and tokenizer.

The vocabulary is computed from the rules of chess rather than harvested
from a corpus, so it should be exactly reproducible -- these tests exist
to catch silent drift in move-space enumeration, which would invalidate
every checkpoint at once.
"""
import json
import os

import chess
import pytest

from src.move_space import build_theoretical_uci_space
from src.dataset import MoveTokenizer
from src.constants import START_TOKEN, RESULT_TOKENS

VOCAB_PATH = "vocab_fixed.json"
EXPECTED_VOCAB_SIZE = 1973
N_SPECIAL_TOKENS = 5  # <|SOM|> plus four result tokens


def test_committed_vocab_has_expected_size():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    assert len(vocab) == EXPECTED_VOCAB_SIZE


def test_committed_vocab_has_no_duplicates():
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    assert len(vocab) == len(set(vocab))


def test_committed_vocab_contains_all_special_tokens():
    with open(VOCAB_PATH) as f:
        vocab = set(json.load(f))
    assert START_TOKEN in vocab
    for token in RESULT_TOKENS:
        assert token in vocab


def test_theoretical_space_matches_committed_vocab():
    """The committed file should be exactly the theoretical space plus
    special tokens -- nothing harvested, nothing missing."""
    all_uci, _, _ = build_theoretical_uci_space()
    with open(VOCAB_PATH) as f:
        vocab = set(json.load(f))
    assert all_uci == vocab - set(RESULT_TOKENS) - {START_TOKEN}


def test_theoretical_space_size_accounts_for_specials():
    all_uci, _, _ = build_theoretical_uci_space()
    assert len(all_uci) + N_SPECIAL_TOKENS == EXPECTED_VOCAB_SIZE


@pytest.mark.parametrize("uci", [
    "e2e4",    # ordinary pawn push
    "g1f3",    # knight
    "e1g1",    # white kingside castling, as a king move
    "e8c8",    # black queenside castling, as a king move
    "a7a8q",   # queen promotion
    "a7a8n",   # knight underpromotion
    "b7c8r",   # capturing rook underpromotion
    "h1a8",    # long diagonal
])
def test_known_legal_moves_are_in_vocab(uci):
    all_uci, _, _ = build_theoretical_uci_space()
    assert uci in all_uci


def test_underpromotions_are_a_subset_of_promotions():
    _, promo_uci, underpromo_uci = build_theoretical_uci_space()
    assert underpromo_uci <= promo_uci


def test_underpromotions_never_end_in_q():
    _, _, underpromo_uci = build_theoretical_uci_space()
    assert all(not u.endswith("q") for u in underpromo_uci)


def test_all_promotions_are_five_characters():
    _, promo_uci, _ = build_theoretical_uci_space()
    assert all(len(u) == 5 for u in promo_uci)


def test_tokenizer_round_trips_every_token():
    tokenizer = MoveTokenizer.from_vocab_file(VOCAB_PATH)
    for token, idx in tokenizer.stoi.items():
        assert tokenizer.itos[idx] == token


def test_tokenizer_vocab_size_matches_file():
    tokenizer = MoveTokenizer.from_vocab_file(VOCAB_PATH)
    assert tokenizer.vocab_size == EXPECTED_VOCAB_SIZE


def test_every_legal_opening_move_is_encodable():
    """Every move legal from the starting position must have a token --
    a gap here would mean the model literally cannot express an opening."""
    tokenizer = MoveTokenizer.from_vocab_file(VOCAB_PATH)
    board = chess.Board()
    for move in board.legal_moves:
        assert move.uci() in tokenizer.stoi
