"""
sweep_games.py

Plays multiple games across a grid of (temperature, Stockfish skill)
combinations, summarizing each one -- opening moves, whether either side
castled, final result, game length -- so you can quickly scan for
interesting games worth turning into an actual GIF via make_game_gif.py,
rather than generating full animations for every combination up front.

Usage:
    python sweep_games.py
"""
import json
import os
import random
import chess
import torch
from src.config import GPTConfig
from src.self_play_engine import PlayerConfig, play_game

CHECKPOINT_DIR = "checkpoints/1.2m_L12E384H6"
DEVICE = "cuda"
STOCKFISH_PATH = "/usr/games/stockfish"
FALLBACK_VOCAB_PATH = "vocab_fixed.json"

TEMPERATURES = [0.5, 0.7, 1.0, 1.2]
STOCKFISH_SKILLS = [0, 5, 10, 15, 20]

MAX_PLIES = 100
OPENING_PLIES_SHOWN = 10

UNDERPROMOTION_SUFFIXES = {"n", "b", "r"}  # queen ('q') is the common case, not flagged as notable

FALLBACK_CONFIG = None  # loaded lazily in main() -- see note there


MODEL_PLAYER_NAME = "model"  # must match the `name=` used in play_one()'s PlayerConfig for the model side


def classify_notable(event) -> str:
    """Labels en passant, underpromotion, or castling -- only for the
    MODEL's own moves, since Stockfish already knows every rule perfectly
    and its moves prove nothing about the model. Castling is included but
    is common enough in real games to not be the interesting part; en
    passant and underpromotion are the signals that actually matter."""
    if event.player_name != MODEL_PLAYER_NAME:
        return ""
    if not event.is_legal or event.san is None:
        return ""
    if len(event.uci) == 5 and event.uci[-1] in UNDERPROMOTION_SUFFIXES:
        return f"underpromotion ({event.san})"
    # Use board.is_castling(), not a naive UCI-string match against
    # {"e1g1", ...} -- a rook that happens to be sitting on e1/e8 could
    # produce the exact same from-square/to-square UCI string via an
    # entirely ordinary move, with no castling involved at all. Verified
    # directly: a rook e8->c8 produces uci "e8c8", identical to Black's
    # queenside castling string, despite being unrelated.
    board_before = chess.Board(event.board_fen_before)
    move = chess.Move.from_uci(event.uci)
    if board_before.is_castling(move):
        return f"castled ({event.san})"
    if board_before.is_en_passant(move):
        return f"en passant ({event.san})"
    return ""


def play_one(temperature: float, skill: int):
    seed = random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)

    white = PlayerConfig(name="model", player_type="model", checkpoint_dir=CHECKPOINT_DIR)
    black = PlayerConfig(name="stockfish", player_type="stockfish",
                          stockfish_path=STOCKFISH_PATH, stockfish_skill_level=skill, stockfish_time_limit=0.1)

    gen = play_game(white, black, DEVICE, fallback_config=FALLBACK_CONFIG,
                     fallback_vocab_path=FALLBACK_VOCAB_PATH,
                     max_plies=MAX_PLIES, temperature=temperature)

    opening_moves = []
    notable_events = []
    all_moves = []
    ply_count = 0
    result = None

    try:
        for event in gen:
            if event.player_name == "" and not event.uci:
                continue
            ply_count = event.ply
            if event.is_legal and event.san is not None:
                all_moves.append(event.uci)
            label = classify_notable(event)
            if label:
                notable_events.append(f"ply {event.ply}: {label}")
            if len(opening_moves) < OPENING_PLIES_SHOWN and event.san is not None:
                opening_moves.append(event.san)
            if event.game_over:
                result = event.result
                break
    finally:
        gen.close()

    saved_game_path = None
    if notable_events:
        os.makedirs("games", exist_ok=True)
        saved_game_path = f"games/temp{temperature}_skill{skill}_seed{seed}.json"
        with open(saved_game_path, "w") as f:
            json.dump({
                "temperature": temperature, "skill": skill, "seed": seed,
                "moves": all_moves, "result": result, "notable": notable_events,
            }, f, indent=2)

    return {
        "temperature": temperature,
        "skill": skill,
        "seed": seed,
        "opening": " ".join(opening_moves),
        "notable": ", ".join(notable_events) if notable_events else "-",
        "result": result,
        "plies": ply_count,
        "saved_game_path": saved_game_path,
    }


def main():
    # Loaded here rather than at module scope so that importing this module
    # -- to test classify_notable(), for instance -- doesn't require a
    # checkpoint on disk. checkpoints/ is gitignored, so a module-level load
    # breaks anywhere the weights aren't present, including CI.
    global FALLBACK_CONFIG
    FALLBACK_CONFIG = GPTConfig.load(f"{CHECKPOINT_DIR}/config.json")

    print(f"{'Temp':<6}{'Skill':<7}{'Seed':<12}{'Plies':<7}{'Result':<14}{'Opening'}")
    print("-" * 110)
    for temperature in TEMPERATURES:
        for skill in STOCKFISH_SKILLS:
            summary = play_one(temperature, skill)
            print(f"{summary['temperature']:<6}{summary['skill']:<7}{summary['seed']:<12}{summary['plies']:<7}"
                  f"{str(summary['result']):<14}{summary['opening']}")
            if summary["notable"] != "-":
                print(f"      -> notable: {summary['notable']}  (saved: {summary['saved_game_path']})")


if __name__ == "__main__":
    main()
