"""
self_play_engine.py

Core game engine, decoupled from any UI framework. Each side is
independently configured as "model" (a trained checkpoint, or a fresh
random-init baseline), "stockfish" (a real engine via python-chess), or
"human" (moves supplied externally) -- covering self-play, head-to-head,
and model-vs-engine/human through one code path.

Model players load their own config/tokenizer independently, since two
players can be different architectures (e.g. 6-layer vs 12-layer); all
checkpoints share the same closed-form vocabulary, so token IDs stay
directly comparable across them regardless. Weights load from either
model.safetensors (as published on HuggingFace) or model_weights.pt (as
written by train.py), so the same engine works with both.

An illegal move ends the game immediately for a MODEL (same convention as
eval_legality.py -- the board state is undefined past that point).
Stockfish never produces one. A human's illegal move is just re-requested,
not game-ending, since a typo isn't the same kind of failure as a model's.

Human input works via generator .send(): the engine yields a MoveEvent
with awaiting_human_input=True and pauses, resuming when the caller sends
back a move string.

Cleanup: a Stockfish player holds a real subprocess, closed in `finally`.
If a game might be abandoned mid-way with Stockfish involved, the caller
MUST call generator.close() -- otherwise the subprocess is left orphaned,
relying on non-deterministic garbage collection instead.
"""
from dataclasses import dataclass
from typing import Optional, Generator
import os
import torch
import chess
import chess.engine
from safetensors.torch import load_model
from src.config import GPTConfig
from src.model import GPT
from src.dataset import MoveTokenizer
from src.constants import START_TOKEN, RESULT_TOKENS


@dataclass
class PlayerConfig:
    name: str                              # display name
    player_type: str                       # "model" | "stockfish" | "human"
    checkpoint_dir: Optional[str] = None   # required if player_type == "model"; None => fresh random init
    stockfish_path: Optional[str] = None   # required if player_type == "stockfish"
    stockfish_skill_level: int = 20        # 0-20, python-chess/Stockfish convention
    stockfish_time_limit: float = 0.1      # seconds per move

    def __post_init__(self):
        if self.player_type not in ("model", "stockfish", "human"):
            raise ValueError(f"PlayerConfig '{self.name}': unknown player_type '{self.player_type}'")
        if self.player_type == "stockfish" and not self.stockfish_path:
            raise ValueError(f"PlayerConfig '{self.name}': player_type='stockfish' requires stockfish_path")


@dataclass
class MoveEvent:
    """One step of the game, yielded to the caller as it happens."""
    ply: int
    player_name: str
    uci: str
    san: Optional[str]
    is_legal: bool
    board_fen_before: str
    board_fen_after: Optional[str]
    game_over: bool
    result: Optional[str]                  # "1-0" / "0-1" / "1/2-1/2" / "illegal_move" / "max_plies_reached" / None
    awaiting_human_input: bool = False     # True when the engine is paused, waiting for .send(uci_string)


class _ModelPlayer:
    """Loads its own model/config/tokenizer independently, so different
    model players can be different architectures in the same game.
    Random init (checkpoint_dir=None) still produces a fully functional,
    if untrained, model -- GPT.__init__ initializes every weight
    regardless of whether a checkpoint is ever loaded on top."""
    def __init__(self, checkpoint_dir: Optional[str], device: str,
                 fallback_config: GPTConfig, fallback_vocab_path: str):
        if checkpoint_dir is not None:
            self.config = GPTConfig.load(os.path.join(checkpoint_dir, "config.json"))
            tokenizer = MoveTokenizer.from_vocab_file(os.path.join(checkpoint_dir, "vocab.json"))
        else:
            self.config = fallback_config
            tokenizer = MoveTokenizer.from_vocab_file(fallback_vocab_path)
        self.config.vocab_size = tokenizer.vocab_size
        self.model = GPT(self.config).to(device)
        if checkpoint_dir is not None:
            safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
            pt_path = os.path.join(checkpoint_dir, "model_weights.pt")
            if os.path.exists(safetensors_path):
                # load_model(), not load_file() -- wte.weight and lm_head.weight
                # are tied, and load_model reconstructs that tie rather than
                # leaving two independent tensors.
                load_model(self.model, safetensors_path, device=device)
            elif os.path.exists(pt_path):
                self.model.load_state_dict(torch.load(pt_path, map_location=device))
            else:
                raise FileNotFoundError(
                    f"No weights found in {checkpoint_dir}: expected either "
                    f"model.safetensors or model_weights.pt"
                )
        self.model.eval()
        self.device = device

    def sample_next_token(self, token_ids, temperature=1.0):
        idx = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        with torch.no_grad():
            logits, _ = self.model(idx_cond)
            logits = logits.squeeze(1) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token


def _build_player(player: PlayerConfig, device: str, fallback_config: GPTConfig, fallback_vocab_path: str):
    """Returns whatever backing object this player's type needs -- a
    _ModelPlayer, a live Stockfish engine, or None for a human (moves come
    from outside the engine entirely, nothing to build)."""
    if player.player_type == "model":
        return _ModelPlayer(player.checkpoint_dir, device, fallback_config, fallback_vocab_path)
    elif player.player_type == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(player.stockfish_path)
        engine.configure({"Skill Level": player.stockfish_skill_level})
        return engine
    else:  # human
        return None


def play_game(
    white: PlayerConfig,
    black: PlayerConfig,
    device: str,
    fallback_config: Optional[GPTConfig] = None,
    fallback_vocab_path: str = "vocab_fixed.json",
    max_plies: int = 300,
    temperature: float = 1.0,
    max_resample_tries: int = 1,
    respect_result_tokens: bool = True,
) -> Generator[MoveEvent, str, None]:
    """
    Plays a full game, White vs Black, yielding a MoveEvent after every ply.

    fallback_config: architecture to use for a "model" player with
        checkpoint_dir=None (fresh random init). Required only if such a
        player is actually configured; ignored otherwise.

    max_resample_tries: for MODEL players only. If 1 (default), the FIRST
        illegal draw ends the game immediately -- "strict" mode, showing
        the model's raw, unfiltered behavior. If >1, an illegal draw is
        discarded and resampled up to this many times before giving up.
        Stockfish never needs this (never produces illegal moves); a
        human's illegal move is always just re-requested, independent of
        this setting.

    respect_result_tokens: for MODEL players only. If True (default), a
        sampled result token ends the game immediately. If False, result
        tokens are ignored and the game only ends on a real terminal board
        state per python-chess. Irrelevant when both players are
        Stockfish/human, since neither ever produces a result token.
    """
    tokenizer = MoveTokenizer.from_vocab_file(fallback_vocab_path)
    result_token_ids = {tokenizer.stoi[t] for t in RESULT_TOKENS if t in tokenizer.stoi}

    white_backend = _build_player(white, device, fallback_config, fallback_vocab_path)
    black_backend = _build_player(black, device, fallback_config, fallback_vocab_path)

    board = chess.Board()
    token_ids = tokenizer.encode(START_TOKEN)

    try:
        for ply in range(1, max_plies + 1):
            is_white_turn = board.turn == chess.WHITE
            current_config = white if is_white_turn else black
            current_backend = white_backend if is_white_turn else black_backend
            current_name = white.name if is_white_turn else black.name
            fen_before = board.fen()

            move = None
            uci_str = None
            legal = False

            if current_config.player_type == "stockfish":
                result = current_backend.play(board, chess.engine.Limit(time=current_config.stockfish_time_limit))
                move = result.move
                uci_str = move.uci()
                legal = True  # a real engine never proposes an illegal move
                token_ids.append(tokenizer.stoi[uci_str])

            elif current_config.player_type == "human":
                proposed_uci = yield MoveEvent(
                    ply=ply, player_name=current_name, uci="", san=None, is_legal=True,
                    board_fen_before=fen_before, board_fen_after=None,
                    game_over=False, result=None, awaiting_human_input=True,
                )
                while True:
                    try:
                        candidate = chess.Move.from_uci(proposed_uci)
                        if candidate in board.legal_moves:
                            move = candidate
                            uci_str = proposed_uci
                            legal = True
                            token_ids.append(tokenizer.stoi[uci_str])
                            break
                    except ValueError:
                        pass
                    proposed_uci = yield MoveEvent(
                        ply=ply, player_name=current_name, uci=proposed_uci or "", san=None, is_legal=False,
                        board_fen_before=fen_before, board_fen_after=None,
                        game_over=False, result=None, awaiting_human_input=True,
                    )

            else:  # "model"
                illegal_attempts = 0
                total_attempts = 0
                max_total_attempts = max(50, max_resample_tries * 10)
                was_result_token = False
                next_id = None
                while total_attempts < max_total_attempts:
                    total_attempts += 1
                    next_id = current_backend.sample_next_token(token_ids, temperature)
                    uci_str = tokenizer.itos[next_id]
                    if next_id in result_token_ids:
                        if respect_result_tokens:
                            was_result_token = True
                            break
                        else:
                            continue
                    try:
                        candidate = chess.Move.from_uci(uci_str)
                        legal = candidate in board.legal_moves
                    except ValueError:
                        legal = False
                    if legal:
                        move = candidate
                        token_ids.append(next_id)
                        break
                    illegal_attempts += 1
                    if illegal_attempts >= max_resample_tries:
                        break
                if was_result_token:
                    token_ids.append(next_id)
                    yield MoveEvent(
                        ply=ply, player_name=current_name, uci=uci_str, san=None,
                        is_legal=True, board_fen_before=fen_before, board_fen_after=fen_before,
                        game_over=True, result=uci_str.strip("<|>"),
                    )
                    return
                if not legal:
                    yield MoveEvent(
                        ply=ply, player_name=current_name, uci=uci_str, san=None,
                        is_legal=False, board_fen_before=fen_before, board_fen_after=None,
                        game_over=True, result="illegal_move",
                    )
                    return

            san = board.san(move)
            board.push(move)

            # Check to see if the game is over after each move.
            if board.is_game_over():
                outcome = board.outcome()
                yield MoveEvent(
                    ply=ply, player_name=current_name, uci=uci_str, san=san,
                    is_legal=True, board_fen_before=fen_before, board_fen_after=board.fen(),
                    game_over=True, result=outcome.result() if outcome else "game_over",
                )
                return

            yield MoveEvent(
                ply=ply, player_name=current_name, uci=uci_str, san=san,
                is_legal=True, board_fen_before=fen_before, board_fen_after=board.fen(),
                game_over=False, result=None,
            )

        yield MoveEvent(
            ply=max_plies, player_name="", uci="", san=None, is_legal=True,
            board_fen_before=board.fen(), board_fen_after=board.fen(),
            game_over=True, result="max_plies_reached",
        )
    finally:
        for backend in (white_backend, black_backend):
            if isinstance(backend, chess.engine.SimpleEngine):
                backend.close()
