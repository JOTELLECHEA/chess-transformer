"""
app.py

Gradio front-end for the chess self-play engine. Both sides are trained
checkpoints, so the app is a head-to-head comparison: watch a checkpoint
play itself, or watch two different architectures play each other.

Weights come from HuggingFace rather than being bundled here, so this file
plus requirements.txt is enough to deploy. snapshot_download() pulls all
four checkpoints once at startup and caches them; subsequent restarts on a
warm machine skip the network entirely.

Legality is strict (max_resample_tries=1, the engine default): the first
illegal move a model samples ends the game. That is deliberate. It's the
same convention every legality number in the README measures, and watching
a weaker checkpoint break down after fifteen moves while the flagship
plays sixty coherent ones is the point of the comparison.

Usage:
    python app.py
"""
import os
import time

import chess
import chess.svg
import gradio as gr
import torch
from huggingface_hub import snapshot_download

from src.config import GPTConfig
from src.self_play_engine import PlayerConfig, play_game

# ----------------------------------------------------------------------
HF_REPO_ID = "Jotellechea/chess-transformer"

# Display label -> subfolder name within the HF repo.
CHECKPOINT_FOLDERS = {
    "98k games, 6-layer": "98k_L6E256H4",
    "1.2m games, 6-layer": "1.2m_L6E256H4",
    "98k games, 12-layer": "98k_L12E384H6",
    "1.2m games, 12-layer (flagship)": "1.2m_L12E384H6",
}
DEFAULT_CHECKPOINT = "1.2m games, 12-layer (flagship)"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOARD_SIZE = 400
MOVE_DELAY_SECONDS = 0.3

print(f"Downloading checkpoints from {HF_REPO_ID} (cached after first run)...")
REPO_ROOT = snapshot_download(repo_id=HF_REPO_ID)
print(f"  checkpoints available at {REPO_ROOT}")

CHECKPOINTS = {
    label: os.path.join(REPO_ROOT, folder)
    for label, folder in CHECKPOINT_FOLDERS.items()
}

# Architecture used only if a player somehow has checkpoint_dir=None.
# Every player here has a real checkpoint, so this is never applied --
# play_game() just requires the argument.
FALLBACK_CONFIG = GPTConfig.load(
    os.path.join(CHECKPOINTS[DEFAULT_CHECKPOINT], "config.json")
)
FALLBACK_VOCAB_PATH = os.path.join(CHECKPOINTS[DEFAULT_CHECKPOINT], "vocab.json")


def render_board(fen: str, last_move_uci: str = None) -> str:
    board = chess.Board(fen)
    last_move = chess.Move.from_uci(last_move_uci) if last_move_uci else None
    return chess.svg.board(board=board, lastmove=last_move, size=BOARD_SIZE)


def play(white_checkpoint, black_checkpoint, temperature, max_plies):
    white = PlayerConfig(name=white_checkpoint, player_type="model",
                         checkpoint_dir=CHECKPOINTS[white_checkpoint])
    black = PlayerConfig(name=black_checkpoint, player_type="model",
                         checkpoint_dir=CHECKPOINTS[black_checkpoint])

    log_lines = [f"White: {white.name}", f"Black: {black.name}", ""]
    yield render_board(chess.Board().fen()), "\n".join(log_lines)

    gen = play_game(white, black, DEVICE,
                    fallback_config=FALLBACK_CONFIG,
                    fallback_vocab_path=FALLBACK_VOCAB_PATH,
                    max_plies=int(max_plies), temperature=temperature)

    move_number = 1
    try:
        for event in gen:
            # max-plies sentinel: no real move attached.
            if not event.uci and not event.player_name:
                log_lines.append(f"\nStopped at the {int(max_plies)}-ply cap "
                                 f"without a natural end.")
                yield render_board(event.board_fen_after), "\n".join(log_lines)
                return

            prefix = f"{move_number}." if event.ply % 2 == 1 else f"{move_number}..."

            # A self-terminating result token: legal, but not a board move.
            if event.is_legal and event.san is None:
                log_lines.append(f"\nGame over: {event.result} "
                                 f"({event.player_name} predicted the game had ended)")
                yield render_board(event.board_fen_after or event.board_fen_before), "\n".join(log_lines)
                return

            if not event.is_legal:
                log_lines.append(f"{prefix} {event.player_name} attempted {event.uci}")
                log_lines.append(f"\nGame over: illegal move.")
                log_lines.append(f"{event.player_name} sampled {event.uci}, which is not "
                                 f"legal in this position, at ply {event.ply}.")
                yield render_board(event.board_fen_before), "\n".join(log_lines)
                return

            log_lines.append(f"{prefix} {event.san}  ({event.player_name}, {event.uci})")
            if event.ply % 2 == 0:
                move_number += 1

            yield render_board(event.board_fen_after, event.uci), "\n".join(log_lines)

            if event.game_over:
                log_lines.append(f"\nGame over: {event.result}")
                yield render_board(event.board_fen_after, event.uci), "\n".join(log_lines)
                return

            time.sleep(MOVE_DELAY_SECONDS)
    finally:
        gen.close()


with gr.Blocks(title="Chess Move-Prediction Transformer") as demo:
    gr.Markdown(
        "# Chess Move-Prediction Transformer\n"
        "Four checkpoints trained on Lichess GM games, playing each other. Each was "
        "trained only to predict the next move — none were given the rules of chess.\n\n"
        "**A game ends the moment a model samples an illegal move.** That's the "
        "comparison: the flagship completes a full game roughly half the time, the "
        "smallest checkpoint almost never does. "
        "[Full results →](https://github.com/JOTELLECHEA/chess-transformer)"
    )

    with gr.Row():
        white_dd = gr.Dropdown(choices=list(CHECKPOINTS.keys()),
                               value=DEFAULT_CHECKPOINT, label="White")
        black_dd = gr.Dropdown(choices=list(CHECKPOINTS.keys()),
                               value=DEFAULT_CHECKPOINT, label="Black")

    with gr.Row():
        temperature_slider = gr.Slider(
            0.1, 1.5, value=0.5, step=0.1, label="Temperature",
            info="Lower = the model's most confident move. Higher = more varied, more mistakes."
        )
        max_plies_input = gr.Number(value=150, precision=0, label="Max plies (safety cap)")

    start_btn = gr.Button("Play", variant="primary")

    with gr.Row():
        board_display = gr.HTML(render_board(chess.Board().fen()))
        move_log = gr.Textbox(label="Moves", lines=22, interactive=False)

    start_btn.click(
        fn=play,
        inputs=[white_dd, black_dd, temperature_slider, max_plies_input],
        outputs=[board_display, move_log],
    )


if __name__ == "__main__":
    demo.launch()
