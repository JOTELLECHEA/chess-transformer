"""
make_game_gif.py

Plays one game via self_play_engine.py and stitches the resulting board
states into an animated GIF.

Usage:
    # live generation, model (White) vs Stockfish (Black)
    python -m src.make_game_gif \
        --white-type model --white-checkpoint-dir checkpoints/1.2m_L12E384H6 \
        --black-type stockfish --black-stockfish-skill 5 \
        --output plots/demo.gif

    # model vs model, two different checkpoints
    python -m src.make_game_gif \
        --white-type model --white-checkpoint-dir checkpoints/1.2m_L12E384H6 \
        --black-type model --black-checkpoint-dir checkpoints/1.2m_L6E256H4 \
        --output plots/demo.gif

"""
import argparse
import io
import os
import json
import chess
import chess.svg
import cairosvg
import torch
from PIL import Image
from src.config import GPTConfig
from src.self_play_engine import PlayerConfig, play_game


def svg_to_png_bytes(svg_string: str) -> bytes:
    return cairosvg.svg2png(bytestring=svg_string.encode("utf-8"))


def render_frame(fen: str, last_move_uci: str = None, board_size: int = 400) -> Image.Image:
    board = chess.Board(fen)
    last_move = chess.Move.from_uci(last_move_uci) if last_move_uci else None
    svg_string = chess.svg.board(board=board, lastmove=last_move, size=board_size)
    png_bytes = svg_to_png_bytes(svg_string)
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def frame_duration_for_ply(ply: int, fast_forward_until_ply, fast_forward_duration_ms, frame_duration_ms) -> int:
    if fast_forward_until_ply and ply < fast_forward_until_ply:
        return fast_forward_duration_ms
    return frame_duration_ms


def replay_saved_game(path: str, max_plies: int, board_size: int,
                       fast_forward_until_ply, fast_forward_duration_ms, frame_duration_ms):
    """Renders a GIF by replaying an already-known, fixed move sequence"""
    with open(path) as f:
        data = json.load(f)
    moves = data["moves"]
    print(f"Replaying saved game: {path}")
    print(f"  temp={data.get('temperature')} skill={data.get('skill')} result={data.get('result')}")

    board = chess.Board()
    frames = [render_frame(board.fen(), board_size=board_size)]
    durations = [frame_duration_for_ply(1, fast_forward_until_ply, fast_forward_duration_ms, frame_duration_ms)]

    for ply, uci in enumerate(moves, start=1):
        if ply > max_plies:
            break
        move = chess.Move.from_uci(uci)
        board.push(move)
        frames.append(render_frame(board.fen(), uci, board_size=board_size))
        durations.append(frame_duration_for_ply(ply, fast_forward_until_ply, fast_forward_duration_ms, frame_duration_ms))
        print(f"ply {ply}: {uci}")

    return frames, durations


def build_player_config(side_label, player_type, name, checkpoint_dir,
                         stockfish_path, stockfish_skill, stockfish_time_limit):
    if player_type == "model":
        return PlayerConfig(name=name or (checkpoint_dir or f"Random ({side_label})"),
                             player_type="model", checkpoint_dir=checkpoint_dir)
    elif player_type == "stockfish":
        return PlayerConfig(name=name or f"Stockfish (skill {stockfish_skill})",
                             player_type="stockfish", stockfish_path=stockfish_path,
                             stockfish_skill_level=stockfish_skill, stockfish_time_limit=stockfish_time_limit)
    else:
        return PlayerConfig(name=name or f"Human ({side_label})", player_type="human")


def main():
    parser = argparse.ArgumentParser(description="Render a played chess game as an animated GIF.")
    parser.add_argument("--replay-from-file", default=None,
                         help="Path to an already-saved game (from sweep_games.py). If set, "
                              "--white-*/--black-*/--seed/--temperature are ignored entirely.")

    parser.add_argument("--white-type", choices=["model", "stockfish", "human"], default="model")
    parser.add_argument("--white-name", default=None)
    parser.add_argument("--white-checkpoint-dir", default=None,
                         help="Required if --white-type=model, unless you want a random-init baseline "
                              "(omit for that).")
    parser.add_argument("--white-stockfish-skill", type=int, default=5)
    parser.add_argument("--white-stockfish-time-limit", type=float, default=0.1)

    parser.add_argument("--black-type", choices=["model", "stockfish", "human"], default="stockfish")
    parser.add_argument("--black-name", default=None)
    parser.add_argument("--black-checkpoint-dir", default=None)
    parser.add_argument("--black-stockfish-skill", type=int, default=5)
    parser.add_argument("--black-stockfish-time-limit", type=float, default=0.1)

    parser.add_argument("--stockfish-path", default="/usr/games/stockfish")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="plots/demo_game.gif")
    parser.add_argument("--seed", type=int, default=None,
                         help="Only meaningful for live generation, and even then, only makes the "
                              "MODEL side reproducible, not Stockfish's. See --replay-from-file.")
    parser.add_argument("--fast-forward-until-ply", type=int, default=None,
                         help="Every ply before this is still rendered (full context preserved), "
                              "just shown briefly rather than at the normal pace.")
    parser.add_argument("--fast-forward-duration-ms", type=int, default=80)
    parser.add_argument("--frame-duration-ms", type=int, default=700)
    parser.add_argument("--final-frame-hold-ms", type=int, default=3000)
    parser.add_argument("--board-size", type=int, default=400)
    parser.add_argument("--fallback-checkpoint-dir", default="checkpoints/1.2m_L12E384H6",
                         help="Architecture used for a random-init player (checkpoint_dir omitted). "
                              "Irrelevant if neither player ends up random-init.")
    parser.add_argument("--fallback-vocab-path", default="vocab_fixed.json")
    args = parser.parse_args()

    if args.replay_from_file:
        frames, durations = replay_saved_game(
            args.replay_from_file, args.max_plies, args.board_size,
            args.fast_forward_until_ply, args.fast_forward_duration_ms, args.frame_duration_ms)
    else:
        if args.seed is not None:
            torch.manual_seed(args.seed)
            print(f"Using seed {args.seed} for reproducible generation (model side only).")

        fallback_config = GPTConfig.load(f"{args.fallback_checkpoint_dir}/config.json")

        white = build_player_config("White", args.white_type, args.white_name, args.white_checkpoint_dir,
                                     args.stockfish_path, args.white_stockfish_skill, args.white_stockfish_time_limit)
        black = build_player_config("Black", args.black_type, args.black_name, args.black_checkpoint_dir,
                                     args.stockfish_path, args.black_stockfish_skill, args.black_stockfish_time_limit)

        frames = [render_frame(chess.Board().fen(), board_size=args.board_size)]
        durations = [frame_duration_for_ply(1, args.fast_forward_until_ply, args.fast_forward_duration_ms, args.frame_duration_ms)]

        gen = play_game(white, black, args.device, fallback_config=fallback_config,
                         fallback_vocab_path=args.fallback_vocab_path,
                         max_plies=args.max_plies, temperature=args.temperature)

        try:
            for event in gen:
                if event.player_name == "" and not event.uci:
                    continue
                fen = event.board_fen_after or event.board_fen_before
                is_real_move = event.is_legal and event.san is not None
                frames.append(render_frame(fen, event.uci if is_real_move else None, board_size=args.board_size))
                durations.append(frame_duration_for_ply(event.ply, args.fast_forward_until_ply,
                                                          args.fast_forward_duration_ms, args.frame_duration_ms))
                print(f"ply {event.ply}: {event.player_name} played {event.uci!r} ({event.san})")
                if event.game_over:
                    print(f"Game over: {event.result}")
                    break
        finally:
            gen.close()

    durations[-1] = args.final_frame_hold_ms
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"\nSaved {len(frames)}-frame GIF to {args.output}")


if __name__ == "__main__":
    main()
