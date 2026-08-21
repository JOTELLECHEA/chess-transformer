#!/usr/bin/env python3
"""
src/tokenise_lichess_batch.py

Converts Lichess .pgn.zst archives into SAN-tokenized game text, applying
header-skip filtering (time control + GM title) before any movetext is
even parsed.

Cross-file resumability: writes a small JSON checkpoint after each file
completes (and periodically within a file) recording which file index and
how many games into that file have been processed. If interrupted (Ctrl+C,
crash, connection issue, whatever), re-running the exact same command
picks up from there.

Usage:
    python -m src.tokenise_lichess_batch \
        --input-dir lichess_gm_data \
        --output new_games_raw.txt \
        --checkpoint-file batch_progress.json

    # or an explicit file list, processed in the given order:
    python -m src.tokenise_lichess_batch \
        --files lichess_gm_data/lichess_db_standard_rated_2025-01.pgn.zst lichess_gm_data/lichess_db_standard_rated_2025-02.pgn.zst \
        --output new_games_raw.txt

    # or a single file, replacing the old single-file script's usage:
    python -m src.tokenise_lichess_batch \
        --files lichess_gm_data/lichess_db_standard_rated_2025-01.pgn.zst \
        --output new_games_raw.txt
"""
import sys
import argparse
import io
import json
import glob
from pathlib import Path

import zstandard as zstd
import chess.pgn
from tqdm import tqdm
from src.constants import START_TOKEN


RESULT_MAP = {
    "1-0": "<|1-0|>",
    "0-1": "<|0-1|>",
    "1/2-1/2": "<|1/2-1/2|>",
    "*": "<|*|>",
}


def classify_timecontrol(timecontrol: str):
    if not timecontrol:
        return None
    parts = timecontrol.split("+")
    try:
        base = int(parts[0])
    except ValueError:
        return None
    if base <= 120:
        return "bullet"
    if base <= 300:
        return "blitz"
    if base <= 900:
        return "rapid"
    return "classical"

def game_passes_filters(headers, keep_categories: set) -> bool:
    """GM title filter is applied here, so we don't even parse movetext for non-GM games."""
    tc_raw = headers.get("TimeControl", "")
    cat = classify_timecontrol(tc_raw)
    if cat is None or cat not in keep_categories:
        return False
    if headers.get("WhiteTitle", "") != "GM" and headers.get("BlackTitle", "") != "GM":
        return False
    return True

def make_filtering_visitor_factory(keep_categories: set):
    """Build a *factory* that produces a custom `chess.pgn.GameBuilder` capable of
    discarding whole games that do not satisfy our domain-specific filters.
    """
    class _FilterBuilder(chess.pgn.GameBuilder):
        def end_headers(self):
            if not game_passes_filters(self.game.headers, keep_categories):
                return chess.pgn.SKIP
            return super().end_headers()

    def factory():
        return _FilterBuilder()

    return factory


def tokenise_game(game) -> str:
    """Convert a :class:`chess.pgn.Game` object into a single-line
    space-separated token string that a language model can ingest.

    The output always has the form

        <|START|> <SAN-move-1> <SAN-move-2> … <RESULT-TOKEN>
    """
    result_raw = game.headers.get("Result", "*")
    result_token = RESULT_MAP.get(result_raw, "<|*|>")
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    return " ".join([START_TOKEN] + moves + [result_token])


def iter_games_from_zst(path: str, keep_categories: set, skip_games: int = 0, skip_bar=None):
    """Lazily iterate over **GM games** stored in a ``.zst`` (Zstandard-compressed)
    PGN file, yielding only those whose time-control belongs to ``keep_categories``.
    """
    visitor_factory = make_filtering_visitor_factory(keep_categories)
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            txt = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

            skipped = 0
            while skipped < skip_games:
                if not chess.pgn.skip_game(txt):
                    return
                skipped += 1
                if skip_bar is not None:
                    skip_bar.update(1)

            while True:
                game = chess.pgn.read_game(txt, Visitor=visitor_factory)
                if game is None:
                    break
                yield game


def load_checkpoint(path):
    """Load a small JSON “checkpoint” file that remembers where we left off
    when processing a large collection of PGN files.
    """
    if path and Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"file_index": 0, "games_seen_in_current_file": 0}


def write_checkpoint(path, file_index, games_seen_in_current_file):
    """Persist a tiny JSON checkpoint that records where a long‑running PGN
    processing job left off.
    """
    if not path:
        return
    with open(path, "w") as f:
        json.dump({"file_index": file_index, "games_seen_in_current_file": games_seen_in_current_file}, f)


def main():
    """Batch-tokenise many Lichess ``*.pgn.zst`` archives into a single,
    filtered, line-by-line token file.
    """
    parser = argparse.ArgumentParser(
        description="Batch-tokenize multiple Lichess .pgn.zst files into one combined, filtered output."
    )
    parser.add_argument("--input-dir", default=None,
                         help="Directory to glob for *.pgn.zst files (sorted alphabetically).")
    parser.add_argument("--files", nargs="+", default=None,
                         help="Explicit list of .pgn.zst files, processed in the given order.")
    parser.add_argument("--output", required=True,
                         help="Combined output file. Appended to if it already exists (safe to resume).")
    parser.add_argument("--checkpoint-file", default=None,
                         help="Path to a JSON checkpoint for cross-file resume support.")
    parser.add_argument(
        "--keep", nargs="+",
        choices=["bullet", "blitz", "rapid", "classical"],
        default=["bullet", "blitz", "rapid", "classical"],
        help="Which time-control families to keep. Default = all four.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=500_000,
                         help="Write a checkpoint every N games processed within a file (default 500000).")
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    if args.files:
        files = args.files
    elif args.input_dir:
        files = sorted(glob.glob(str(Path(args.input_dir) / "*.pgn.zst")))
    else:
        sys.exit("Must specify either --input-dir or --files")

    if not files:
        sys.exit("No .pgn.zst files found.")

    print(f"Found {len(files)} file(s) to process:")
    for f in files:
        print(f"  {f}")
    print()

    keep_set = set(args.keep)
    checkpoint = load_checkpoint(args.checkpoint_file)
    start_file_index = checkpoint["file_index"]
    start_games_seen = checkpoint["games_seen_in_current_file"]

    if start_file_index > 0 or start_games_seen > 0:
        print(f"Resuming from checkpoint: file #{start_file_index} "
              f"({files[start_file_index] if start_file_index < len(files) else 'N/A'}), "
              f"{start_games_seen:,} games already processed in that file")
        print()

    total_kept_this_run = 0

    with open(args.output, "a") as out_f:
        for file_index in range(start_file_index, len(files)):
            path = files[file_index]
            skip_games = start_games_seen if file_index == start_file_index else 0

            print(f"[{file_index+1}/{len(files)}] Processing {path}")
            if skip_games:
                print(f"  fast-forwarding past {skip_games:,} already-processed games in this file")

            skip_bar = None
            if skip_games:
                skip_bar = tqdm(total=skip_games, desc="  fast-forward", unit="game",
                                 disable=args.no_tqdm, leave=False)

            progress = tqdm(desc="  filtering", unit="game", disable=args.no_tqdm,
                             leave=False, initial=skip_games)

            games_seen = skip_games
            kept_this_file = 0

            try:
                for game in iter_games_from_zst(path, keep_set, skip_games=skip_games, skip_bar=skip_bar):
                    if skip_bar is not None:
                        skip_bar.close()
                        skip_bar = None

                    games_seen += 1
                    progress.update(1)

                    if games_seen % args.checkpoint_interval == 0:
                        write_checkpoint(args.checkpoint_file, file_index, games_seen)

                    if not game_passes_filters(game.headers, keep_set):
                        continue

                    out_f.write(tokenise_game(game) + "\n")
                    out_f.flush()  
                    kept_this_file += 1
                    total_kept_this_run += 1

            except KeyboardInterrupt:
                write_checkpoint(args.checkpoint_file, file_index, games_seen)
                progress.close()
                if skip_bar is not None:
                    skip_bar.close()
                sys.exit(f"\nInterrupted. Checkpoint saved at file #{file_index}, "
                         f"{games_seen:,} games in. Re-run the same command to resume.")
            except Exception as exc:
                write_checkpoint(args.checkpoint_file, file_index, games_seen)
                progress.close()
                if skip_bar is not None:
                    skip_bar.close()
                sys.stderr.write(f"\nError while processing {path}: {exc}\n")
                sys.stderr.write("Checkpoint saved -- re-run the same command to resume.\n")
                sys.exit(1)

            progress.close()
            print(f"  done: {games_seen:,} games seen, {kept_this_file:,} kept")
            print()

            write_checkpoint(args.checkpoint_file, file_index + 1, 0)

    print("=" * 60)
    print(f"All files processed. {total_kept_this_run:,} games kept this run.")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
