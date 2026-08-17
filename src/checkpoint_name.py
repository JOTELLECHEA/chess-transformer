"""
src/checkpoint_name.py

Generates a checkpoint folder name automatically from a config.json and a
corpus file -- e.g. "98k_L6E256H4" -- rather than typing it by hand.Dataset
size is derived by counting lines in the corpus (one game per line).

Usage:
    python -m src.checkpoint_name --config-file staging/config.json --corpus data/chessDataset_98k.txt
    python -m src.checkpoint_name --config-file staging/config.json --corpus data/chessDataset_98k.txt --promote
"""

import argparse
import os
import shutil
from src.config import GPTConfig


def format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}m".replace(".0m", "m")
    elif n >= 1_000:
        return f"{round(n/1000)}k"
    return str(n)


def count_games(corpus_path: str) -> int:
    with open(corpus_path, "r") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Generate a checkpoint folder name from config + corpus, optionally promoting staging/ into it directly."
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--corpus", required=True,
                         help="Corpus file to count games from (one game per line).")
    parser.add_argument("--staging-dir", default="staging",
                         help="Directory to promote FROM, if --promote is set.")
    parser.add_argument("--checkpoints-dir", default="checkpoints",
                         help="Directory to promote INTO, if --promote is set.")
    parser.add_argument("--promote", action="store_true",
                         help="Also perform the actual promotion: create checkpoints/<name>/ "
                              "and move everything out of staging/ into it. Without this flag, "
                              "only prints the name -- no side effects.")
    args = parser.parse_args()

    config = GPTConfig.load(args.config_file)
    n_games = count_games(args.corpus)
    name = f"{format_count(n_games)}_L{config.n_layer}E{config.n_embd}H{config.n_head}"
    print(name)

    if not args.promote:
        return

    target_dir = os.path.join(args.checkpoints_dir, name)

    if not os.path.isdir(args.staging_dir):
        raise SystemExit(f"ERROR: staging directory '{args.staging_dir}' does not exist -- nothing to promote.")

    staged_files = os.listdir(args.staging_dir)
    if not staged_files:
        raise SystemExit(f"ERROR: staging directory '{args.staging_dir}' is empty -- nothing to promote.")

    expected = {"model_weights.pt", "config.json", "vocab.json"}
    missing = expected - set(staged_files)
    if missing:
        raise SystemExit(f"ERROR: staging directory is missing expected file(s): {sorted(missing)}. "
                          f"Refusing to promote an incomplete checkpoint.")

    if os.path.isdir(target_dir) and os.listdir(target_dir):
        raise SystemExit(f"ERROR: '{target_dir}' already exists and is non-empty -- "
                          f"refusing to overwrite an existing checkpoint. "
                          f"Remove it manually first if you really want to replace it.")

    os.makedirs(target_dir, exist_ok=True)
    for filename in staged_files:
        shutil.move(os.path.join(args.staging_dir, filename), os.path.join(target_dir, filename))

    print(f"Promoted: {args.staging_dir}/* -> {target_dir}/")


if __name__ == "__main__":
    main()
