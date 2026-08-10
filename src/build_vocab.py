"""
build_vocab.py

Builds a vocab.json from a given corpus file.

Usage:
    python -m src.build_vocab --corpus data/kept_games_uci_clean.txt --output checkpoints/smallmodel/vocab.json
    python -m src.build_vocab --corpus combined_games_uci_clean.txt --output checkpoints/bigmodel/vocab.json
    python -m src.build_vocab --closed-form --output checkpoints/fixedmodel/vocab_fixed.json
"""

import argparse
from src.dataset import MoveTokenizer


def main():
    parser = argparse.ArgumentParser(description="Build a vocab.json from a move corpus file.")
    parser.add_argument("--corpus", required=True, help="Path to the UCI-tokenized corpus file.")
    parser.add_argument("--output", default="vocab.json", help="Where to save the resulting vocab.json.")
    parser.add_argument("--closed-form", action="store_true",
                     help="Build the fixed, corpus-independent vocab instead of "
                          "deriving one from a corpus. Recommended going forward.")
    args = parser.parse_args()

    if args.closed_form:
        tokenizer = MoveTokenizer.build_closed_form()
    else:
        print(f"Reading corpus from {args.corpus}...")
        with open(args.corpus, "r", encoding="utf-8") as f:
            tokenizer = MoveTokenizer(f.read())
    tokenizer.save_vocab(args.output)


if __name__ == "__main__":
    main()
