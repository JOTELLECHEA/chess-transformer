"""
build_vocab.py

Builds a vocab.json either from a corpus file (legacy, corpus-derived vocab)
or from the fixed closed-form move space (recommended going forward -- see
MoveTokenizer.build_closed_form() for why this is preferable).

Usage:
    python -m src.build_vocab --closed-form --output vocab_fixed.json
    python -m src.build_vocab --corpus data/chessDataset_98k.txt --output vocab.json
"""

import argparse
from src.dataset import MoveTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Build a vocab.json from a move corpus file, or from the fixed closed-form move space."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", help="Path to the UCI-tokenized corpus file.")
    group.add_argument("--closed-form", action="store_true",
                        help="Build the fixed, corpus-independent vocab instead of "
                             "deriving one from a corpus. Recommended going forward.")
    parser.add_argument("--output", default="vocab.json", help="Where to save the resulting vocab.json.")
    args = parser.parse_args()

    if args.closed_form:
        print("Building closed-form vocab (fixed, corpus-independent)...")
        tokenizer = MoveTokenizer.build_closed_form()
    else:
        print(f"Reading corpus from {args.corpus}...")
        with open(args.corpus, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = MoveTokenizer(text)

    tokenizer.save_vocab(args.output)
    print(f"Saved vocab.json with {tokenizer.vocab_size} tokens to {args.output}")


if __name__ == "__main__":
    main()
