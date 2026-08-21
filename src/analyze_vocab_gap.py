#!/usr/bin/env python3
"""
analyze_vocab_gap.py

Compares the UCI move vocabulary actually observed in your corpus against
the full theoretical space of UCI strings the closed-form vocab covers.


Usage:
    python -m src.analyze_vocab_gap data/chessDataset_1.2m.txt
    python -m src.analyze_vocab_gap data/chessDataset_1.2m.txt --dump-missing missing.txt
"""
import sys
import argparse
from pathlib import Path
from src.move_space import build_theoretical_uci_space


def load_observed_vocab(path):
    """Reads a UCI corpus file and returns the set of unique move tokens
    (special tokens <|SOM|> / result tokens excluded)."""
    if not Path(path).exists():
        sys.exit(f"No such file: {path}")
    vocab = set()
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            vocab.update(tokens[1:-1])  # drop <|SOM|> and the result token
    if not vocab:
        sys.exit(f"No games found in {path}.")
    return vocab


def main():
    parser = argparse.ArgumentParser(
        description="Compare observed UCI vocab against the full theoretical move space."
    )
    parser.add_argument("uci_corpus", help="Path to your UCI-converted corpus file")
    parser.add_argument(
        "--dump-missing",
        default=None,
        help="Optional path to write the full list of missing moves, one per line",
    )
    args = parser.parse_args()

    print("Building theoretical move space")
    all_uci, promo_uci, underpromo_uci = build_theoretical_uci_space()

    print(f"Loading observed vocab from {args.uci_corpus}")
    observed = load_observed_vocab(args.uci_corpus)

    missing = all_uci - observed
    unexpected = observed - all_uci 

    missing_underpromo = missing & underpromo_uci
    missing_queen_promo = missing & (promo_uci - underpromo_uci)
    missing_ordinary = missing - promo_uci

    print()
    print(f"Theoretical move space:     {len(all_uci):,}")
    print(f"Observed in corpus:         {len(observed):,}")
    print(f"Coverage:                   {len(observed) / len(all_uci) * 100:.1f}%")
    print(f"Missing:                    {len(missing):,}")
    print()
    print("Missing, broken down:")
    print(f"  underpromotions (N/B/R):  {len(missing_underpromo):,}  (of {len(underpromo_uci):,} possible)")
    print(f"  queen promotions:         {len(missing_queen_promo):,}  (of {len(promo_uci) - len(underpromo_uci):,} possible)")
    print(f"  ordinary (non-promotion): {len(missing_ordinary):,}")

    if unexpected:
        print()
        print(f"WARNING: {len(unexpected)} tokens in your corpus aren't in the theoretical")
        print("space at all - this would indicate a real bug (e.g. a malformed UCI string).")
        print("Examples:", sorted(unexpected)[:10])

    if args.dump_missing:
        with open(args.dump_missing, "w") as f:
            f.write("# underpromotions\n")
            for m in sorted(missing_underpromo):
                f.write(m + "\n")
            f.write("\n# queen promotions\n")
            for m in sorted(missing_queen_promo):
                f.write(m + "\n")
            f.write("\n# ordinary moves\n")
            for m in sorted(missing_ordinary):
                f.write(m + "\n")
        print(f"\nFull missing list written to {args.dump_missing}")


if __name__ == "__main__":
    main()
