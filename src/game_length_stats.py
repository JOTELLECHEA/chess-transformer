#!/usr/bin/env python3
"""
src/game_length_stats.py

Utility for analysing a UCI-tokenised chess-games corpus.
Reports the real distribution of moves (plies) per game in your UCI-tokenized
corpus, so block_size choices are grounded in actual data rather than a guess.

Typical usage
-------------
    python -m src.game_length_stats
"""
import os
import sys
import argparse
import statistics

def load_lengths(path):
    """Reads a UCI corpus file and returns a sorted list of move-counts per game."""
    lengths = []
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            lengths.append(len(tokens) - 2)
    lengths.sort()
    return lengths

def print_stats(lengths, label):
    """
    Print a compact, human-readable report about a list of game lengths.
    The report includes
    * total number of games,
    * mean, standard deviation, min and max,
    * the 50th, 75th, 90th, 95th and 99th percentiles,
    * the fraction of games that would completely fit inside a few typical
      ``block_size`` values (useful when deciding the context window for a
      transformer model),
    * a quick “near-empty” check that counts games with 0-4 moves.
    """
    n = len(lengths)
    mean = statistics.mean(lengths)
    stdev = statistics.stdev(lengths) if n > 1 else 0.0
    minimum = lengths[0]
    maximum = lengths[-1]

    def pct(p):
        idx = min(n - 1, int(p / 100 * n))
        return lengths[idx]
    
    print(f"{label:-^5}")
    print(f"Games analyzed: {n:,}")
    print()
    print(f"Mean moves/game:   {mean:.1f}")
    print(f"Std dev:           {stdev:.1f}")
    print(f"Min:               {minimum}")
    print(f"Max:               {maximum}")
    print()
    print("Percentiles:")

    for p in (50, 75, 90, 95, 99):
        print(f"  {p}th: {pct(p)}")

    print()
    print("Block-size coverage reference:")
    print("  (what fraction of games would fit ENTIRELY within this block_size)")

    for candidate in [64, 128, 192, 256]:
        covered = sum(1 for l in lengths if l <= candidate)
        pct_covered = covered / n * 100
        print(f"  block_size={candidate:4d} -> fully covers {pct_covered:5.1f}% of games")

    print()
    print("Near-empty game check:")
    thresholds = [0, 1, 2, 4]

    for t in thresholds:
        count = sum(1 for l in lengths if l == t) if t < 4 else sum(1 for l in lengths if 3 <= l <= 4)
        label2 = f"exactly {t} moves" if t < 4 else "3-4 moves"
        print(f"  games with {label2}: {count}")

    print()

def plot_distribution(lengths_a, label_a, lengths_b=None, label_b=None,
                       output_path="plots/game_length_distribution.png", block_size_ref=None, xlim=None):
    """Render a density histogram of one (or two) corpora's game-length distribution and write it to an image file."""

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    black = "#000000"
    hist1 = "#FF0000"
    hist2 = "#008CFF"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_lengths = lengths_a + (lengths_b or [])
    bin_width = 10
    bins = np.arange(0, max(all_lengths) + bin_width, bin_width)
    ax.hist(lengths_a, bins=bins, density=True, histtype="step", linewidth=2.2, color=hist1,
            label=f"{label_a[5:-4]}")
    
    if lengths_b is not None:
        ax.hist(lengths_b, bins=bins, density=True, histtype="step", linewidth=2.2, color=hist2,
                label=f"{label_b[5:-4]}")
    if block_size_ref is not None:
        ax.axvline(block_size_ref, linestyle="--", color=black, linewidth=1.3,
                   label=f"block_size={block_size_ref}")
    if xlim is not None:
        ax.set_xlim(0, xlim)

    ax.set_xlabel("Moves per game", fontsize=11)
    ax.set_ylabel("Density (normalized)", fontsize=11)
    ax.set_title("Game length distribution", fontsize=13)
    ax.tick_params(colors=black)

    for spine in ax.spines.values():
        spine.set_color(black)
        spine.set_linewidth(0.6)

    ax.legend(frameon=False, labelcolor=black)
    plt.tight_layout()
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare per-game move-count distribution across the two corpora.")
    parser.add_argument("--corpus-a", default="data/chessDataset_98k.txt",
                         help="First corpus (default: the 98k corpus)")
    parser.add_argument("--corpus-b", default="data/chessDataset_1.2m.txt",
                         help="Second corpus (default: the 1.2m corpus)")
    parser.add_argument("--plot-output", default="plots/game_length_distribution.png",
                         help="Where to save the comparison histogram.")
    parser.add_argument("--block-size-ref", type=int, default=192,
                         help="Draw a vertical reference line at this block_size value.")
    parser.add_argument("--xlim", type=int, default=300,
                         help="Trim the plot's x-axis to (0, this value).")
    args = parser.parse_args()

    lengths_a = load_lengths(args.corpus_a)
    lengths_b = load_lengths(args.corpus_b)

    if not lengths_a:
        sys.exit(f"No games found in {args.corpus_a}.")
    if not lengths_b:
        sys.exit(f"No games found in {args.corpus_b}.")

    print_stats(lengths_a, args.corpus_a)
    print_stats(lengths_b, args.corpus_b)

    # Show a few examples of games that have zero moves (they are usually errors).
    for path, lengths in [(args.corpus_a, lengths_a), (args.corpus_b, lengths_b)]:
        zero_move_count = sum(1 for l in lengths if l == 0)
        if zero_move_count:
            print(f"Found {zero_move_count} game(s) with ZERO moves in {path} — showing up to 5 examples:")
            shown = 0
            with open(path, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    tokens = line.strip().split()
                    if len(tokens) == 2:
                        print(f"    line {line_num}: {line.strip()!r}")
                        shown += 1
                        if shown >= 5:
                            break
            print()

    plot_distribution(
        lengths_a, args.corpus_a,
        lengths_b=lengths_b, label_b=args.corpus_b,
        output_path=args.plot_output,
        block_size_ref=args.block_size_ref,
        xlim=args.xlim,
    )
if __name__ == "__main__":
    main()
