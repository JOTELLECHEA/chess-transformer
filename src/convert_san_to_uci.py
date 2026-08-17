#!/usr/bin/env python3
"""
src/convert_san_to_uci.py

Converts a corpus of tokenized games from SAN to UCI move notation.

Input format (one game per line):
    <|SOM|> e4 e5 Nf3 Nc6 ... <|1-0|>

Output format (same shape, moves swapped to UCI):
    <|SOM|> e2e4 e7e5 g1f3 b8c6 ... <|1-0|>

Usage:
    python convert_san_to_uci.py input.txt output.txt
"""

import argparse
import chess
from tqdm import tqdm
from src.constants import START_TOKEN, RESULT_TOKENS



def convert_line(line: str):
    """
    Convert one line of SAN tokens to UCI tokens.
    Returns (converted_line, None) on success, or (None, error_message) on failure.
    """
    tokens = line.strip().split()
    if len(tokens) < 2:
        return None, "line too short (missing SOM/result tokens)"

    if tokens[0] != START_TOKEN:
        return None, f"expected {START_TOKEN} at start, got {tokens[0]!r}"

    if tokens[-1] not in RESULT_TOKENS:
        return None, f"unrecognized result token {tokens[-1]!r}"

    san_moves = tokens[1:-1]
    result_token = tokens[-1]

    board = chess.Board()
    uci_moves = []
    for ply_num, san in enumerate(san_moves, start=1):
        try:
            move = board.parse_san(san)
        except ValueError as exc:
            return None, f"ply {ply_num} ({san!r}) failed to parse: {exc}"
        uci_moves.append(move.uci())
        board.push(move)

    converted = " ".join([START_TOKEN] + uci_moves + [result_token])
    return converted, None


def main():
    parser = argparse.ArgumentParser(
        description="Convert a SAN-tokenized game corpus to UCI notation."
    )
    parser.add_argument("input", help="Path to input corpus (SAN-tokenized, one game per line)")
    parser.add_argument("output", help="Path to write UCI-tokenized output")
    parser.add_argument(
        "--errors-file",
        default=None,
        help="Optional path to write skipped lines + reasons (for debugging bad data)",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Suppress the progress bar (useful for non-interactive runs)",
    )
    args = parser.parse_args()

    # Count lines up front so tqdm can show a real percentage/ETA.
    with open(args.input, "r") as f:
        total_lines = sum(1 for _ in f)

    kept = 0
    skipped = 0
    san_vocab = set()
    uci_vocab = set()
    errors_fh = open(args.errors_file, "w") if args.errors_file else None

    try:
        with open(args.input, "r") as fin, open(args.output, "w") as fout:
            for line_num, line in enumerate(
                tqdm(fin, total=total_lines, desc="Converting", unit="game", disable=args.no_tqdm),
                start=1,
            ):
                if not line.strip():
                    continue

                converted, error = convert_line(line)
                if error is not None:
                    skipped += 1
                    if errors_fh:
                        errors_fh.write(f"line {line_num}: {error}\n")
                    continue

                fout.write(converted + "\n")
                kept += 1

                # Track vocab on the same kept lines so the two counts
                # are a fair apples-to-apples comparison.
                san_vocab.update(line.strip().split()[1:-1])
                uci_vocab.update(converted.split()[1:-1])
    finally:
        if errors_fh:
            errors_fh.close()

    print(f"\nDone. Converted: {kept:,}  Skipped: {skipped:,}  Total: {total_lines:,}")
    if skipped and not args.errors_file:
        print("Tip: re-run with --errors-file skipped.txt to see why lines were skipped.")

    print(f"\nUnique SAN move tokens: {len(san_vocab):,}")
    print(f"Unique UCI move tokens: {len(uci_vocab):,}")
    print("(special tokens like <|SOM|> and result tokens excluded from these counts,")
    print(" since they're fixed and identical in both versions)")


if __name__ == "__main__":
    main()
