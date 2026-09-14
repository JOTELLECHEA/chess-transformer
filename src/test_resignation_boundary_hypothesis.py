"""
test_resignation_boundary_hypothesis.py

Tests a specific hypothesis: illegal moves happen because generated games
get pushed PAST where a typical real (GM) game in the training corpus
would have ended via resignation, into territory the model saw little of
during training -- rather than illegal-move rate just being a generic
"more moves = more chances to fail" counting effect.

Method: collect the first-illegal-move ply across many games against
Stockfish, and overlay that distribution against the REAL corpus's
game-length percentiles (reusing game_length_stats.py's own logic, so
this is the same ground truth already used to justify block_size).

If the hypothesis holds, first-illegal-move plies should cluster around
the corpus's own upper percentiles (75th-90th and beyond) -- the exact
region where real training games become rare. If illegal moves are just
as likely at any point in a game regardless of that boundary, the
simpler "shorter games fail less" explanation is the more honest one to
report instead.

Usage:
    python -m src.test_resignation_boundary_hypothesis
    python -m src.test_resignation_boundary_hypothesis --stockfish-skill 20 --n-games 100
    python -m src.test_resignation_boundary_hypothesis --temperature 0.1
"""
import os
import argparse
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import GPTConfig
from src.self_play_engine import PlayerConfig, play_game

FALLBACK_VOCAB_PATH = "vocab_fixed.json"


def load_real_game_lengths(corpus_path):
    """Same logic as game_length_stats.py's load_lengths() -- real
    move-count-per-game from the actual training corpus."""
    lengths = []
    with open(corpus_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            lengths.append(len(tokens) - 2)  # drop <|SOM|> and result token
    lengths.sort()
    return lengths


def collect_first_illegal_plies(checkpoint_dir, n_games, skill, stockfish_path, device,
                                 max_plies, temperature, fallback_config):
    plies = []
    white = PlayerConfig(name="model", player_type="model", checkpoint_dir=checkpoint_dir)
    black = PlayerConfig(name="stockfish", player_type="stockfish",
                          stockfish_path=stockfish_path, stockfish_skill_level=skill,
                          stockfish_time_limit=0.1)
    for i in range(n_games):
        gen = play_game(white, black, device, fallback_config=fallback_config,
                         fallback_vocab_path=FALLBACK_VOCAB_PATH,
                         max_plies=max_plies, temperature=temperature)
        try:
            for event in gen:
                if event.result == "illegal_move":
                    plies.append(event.ply)
                if event.game_over:
                    break
        finally:
            gen.close()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_games} games played, {len(plies)} illegal-move events so far")
    return plies


def main():
    parser = argparse.ArgumentParser(
        description="Test whether illegal moves cluster near the real corpus's resignation boundary."
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints/1.2m_L12E384H6")
    parser.add_argument("--corpus", default="data/chessDataset_1.2m.txt")
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish")
    parser.add_argument("--stockfish-skill", type=int, default=0,
                         help="Lower skill -> longer games -> more illegal-move events to sample from.")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-plies", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="plots/resignation_boundary_test.png")
    args = parser.parse_args()

    fallback_config = GPTConfig.load(f"{args.checkpoint_dir}/config.json")

    print(f"Loading real game lengths from {args.corpus}...")
    real_lengths = load_real_game_lengths(args.corpus)
    n = len(real_lengths)
    p50 = real_lengths[int(n * 0.50)]
    p75 = real_lengths[int(n * 0.75)]
    p90 = real_lengths[int(n * 0.90)]
    p95 = real_lengths[int(n * 0.95)]
    print(f"  Real corpus percentiles: 50th={p50}, 75th={p75}, 90th={p90}, 95th={p95}")

    print(f"\nPlaying {args.n_games} games at Stockfish skill={args.stockfish_skill}, "
          f"temperature={args.temperature}, collecting first-illegal-move plies...")
    illegal_plies = collect_first_illegal_plies(
        args.checkpoint_dir, args.n_games, args.stockfish_skill, args.stockfish_path,
        args.device, args.max_plies, args.temperature, fallback_config)
    print(f"  {len(illegal_plies)} illegal-move events collected out of {args.n_games} games")

    if not illegal_plies:
        print("No illegal-move events collected -- can't test the hypothesis with this run. "
              "Try a lower --stockfish-skill or more --n-games.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bins = range(0, max(max(illegal_plies), p95) + 20, 10)
    ax.hist(real_lengths, bins=bins, density=True, alpha=0.4, color="tab:green",
            label=f"Real corpus game lengths (n={n:,})")
    ax.hist(illegal_plies, bins=bins, density=True, alpha=0.5, color="tab:red",
            label=f"First-illegal-move ply (n={len(illegal_plies)})")

    for p, label in [(p50, "50th"), (p75, "75th"), (p90, "90th"), (p95, "95th")]:
        ax.axvline(p, color="tab:green", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(p, ax.get_ylim()[1] * 0.95, label, rotation=90, fontsize=8,
                color="tab:green", ha="right", va="top")

    ax.set_xlabel("Ply")
    ax.set_ylabel("Density")
    ax.set_title(f"First-illegal-move ply vs. real corpus game-length distribution "
                 f"(skill={args.stockfish_skill}, temp={args.temperature})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")

    median_illegal_ply = statistics.median(illegal_plies)
    print(f"\nMedian first-illegal-move ply: {median_illegal_ply}")
    print(f"Real corpus 75th percentile: {p75}, 90th percentile: {p90}")
    if p75 <= median_illegal_ply <= p90 * 1.3:
        print("-> Illegal moves cluster in/near the range where real games become rare -- "
              "consistent with the resignation-boundary hypothesis.")
    else:
        print("-> Illegal-move onset doesn't clearly align with the real corpus's "
              "resignation boundary -- the simpler 'more moves = more chances to fail' "
              "explanation may be the more honest one to report.")


if __name__ == "__main__":
    main()
