"""
plot_win_rate_by_skill.py

Runs many games per Stockfish skill level, for each of two checkpoints
(e.g. small and big architecture), and plots win rate as a function of
skill level -- the same style of comparison used in Karvonen (2024)'s
"Emergent World Models and Latent Variable Estimation in Chess-Playing
Language Models".

Win rate = wins / total games at that skill level, where "total games"
includes EVERY outcome (win, loss, draw, illegal-move ending, max-plies
reached) in the denominator. Illegal-move endings are deliberately NOT
excluded -- excluding them would artificially inflate the apparent win
rate by only counting games the model didn't immediately fail at, which
misrepresents real-world performance against that skill level.

Elo correspondence for Stockfish skill levels is inherently approximate --
Stockfish's own documentation states precise Elo calibration against a
human/engine scale isn't reliably achievable. If you cite an Elo axis
alongside skill level, attribute it explicitly to whatever source you're
using (e.g. Karvonen 2024) rather than presenting it as independently
verified here.

Sample size note: given how low this project's overall win rate has
already shown to be (see the sweep_games.py results), GAMES_PER_SKILL_LEVEL
needs to be large enough to distinguish a real signal from noise. The
default below is a reasonable starting point for a first look, not a
guarantee of statistically tight results -- consider increasing it for a
final, publishable number.

Usage:
    python plot_win_rate_by_skill.py
"""
import os
import sys
import json
import math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import GPTConfig
from src.self_play_engine import PlayerConfig, play_game

# ----------------------------------------------------------------------
SMALL_CHECKPOINT_DIR = "checkpoints/1.2m_L6E256H4"
BIG_CHECKPOINT_DIR = "checkpoints/1.2m_L12E384H6"
SMALL_LABEL = "6-layer"
BIG_LABEL = "12-layer"

STOCKFISH_PATH = "/usr/games/stockfish"
SKILL_LEVELS = [0, 5, 10, 15, 20]
GAMES_PER_SKILL_LEVEL = 50
TEMPERATURE = 1.0
MAX_PLIES = 150
DEVICE = "cuda"
OUTPUT_PLOT = "plots/win_rate_by_skill.png"
OUTPUT_RAW = "results/win_rate_by_skill.json"

FALLBACK_VOCAB_PATH = "vocab_fixed.json"


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def make_plot(skill_levels, games_per_level, small_label, big_label, small_outcomes, big_outcomes, output_path):
    """
    Two panels:
      Left:  win rate with 95% Wilson CI error bars -- overlapping bars
             between architectures at a given skill level mean that
             difference isn't statistically distinguishable, visible
             directly rather than needing a separate table to explain it.
      Right: illegal-move rate by skill level -- a separate, real finding
             in its own right (both architectures show illegal-move rate
             cleanly decreasing as Stockfish gets stronger, plausibly
             because stronger play produces shorter games).
    """
    def series(outcomes_by_skill, outcome_key):
        rates, los, his = [], [], []
        for skill in skill_levels:
            key = skill if skill in outcomes_by_skill else str(skill)
            outcomes = outcomes_by_skill[key]
            count = outcomes.get(outcome_key, 0)
            total = sum(outcomes.values())
            rate = count / total * 100 if total > 0 else 0.0
            lo, hi = wilson_ci(count, total)
            rates.append(rate)
            los.append(rate - lo * 100)
            his.append(hi * 100 - rate)
        return rates, [los, his]

    small_win_rates, small_win_err = series(small_outcomes, "1-0")
    big_win_rates, big_win_err = series(big_outcomes, "1-0")
    small_illegal_rates, _ = series(small_outcomes, "illegal_move")
    big_illegal_rates, _ = series(big_outcomes, "illegal_move")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.errorbar(skill_levels, small_win_rates, yerr=small_win_err, fmt="o-",
                 label=small_label, color="tab:blue", capsize=4)
    ax1.errorbar(skill_levels, big_win_rates, yerr=big_win_err, fmt="s-",
                 label=big_label, color="tab:red", capsize=4)
    ax1.set_xlabel("Stockfish skill level")
    ax1.set_ylabel("Win rate (%)")
    ax1.set_title(f"Win rate, 95% CI (n={games_per_level}/level)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(skill_levels, small_illegal_rates, "o-", label=small_label, color="tab:blue")
    ax2.plot(skill_levels, big_illegal_rates, "s-", label=big_label, color="tab:red")
    ax2.set_xlabel("Stockfish skill level")
    ax2.set_ylabel("Illegal-move ending rate (%)")
    ax2.set_title("Illegal-move rate by opponent strength")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def play_one_game(checkpoint_dir, skill_level, fallback_config):
    white = PlayerConfig(name="model", player_type="model", checkpoint_dir=checkpoint_dir)
    black = PlayerConfig(name="stockfish", player_type="stockfish",
                          stockfish_path=STOCKFISH_PATH, stockfish_skill_level=skill_level,
                          stockfish_time_limit=0.1)
    gen = play_game(white, black, DEVICE, fallback_config=fallback_config,
                     fallback_vocab_path=FALLBACK_VOCAB_PATH,
                     max_plies=MAX_PLIES, temperature=TEMPERATURE)
    result = None
    try:
        for event in gen:
            if event.game_over:
                result = event.result
                break
    finally:
        gen.close()
    return result


def compute_win_rates(checkpoint_dir, skill_levels, games_per_level, fallback_config):
    win_rates = []
    all_outcomes = {}
    for skill in skill_levels:
        outcomes = defaultdict(int)
        for _ in range(games_per_level):
            result = play_one_game(checkpoint_dir, skill, fallback_config)
            outcomes[str(result)] += 1
        wins = outcomes.get("1-0", 0)
        total = sum(outcomes.values())
        win_rate = wins / total * 100 if total > 0 else 0.0
        win_rates.append(win_rate)
        all_outcomes[skill] = dict(outcomes)
        print(f"  skill={skill}: {wins}/{total} wins ({win_rate:.1f}%) -- outcomes: {dict(outcomes)}")
    return win_rates, all_outcomes


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--from-json":
        json_path = sys.argv[2]
        with open(json_path) as f:
            data = json.load(f)
        # Support both this script's own {label: {outcomes_by_skill: ...}} shape
        # and the flat two-key shape (whichever two top-level keys aren't
        # skill_levels/games_per_skill_level).
        labels = [k for k in data if k not in ("skill_levels", "games_per_skill_level")]
        small_label, big_label = labels[0], labels[1]
        small_outcomes = data[small_label]["outcomes_by_skill"] if "outcomes_by_skill" in data[small_label] else data[small_label]
        big_outcomes = data[big_label]["outcomes_by_skill"] if "outcomes_by_skill" in data[big_label] else data[big_label]
        make_plot(data["skill_levels"], data["games_per_skill_level"],
                  small_label, big_label, small_outcomes, big_outcomes, OUTPUT_PLOT)
        return

    # Only used if a player has checkpoint_dir=None -- neither does in this
    # script, so this is never actually applied, just required by the
    # function signature.
    fallback_config = GPTConfig.load(f"{BIG_CHECKPOINT_DIR}/config.json")

    print(f"Running {SMALL_LABEL} ({SMALL_CHECKPOINT_DIR})...")
    small_win_rates, small_outcomes = compute_win_rates(
        SMALL_CHECKPOINT_DIR, SKILL_LEVELS, GAMES_PER_SKILL_LEVEL, fallback_config)

    print(f"Running {BIG_LABEL} ({BIG_CHECKPOINT_DIR})...")
    big_win_rates, big_outcomes = compute_win_rates(
        BIG_CHECKPOINT_DIR, SKILL_LEVELS, GAMES_PER_SKILL_LEVEL, fallback_config)

    os.makedirs(os.path.dirname(OUTPUT_RAW) or ".", exist_ok=True)
    with open(OUTPUT_RAW, "w") as f:
        json.dump({
            "skill_levels": SKILL_LEVELS,
            "games_per_skill_level": GAMES_PER_SKILL_LEVEL,
            SMALL_LABEL: {"win_rates": small_win_rates, "outcomes_by_skill": small_outcomes},
            BIG_LABEL: {"win_rates": big_win_rates, "outcomes_by_skill": big_outcomes},
        }, f, indent=2)
    print(f"Raw results saved to {OUTPUT_RAW}")

    make_plot(SKILL_LEVELS, GAMES_PER_SKILL_LEVEL, SMALL_LABEL, BIG_LABEL,
              small_outcomes, big_outcomes, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
