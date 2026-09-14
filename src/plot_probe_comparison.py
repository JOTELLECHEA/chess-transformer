"""
plot_probe_comparison.py
Plots the layer-by-layer probe accuracy comparison between two checkpoints
(e.g. the small and big model), reading directly from the JSON files
produced by probe_board_state.py -- never hardcodes numbers, so the plot
always reflects whatever your saved results actually say.
Two panels:
    Left:  raw probe accuracy vs. RELATIVE depth (layer_index / (n_layer-1),
           so a 6-layer and 12-layer model's first layer both sit at x=0
           and last layer both sit at x=1 -- the fair way to overlay
           architectures of different depths on one axis), trained and
           random-baseline shown for each model.
    Right: "headroom captured" = (trained - random) / (100 - random), the
           genuinely-learned signal with the random floor subtracted out --
           this is the cleaner, single-line-per-model view of whether a
           model has saturated (plateaued) or is still improving when it
           runs out of layers.
Usage:
    python -m src.plot_probe_comparison \
        --small-trained results/probe_98k_L6E256H4.json \
        --small-random results/probe_98k_L6E256H4_random.json \
        --big-trained results/probe_1.2m_L12E384H6.json \
        --big-random results/probe_1.2m_L12E384H6_random.json \
        --small-label "98k / 6 layers" \
        --big-label "1.2m / 12 layers" \
        --output plots/probe_comparison.png
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


def load_layer_accuracies(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    layer_acc = {int(k): v for k, v in raw.items()}
    sorted_layers = sorted(layer_acc.keys())
    accuracies = [layer_acc[l] * 100 if layer_acc[l] <= 1.0 else layer_acc[l] for l in sorted_layers]
    return sorted_layers, accuracies


def relative_depth(layer_indices):
    n = len(layer_indices)
    if n == 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


def headroom_captured(trained, random_baseline):
    return [
        (t - r) / (100 - r) * 100 if r < 100 else 0.0
        for t, r in zip(trained, random_baseline)
    ]


def main():
    parser = argparse.ArgumentParser(description="Plot trained-vs-random probe accuracy comparison across two models.")
    parser.add_argument("--results-dir", default=None,
                         help="Shortcut: a folder containing probe_{name}_trained.json and "
                              "probe_{name}_random.json for both --small-checkpoint and "
                              "--big-checkpoint. Cannot be combined with the explicit "
                              "--small-trained/--small-random/--big-trained/--big-random flags.")
    parser.add_argument("--small-checkpoint", default=None,
                         help="Name used in the results-dir filename convention, e.g. '1.2m_L6'.")
    parser.add_argument("--big-checkpoint", default=None,
                         help="Name used in the results-dir filename convention, e.g. '1.2m_L12'.")
    parser.add_argument("--small-trained", default=None)
    parser.add_argument("--small-random", default=None)
    parser.add_argument("--big-trained", default=None)
    parser.add_argument("--big-random", default=None)
    parser.add_argument("--small-label", default="Small model")
    parser.add_argument("--big-label", default="Big model")
    parser.add_argument("--output", default="plots/probe_comparison.png")
    args = parser.parse_args()

    explicit_flags_given = any([args.small_trained, args.small_random, args.big_trained, args.big_random])
    if args.results_dir:
        if explicit_flags_given:
            parser.error("--results-dir cannot be combined with --small-trained/--small-random/"
                          "--big-trained/--big-random.")
        if not (args.small_checkpoint and args.big_checkpoint):
            parser.error("--results-dir requires both --small-checkpoint and --big-checkpoint.")
        args.small_trained = os.path.join(args.results_dir, f"probe_{args.small_checkpoint}_trained.json")
        args.small_random = os.path.join(args.results_dir, f"probe_{args.small_checkpoint}_random.json")
        args.big_trained = os.path.join(args.results_dir, f"probe_{args.big_checkpoint}_trained.json")
        args.big_random = os.path.join(args.results_dir, f"probe_{args.big_checkpoint}_random.json")
    elif not explicit_flags_given:
        parser.error("Provide either --results-dir with --small-checkpoint/--big-checkpoint, "
                      "or all four of --small-trained/--small-random/--big-trained/--big-random.")
    elif not all([args.small_trained, args.small_random, args.big_trained, args.big_random]):
        parser.error("All four of --small-trained/--small-random/--big-trained/--big-random "
                      "are required when not using --results-dir.")

    small_layers, small_trained_acc = load_layer_accuracies(args.small_trained)
    _, small_random_acc = load_layer_accuracies(args.small_random)
    big_layers, big_trained_acc = load_layer_accuracies(args.big_trained)
    _, big_random_acc = load_layer_accuracies(args.big_random)

    small_x = relative_depth(small_layers)
    big_x = relative_depth(big_layers)

    small_headroom = headroom_captured(small_trained_acc, small_random_acc)
    big_headroom = headroom_captured(big_trained_acc, big_random_acc)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(small_x, small_trained_acc, "o-", color="tab:blue", label=f"{args.small_label} (trained)")
    ax1.plot(small_x, small_random_acc, "o--", color="tab:blue", alpha=0.4, label=f"{args.small_label} (random)")
    ax1.plot(big_x, big_trained_acc, "s-", color="tab:red", label=f"{args.big_label} (trained)")
    ax1.plot(big_x, big_random_acc, "s--", color="tab:red", alpha=0.4, label=f"{args.big_label} (random)")
    ax1.set_xlabel("Relative depth (0 = first layer, 1 = last layer)")
    ax1.set_ylabel("Probe accuracy (%)")
    ax1.set_title("Board-state probe accuracy by depth")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(small_x, small_headroom, "o-", color="tab:blue", label=args.small_label)
    ax2.plot(big_x, big_headroom, "s-", color="tab:red", label=args.big_label)
    ax2.set_xlabel("Relative depth (0 = first layer, 1 = last layer)")
    ax2.set_ylabel("Headroom captured (%)\n[(trained - random) / (100 - random)]")
    ax2.set_title("Genuinely-learned signal, random floor subtracted out")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    fig.suptitle("Internal board-state representation: does the model saturate before running out of layers?")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    print()
    print(f"{args.small_label}: headroom captured goes from {small_headroom[0]:.1f}% to {small_headroom[-1]:.1f}% "
          f"({'still rising' if small_headroom[-1] > small_headroom[-2] else 'flat/plateaued'} at last layer)")
    print(f"{args.big_label}:   headroom captured goes from {big_headroom[0]:.1f}% to {big_headroom[-1]:.1f}% "
          f"({'still rising' if big_headroom[-1] > big_headroom[-2] else 'flat/plateaued'} at last layer)")


if __name__ == "__main__":
    main()
