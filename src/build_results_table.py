"""
Scans checkpoints/ and prints a markdown table with
architecture, corpus, epochs trained, and final val_loss -- pulled directly
from each checkpoint's own saved training_checkpoint.pt.

Legal-move-rate / fully-legal-games columns are loaded from the eval_results.json file.

Usage:
python -m src.build_results_table
"""
import os
import json
import torch


CHECKPOINTS_DIR = "checkpoints"

rows = []
for name in sorted(os.listdir(CHECKPOINTS_DIR)):
    folder = os.path.join(CHECKPOINTS_DIR, name)
    config_path = os.path.join(folder, "config.json")
    ckpt_path = os.path.join(folder, "training_checkpoint.pt")
    eval_path = os.path.join(folder, "eval_results.json")
    if not (os.path.exists(config_path) and os.path.exists(ckpt_path)):
        continue

    config = json.load(open(config_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt["history"]

    eval_results = json.load(open(eval_path)) if os.path.exists(eval_path) else None

    rows.append({
        "name": name,
        "n_layer": config["n_layer"],
        "n_embd": config["n_embd"],
        "n_head": config["n_head"],
        "epochs_trained": len(history),
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "legal_move_rate_pct": eval_results["legal_move_rate_pct"] if eval_results else None,
        "fully_legal_pct": eval_results["fully_legal_pct"] if eval_results else None,
        "n_games_evaluated": eval_results["n_games"] if eval_results else None,
    })

print(f"| Checkpoint | Architecture | Epochs | Final Val Loss | Legal-move rate | Fully-legal games |")
print(f"|---|---|---|---|---|---|")
for r in rows:
    arch = f"{r['n_layer']}L / {r['n_embd']}E / {r['n_head']}H"
    val_loss = f"{r['final_val_loss']:.4f}" if r['final_val_loss'] is not None else "?"
    if r["legal_move_rate_pct"] is not None:
        legal_rate = f"{r['legal_move_rate_pct']}%"
        fully_legal = f"{r['fully_legal_pct']}% (n={r['n_games_evaluated']})"
    else:
        legal_rate = "_run eval --save-results_"
        fully_legal = "_run eval --save-results_"
    print(f"| `{r['name']}` | {arch} | {r['epochs_trained']} | {val_loss} | {legal_rate} | {fully_legal} |")
