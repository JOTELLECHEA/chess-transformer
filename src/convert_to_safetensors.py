#!/usr/bin/env python3
"""
convert_to_safetensors.py

Converts each checkpoint's model_weights.pt into a .safetensors file, ready
for upload to HuggingFace.

Usage:
    # all four checkpoints
    python -m src.convert_to_safetensors --checkpoints-dir checkpoints --output-dir hf_upload

    # a single checkpoint
    python -m src.convert_to_safetensors --checkpoint-dir checkpoints/1.2m_L12E384H6 --output-dir hf_upload
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_model, load_model

from src.config import GPTConfig
from src.model import GPT
from src.dataset import MoveTokenizer


def convert_one(checkpoint_dir: Path, output_dir: Path, verify: bool = True) -> bool:
    """Converts one checkpoint. Returns True on success."""
    name = checkpoint_dir.name
    weights_path = checkpoint_dir / "model_weights.pt"
    config_path = checkpoint_dir / "config.json"
    vocab_path = checkpoint_dir / "vocab.json"

    for required in (weights_path, config_path, vocab_path):
        if not required.exists():
            print(f"  SKIP {name}: missing {required.name}")
            return False

    config = GPTConfig.load(str(config_path))
    tokenizer = MoveTokenizer.from_vocab_file(str(vocab_path))
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config)
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()

    dest_dir = output_dir / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = dest_dir / "model.safetensors"

    # save_model(), NOT save_file(model.state_dict(), ...) -- see module docstring.
    save_model(model, str(safetensors_path))

    # config.json and vocab.json go alongside the weights, so a downloader
    # gets everything needed to reconstruct the model from one place.
    shutil.copy(str(config_path), str(dest_dir / "config.json"))
    shutil.copy(str(vocab_path), str(dest_dir / "vocab.json"))

    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"  {name}: {size_mb:.1f} MB -> {safetensors_path}")

    if verify:
        reloaded = GPT(config)
        load_model(reloaded, str(safetensors_path))
        reloaded.eval()

        original_sd = model.state_dict()
        reloaded_sd = reloaded.state_dict()

        if original_sd.keys() != reloaded_sd.keys():
            print(f"    VERIFY FAILED: key mismatch after reload")
            return False
        for key in original_sd:
            if not torch.equal(original_sd[key], reloaded_sd[key]):
                print(f"    VERIFY FAILED: tensor '{key}' differs after reload")
                return False

        # The tie must be reconstructed, not just duplicated -- check the
        # two tensors are the same object, as they are in a fresh model.
        tie_intact = reloaded.transformer.wte.weight is reloaded.lm_head.weight
        if not tie_intact:
            print(f"    VERIFY FAILED: wte/lm_head tie not reconstructed on load")
            return False

        print(f"    verified: all tensors identical, weight tie intact")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint weights from .pt to .safetensors for HuggingFace upload."
    )
    parser.add_argument("--checkpoints-dir", default=None,
                        help="Convert every subfolder containing a model_weights.pt.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Convert a single checkpoint folder. Mutually exclusive with --checkpoints-dir.")
    parser.add_argument("--output-dir", default="hf_upload",
                        help="Where the converted files are written. One subfolder per checkpoint.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the reload-and-compare check. Not recommended -- the verification "
                             "is what confirms the tied-weight handling actually worked.")
    args = parser.parse_args()

    if args.checkpoints_dir and args.checkpoint_dir:
        sys.exit("Pass either --checkpoints-dir or --checkpoint-dir, not both.")
    if not args.checkpoints_dir and not args.checkpoint_dir:
        args.checkpoints_dir = "checkpoints"

    output_dir = Path(args.output_dir)

    if args.checkpoint_dir:
        targets = [Path(args.checkpoint_dir)]
    else:
        root = Path(args.checkpoints_dir)
        if not root.exists():
            sys.exit(f"No such directory: {root}")
        targets = sorted(d for d in root.iterdir() if d.is_dir() and (d / "model_weights.pt").exists())
        if not targets:
            sys.exit(f"No checkpoints with a model_weights.pt found under {root}")

    print(f"Converting {len(targets)} checkpoint(s) to {output_dir.resolve()}")
    print()

    succeeded = 0
    for checkpoint_dir in targets:
        if convert_one(checkpoint_dir, output_dir, verify=not args.no_verify):
            succeeded += 1

    print()
    print("=" * 60)
    print(f"Done: {succeeded}/{len(targets)} converted")
    print("=" * 60)

    if succeeded < len(targets):
        sys.exit(1)


if __name__ == "__main__":
    main()
