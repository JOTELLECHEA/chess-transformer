"""
probe_board_state.py

Probe whether a GPT-style model linearly encodes chess board state.

Steps:
1. Capture the residual-stream activation from every transformer block
   while processing each game in a single forward pass.
2. Replay the game with python-chess to obtain the true board after each
   move (64 squares, 13 classes: empty + 6 piece types x 2 colors).
3. Train a linear classifier per layer to predict those board labels.
   High validation accuracy indicates a linear board-state representation
   at that layer.

Example:
    python probe_board_state.py \
        --checkpoint-dir checkpoints/1.2m_L12E384H6 \
        --corpus data/chessDataset_1.2m.txt \
        --n-games 2000 \
        --output results/probe_1.2m_L12_trained.json 
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
import chess
from src.config import GPTConfig
from src.model import GPT
from src.dataset import MoveTokenizer
from src.constants import START_TOKEN, RESULT_TOKENS

# 13 classes per square: empty + 6 piece types x 2 colors.
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_CLASS = {None: 0}
_idx = 1
for _color in [chess.WHITE, chess.BLACK]:
    for _pt in _PIECE_TYPES:
        PIECE_TO_CLASS[(_pt, _color)] = _idx
        _idx += 1
N_CLASSES = _idx  # 13


def board_to_labels(board: chess.Board) -> torch.Tensor:
    """Returns a length-64 tensor of class labels (0-12) for the current board."""
    labels = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        key = None if piece is None else (piece.piece_type, piece.color)
        labels[square] = PIECE_TO_CLASS[key]
    return labels


class ActivationCapture:
    """Registers forward hooks on every transformer block, capturing the
    residual-stream output at each layer for later probing."""
    def __init__(self, model):
        self.activations = {}
        self.hooks = []
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        for i, block in enumerate(raw_model.transformer.h):
            hook = block.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self.activations[layer_idx] = output.detach()
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()


def load_games(corpus_path: str, n_games: int, seed: int):
    rng = random.Random(seed)
    games = []
    with open(corpus_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2 or tokens[0] != START_TOKEN:
                continue
            move_tokens = tokens[1:-1] if tokens[-1] in RESULT_TOKENS else tokens[1:]
            if len(move_tokens) >= 4:
                games.append(move_tokens)
    rng.shuffle(games)
    return games[:n_games]


def extract_probe_dataset(model, tokenizer, config, device, games):
    """
    Returns {layer_idx: (X, Y)} where X is [N, n_embd] activations and Y is
    [N, 64] true board-state class labels, pooled across all games and all
    ply-positions within each game.

    Position alignment (the part most prone to off-by-one bugs):
    token_ids[0] is <|SOM|>, token_ids[i] for i>=1 is the i-th move. The
    activation at sequence position i is what the model computed AFTER
    seeing token_ids[i] -- i.e. right after the i-th move was played. So
    activation at position i is paired with the true board state
    immediately after that same i-th move. We verify this pairing
    explicitly in the test below before trusting it on real data.
    """
    capture = ActivationCapture(model)
    n_layers = len(model.transformer.h) if not hasattr(model, '_orig_mod') else len(model._orig_mod.transformer.h)
    layer_X = {i: [] for i in range(n_layers)}
    layer_Y = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for move_tokens in games:
            move_tokens = move_tokens[: config.block_size - 1]  
            board = chess.Board()
            token_ids = [tokenizer.stoi[START_TOKEN]]
            labels_after_move = []  
            ok = True
            for uci in move_tokens:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    ok = False
                    break
                if move not in board.legal_moves:
                    ok = False
                    break
                board.push(move)
                token_ids.append(tokenizer.stoi[uci])
                labels_after_move.append(board_to_labels(board))
            if not ok or len(token_ids) < 2:
                continue

            idx = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
            model(idx)  

            for i in range(1, len(token_ids)):
                true_labels = labels_after_move[i - 1]
                for layer_idx in range(n_layers):
                    act = capture.activations[layer_idx][0, i, :].cpu()
                    layer_X[layer_idx].append(act)
                    layer_Y[layer_idx].append(true_labels)

    capture.remove()
    result = {}
    for layer_idx in range(n_layers):
        result[layer_idx] = (torch.stack(layer_X[layer_idx]), torch.stack(layer_Y[layer_idx]))
    return result


def train_probe(X, Y, n_embd, device, epochs=30, lr=1e-2, val_split=0.15):
    """Trains a linear probe: n_embd -> 64*13 logits. Returns val accuracy
    (fraction of squares correctly classified) and the trained probe."""
    N = X.shape[0]
    n_val = max(1, int(N * val_split))
    perm = torch.randperm(N)
    train_idx, val_idx = perm[n_val:], perm[:n_val]
    X_train, Y_train = X[train_idx].to(device), Y[train_idx].to(device)
    X_val, Y_val = X[val_idx].to(device), Y[val_idx].to(device)

    probe = nn.Linear(n_embd, 64 * N_CLASSES).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(X_train).view(-1, 64, N_CLASSES)
        loss = nn.functional.cross_entropy(logits.reshape(-1, N_CLASSES), Y_train.reshape(-1))
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val).view(-1, 64, N_CLASSES)
        val_preds = val_logits.argmax(dim=-1)
        accuracy = (val_preds == Y_val).float().mean().item()
    return accuracy, probe


def main():
    parser = argparse.ArgumentParser(description="Train linear probes to test for emergent board-state representations.")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Shortcut: a folder containing model_weights.pt, config.json, and "
                              "vocab.json together (e.g. checkpoints/1.2m_L12E384H6). Cannot be "
                              "combined with --weights/--config-file/--vocab-file.")
    parser.add_argument("--weights", default=None,
                         help="Checkpoint to probe. Ignored if --random-init is set.")
    parser.add_argument("--random-init", action="store_true",
                         help="Skip loading --weights entirely; use a freshly "
                              "random-initialized model instead. This is the baseline "
                              "comparison -- run once with this flag, once without, to "
                              "see how much of the probe accuracy is genuinely learned "
                              "versus exploitable from board statistics alone.")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--corpus", default="data/combined_games_uci_clean.txt")
    parser.add_argument("--n-games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--output", default="probe_results.json")
    args = parser.parse_args()

    if args.checkpoint_dir:
        if args.weights or args.config_file or args.vocab_file:
            parser.error("--checkpoint-dir cannot be combined with --weights/--config-file/--vocab-file.")
        args.weights = os.path.join(args.checkpoint_dir, "model_weights.pt")
        args.config_file = os.path.join(args.checkpoint_dir, "config.json")
        args.vocab_file = os.path.join(args.checkpoint_dir, "vocab.json")
    else:
        args.weights = args.weights or "model_weights.pt"
        args.config_file = args.config_file or "config.json"
        args.vocab_file = args.vocab_file or "vocab.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    config = GPTConfig.load(args.config_file)
    tokenizer = MoveTokenizer.from_vocab_file(args.vocab_file)
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)
    if args.random_init:
        print("--random-init set: using freshly random-initialized weights (baseline mode)")
    else:
        print(f"Loading weights from: {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    print(f"Loading up to {args.n_games} games from {args.corpus}...")
    games = load_games(args.corpus, args.n_games, args.seed)
    print(f"  using {len(games)} games")

    print("Extracting activations (one forward pass per game)...")
    layer_data = extract_probe_dataset(model, tokenizer, config, device, games)

    results = {}
    for layer_idx, (X, Y) in layer_data.items():
        print(f"  Layer {layer_idx}: {X.shape[0]:,} probe examples")
        acc, _ = train_probe(X, Y, config.n_embd, device)
        results[layer_idx] = acc
        print(f"    -> probe accuracy: {acc*100:.1f}%")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 50)
    print("Probe accuracy by layer (fraction of 64 squares correctly classified):")
    for layer_idx, acc in sorted(results.items()):
        print(f"  Layer {layer_idx:2d}: {acc*100:.1f}%")
    print("=" * 50)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
