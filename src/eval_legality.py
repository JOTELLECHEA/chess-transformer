"""
src/eval_legality.py
Evaluates a trained model's legal-move rate by generating games and checking each move
against a real chess.Board() to see if it was legal in context.


usage: eval_legality.py [-h] [--checkpoint-dir CHECKPOINT_DIR] [--weights WEIGHTS] [--vocab-file VOCAB_FILE]
                        [--config-file CONFIG_FILE] [--n-games N_GAMES] [--max-new-tokens MAX_NEW_TOKENS] [--temperature TEMPERATURE]
                        [--save-results [SAVE_RESULTS]]

example usage:
    python -m src.eval_legality --checkpoint-dir checkpoints/1.2m_L6E256H4 --n-games 500

"""
import argparse
import os
import json
import torch
import chess
from src.config import GPTConfig
from src.model import GPT
from src.dataset import MoveTokenizer
from src.constants import RESULT_TOKENS, START_TOKEN


def generate_game(model, tokenizer, prompt: str, max_new_tokens: int, block_size: int,
                   device: str, temperature: float = 1.0) -> str:
    """Runs the generation loop, stopping early if a result token is sampled."""
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    result_token_ids = {tokenizer.stoi[t] for t in RESULT_TOKENS if t in tokenizer.stoi}

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits.squeeze(1) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() in result_token_ids:
                break

    return tokenizer.decode(idx[0].tolist())

def check_game_legality(game_text: str):
    """
    Replays a generated game's move tokens against a real chess.Board(),
    checking each one for legality given the actual current position.

    Returns a dict with:
        total_moves       - how many move tokens were generated (excluding SOM/result)
        legal_moves        - how many of those were legal in context
        first_illegal_ply   - ply number (1-indexed) of the first illegal move, or None
        self_terminated     - True if the game ended with a real result token
                               (as opposed to hitting max_new_tokens with no result)
    """
    tokens = game_text.strip().split()

    if not tokens or tokens[0] != START_TOKEN:
        return None

    self_terminated = tokens[-1] in RESULT_TOKENS
    move_tokens = tokens[1:-1] if self_terminated else tokens[1:]

    board = chess.Board()
    legal_count = 0
    first_illegal_ply = None

    for ply, uci_str in enumerate(move_tokens, start=1):
        try:
            move = chess.Move.from_uci(uci_str)
        except ValueError:
            # Malformed UCI string entirely (shouldn't happen — vocab is closed
            # to valid square-pair strings — but guard against it anyway).
            if first_illegal_ply is None:
                first_illegal_ply = ply
            continue

        if move in board.legal_moves:
            legal_count += 1
            board.push(move)
        else:
            if first_illegal_ply is None:
                first_illegal_ply = ply
            # Stop replaying further — once a move is illegal, the board state
            # from here on is undefined, so counting subsequent tokens as
            # "legal" or "illegal" against a board that no longer reflects
            # what actually happened would be meaningless.
            break

    return {
        "total_moves": len(move_tokens),
        "checked_moves": legal_count + (1 if first_illegal_ply is not None else 0),
        "legal_moves": legal_count,
        "first_illegal_ply": first_illegal_ply,
        "self_terminated": self_terminated,
    }

def run_eval(model, tokenizer, config, device,
             n_games: int = 50, max_new_tokens: int = 250, temperature: float = 1.0):
    results = []
    for i in range(n_games):
        game_text = generate_game(
            model=model,
            tokenizer=tokenizer,
            prompt=START_TOKEN,
            max_new_tokens=max_new_tokens,
            block_size=config.block_size,
            device=device,
            temperature=temperature,
        )
        result = check_game_legality(game_text)
        if result is not None:
            results.append(result)
        print(f"  game {i+1}/{n_games}: "
              f"{result['legal_moves']}/{result['checked_moves']} legal moves (of moves checked)"
              + (f", first illegal at ply {result['first_illegal_ply']}" if result['first_illegal_ply'] else ", fully legal")
              + (", self-terminated" if result['self_terminated'] else ", hit token ceiling")
              + f" [{result['total_moves']} total tokens generated]")

    # ---- aggregate stats ----
    total_moves_generated = sum(r["total_moves"] for r in results)
    total_moves_checked = sum(r["checked_moves"] for r in results)
    total_legal_moves = sum(r["legal_moves"] for r in results)
    fully_legal_games = sum(1 for r in results if r["first_illegal_ply"] is None)
    self_terminated_games = sum(1 for r in results if r["self_terminated"])
    first_illegal_plies = [r["first_illegal_ply"] for r in results if r["first_illegal_ply"] is not None]

    print("\n" + "=" * 60)
    print(f"Games generated:                  {len(results)}")
    print(f"Fully legal games (no illegal move): {fully_legal_games} ({fully_legal_games/len(results)*100:.1f}%)")
    print(f"Self-terminated (real result token): {self_terminated_games} ({self_terminated_games/len(results)*100:.1f}%)")
    print()
    print("Legal-move rate (of moves actually checked -- i.e. up to and")
    print("including each game's first illegal move, or the whole game if none):")
    print(f"  {total_legal_moves:,} / {total_moves_checked:,} moves legal "
          f"({total_legal_moves/total_moves_checked*100:.1f}%)")
    print()
    print(f"(For reference, total raw tokens generated across all games -- including")
    print(f" tokens after a game's first illegal move, which are never checked -- was")
    print(f" {total_moves_generated:,}. This number is NOT used in the ratio above,")
    print(f" since checking correctly stops at the first illegal move.)")

    summary = {
        "n_games": len(results),
        "fully_legal_games": fully_legal_games,
        "fully_legal_pct": round(fully_legal_games / len(results) * 100, 2),
        "self_terminated_games": self_terminated_games,
        "total_legal_moves": total_legal_moves,
        "total_moves_checked": total_moves_checked,
        "legal_move_rate_pct": round(total_legal_moves / total_moves_checked * 100, 2),
        "total_moves_generated": total_moves_generated,
    }

    if first_illegal_plies:
        first_illegal_plies.sort()
        n = len(first_illegal_plies)
        print()
        print(f"Of the {n} games with an illegal move, first-illegal-move ply:")
        print(f"  min: {first_illegal_plies[0]}   "
              f"median: {first_illegal_plies[n//2]}   "
              f"max: {first_illegal_plies[-1]}")
        summary["first_illegal_ply_min"] = first_illegal_plies[0]
        summary["first_illegal_ply_median"] = first_illegal_plies[n//2]
        summary["first_illegal_ply_max"] = first_illegal_plies[-1]
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate legal-move rate of a trained checkpoint.")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="Shortcut: a folder containing model_weights.pt, config.json, and "
                              "vocab.json together (e.g. checkpoints/1.2m_L6E256H4). Cannot be "
                              "combined with --weights/--config-file/--vocab-file.")
    parser.add_argument("--weights", default=None,
                         help="Which checkpoint to evaluate. Defaults to model_weights.pt if "
                              "neither this nor --checkpoint-dir is given.")
    parser.add_argument("--vocab-file", default=None)
    parser.add_argument("--config-file", default=None,
                         help="Which saved config to reconstruct the architecture from -- "
                              "must match whatever config produced --weights.")
    parser.add_argument("--n-games", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save-results", nargs="?", const="__derive__", default=None,
                         help="Save results as JSON. Give a path explicitly, or use bare "
                              "--save-results with --checkpoint-dir to save as "
                              "eval_results.json inside that same checkpoint folder.")
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

    if args.save_results == "__derive__" and not args.checkpoint_dir:
        parser.error("Bare --save-results requires --checkpoint-dir, so the save "
                      "location can be derived. Pass an explicit path otherwise.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eval engine running on: {device}")
    print(f"Loading weights from: {args.weights}")

    config = GPTConfig.load(args.config_file)

    tokenizer = MoveTokenizer.from_vocab_file(args.vocab_file)
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    summary = run_eval(model, tokenizer, config, device,
                        n_games=args.n_games, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    if args.save_results:
        save_path = os.path.join(args.checkpoint_dir, "eval_results.json") if args.save_results == "__derive__" else args.save_results
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
