import argparse
import torch
from src.config import GPTConfig
from src.model import GPT
from src.dataset import MoveTokenizer
from src.constants import START_TOKEN, RESULT_TOKENS

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, block_size: int,
                   device: str, temperature: float = 1.0) -> str:
    """
    Runs the generation loop token-by-token using multinomial sampling.
    Stops early if a result token is sampled (a game genuinely ended),
    rather than always running to max_new_tokens and risking a second
    fabricated game getting glued onto the first.
    
    python -m src.inference \
    --weights checkpoints/1.2m_L12E384H6/model_weights.pt \
    --config-file checkpoints/1.2m_L12E384H6/config.json \
    --vocab-file checkpoints/1.2m_L12E384H6/vocab.json \
    --prompt "<|SOM|> d2d4 g8f6"
    
    """

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

def main():
    parser = argparse.ArgumentParser(description="Generate a chess game from a trained checkpoint.")
    parser.add_argument("--weights", default="model_weights.pt",
                         help="Which checkpoint to load -- this is the switch between "
                              "base and DPO models.")
    parser.add_argument("--config-file", default="config.json",
                         help="Which saved config to reconstruct the architecture from -- "
                              "must match whatever config produced --weights.")
    parser.add_argument("--vocab-file", default="vocab.json")
    parser.add_argument("--prompt", default="<|SOM|> e2e4 e7e5",
                         help="Seed sequence to continue from. Must start with <|SOM|>.")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference engine running on: {device}")
    print(f"Loading weights from: {args.weights}")

    config = GPTConfig.load(args.config_file)

    tokenizer = MoveTokenizer.from_vocab_file(args.vocab_file)
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    print(f"\n--- Generating from: '{args.prompt}' ---")
    generated_game = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        block_size=config.block_size,
        device=device,
        temperature=args.temperature,
    )
    print(generated_game)
    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    main()
