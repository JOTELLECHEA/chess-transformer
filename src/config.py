# src/config.py
from dataclasses import dataclass, asdict
import json

@dataclass
class GPTConfig:
    # Training Loop Iterations
    max_epochs: int = 5 
    learning_rate: float = 5e-4
    seed: int = 137
    
    # Dataset / Data Loading
    block_size: int = 192
    batch_size: int = 256
    
    # Architecture Structure
    vocab_size: int = 1973
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.2

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GPTConfig":
        with open(path) as f:
            return cls(**json.load(f))
