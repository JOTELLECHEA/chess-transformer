import json
import torch
import chess
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

from src.constants import START_TOKEN, RESULT_TOKENS


def _build_theoretical_uci_space():
    """Every UCI string producible for any piece/square/promotion combo --
    same logic as analyze_vocab_gap.py's build_theoretical_space(), reused
    here so the closed-form vocab and the coverage-analysis tool never drift
    apart from each other."""
    all_uci = set()

    def add(frm, to, promo=None):
        all_uci.add(chess.Move(frm, to, promotion=promo).uci())

    for from_sq in chess.SQUARES:
        f_file, f_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
        for df, dr in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            for dist in range(1, 8):
                tf, tr = f_file + df*dist, f_rank + dr*dist
                if 0 <= tf < 8 and 0 <= tr < 8:
                    add(from_sq, chess.square(tf, tr))
        for df, dr in [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]:
            tf, tr = f_file + df, f_rank + dr
            if 0 <= tf < 8 and 0 <= tr < 8:
                add(from_sq, chess.square(tf, tr))

    promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    for f_file in range(8):
        from_sq = chess.square(f_file, 6)
        for df in (-1, 0, 1):
            tf = f_file + df
            if 0 <= tf < 8:
                for p in promo_pieces:
                    add(from_sq, chess.square(tf, 7), p)
        from_sq = chess.square(f_file, 1)
        for df in (-1, 0, 1):
            tf = f_file + df
            if 0 <= tf < 8:
                for p in promo_pieces:
                    add(from_sq, chess.square(tf, 0), p)
    return all_uci


class MoveTokenizer:
    def __init__(self, text: Optional[str] = None, vocab: Optional[List[str]] = None):
        if vocab is not None:
            vocab_list = vocab
            self.text = text
        else:
            if text is None:
                raise ValueError("MoveTokenizer requires either `text` or `vocab`.")
            self.text = text
            vocab_list = sorted(set(text.split()))
        self.vocab_size = len(vocab_list)
        self.stoi = {s: i for i, s in enumerate(vocab_list)}
        self.itos = {i: s for i, s in enumerate(vocab_list)}

    @classmethod
    def build_closed_form(cls) -> "MoveTokenizer":
        """Builds a FIXED vocab from the full theoretical UCI move space plus
        the special tokens -- independent of any specific corpus. This means
        the vocab (and therefore every token's index) stays identical across
        any future corpus or retrain, permanently, rather than shifting
        based on which moves happened to appear in a given dataset."""
        move_space = _build_theoretical_uci_space()
        full_vocab = sorted(move_space | RESULT_TOKENS | {START_TOKEN})
        return cls(vocab=full_vocab)

    def save_vocab(self, path: str) -> None:
        vocab_list = [self.itos[i] for i in range(self.vocab_size)]
        with open(path, 'w') as f:
            json.dump(vocab_list, f)

    @classmethod
    def from_vocab_file(cls, path: str) -> "MoveTokenizer":
        with open(path, 'r') as f:
            vocab_list = json.load(f)
        return cls(vocab=vocab_list)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[s] for s in text.split()]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join([self.itos[i] for i in tokens])


class GPTDataset(Dataset):
    def __init__(self, corpus, block_size):
        self.data = torch.tensor(corpus, dtype=torch.long)
        self.block_size = block_size
    def __len__(self):
        return max(1, len(self.data) - self.block_size)
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def prepare_datasets(file_path, block_size, batch_size, split_ratio=0.9, tokenizer=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if tokenizer is None:
        tokenizer = MoveTokenizer(text)
    encoded_text = tokenizer.encode(text)
    n = int(split_ratio * len(encoded_text))
    train_data = encoded_text[:n]
    val_data = encoded_text[n:] if encoded_text[n:] else encoded_text[-block_size-1:]
    train_dataset = GPTDataset(train_data, block_size)
    val_dataset = GPTDataset(val_data, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
