# chess-transformer

A GPT-style transformer trained to predict chess moves from UCI-tokenized move sequences — and, more specifically, a controlled investigation into *why* such models produce illegal moves, and what actually fixes it.

### This repo is in progress; comments and write-up are being actively worked on.

This project trains on data derived from the [Lichess open database](https://database.lichess.org), released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Games were filtered to GM-titled players across standard time controls and converted to UCI notation via the pipeline in `src/`.

`vocab_fixed.json` is the canonical, closed-form move vocabulary; every model in this repo trains against this exact file.

---

## Motivation

This project began as a straightforward tokenizer-and-training exercise, extending prior character-level language modeling work into the chess domain. What started as an engineering exercise evolved into a controlled empirical investigation once early evaluation revealed specific, reproducible failure modes — prompting a systematic comparison of data scale, preference optimization, and model capacity as candidate explanations for why the model occasionally produces illegal moves, and ultimately a mechanistic investigation into whether the model's internal representations explain its behavioral improvements.

The tokenization choice made early on turned out to matter a lot: moves are encoded in **UCI notation** (`e2e4`, not `Nf3`), which — unlike SAN — carries no explicit piece-identity information. The model has to *infer* what piece is moving from move history alone. This is what makes "does the model track the board state internally" a genuine, non-trivial question rather than something handed to it for free.

<!-- ## Key findings -->

<!-- | Experiment | Legal-move rate | Fully-legal games | Result | -->
<!-- |---|---|---|---|
| Baseline (98k games, 6-layer) | 97.2% | 8% | — |
| 12.5x more data, same architecture | 97.6% | 10% | No significant change |
| DPO (naive, no legal/quality split) | — | — | **Regression** — diagnosed and fixed |
| DPO (illegal/quality beta split) | 97.3% | 8% | Regression fixed; legality unchanged from baseline |
| **12.5x more data + 4.2x more parameters** | **98.9%** | **42%** | **Statistically significant improvement** |
| Same big architecture, small (98k) corpus | 97.5-97.8% | 6-14% | Interaction effect — capacity alone isn't sufficient without data |
| DPO on the capacity-scaled model | 98.9% | 42% | Legality preserved; quality improvement suggestive, not yet statistically confirmed | -->

<!-- **The headline result:** across three candidate explanations for illegal-move generation — more data, preference optimization (DPO), and model capacity — only capacity produced a statistically significant improvement, and only when paired with sufficient data. Neither lever alone was sufficient.

**Mechanistic confirmation, via linear probing:** following the methodology of Li et al.'s OthelloGPT work and its chess-specific extensions, we trained linear probes to test whether the model's internal activations linearly encode true board state. The smaller (6-layer) architecture's probe accuracy is *still rising* at its final layer — it runs out of depth before finishing the job. The larger (12-layer) architecture's probe accuracy *plateaus* three layers before the end — it has genuine spare capacity. This is a direct, measured account of *why* the capacity intervention worked, not just confirmation that it did.

**A real regression, found and fixed:** an early DPO attempt, without separating "illegal move" preference pairs from "legal but suboptimal move" pairs, caused a measurable *decrease* in legal-move rate — the model was, in effect, punished for generating a move at all whenever its sampled move happened to be illegal, without a compensating signal that legal moves were the actual goal. Splitting the DPO loss into separate `beta` terms for illegal-move pairs (strong penalty) and quality pairs (gentle preference) resolved the regression, validated across two independently-trained architectures.

### `⚠️ Pending: big-architecture (12-layer) / full-corpus (1.2M games) multi-epoch run`
*Currently training on a rented A100 via Google Colab, testing whether the capacity-scaled model's validation loss continues improving past epoch 1 (the original 98.9%-legal result only ever tested a single epoch) or begins overfitting the way the same architecture did on the smaller corpus. Results will be added here once the run completes.* -->

<!-- ## Related work

This project replicates and extends an established line of interpretability research, rather than introducing a novel finding of its own:

- Li, Wu, Nathan, Andreas, Belinkov, Bau — [Emergent World Representations](https://arxiv.org/abs/2210.13382) (OthelloGPT, ICLR 2023) — the foundational demonstration that a transformer trained purely on next-move prediction, with no explicit rules or board supervision, develops an internal board-state representation.
- Nanda, Lee, Wattenberg — [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941) — showed the OthelloGPT representation is *linearly* decodable, and causally used by the model, not just incidentally present.
- Karvonen — [Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498) (COLM 2024) — extends the same methodology to chess specifically.
- Rafailov, Sharma, Mitchell, Ermon, Manning, Finn — [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — the DPO formulation used for the preference-tuning experiments here. -->

## Reproducing this work

```bash
pip install -r requirements.txt
# Stockfish is a separate system dependency, required for DPO pair generation:
# apt install stockfish  (or your platform's equivalent)

# Build the vocabulary (one-time, fixed forever after):
python -m src.build_vocab --closed-form --output vocab_fixed.json

# Train:
python train.py

# Evaluate legality:
python -m src.eval_legality --weights <path> --config-file <path> --vocab-file <path>
```

See `src/README.md` for the full pipeline breakdown and script-by-script documentation.

## License

Code, weights, and the closed-form tokenizer are released under the MIT License (see `LICENSE`). Training data is derived from Lichess's CC0-licensed database, per the attribution above.