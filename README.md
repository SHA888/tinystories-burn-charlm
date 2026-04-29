# tinystories-burn-charlm

Character-level transformer language model in Rust, trained end-to-end on a
single CPU using [Burn], on the [TinyStories] dataset, with the resulting
weights published as a `safetensors` artifact on the Hugging Face Hub.

> **Status:** `v0.0.1` — Phase 0 complete. Workspace skeleton builds clean,
> CI configured, no model code yet. See [`TODO.md`](./TODO.md) for the path
> to `v0.1.0`.

## Why this exists

A learning artifact that exercises the full Rust ML loop end-to-end:
dataset → tokenizer → model → training → checkpoint → publish. The point is
to **own** every step, not to reach SOTA. Once the loop is understood, the
same scaffold can be retargeted at domain-specific data without re-learning
the mechanics.

## Goals

- Single-machine, single-CPU training run completes overnight (~hours).
- ~1M parameters, 128-token context, character-level vocabulary.
- Reproducible: pinned dataset revision, pinned seed, pinned dependency
  versions via `Cargo.lock`.
- Publishable artifact: `model.safetensors` + `config.json` + model card
  with valid YAML frontmatter, loadable from Rust (Burn) and Python
  (`safetensors` lib) without modification.

## Non-goals

- **Not** competitive with PyTorch-trained small LMs on perplexity. CPU and
  Burn's young training stack both bound the achievable quality.
- **Not** a proof that Rust is a better training language than Python at
  this scale. It isn't, and that isn't the question being asked.
- **Not** a multi-GPU, distributed, or AMP-optimized training harness.
  YAGNI applies hard at this scope.

## Workspace

```
tinystories-burn-charlm/
├── crates/
│   ├── model-core/   # lib  — model, config, ports (traits)
│   ├── data/         # lib  — dataset pull, tokenizer, batching
│   ├── train/        # bin  — training loop, checkpointing
│   └── infer/        # bin  — load checkpoint, sample text
├── docs/
├── ARCHITECTURE.md
├── DECISIONS.md      # consolidated ADRs (split per-file when project grows)
├── TODO.md
└── README.md
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the ports-and-adapters layout
and the rationale for splitting four crates at this scope.

## Quick start (planned — not yet runnable)

**MSRV:** Rust `1.89` (enforced by `rust-toolchain.toml`).

```bash
# one-time tooling
bash scripts/setup.sh

# pull dataset, train, generate
cargo run -p data    --release -- prepare
cargo run -p train   --release -- --config configs/charlm-1m.toml
cargo run -p infer   --release -- --prompt "Once upon a time"
```

## Licensing

- **Code:** dual `MIT OR Apache-2.0` (Rust ecosystem default; downstream
  picks). See [`LICENSE-MIT`](./LICENSE-MIT) and
  [`LICENSE-APACHE`](./LICENSE-APACHE).
- **Trained weights:** `Apache-2.0` (declared in the model card YAML
  frontmatter when published). No use restrictions.
- **Dataset:** TinyStories — license verified at pull time before
  redistributing any derived artifacts. The trained weights are a
  derivative work; check carefully before relicensing more permissively
  than the source.

## Acknowledgements

- [Burn] — pure-Rust deep learning framework.
- [TinyStories] — Eldan & Li, the dataset that made small-model training
  pedagogically tractable.
- The smol-models / nanoGPT lineage for showing that a few thousand lines
  is enough to build a real LM.

[Burn]: https://burn.dev
[TinyStories]: https://huggingface.co/datasets/roneneldan/TinyStories
