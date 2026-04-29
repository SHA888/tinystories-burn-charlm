# Architecture

## Design principles applied

The scaffold follows your cross-stack non-negotiables. Each principle is
mapped to a **mechanism** that fails the build when violated, per the
load-bearing meta-rule that "principles without CI enforcement are
decoration."

| Principle                         | Mechanism (CI-enforced)                              |
| --------------------------------- | ---------------------------------------------------- |
| KISS, YAGNI                       | Code review + ADR pressure-test before new crate     |
| Separation of Concerns            | Crate-per-concern (4 crates); `cargo-modules` graph  |
| High cohesion, low coupling       | `cargo deny` on cross-crate cycles                   |
| Hexagonal (ports & adapters)      | Ports = traits in `model-core`; adapters elsewhere   |
| Parse-don't-validate              | Config parsed once into typed structs at edges       |
| Make-illegal-states-unrepresentable | Newtypes (`SeqLen`, `VocabSize`, `LayerCount`)     |
| Principle of least privilege      | Workspace-narrow `pub` exports; `cargo-public-api`   |
| Chesterton's Fence                | ADRs in `DECISIONS.md` justify every removal         |
| Boy Scout Rule                    | Pre-commit `fmt` + `clippy -D warnings`              |
| SemVer                            | `cargo-semver-checks` gate on `0.x` minor bumps      |
| 12-Factor (where service-shaped)  | N/A pre-publish; revisit if `infer` grows an HTTP API |

## Workspace topology

```
                ┌─────────────────────────────┐
                │       model-core (lib)      │
                │                             │
                │  • CharLm transformer       │
                │  • ModelConfig (typed)      │
                │  • Tokenizer trait (port)   │
                │  • Dataset trait (port)     │
                │  • Checkpoint I/O           │
                └──────────────┬──────────────┘
                               │ traits + types
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│   data (lib)   │    │   train (bin)   │    │   infer (bin)   │
│                │    │                 │    │                 │
│ • HF Hub pull  │    │ • Loop          │    │ • Load weights  │
│ • CharTokenizer│    │ • Optimizer     │    │ • Sampling      │
│ • Batcher      │    │ • Checkpointing │    │ • Stdout / API  │
│ • Dataset impl │    │ • Metrics log   │    │                 │
└────────────────┘    └─────────────────┘    └─────────────────┘
       adapter              adapter                 adapter
```

`model-core` is the **inner hexagon** — pure types and traits, no I/O, no
device assumptions, no logging. The outer crates are **adapters** plugged
into the ports `model-core` exposes. This means:

- The model is testable without a dataset (in-memory dummy data).
- The dataset is swappable without touching the model (port: `Dataset`).
- The tokenizer is swappable (port: `Tokenizer`) — char today, BPE later.
- The training loop doesn't import `data` directly; it sees a `Dataset`.

### Why 4 crates and not 1?

YAGNI would argue for one crate at this scope. Two reasons to split anyway:

1. **It's a learning project.** The split forces clarity about what
   belongs in the model vs. what belongs in the loop vs. what belongs in
   the data pipeline. Collapsing them later is mechanical; pulling them
   apart later is an architectural rewrite.
2. **It establishes the pattern for retargeting.** When the next project
   (domain-specific data, different tokenizer) reuses this scaffold, the
   seams are already where they need to be.

Documented in `DECISIONS.md` under ADR-0002.

## Type design — illegal states unrepresentable

Newtypes used to make wrong code not compile:

```rust
// model-core/src/types.rs (sketch)
pub struct VocabSize(NonZeroU32);
pub struct SeqLen(NonZeroU16);
pub struct EmbedDim(NonZeroU16);
pub struct LayerCount(NonZeroU8);
pub struct HeadCount(NonZeroU8);

// EmbedDim must be divisible by HeadCount — encoded in constructor:
impl ModelConfig {
    pub fn new(embed: EmbedDim, heads: HeadCount, ...) -> Result<Self, ConfigError>;
}
```

Config is parsed once at the binary edge (`train`, `infer`) into
`ModelConfig`. Inside `model-core`, no function ever takes a raw `u32`
that "should be" a vocab size.

## Backend abstraction

Burn parameterises models over a `Backend` trait. `model-core` is generic
over `B: Backend`. Binaries pick the concrete backend:

- `train` and `infer` use `burn::backend::NdArray` (CPU, no native deps,
  cross-platform). See ADR-0005.
- A future `train-gpu` binary could swap in `burn::backend::Wgpu` or
  `burn::backend::LibTorch` without changing `model-core`.

## CI enforcement matrix

```yaml
# .github/workflows/ci.yml — planned
jobs:
  fmt:        cargo fmt --all -- --check
  clippy:     cargo clippy --all-targets -- -D warnings
  test:       cargo nextest run --all
  doc:        cargo doc --no-deps --all
  semver:     cargo semver-checks            # on PRs touching public API
  deny:       cargo deny check               # licenses, advisories, bans
  audit:      cargo audit                    # RUSTSEC
  machete:    cargo machete                  # unused deps
  msrv:       cargo +<msrv> check            # pin once chosen
```

Branch protection on `main`: all jobs green, 1 review, no force-push.

## What is explicitly out of scope (Chesterton's Fence pre-applied)

- Distributed training. No DDP, no FSDP, no gradient accumulation
  abstractions. Add only when single-CPU is provably insufficient for
  declared goals — and document why in an ADR.
- GPU support. The `ndarray` backend is the contract. Adding `wgpu` or
  `cuda` later is a feature, not a bugfix.
- Custom CUDA kernels. We are not in that business at this scope.
- An HTTP serving layer. `infer` writes to stdout. If serving becomes a
  goal, it's a new crate (`serve`), not a flag on `infer`.

## Future extension points

When (not if) this scaffold is retargeted:

- New tokenizer → new module in `data/`, implements `Tokenizer` port.
- New dataset → new module in `data/`, implements `Dataset` port.
- New architecture variant → new module in `model-core/`; existing
  trainer/inferer reused unchanged.
- Different backend → swap one line in the binary; `model-core` untouched.
