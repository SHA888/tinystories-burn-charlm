# Architecture Decision Records

Consolidated ADR log. Per the OSS convention each decision will be split
into its own file under `docs/adr/NNNN-slug.md` once we exceed ~10
decisions. Until then, one file is easier to scan.

Format per record: **Context → Decision → Consequences → Alternatives
considered → Status**.

---

## ADR-0001 — Choose Burn as the training framework

**Context.** We need a Rust deep-learning framework that supports
end-to-end training (forward, autodiff, optimizer, checkpoint) on CPU.
Three real candidates: `burn`, `candle`, `tch`.

**Decision.** Use `burn`.

**Consequences.**
- Pure Rust; no system libtorch, no CUDA toolchain required.
- Idiomatic Rust API (derive macros for `Module`, typed config via
  `Config` derive, `AutodiffBackend` parameterisation).
- Smaller pretrained-model ecosystem than `candle`. We accept this — we
  are training from scratch, not loading reference checkpoints.

**Alternatives considered.**
- `candle` — inference-leaning. Training works but the abstractions are
  thinner; we'd be writing more glue. Rejected for *this* project; would
  reconsider if the goal were Hub-loading existing models.
- `tch` — fast and complete, but it's PyTorch-with-a-Rust-frontend. Fails
  the spirit of "Rust-trained end-to-end." Rejected.

**Status.** Accepted, 2026-04-29.

---

## ADR-0002 — Four-crate workspace layout

**Context.** Project is small (~1M params, single training script). YAGNI
suggests one crate. Cross-stack principle says crate-per-concern.

**Decision.** Four crates: `model-core` (lib), `data` (lib), `train`
(bin), `infer` (bin).

**Consequences.**
- More boilerplate up front (4 `Cargo.toml`s, workspace members list,
  inter-crate version pins).
- Hexagonal seams enforced by the compiler — `model-core` cannot
  accidentally import `data` because there's no path.
- Re-targeting cost is near-zero: replace `data` and re-run `train`.

**Alternatives considered.**
- Single crate with modules. Cheaper now; the seams become conventions,
  not invariants. Rejected — conventions are wishes per the meta-rule.
- Three crates (merge `data` into `model-core`). Considered. Rejected
  because dataset I/O brings `tokio`, `reqwest`, file parsing — all of
  which would pollute `model-core`'s dependency footprint.

**Status.** Accepted, 2026-04-29.

---

## ADR-0003 — Licensing posture

**Context.** Two artifacts to license: source code and trained weights.
Each has different ecosystem norms.

**Decision.**
- **Code:** dual `MIT OR Apache-2.0`.
- **Weights:** `Apache-2.0` (single, not dual — to match common HF model
  card patterns and avoid downstream confusion about which terms apply).

**Consequences.**
- Maximally permissive. Anyone can use, modify, redistribute, and embed
  in commercial products without notifying us.
- Cannot tighten retroactively. Future restricted releases would need a
  new repo/branch with new terms.
- Weights inherit any restrictions from the training dataset's license.
  Verified at pull time (TODO Phase 1).

**Alternatives considered.**
- `OpenRAIL` / `Llama-Community` style use-restricted licenses. Rejected
  because (a) at this scale the restrictions are unenforceable theatre,
  (b) they break HF Inference API for many consumers, (c) downstream
  compatibility headaches outweigh any signaled intent.
- `CC-BY-NC` for weights. Rejected — non-commercial clauses are
  notoriously ambiguous and chill legitimate research use.
- `AGPL`. Rejected — wrong tool for a model artifact; mostly defends
  network services.

**Status.** Accepted, 2026-04-29.

---

## ADR-0004 — TinyStories as bootstrap dataset

**Context.** Need a dataset small enough to train on CPU overnight, large
enough that a 1M-param model can demonstrate coherent learning, and
licensed permissively enough to redistribute derivative weights.

**Decision.** `roneneldan/TinyStories` from Hugging Face Hub.

**Consequences.**
- Synthetic (GPT-3.5-generated) short children's stories. Vocabulary is
  small, sentences are short — exactly the regime tiny models can fit.
- ~500 MB. Fits in memory on commodity hardware; can also stream.
- License verified: TinyStories is released under **Apache-2.0**
  (Hugging Face dataset card, 2023). Compatible with weight redistribution
  under our declared Apache-2.0 terms. No attribution conflict.
  (Verified via `hf_hub` API at dataset metadata, 2026-04-29).

**Alternatives considered.**
- `enwik8` / `enwik9` — Wikipedia-derived, larger, harder. Rejected for
  bootstrap; revisit for a v0.2 retraining run.
- A Balinese/Indonesian corpus the user has on hand. Deferred to a
  follow-up project once the loop is understood end-to-end.

**Status.** Accepted, 2026-04-29.

---

## ADR-0005 — `ndarray` backend (CPU-only)

**Context.** User has CPU only. Burn supports several backends:
`ndarray`, `wgpu`, `candle`, `cuda`, `libtorch`, `tch`.

**Decision.** Use `burn::backend::NdArray` for both `train` and `infer`
binaries.

**Consequences.**
- Zero native dependencies (no CUDA toolkit, no libtorch download, no
  Vulkan SDK). `cargo build` and run.
- Slower than `wgpu` on machines with integrated GPUs, but eliminates a
  whole class of "works on my machine" issues.
- `model-core` remains generic over `B: Backend`. Switching backends
  later is a one-line change in the binaries plus a `Cargo.toml` feature
  flip — no API change.

**Alternatives considered.**
- `wgpu` — would let users with iGPUs accelerate without vendor lock-in.
  Rejected for v0.1.0 only; revisit once the loop is proven.
- `candle` backend (Burn-on-Candle). Adds a layer. Rejected for clarity.

**Status.** Accepted, 2026-04-29. Revisit at v0.2.0.

---

## Template for new ADRs

```markdown
## ADR-NNNN — <short imperative title>

**Context.** Why is this decision live right now?

**Decision.** What did we choose?

**Consequences.** What follows from this — both gains and costs?

**Alternatives considered.** What was rejected and why?

**Status.** Proposed | Accepted (date) | Superseded by ADR-MMMM.
```
