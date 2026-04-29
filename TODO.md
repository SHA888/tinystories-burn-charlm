# TODO

Atomic, granular, tasks→subtasks. Each leaf is small enough to be a single
commit / single PR with a clear definition of done (DoD). Phases gate on
SemVer milestones.

Legend: `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked

---

## Phase 0 — Pre-code scaffolding (target: tag `v0.0.1`)

Goal: a green CI pipeline on an empty workspace. No model code yet.

- [x] **0.1 Repository hygiene**
  - [x] 0.1.1 Initialise `git` repo; first commit = these markdown files only.
  - [x] 0.1.2 Add `.gitignore` (Rust + `target/` + `.env` + `data/cache/`).
  - [x] 0.1.3 Add `.gitattributes` for line endings and LFS hints (no LFS use yet).
  - [x] 0.1.4 Add `CODEOWNERS` (`* @SHA888`).
  - [x] 0.1.5 Add `SECURITY.md` (responsible disclosure stub, email).
  - [x] 0.1.6 Add `CONTRIBUTING.md` (code style, commit format, ADR process).
  - [x] 0.1.7 Add Conventional Commits convention statement to `CONTRIBUTING.md`.

- [x] **0.2 LICENSE files**
  - [x] 0.2.1 Add `LICENSE-MIT` (verbatim MIT text, copyright "SHA888").
  - [x] 0.2.2 Add `LICENSE-APACHE` (verbatim Apache-2.0 text, no NOTICE yet).
  - [x] 0.2.3 Reference both in every crate `Cargo.toml` (`license = "MIT OR Apache-2.0"`).
  - [ ] 0.2.4 Add `NOTICE` only if/when third-party Apache-2.0 attribution is required.

- [x] **0.3 Cargo workspace skeleton**
  - [x] 0.3.1 Root `Cargo.toml` with `[workspace]` and `members = ["crates/*"]`.
  - [x] 0.3.2 `[workspace.package]` with shared `version = "0.0.1"`, `edition = "2021"`, `license`, `repository`, `rust-version` (MSRV).
  - [x] 0.3.3 `[workspace.lints]` enabling `clippy::pedantic` (selectively allow, not deny-all).
  - [x] 0.3.4 Empty `crates/{model-core,data,train,infer}` with `cargo new --lib` / `--bin` and minimal `lib.rs`/`main.rs`.
  - [x] 0.3.5 Each crate `Cargo.toml` inherits via `version.workspace = true` etc.
  - [x] 0.3.6 Pin Burn at latest stable; verify version at write time, not from memory.
  - [x] 0.3.7 `Cargo.lock` committed (this is a binary-producing project).

- [x] **0.4 Tooling install script**
  - [x] 0.4.1 `scripts/setup.sh` that runs `cargo install cargo-skill cargo-semver-checks cargo-deny cargo-audit cargo-machete cargo-nextest`.
  - [x] 0.4.2 Document MSRV in `README.md` and `rust-toolchain.toml`.
  - [x] 0.4.3 Pre-commit hook (`.githooks/pre-commit`) running `fmt + clippy + nextest`.

- [x] **0.5 CI pipeline (GitHub Actions)**
  - [x] 0.5.1 `.github/workflows/ci.yml` jobs: `fmt`, `clippy`, `test`, `doc`.
  - [x] 0.5.2 `.github/workflows/quality.yml` jobs: `deny`, `audit`, `machete`.
  - [x] 0.5.3 `.github/workflows/semver.yml` running `cargo-semver-checks` on PRs to `main`.
  - [x] 0.5.4 Cache strategy: `Swatinem/rust-cache` keyed on lockfile.
  - [x] 0.5.5 Branch protection on `main`: require all checks green, 1 review, no force-push. *(Configure in GitHub repo settings.)*
  - [x] 0.5.6 Dependabot config: weekly cargo updates, grouped minor+patch.

- [x] **0.6 Quality config files**
  - [x] 0.6.1 `deny.toml` — license allowlist (MIT, Apache-2.0, BSD-3, MPL-2.0); deny GPL/AGPL.
  - [x] 0.6.2 `clippy.toml` — cognitive complexity, MSRV.
  - [x] 0.6.3 `rustfmt.toml` — `edition = "2021"`, otherwise stdlib defaults.
  - [x] 0.6.4 `.cargo/config.toml` if any custom alias/profile needed (defer if not).

- [x] **0.7 Tag `v0.0.1`** — DoD: empty workspace builds clean, CI is green.

---

## Phase 1 — `data` crate (target: `v0.0.2`)

- [x] **1.1 Dataset acquisition**
  - [x] 1.1.1 Add `hf-hub` dependency for HF Hub pulls (verify latest stable version at write time).
  - [x] 1.1.2 Implement `pull_tinystories(cache_dir)` returning local paths.
  - [x] 1.1.3 Pin dataset revision (`refs/convert/parquet`) for reproducibility.
  - [x] 1.1.4 Verify and record TinyStories license tag in `DECISIONS.md` ADR-0004 update.
  - [x] 1.1.5 Streaming-friendly read path (row-group at a time via `parquet` arrow reader).

- [x] **1.2 Tokenizer (char-level)**
  - [x] 1.2.1 Define `Tokenizer` port (trait) in `model-core`.
  - [x] 1.2.2 Implement `CharTokenizer` in `data`: byte-level fallback for non-ASCII.
  - [x] 1.2.3 `VocabSize` newtype derived from observed corpus.
  - [x] 1.2.4 Persist tokenizer to/from JSON for inference reproducibility.
  - [x] 1.2.5 Property test: `decode(encode(s)) == s` for arbitrary strings.

- [x] **1.3 Dataset port + impl**
  - [x] 1.3.1 `TinyStoriesDataset` in `data` returning chunked tokenized pairs.
  - [x] 1.3.2 Implement `TinyStoriesDataset` in `data`.
  - [x] 1.3.3 Chunked `(input_ids, target_ids)` shifted pairs.
  - [x] 1.3.4 Train/val split via separate HF files (train/validation).
  - [x] 1.3.5 Unit test: shapes correct for `(batch inferred, seq=4)`.

- [x] **1.4 Tag `v0.0.2`** — DoD: `cargo run -p data -- prepare` produces tokenized cache + tokenizer JSON.

---

## Phase 2 — `model-core` crate (target: `v0.0.3`)

- [ ] **2.1 Type foundations**
  - [ ] 2.1.1 Newtypes: `VocabSize`, `SeqLen`, `EmbedDim`, `LayerCount`, `HeadCount`, `DropoutProb`.
  - [ ] 2.1.2 `ModelConfig::new` validates `EmbedDim % HeadCount == 0`.
  - [ ] 2.1.3 `ConfigError` enum (parse errors, invariant violations).
  - [ ] 2.1.4 `serde` round-trip test on `ModelConfig`.

- [ ] **2.2 Model modules (Burn `Module` derive)**
  - [ ] 2.2.1 Token + positional embedding.
  - [ ] 2.2.2 Multi-head causal self-attention.
  - [ ] 2.2.3 Feed-forward (GELU).
  - [ ] 2.2.4 Pre-norm transformer block (RMSNorm or LayerNorm — pick, document).
  - [ ] 2.2.5 `CharLm` top module with LM head (weight-tied to token embedding).

- [ ] **2.3 Forward pass**
  - [ ] 2.3.1 `forward(ids) -> logits` generic over `B: Backend`.
  - [ ] 2.3.2 Causal mask correctness test.
  - [ ] 2.3.3 Loss helper: `cross_entropy(logits, targets, ignore_index)`.

- [ ] **2.4 Checkpoint I/O**
  - [ ] 2.4.1 Save/load via Burn's `Recorder` to `safetensors` format.
  - [ ] 2.4.2 Round-trip test: train 1 step, save, load, assert weights equal.

- [ ] **2.5 Tag `v0.0.3`** — DoD: untrained model forward-passes a batch without error; doc-tested.

---

## Phase 3 — `train` crate (target: `v0.0.4`)

- [ ] **3.1 Loop**
  - [ ] 3.1.1 Wire `data` + `model-core` via Burn's `Learner`.
  - [ ] 3.1.2 AdamW optimizer; cosine LR schedule with warmup.
  - [ ] 3.1.3 Gradient clipping at `1.0`.
  - [ ] 3.1.4 Loss + perplexity metrics; log to stdout + a CSV file.
  - [ ] 3.1.5 Checkpoint every N steps; keep last K.

- [ ] **3.2 Config**
  - [ ] 3.2.1 `configs/charlm-1m.toml` with all hyperparameters.
  - [ ] 3.2.2 `clap`-based CLI: `--config`, `--resume`, `--seed`, `--output`.
  - [ ] 3.2.3 Parse-don't-validate: TOML → typed `TrainConfig` at startup.

- [ ] **3.3 Reproducibility**
  - [ ] 3.3.1 Seed PRNG (data shuffle, dropout, init).
  - [ ] 3.3.2 Log full resolved config + dataset revision + git SHA into checkpoint dir.

- [ ] **3.4 Tag `v0.0.4`** — DoD: a 100-step training run completes on CPU, loss visibly decreases.

---

## Phase 4 — `infer` crate (target: `v0.0.5`)

- [ ] **4.1 Loading**
  - [ ] 4.1.1 Load `safetensors` weights + tokenizer + config from a checkpoint dir.
  - [ ] 4.1.2 Validate config matches at load (no silent shape coercion).

- [ ] **4.2 Sampling**
  - [ ] 4.2.1 Greedy decoding.
  - [ ] 4.2.2 Top-k and top-p sampling.
  - [ ] 4.2.3 Temperature.
  - [ ] 4.2.4 Stop conditions (max tokens, optional stop string).

- [ ] **4.3 CLI**
  - [ ] 4.3.1 `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--seed`.
  - [ ] 4.3.2 Stream tokens to stdout as generated.

- [ ] **4.4 Tag `v0.0.5`** — DoD: untrained-checkpoint inference runs end-to-end (gibberish is fine — proves the pipe).

---

## Phase 5 — Quality gates (target: `v0.0.6`)

- [ ] **5.1 Tests**
  - [ ] 5.1.1 Coverage report via `cargo-llvm-cov`; aim ≥70% on `model-core`.
  - [ ] 5.1.2 Property tests on tokenizer round-trip.
  - [ ] 5.1.3 Integration test: tiny synthetic dataset → 5-step train → infer → assert non-panic.
  - [ ] 5.1.4 Doc-tests on every public type.

- [ ] **5.2 Benchmarks**
  - [ ] 5.2.1 `criterion` bench on forward pass; baseline numbers in README.
  - [ ] 5.2.2 Track regression in CI (warn-only on PR).

- [ ] **5.3 Documentation**
  - [ ] 5.3.1 `cargo doc --no-deps` warning-free.
  - [ ] 5.3.2 Each public module has a `//!` doc with example.
  - [ ] 5.3.3 README "Quick start" actually works on a fresh clone.

- [ ] **5.4 Tag `v0.0.6`** — DoD: all CI green, coverage ≥70%, docs clean.

---

## Phase 6 — Real training run (target: `v0.0.7`)

- [ ] **6.1 Run the model**
  - [ ] 6.1.1 Full training run on TinyStories. Capture: wall-clock, final loss, sample generations.
  - [ ] 6.1.2 Save final checkpoint + tokenizer + config + training log.
  - [ ] 6.1.3 Hand-evaluate 20 generations for coherence; record observations.
  - [ ] 6.1.4 If output is incoherent: pre-publish triage (LR too high? vocab wrong? mask bug?).

- [ ] **6.2 Tag `v0.0.7`** — DoD: a real, evaluated checkpoint exists locally.

---

## Phase 7 — Hugging Face publish (target: `v0.1.0`)

- [ ] **7.1 Hub repo**
  - [ ] 7.1.1 Create public model repo `SHA888/tinystories-burn-charlm` on HF.
  - [ ] 7.1.2 `hf auth login` with a fine-grained token scoped to that repo only.

- [ ] **7.2 Model card**
  - [ ] 7.2.1 YAML frontmatter: `license: apache-2.0`, `library_name: burn`, `tags: [rust, burn, char-lm, tinystories]`, `datasets: [roneneldan/TinyStories]`, `language: en`.
  - [ ] 7.2.2 Body sections: Intended use, Out-of-scope use, Training data, Training procedure, Eval results (perplexity), Limitations, How to load (Rust + Python snippets).
  - [ ] 7.2.3 Reproducibility block: dataset revision, seed, hyperparameters, hardware, wall-clock.

- [ ] **7.3 Artifact contents**
  - [ ] 7.3.1 `model.safetensors`.
  - [ ] 7.3.2 `config.json` (the same `ModelConfig` `train` consumed).
  - [ ] 7.3.3 `tokenizer.json`.
  - [ ] 7.3.4 `README.md` (the model card).
  - [ ] 7.3.5 `training.log` excerpt (final epoch).
  - [ ] 7.3.6 LICENSE files copied into the model repo (HF convention).

- [ ] **7.4 Push**
  - [ ] 7.4.1 `git lfs install`; `huggingface-cli lfs-enable-largefiles .`.
  - [ ] 7.4.2 `git push` to HF remote.
  - [ ] 7.4.3 Smoke test: `safetensors` loads in a Python notebook (proves cross-language compatibility).
  - [ ] 7.4.4 Smoke test: `infer` loads weights from `hf-hub` cache directly (proves Rust round-trip).

- [ ] **7.5 Tag `v0.1.0`** — DoD: the model is on the Hub, downloadable, loadable, and the model card is honest about what it can and cannot do.

---

## Future (post-`v0.1.0`, not on the critical path)

- [ ] BPE tokenizer adapter (8K vocab) for comparison.
- [ ] `wgpu` backend feature flag.
- [ ] Domain-specific dataset (Balinese, medical, etc.) — requires its own ADR.
- [ ] Per-file ADR split once `DECISIONS.md` exceeds ~10 entries.
- [ ] HTTP serving via a new `serve` crate (only if a real consumer asks).
