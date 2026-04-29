#!/usr/bin/env bash
set -euo pipefail

echo "Installing cargo tooling..."

# Install cargo tooling. `cargo-nextest` requires a Rust version matching its
# MSRV; if it fails, the pre-commit hook falls back to `cargo test`.
cargo install cargo-semver-checks
cargo install --locked cargo-deny
cargo install cargo-audit
cargo install cargo-machete
cargo install cargo-nextest || echo "cargo-nextest failed (will use cargo test fallback)"

echo "Done."
