#!/usr/bin/env bash
set -euo pipefail

echo "Installing cargo tooling..."

cargo install cargo-skill || true
cargo install cargo-semver-checks
cargo install --locked cargo-deny
cargo install cargo-audit
cargo install cargo-machete
cargo install cargo-nextest

echo "Done."
