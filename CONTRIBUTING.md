# Contributing

## Code Style

- Follow the Rust Style Guide enforced by `rustfmt`.
- Run `cargo clippy --all-targets --all-features` and resolve warnings before submitting.
- Prefer explicit types over heavy type inference in public APIs.

## Commit Format

This repository uses **Conventional Commits**.

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`
- **scope**: optional crate name (`model-core`, `data`, `train`, `infer`) or area (`ci`, `docs`)
- **subject**: lowercase, imperative mood, no trailing period

Examples:
- `feat(model-core): add RMSNorm layer`
- `fix(train): correct LR warmup calculation`
- `docs: update README quickstart`

## ADR Process

Architectural decisions are recorded in `DECISIONS.md` using the ADR format:
- Sequential numbering (`ADR-0001`, `ADR-0002`, ...)
- Status: `proposed`, `accepted`, `superseded`
- Context, decision, consequences sections

Open a PR to propose a new ADR before landing cross-crate design changes.
