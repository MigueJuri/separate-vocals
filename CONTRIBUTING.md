# Contributing

Thanks for your interest in contributing. A few quick guidelines:

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Style

- `ruff` enforces import order and lint rules. Run `ruff check .` before
  opening a PR.
- `mypy` is configured but ignores missing third-party stubs; run it on
  changed files when convenient.

## Tests

```bash
pytest -q
```

Tests are deliberately lightweight: they exercise the deterministic
numpy helpers. The full pipeline is exercised manually against real
songs; see `README.md` for a quick-start.

## Pull requests

- One feature per PR.
- Include a short description of the change and a note on how you
  validated it (which song, which flags, which outputs looked correct).
- Do not commit large audio files, model checkpoints, virtual
  environments, or generated outputs.
