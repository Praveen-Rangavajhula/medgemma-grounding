# medgemma-grounding

Minimal Python research scaffold for mechanistic interpretability experiments on visual grounding in MedGemma 1.5.

## Stack

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- `src/` package layout

## Quick start

```bash
uv python install 3.11
uv sync --dev
uv run pytest
```

## Project layout

```text
.
├── pyproject.toml
├── src/medgemma_grounding/
└── tests/
```

This repository intentionally includes only the reproducible scaffold and no research methodology implementation yet.
