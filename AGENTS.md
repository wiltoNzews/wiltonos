# Repository Guidelines

## Project Structure & Module Organization
- `core/` holds the main engine modules (routing, coherence, session, write-back).
- `tools/` contains integration helpers (OpenWebUI pipes, voice, OpenRouter tooling).
- `scripts/` is for one-off utilities such as audits (`scripts/deep_audit.py`).
- `data/` stores local SQLite databases and JSON assets (runtime data).
- `logs/` is runtime output; treat as transient.
- `analysis/` and `docs/` are project notes, specs, and context references.
- Top-level entry points: `talk_v2.py`, `talk.py`, `gateway.py`, `web.py`.

## Build, Test, and Development Commands
- `python talk_v2.py` runs the primary CLI engine (interactive mode).
- `python talk_v2.py "your question"` runs a single query.
- `python talk.py` runs the lightweight companion mode.
- `python gateway.py --port 8000` starts the FastAPI gateway (default port 8000).
- `python web.py` starts the Flask web UI at `http://localhost:7777`.
- `python scripts/deep_audit.py` runs the repository audit utility.

Notes: The runtime expects local data at `~/wiltonos/data/crystals_unified.db`. OpenRouter usage reads `~/.openrouter_key`. Ollama is assumed at `http://localhost:11434`.

## Coding Style & Naming Conventions
- Python-only codebase; use 4-space indentation and PEP 8 naming.
- Prefer `snake_case` for functions/variables and `PascalCase` for classes.
- Keep module boundaries clear: core logic in `core/`, entry points at repo root.
- Use `Path` utilities for filesystem paths (see `talk_v2.py` and `web.py`).

## Testing Guidelines
- No automated test suite is present in this repository.
- Validate changes by running the relevant entry point and confirming expected behavior against the local SQLite database.
- If you add tests, document how to run them in this file and place them under a new `tests/` directory.

## Commit & Pull Request Guidelines
- This folder has no `.git` history; follow the parent repo convention if applicable.
- If you introduce version control, use short imperative messages (e.g., "Add gateway auth guard").
- Avoid committing `data/` or `logs/` artifacts unless explicitly needed.

## Security & Configuration Tips
- Do not hardcode API keys; use `~/.openrouter_key` for cloud access.
- Treat `data/` as sensitive user memory; avoid copying or sharing without approval.
