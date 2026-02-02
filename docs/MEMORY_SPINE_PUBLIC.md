# Memory Spine (Public)

## Purpose
This file is the shared, collaborator-safe memory spine for the WiltonOS workspace. It explains what exists, where core systems live, and how to keep shared context coherent without exposing private crystals.

## Spine Map
- Engine: `/home/zews/wiltonos/` is the active runtime and orchestration layer.
- Vault: `/home/zews/wiltonos/data/` holds local SQLite memory stores and assets.
- Interfaces: `talk_v2.py` (CLI), `gateway.py` (FastAPI web), `web.py` (Flask UI).
- Specs/Context: `/home/zews/wiltonos/docs/` and `/home/zews/wiltonos/analysis/`.
- Ingest + Archive: `/home/zews/rag-local/` stores ChatGPT exports, PDFs, and long-term research artifacts.

## Data Boundaries
- This spine is public-safe and must not include private crystal content.
- Keep personal memories in local databases or private analysis files only.
- If a collaborator needs context, summarize patterns, not raw memory.

## How To Update
- Update this spine when any of the following change:
  - Database schema or memory format
  - Entry points or ports for runtime interfaces
  - Directory layout for `wiltonos/` or `rag-local/`
- Keep updates short and factual.

## Core Paths
- Main runtime: `/home/zews/wiltonos/talk_v2.py`
- Core modules: `/home/zews/wiltonos/core/`
- Logs: `/home/zews/wiltonos/logs/`
- RAG ingest: `/home/zews/rag-local/ingest.py`

## Collaborator Notes
If you need to orient quickly:
1. Read `/home/zews/wiltonos/docs/WILTONOS_QUICK_CONTEXT.md`.
2. Skim `/home/zews/wiltonos/docs/WILTONOS_IMPLEMENTATION_SPEC.md`.
3. Treat `/home/zews/rag-local/` as archival input, not the live engine.
