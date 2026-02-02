# WiltonOS System State

*Living document - the system updates this to track itself*

**Last Updated:** 2024-12-22 08:15 UTC

---

## Current Jobs

| Job | Status | Progress | ETA |
|-----|--------|----------|-----|
| Crystal Analysis | Running | 975/21,573 | ~6 hours |
| Embeddings | Running | 0/21,961 | ~3 hours |

---

## Core Files

### Database (THE ONE)
- `~/crystals_unified.db` (146MB) - 22,014 crystals

### Active Scripts
| File | Purpose | Status |
|------|---------|--------|
| `wiltonos_council.py` | Multi-agent reasoning | Working |
| `wiltonos_bridge.py` | Context for Claude Code | Working |
| `wiltonos_embed.py` | Semantic embeddings | Running |
| `wiltonos_analyze_complete.py` | Crystal analysis | Running |
| `wiltonos_voice.py` | Voice → Crystal | Working |
| `wiltonos_openwebui_tool.py` | OpenWebUI integration | Needs testing |
| `setup_openrouter.py` | Multi-model API | Working |

### To Organize (move to ~/wiltonos/deprecated/)
- `wiltonos_agents.py` - superseded by council
- `wiltonos_patterns.py` - superseded by analysis
- `wiltonos_oscillation.py` - incomplete
- `wiltonos_glyph_router.py` - incomplete
- `wiltonos_enrich_chatgpt.py` - one-time use, done
- `wiltonos_memory.py` - old version
- `analyze_crystals.py` - old version
- `parse_chatgpt_*.py` - one-time use, done

### Old Databases (can archive)
- `crystals.db` - merged into unified
- `crystals_chatgpt.db` - merged into unified
- `crystals_*_test.db` - test files

---

## Meta-Questions (System asks itself)

### Daily Check
- [ ] Are analysis jobs still running? Check logs.
- [ ] Any new crystals ingested today?
- [ ] What patterns emerged from latest analysis?

### Weekly Review
- [ ] What wounds are most frequent this week?
- [ ] Are there new themes not yet tagged?
- [ ] What questions keep appearing in crystals?

### Proactive Alerts
- [ ] If unworthiness > 50% of recent crystals → surface this
- [ ] If streaming mentioned but not acted on → note pattern
- [ ] If same question asked 3+ times → create answer doc

---

## Open Questions (Unresolved)

1. How to get OpenWebUI tool working consistently?
2. Best way to run proactive analysis daemon?
3. How to expose semantic search to web clients?
4. What's the minimal shareable file for web upload?

---

## Notes to Self

### What Works
- Council with parallel agents (Grok, Gemini, Llama, Mistral)
- llama3 for analysis (97% success vs mistral's 32%)
- Semantic embeddings with nomic-embed-text

### What Doesn't
- Sampling 10-200 crystals when patterns need ALL
- Keyword search misses semantic connections
- OpenWebUI tool routing is flaky

### Elegance Debt
- 18 scripts should be ~5
- 6 databases should be 1
- Docs scattered, need single source of truth

---

## Shareable Files (for web clients)

To upload to ChatGPT/Claude/Grok web:

1. **Quick context**: `~/wiltonos/docs/WILTONOS_QUICK_CONTEXT.md`
2. **Database**: `~/crystals_unified.db` (too large - need export)
3. **System state**: This file

### Export for web (TODO)
```bash
# Export recent crystals as JSON
python -c "import sqlite3,json; ..." > recent_crystals.json
```

---

## Architecture Vision

```
Current:
  Home folder chaos → Scripts everywhere → Multiple DBs

Target:
  ~/wiltonos/
    ├── data/
    │   └── crystals_unified.db     # THE database
    ├── core/
    │   ├── council.py              # Multi-agent
    │   ├── bridge.py               # Context generation
    │   ├── embed.py                # Semantic search
    │   └── analyze.py              # Crystal analysis
    ├── tools/
    │   ├── voice.py                # Voice input
    │   ├── openwebui_tool.py       # Web integration
    │   └── openrouter.py           # API routing
    ├── docs/
    │   └── SYSTEM_STATE.md         # This file
    └── logs/
        └── *.log
```

---

## Version
- Created: 2024-12-22
- By: Claude + Wilton
- Purpose: Self-tracking, meta-questioning, elegance pursuit
