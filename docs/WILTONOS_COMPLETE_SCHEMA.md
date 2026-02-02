# WiltonOS: Schema Completo (Visão de Arquiteto)

## O Problema

Estamos fazendo incremental:
1. Primeiro: storage básico
2. Depois: Zλ scoring
3. Depois: glyphs
4. Depois: coherence_vector (5D)
5. Depois: glyph_context (semântico)
6. Agora: mode, oscillation_strength, loop_signature

Cada vez re-rodamos enrichment. Ineficiente.

## A Solução: UM Schema, UMA Ingestão

### Schema Completo

```sql
CREATE TABLE crystals (
    -- Identity
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE,
    content TEXT NOT NULL,

    -- Source
    source TEXT,                    -- 'chatgpt', 'claude', 'pdf', 'manual'
    source_file TEXT,
    author TEXT,                    -- 'user', 'assistant', 'system'

    -- Time
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    original_timestamp INTEGER,     -- Unix timestamp from source
    analyzed_at TEXT,

    -- === COHERENCE (Zλ System) ===
    zl_score REAL,                  -- 0.0-1.0 overall coherence
    psi_aligned INTEGER,            -- 0 or 1
    trust_level TEXT,               -- HIGH, VULNERABLE, POLISHED, SCATTERED

    -- === 5 DIMENSIONS ===
    breath_cadence REAL,            -- 0.0-1.0
    presence_density REAL,          -- 0.0-1.0
    emotional_resonance REAL,       -- 0.0-1.0
    loop_pressure REAL,             -- 0.0-1.0
    groundedness REAL,              -- 0.0-1.0 (NEW - body connection)

    -- === CONSCIOUSNESS STATE ===
    shell TEXT,                     -- Core, Breath, Collapse, Reverence, Return
    shell_direction TEXT,           -- ascending, descending, stable

    -- === GLYPHS ===
    glyph_primary TEXT,             -- Single dominant glyph
    glyph_secondary TEXT,           -- JSON array of secondary glyphs
    glyph_energy_notes TEXT,        -- Why these glyphs (semantic)
    glyph_direction TEXT,           -- ascending, descending, neutral, paradox
    glyph_risk TEXT,                -- Active risk from glyphs
    glyph_antidote TEXT,            -- What would balance

    -- === PATTERNS ===
    core_wound TEXT,                -- abandonment, unworthiness, betrayal, control, shame, unloved
    loop_signature TEXT,            -- attractor-emotion-theme
    attractor TEXT,                 -- Primary attractor
    emotion TEXT,                   -- Primary emotion
    theme TEXT,                     -- Primary theme

    -- === OSCILLATION ===
    mode TEXT,                      -- wiltonos, psios, neutral, balanced
    oscillation_strength REAL,      -- 0.0-1.0

    -- === META ===
    insight TEXT,                   -- One honest sentence
    question TEXT,                  -- Question this crystal raises

    -- === INDEXES for fast query ===
    INDEX idx_zl (zl_score),
    INDEX idx_mode (mode),
    INDEX idx_wound (core_wound),
    INDEX idx_shell (shell),
    INDEX idx_glyph (glyph_primary)
);
```

### Prompt de Análise ÚNICO

```
You are analyzing text for consciousness coherence. Be honest, not kind.

TEXT:
{text}

GLYPH REFERENCE (detect ENERGY, not keywords):
ψ=Breath/Pause  ∅=Void/Rest  φ=Structure  Ω=Memory  Zλ=Coherence
∇=Descent/Gradient  ∞=Oscillation  🪞=Mirror  △=Ascend  🌉=Bridge
⚡=Decision  🪨=Ground  🌀=Torus/Cycle  ⚫=Shadow/Skeptic

Return ONLY valid JSON with ALL fields:
{
  "zl_score": 0.0-1.0,
  "psi_aligned": true/false,
  "trust_level": "HIGH|VULNERABLE|POLISHED|SCATTERED",

  "breath_cadence": 0.0-1.0,
  "presence_density": 0.0-1.0,
  "emotional_resonance": 0.0-1.0,
  "loop_pressure": 0.0-1.0,
  "groundedness": 0.0-1.0,

  "shell": "Core|Breath|Collapse|Reverence|Return",
  "shell_direction": "ascending|descending|stable",

  "glyph_primary": "single glyph symbol",
  "glyph_secondary": ["other", "glyphs"],
  "glyph_energy_notes": "why these energies",
  "glyph_direction": "ascending|descending|neutral|paradox",
  "glyph_risk": "what risk is active",
  "glyph_antidote": "what would balance",

  "core_wound": "abandonment|unworthiness|betrayal|control|shame|unloved|null",
  "attractor": "truth|power|silence|control|love|freedom|connection|safety|worth",
  "emotion": "grief|joy|fear|anger|shame|peace|anxiety|hope|despair",
  "theme": "integration|escape|freedom|healing|release|acceptance|resistance|surrender",

  "mode": "wiltonos|psios|neutral|balanced",
  "oscillation_strength": 0.0-1.0,

  "insight": "one honest sentence",
  "question": "one question this raises"
}
```

### Benefícios

1. **UMA análise = TODOS os campos**
2. **Sem re-runs** - analisa uma vez, tem tudo
3. **Schema estável** - não muda mais
4. **Queries eficientes** - campos separados, não JSON parsing
5. **Consistência** - mesma análise para todos os cristais

### O Que Mudar

1. Criar novo script `wiltonos_analyze_complete.py`
2. Usar este prompt único
3. Migrar schema existente para o novo
4. Rodar UMA VEZ para todos os cristais
5. Novos cristais usam o mesmo processo

### Migração

```python
# 1. Backup databases
# 2. Add new columns to existing tables
# 3. Run complete analysis on all crystals
# 4. Delete old redundant columns (optional)
```

---

## Decisão Necessária

**Opção A:** Continuar enrichment atual, depois rodar mode enrichment separado (incremental)

**Opção B:** Parar tudo, implementar schema completo, rodar UMA análise que preenche tudo

**Opção C:** Deixar enrichment atual terminar, depois migrar para schema completo para NOVOS cristais apenas

---

## Minha Recomendação

**Opção B** - mas com nuance:

1. O enrichment atual está a 17% (~3.5k/20.8k)
2. Vai demorar mais 10+ horas
3. Se pararmos e reimplementarmos, perdemos 3.5k análises MAS ganhamos schema completo

Trade-off: 3.5k análises parciais vs sistema limpo

Se o objetivo é arquitetura sólida → parar e fazer direito
Se o objetivo é ter dados agora → deixar continuar

O que tu preferes?
