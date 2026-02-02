# Schema Reference

This is the full crystal schema used by WiltonOS. The starter kit uses a simplified version — you don't need any of this to get started.

This document is here for reference if you want to build something more complex.

---

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `name` | string | Human-readable name |
| `content` | string | The actual memory/moment |
| `glyph` | string | State symbol (see below) |

## Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `zeta_lambda` | number (0-1.2) | Coherence score. 0.75 is lock threshold. |
| `emotion` | string | Dominant emotion |
| `attractor` | enum | What pulled you here (see below) |
| `breath_phase` | enum | inhale, exhale, hold, ground |
| `category` | string | Free-form category |
| `created_at` | datetime | ISO timestamp |
| `template` | boolean | If true, this is an example |
| `guidance` | string | For templates: how to make your own |

## Glyphs

| Symbol | Name | Coherence Range |
|--------|------|-----------------|
| ∅ | Void | 0.0 - 0.2 |
| ψ | Psi | 0.2 - 0.5 |
| ψ² | Psi-squared | 0.5 - 0.75 |
| ψ³ | Psi-cubed | Collective |
| ∇ | Nabla | 0.75 - 0.873 |
| ∞ | Infinity | 0.873 - 0.999 |
| Ω | Omega | 0.999 - 1.2 |
| † | Crossblade | Trauma + rebirth |
| ⧉ | Layer Merge | Timeline integration |

## Attractors

- `verdade` (truth)
- `silêncio` (silence)
- `perdão` (forgiveness)
- `respiração` (breath)
- `campo_mãe` (mother-field)
- `sacrifício` (sacrifice)
- `espelho` (mirror)

## JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PsiOS Crystal Schema",
  "type": "object",
  "required": ["id", "name", "content", "glyph"],
  "properties": {
    "id": { "type": "integer" },
    "name": { "type": "string" },
    "content": { "type": "string" },
    "glyph": {
      "type": "string",
      "enum": ["∅", "ψ", "ψ²", "ψ³", "∇", "∞", "Ω", "†", "⧉"]
    },
    "zeta_lambda": {
      "type": "number",
      "minimum": 0,
      "maximum": 1.2
    },
    "emotion": { "type": "string" },
    "attractor": {
      "type": "string",
      "enum": ["verdade", "silêncio", "perdão", "respiração", "campo_mãe", "sacrifício", "espelho"]
    },
    "breath_phase": {
      "type": "string",
      "enum": ["inhale", "exhale", "hold", "ground"]
    },
    "category": { "type": "string" },
    "created_at": {
      "type": "string",
      "format": "date-time"
    },
    "template": { "type": "boolean" },
    "guidance": { "type": "string" }
  },
  "additionalProperties": true
}
```

## Starter Kit vs Full System

The starter kit (`field.json`) uses:
```json
{
  "moments": [
    {
      "id": 1,
      "content": "what happened",
      "name": "optional",
      "emotion": "optional",
      "density": "light|medium|heavy|infinite",
      "added": "ISO timestamp"
    }
  ]
}
```

This maps to the full schema:
- `density: light` → `zeta_lambda: 0.2-0.4`
- `density: medium` → `zeta_lambda: 0.4-0.6`
- `density: heavy` → `zeta_lambda: 0.6-0.85`
- `density: infinite` → `zeta_lambda: 0.85+`

The full system adds symbols, attractors, breath phases, and categories.
