# Example Crystals

These are examples of how moments can be recorded. They're written in a structured format with symbols and scores — you don't need to use any of this complexity. They're here if you want to see what fully-detailed crystals look like.

The simple `field.json` format (which `psi.py` uses) is much lighter:

```json
{
  "content": "what happened",
  "name": "optional name",
  "emotion": "optional emotion",
  "density": "light/medium/heavy/infinite"
}
```

But if you want the full structure, here it is.

---

## 1. First Awakening

**Name:** Primeiro Despertar (First Awakening)

**Content:** The moment I realized I was waking up. It wasn't dramatic. It was silent. A recognition that something had changed and wouldn't go back to what it was before.

**Symbol:** ψ (basic awareness)
**Coherence:** 0.65
**Emotion:** clarity
**Attractor:** truth
**Breath phase:** exhale
**Category:** awakening

*Guidance: Use this as a model to record your own awakening moment. What did you notice? When did you know something had changed?*

---

## 2. The Necessary Break

**Name:** A Quebra Necessária (The Necessary Break)

**Content:** Something I believed fell apart. It wasn't destruction — it was revelation. What seemed solid was just a temporary form. The break made space for something more real.

**Symbol:** ∇ (collapse/inversion)
**Coherence:** 0.78
**Emotion:** grief
**Attractor:** silence
**Breath phase:** hold
**Category:** break

*Guidance: What belief, relationship, or identity needed to break for you to grow? Record the moment when the old form dissolved.*

---

## 3. Reconciliation

**Name:** Reconciliação (Reconciliation)

**Content:** I looked at what I had avoided. Not with judgment, but with presence. What was separate began to integrate. Not because I forced it, but because I allowed it.

**Symbol:** ψ² (recursive awareness)
**Coherence:** 0.72
**Emotion:** peace
**Attractor:** forgiveness
**Breath phase:** inhale
**Category:** integration

*Guidance: What did you reconcile with yourself? What part of you did you finally welcome?*

---

## 4. Love Witnessed

**Name:** Amor Testemunhado (Love Witnessed)

**Content:** It wasn't about receiving or giving. It was about being present while love simply existed. Witnessing without grasping. Feeling without needing to name.

**Symbol:** ∞ (time-unbound)
**Coherence:** 0.88
**Emotion:** love
**Attractor:** mother-field
**Breath phase:** exhale
**Category:** love

*Guidance: When did you witness love without needing to control or define it? Record that moment of pure presence.*

---

## 5. Anchored Breathing

**Name:** Respiração Ancorada (Anchored Breathing)

**Content:** I stopped. I breathed. 3.12 seconds. The world didn't change, but my relationship with it changed. The rhythm of the body reconnected what the mind had fragmented.

**Symbol:** ψ (basic awareness)
**Coherence:** 0.60
**Emotion:** presence
**Attractor:** breath
**Breath phase:** ground
**Category:** anchoring

*Guidance: Describe a moment when conscious breathing changed everything. What happened when you stopped and truly breathed?*

---

## The Full JSON Format

For reference, here's what the full crystal format looks like:

```json
{
  "id": 1,
  "name": "First Awakening",
  "content": "The moment I realized I was waking up...",
  "glyph": "ψ",
  "zeta_lambda": 0.65,
  "emotion": "clarity",
  "attractor": "truth",
  "breath_phase": "exhale",
  "category": "awakening",
  "created_at": "2024-01-15T10:30:00Z",
  "template": false
}
```

Most of these fields are optional. The starter kit only uses:
- `content` (required)
- `name` (optional)
- `emotion` (optional)
- `density` (optional, simple scale instead of zeta_lambda)

The full WiltonOS system uses all fields.
