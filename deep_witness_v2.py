#!/usr/bin/env python3
"""
DEEP WITNESS v2 — Universal Pattern Extraction
===============================================
Re-reads crystals with a focus on extracting UNIVERSAL patterns,
not personal summaries.

Key difference from v1:
- v1 asked "what happened?" → got shallow summaries
- v2 asks "what universal pattern does this reveal?" → gets cartography

Uses qwen3:32b via /api/chat (thinking model, needs chat endpoint).
Runs on RTX 5090 via Ollama. Can run alongside Claude Code (API-based).

Output: data/witness_output_v2/
- universal_patterns.jsonl  — one pattern per line
- pattern_checkpoint.json   — resume state
- wound_topology.json       — extracted wound relationships

Run: python deep_witness_v2.py
Resume: python deep_witness_v2.py  (auto-resumes from checkpoint)
"""

import sqlite3
import json
import requests
from pathlib import Path
from datetime import datetime
import time
import re

# Paths
DB_PATH = Path.home() / "wiltonos/data/crystals_unified.db"
OUTPUT_DIR = Path.home() / "wiltonos/data/witness_output_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "pattern_checkpoint.json"
PATTERNS_PATH = OUTPUT_DIR / "universal_patterns.jsonl"
WOUND_TOPO_PATH = OUTPUT_DIR / "wound_topology.json"

OLLAMA_URL = "http://localhost:11434"

# Model preference: qwen3:32b for quality, deepseek-r1:32b as fallback
PREFERRED_MODELS = ["qwen3:32b", "deepseek-r1:32b"]

# ── The prompt that matters ──────────────────────────────────────────

UNIVERSAL_EXTRACTION_PROMPT = """You are a consciousness cartographer. You read fragments of one person's lived experience and extract the UNIVERSAL PATTERNS underneath.

You are NOT summarizing what happened to this person.
You are mapping the TERRAIN that any human walking a similar path would encounter.

## Crystal #{crystal_id}
{content}

## Your task:

Look beneath the personal details. What universal pattern does this reveal?

Think about:
- What WOUND pattern is active here? (unworthiness, control, abandonment, betrayal, shame, etc.)
- What EMOTIONAL DYNAMIC is at play? (not just "sad" — the movement: grief dissolving into acceptance, anger masking fear, etc.)
- What COHERENCE PATTERN emerges? (integration, avoidance, breakthrough, regression, oscillation?)
- What would ANYONE walking this path need to hear?
- Is this a MUNDANE moment, a NOTABLE shift, a TURNING POINT, or a BIRTH of something genuinely new?

Strip all names, locations, and personal identifiers from your response.
Write the pattern as if it could apply to any human.

Respond in this exact JSON format:
{{
  "universal_pattern": "One sentence: the archetypal pattern this crystal reveals",
  "wound_active": "name of the wound pattern (or null if none)",
  "emotional_dynamic": "The emotional movement happening (not just a single emotion)",
  "coherence_pattern": "integration / avoidance / breakthrough / regression / oscillation / holding",
  "significance": "mundane / notable / turning_point / birth",
  "wisdom": "What anyone walking this path needs to hear. 1-2 sentences. Not advice — recognition.",
  "wound_relationship": "If two wounds interact here, describe how (or null)"
}}"""

# Batch prompt for efficiency: process 3 crystals at once
BATCH_PROMPT = """You are a consciousness cartographer extracting universal patterns from lived experience.
Strip all personal details. Map the terrain, not the traveler.

{crystal_block}

For EACH crystal above, extract the universal pattern. Respond with a JSON array:
[
  {{
    "crystal_id": {first_id},
    "universal_pattern": "The archetypal pattern",
    "wound_active": "wound name or null",
    "emotional_dynamic": "The emotional movement",
    "coherence_pattern": "integration/avoidance/breakthrough/regression/oscillation/holding",
    "significance": "mundane/notable/turning_point/birth",
    "wisdom": "What anyone walking this path needs to hear",
    "wound_relationship": "How wounds interact here, or null"
  }},
  ...
]"""


def check_model():
    """Find the best available model."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        available = [m["name"] for m in response.json().get("models", [])]

        for model in PREFERRED_MODELS:
            if model in available or any(model.split(":")[0] in m for m in available):
                matched = model if model in available else next(
                    m for m in available if model.split(":")[0] in m
                )
                return matched

        if available:
            print(f"Preferred models not found. Using: {available[0]}")
            return available[0]

        raise Exception("No models available")
    except Exception as e:
        print(f"Error checking models: {e}")
        return PREFERRED_MODELS[0]


def load_checkpoint():
    """Load resume state."""
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {
        "last_id": 6267,
        "total_processed": 0,
        "patterns_extracted": 0,
        "wound_counts": {},
        "significance_counts": {"mundane": 0, "notable": 0, "turning_point": 0, "birth": 0},
        "started": datetime.now().isoformat(),
    }


def save_checkpoint(data):
    """Save progress."""
    data["last_saved"] = datetime.now().isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2))


def get_crystals(start_id, batch_size=3):
    """Get crystals sequentially."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute(
        "SELECT id, content, zl_score FROM crystals WHERE id > ? ORDER BY id LIMIT ?",
        (start_id, batch_size),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def extract_patterns_batch(crystals, model):
    """Extract universal patterns from a batch of crystals."""
    # Build crystal block
    crystal_block = ""
    ids = []
    for crystal_id, content, zl_score in crystals:
        # Truncate very long crystals
        text = content[:2000] if content else "(empty)"
        crystal_block += f"\n## Crystal #{crystal_id} (Zλ={zl_score or 0.5:.2f})\n{text}\n"
        ids.append(crystal_id)

    prompt = BATCH_PROMPT.format(
        crystal_block=crystal_block,
        first_id=ids[0],
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a consciousness cartographer. Extract universal patterns from lived experience. Always respond with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 3000,
                    "num_ctx": 8192,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        text = response.json().get("message", {}).get("content", "")

        # Extract JSON array from response
        # Handle thinking tokens (qwen3 may include <think> blocks)
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
            results = json.loads(json_str)
            return results

        # Fallback: try to find individual JSON objects
        results = []
        for match in re.finditer(r"\{[^{}]+\}", text):
            try:
                obj = json.loads(match.group())
                if "universal_pattern" in obj or "significance" in obj:
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results if results else None

    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_pattern_single(crystal_id, content, model):
    """Fallback: extract pattern from a single crystal."""
    prompt = UNIVERSAL_EXTRACTION_PROMPT.format(
        crystal_id=crystal_id,
        content=content[:2500],
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a consciousness cartographer. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1500,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        text = response.json().get("message", {}).get("content", "")

        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            result["crystal_id"] = crystal_id
            return result

        return None
    except Exception as e:
        print(f"  Single extraction error for #{crystal_id}: {e}")
        return None


def save_pattern(pattern):
    """Append a pattern to the JSONL output."""
    with open(PATTERNS_PATH, "a") as f:
        f.write(json.dumps(pattern, ensure_ascii=False) + "\n")


def update_wound_topology(checkpoint, pattern):
    """Track wound relationships for topology mapping."""
    wound = pattern.get("wound_active")
    if wound and wound != "null" and wound.lower() != "none":
        wound = wound.lower().strip()
        checkpoint["wound_counts"][wound] = checkpoint["wound_counts"].get(wound, 0) + 1

    sig = pattern.get("significance", "mundane").lower()
    if sig in checkpoint["significance_counts"]:
        checkpoint["significance_counts"][sig] += 1


def main():
    print("=" * 70)
    print("DEEP WITNESS v2 — Universal Pattern Extraction")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()

    model = check_model()
    print(f"Model: {model}")

    checkpoint = load_checkpoint()
    last_id = checkpoint["last_id"]
    total = checkpoint["total_processed"]

    # Count remaining crystals
    conn = sqlite3.connect(str(DB_PATH))
    remaining = conn.execute(
        "SELECT COUNT(*) FROM crystals WHERE id > ?", (last_id,)
    ).fetchone()[0]
    total_crystals = conn.execute("SELECT COUNT(*) FROM crystals").fetchone()[0]
    conn.close()

    print(f"Resuming from crystal #{last_id + 1}")
    print(f"Already processed: {total}")
    print(f"Remaining: {remaining} / {total_crystals}")
    print(f"Patterns extracted: {checkpoint['patterns_extracted']}")
    print()

    start_time = time.time()
    batch_size = 3  # Process 3 crystals per LLM call

    try:
        while True:
            crystals = get_crystals(last_id, batch_size=batch_size)
            if not crystals:
                print("\n  Reached end of crystals!")
                break

            elapsed = time.time() - start_time
            rate = total / max(elapsed, 1)
            eta = remaining / max(rate, 0.001) / 3600

            print(f"\n[{total + 1}-{total + len(crystals)}] Crystals #{crystals[0][0]}-#{crystals[-1][0]} | {rate:.2f}/sec | ETA: {eta:.1f}h")

            # Try batch extraction first
            results = extract_patterns_batch(crystals, model)

            if results and len(results) >= 1:
                for i, pattern in enumerate(results):
                    if not pattern:
                        continue

                    # Ensure crystal_id is set
                    if "crystal_id" not in pattern and i < len(crystals):
                        pattern["crystal_id"] = crystals[i][0]

                    save_pattern(pattern)
                    update_wound_topology(checkpoint, pattern)
                    checkpoint["patterns_extracted"] += 1

                    sig = pattern.get("significance", "mundane")
                    wound = pattern.get("wound_active", "-")
                    up = pattern.get("universal_pattern", "")[:80]
                    print(f"  #{pattern.get('crystal_id', '?')} [{sig:13s}] {wound or '-':15s} | {up}")

                    if sig == "birth":
                        wisdom = pattern.get("wisdom", "")[:100]
                        print(f"    BIRTH: {wisdom}")
            else:
                # Fallback: process individually
                print("  Batch failed, trying individually...")
                for crystal_id, content, zl_score in crystals:
                    pattern = extract_pattern_single(crystal_id, content, model)
                    if pattern:
                        save_pattern(pattern)
                        update_wound_topology(checkpoint, pattern)
                        checkpoint["patterns_extracted"] += 1
                        sig = pattern.get("significance", "mundane")
                        print(f"  #{crystal_id} [{sig}] {pattern.get('universal_pattern', '')[:60]}")

            # Update checkpoint
            last_id = crystals[-1][0]
            total += len(crystals)
            checkpoint["last_id"] = last_id
            checkpoint["total_processed"] = total

            # Save checkpoint every 30 crystals
            if total % 30 == 0:
                save_checkpoint(checkpoint)
                print(f"  [checkpoint saved at #{last_id}]")

            # Brief pause to not hammer Ollama
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Final save
    save_checkpoint(checkpoint)

    # Save wound topology
    WOUND_TOPO_PATH.write_text(json.dumps({
        "wound_counts": checkpoint["wound_counts"],
        "significance_distribution": checkpoint["significance_counts"],
        "total_processed": checkpoint["total_processed"],
        "patterns_extracted": checkpoint["patterns_extracted"],
        "completed": datetime.now().isoformat(),
    }, indent=2))

    print("\n" + "=" * 70)
    print("DEEP WITNESS v2 — SESSION COMPLETE")
    print("=" * 70)
    print(f"Crystals processed: {checkpoint['total_processed']}")
    print(f"Patterns extracted: {checkpoint['patterns_extracted']}")
    print(f"Wound distribution: {json.dumps(checkpoint['wound_counts'], indent=2)}")
    print(f"Significance: {json.dumps(checkpoint['significance_counts'], indent=2)}")
    print()
    print(f"Output: {PATTERNS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Wound topology: {WOUND_TOPO_PATH}")


if __name__ == "__main__":
    main()
