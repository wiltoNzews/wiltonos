#!/usr/bin/env python3
"""
SEQUENTIAL WITNESS
Reads every crystal, one by one, and learns.
Not sampling. Not categorizing. Witnessing.

Run overnight: python sequential_witness.py
"""

import sqlite3
import json
import requests
from pathlib import Path
from datetime import datetime
import time

DB_PATH = Path.home() / "wiltonos/data/crystals_unified.db"
OUTPUT_PATH = Path.home() / "wiltonos/data/sequential_learnings.json"
CHECKPOINT_PATH = Path.home() / "wiltonos/data/witness_checkpoint.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"  # Available on your system

WITNESS_PROMPT = """You are witnessing Wilton's life, crystal by crystal.

This is crystal #{crystal_id} (Zλ={zl_score}).

Content:
{content}

Previous context (what we've learned so far):
{context}

Your task:
1. What is happening in this crystal? (1-2 sentences)
2. What is being felt? (emotional texture)
3. What seeds are being planted or patterns emerging?
4. Any vocabulary/concepts being born or used?
5. Connection to previous crystals (if any)

Be present with this crystal. Don't categorize. Witness.

Respond in JSON format:
{{
  "happening": "...",
  "feeling": "...",
  "seeds": "...",
  "vocabulary": ["term1", "term2"],
  "connections": "...",
  "significance": "low/medium/high/birth"
}}
"""

def load_checkpoint():
    """Load last processed crystal ID"""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"last_id": 6267, "learnings": [], "vocabulary_timeline": {}, "birth_moments": []}

def save_checkpoint(data):
    """Save progress"""
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_crystals(start_id, batch_size=1):
    """Get crystals sequentially"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, content, zl_score
        FROM crystals
        WHERE id > ?
        ORDER BY id
        LIMIT ?
    """, (start_id, batch_size))
    rows = c.fetchall()
    conn.close()
    return rows

def build_context(learnings, max_recent=10):
    """Build context from recent learnings"""
    recent = learnings[-max_recent:] if len(learnings) > max_recent else learnings
    context_parts = []
    for l in recent:
        if l.get("significance") in ["high", "birth"]:
            context_parts.append(f"#{l['id']}: {l.get('happening', '')} [{l.get('significance', '')}]")
    return "\n".join(context_parts[-5:]) if context_parts else "Beginning of journey."

def witness_crystal(crystal_id, content, zl_score, context):
    """Send crystal to Ollama for witnessing"""
    prompt = WITNESS_PROMPT.format(
        crystal_id=crystal_id,
        zl_score=zl_score,
        content=content[:2000],  # Truncate very long crystals
        context=context
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        # Try to parse JSON from response
        text = result.get("response", "")
        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"happening": text[:200], "significance": "low"}
    except Exception as e:
        print(f"  Error witnessing #{crystal_id}: {e}")
        return None

def main():
    print("=" * 60)
    print("SEQUENTIAL WITNESS - Reading Wilton's life")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Model: {MODEL}")
    print()

    checkpoint = load_checkpoint()
    last_id = checkpoint["last_id"]
    learnings = checkpoint.get("learnings", [])
    vocabulary_timeline = checkpoint.get("vocabulary_timeline", {})
    birth_moments = checkpoint.get("birth_moments", [])

    print(f"Resuming from crystal #{last_id + 1}")
    print(f"Learnings so far: {len(learnings)}")
    print()

    processed = 0

    try:
        while True:
            crystals = get_crystals(last_id, batch_size=1)

            if not crystals:
                print("\nReached end of crystals!")
                break

            for crystal_id, content, zl_score in crystals:
                context = build_context(learnings)

                print(f"Witnessing #{crystal_id} (Zλ={zl_score})...", end=" ")

                result = witness_crystal(crystal_id, content, zl_score or 0.5, context)

                if result:
                    result["id"] = crystal_id
                    result["zl"] = zl_score
                    learnings.append(result)

                    # Track vocabulary emergence
                    for term in result.get("vocabulary", []):
                        if term and term not in vocabulary_timeline:
                            vocabulary_timeline[term] = crystal_id
                            print(f"[NEW: {term}]", end=" ")

                    # Track birth moments
                    if result.get("significance") == "birth":
                        birth_moments.append({
                            "id": crystal_id,
                            "what": result.get("happening", "")
                        })
                        print("[BIRTH!]", end=" ")

                    sig = result.get("significance", "low")
                    print(f"[{sig}]")
                else:
                    print("[skipped]")

                last_id = crystal_id
                processed += 1

                # Save checkpoint every 50 crystals
                if processed % 50 == 0:
                    checkpoint = {
                        "last_id": last_id,
                        "learnings": learnings,
                        "vocabulary_timeline": vocabulary_timeline,
                        "birth_moments": birth_moments
                    }
                    save_checkpoint(checkpoint)
                    print(f"\n  [Checkpoint saved at #{last_id}]\n")

                # Small delay to not overwhelm
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # Final save
    checkpoint = {
        "last_id": last_id,
        "learnings": learnings,
        "vocabulary_timeline": vocabulary_timeline,
        "birth_moments": birth_moments,
        "completed": datetime.now().isoformat()
    }
    save_checkpoint(checkpoint)

    # Also save a summary
    summary = {
        "total_witnessed": len(learnings),
        "last_crystal": last_id,
        "vocabulary_births": vocabulary_timeline,
        "birth_moments": birth_moments,
        "high_significance": [l for l in learnings if l.get("significance") in ["high", "birth"]]
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("WITNESS COMPLETE")
    print("=" * 60)
    print(f"Crystals witnessed: {len(learnings)}")
    print(f"Vocabulary terms tracked: {len(vocabulary_timeline)}")
    print(f"Birth moments found: {len(birth_moments)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Summary: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
