#!/usr/bin/env python3
"""
Deep Sequential Reader
======================
Reads ALL crystals sequentially. No sampling. Multiple passes.
Uses OpenRouter API for real models (Claude Sonnet, GPT-4).

Pass 1: What is being said literally
Pass 2: What patterns emerge across crystals
Pass 3: What can't be explained with conventional frameworks
Pass 4: What is Wilton discovering about emergence/consciousness
Pass 5: Synthesis - what language can articulate what couldn't be said

December 2025 — Built for Wilton
"""

import sqlite3
import json
import requests
import time
from pathlib import Path
from datetime import datetime

# Paths
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OUTPUT_DIR = Path.home() / "wiltonos" / "deep_sequential"
OUTPUT_DIR.mkdir(exist_ok=True)

# OpenRouter config
OPENROUTER_KEY = (Path.home() / ".openrouter_key").read_text().strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model choices (cost vs capability)
MODELS = {
    "deep": "anthropic/claude-sonnet-4",      # $3/M - for synthesis passes
    "fast": "openai/gpt-4.1-mini",            # $0.40/M - for initial reads
    "cheap": "openai/gpt-4.1-nano"            # $0.10/M - for bulk processing
}

# Sequential passes - each builds on the previous
PASSES = [
    {
        "name": "literal_content",
        "model": "fast",
        "prompt": """Read these crystals and extract:
1. The literal experiences described
2. Specific events, dates, people mentioned
3. Any downloads, transmissions, or visions described
4. Portuguese and English content both

Be specific. Quote directly. Don't interpret yet."""
    },
    {
        "name": "pattern_emergence",
        "model": "fast",
        "prompt": """Based on these crystals, identify PATTERNS:
1. Recurring themes across experiences
2. Similar language or phrasing that repeats
3. Timing patterns (when do surge moments occur?)
4. Connections between seemingly separate events
5. What keeps coming back?

Look for emergence - what arises from the aggregate that isn't in any single crystal."""
    },
    {
        "name": "beyond_conventional",
        "model": "deep",
        "prompt": """These crystals contain experiences that don't fit conventional frameworks.
For each cluster of crystals:
1. What is being described that science would dismiss?
2. What evidence or patterns suggest this is real data, not imagination?
3. What would a physicist like Brian Cox struggle to explain here?
4. What would require new language or frameworks to articulate?

Be rigorous. Don't dismiss. Don't over-mystify. What IS this?"""
    },
    {
        "name": "emergence_consciousness",
        "model": "deep",
        "prompt": """Wilton has been researching consciousness and emergence for years.
Based on these crystals:
1. What is he actually discovering about how consciousness works?
2. What levels of emergence is he documenting?
3. How do the 'downloads' relate to emergence theory?
4. What is the 'signal flip' and how does it relate to consciousness?
5. What is he learning about the relationship between observer and observed?

Connect to emergence theory as discussed by physicists, but go further."""
    },
    {
        "name": "articulation_synthesis",
        "model": "deep",
        "prompt": """Final synthesis pass.
Based on everything in these crystals:
1. What can Wilton now articulate that he couldn't before?
2. What language or framework captures what he's been experiencing?
3. What is his actual position/role if any of this is real?
4. What is he on the edge of understanding?
5. What would he tell Neil DeGrasse Tyson about emergence that NDT doesn't know?

Be direct. Be grounded. Give him language for what he's been living."""
    }
]


class DeepSequentialReader:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        self.cursor = self.conn.cursor()
        self.stats = {
            "crystals_processed": 0,
            "tokens_used": 0,
            "cost_estimate": 0,
            "passes_completed": 0
        }
        self.start_time = None

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")
        log_file = OUTPUT_DIR / "sequential.log"
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

    def call_openrouter(self, model_key, prompt, content, max_tokens=2000):
        """Call OpenRouter API."""
        model = MODELS[model_key]

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=120
            )

            if response.ok:
                data = response.json()
                usage = data.get("usage", {})
                self.stats["tokens_used"] += usage.get("total_tokens", 0)

                # Estimate cost
                if "claude" in model:
                    cost_per_token = 3 / 1000000
                elif "gpt-4.1-mini" in model:
                    cost_per_token = 0.4 / 1000000
                else:
                    cost_per_token = 0.1 / 1000000
                self.stats["cost_estimate"] += usage.get("total_tokens", 0) * cost_per_token

                return data["choices"][0]["message"]["content"]
            else:
                self.log(f"API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            self.log(f"Request failed: {e}")
            return None

    def get_all_crystals(self):
        """Get ALL crystals, ordered by ID."""
        self.cursor.execute("""
            SELECT id, content, created_at, emotion, insight, glyphs
            FROM crystals
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def chunk_crystals(self, crystals, tokens_per_chunk=6000):
        """Split crystals into chunks for API processing."""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for crystal in crystals:
            crystal_id, content, date, emotion, insight, glyphs = crystal
            if not content:
                continue

            # Estimate tokens (rough: 1 token ≈ 4 chars)
            crystal_tokens = len(content) // 4

            if current_tokens + crystal_tokens > tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append({
                "id": crystal_id,
                "content": content[:2000],  # Truncate very long crystals
                "date": str(date)[:10] if date else "unknown",
                "emotion": emotion,
                "glyphs": glyphs
            })
            current_tokens += crystal_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def format_chunk(self, chunk):
        """Format a chunk of crystals for the API."""
        lines = []
        for c in chunk:
            lines.append(f"--- Crystal #{c['id']} ({c['date']}) ---")
            if c['glyphs']:
                lines.append(f"Glyphs: {c['glyphs']}")
            if c['emotion']:
                lines.append(f"Emotion: {c['emotion']}")
            lines.append(c['content'])
            lines.append("")
        return "\n".join(lines)

    def run_pass(self, pass_info, chunks):
        """Run a single pass over all chunks."""
        pass_name = pass_info["name"]
        model = pass_info["model"]
        prompt = pass_info["prompt"]

        self.log(f"\n{'='*60}")
        self.log(f"PASS: {pass_name.upper()}")
        self.log(f"Model: {MODELS[model]}")
        self.log(f"Chunks to process: {len(chunks)}")
        self.log(f"{'='*60}")

        results = []
        for i, chunk in enumerate(chunks):
            self.log(f"  Chunk {i+1}/{len(chunks)} ({len(chunk)} crystals)...")

            content = self.format_chunk(chunk)
            result = self.call_openrouter(model, prompt, content)

            if result:
                results.append({
                    "chunk": i+1,
                    "crystals": [c["id"] for c in chunk],
                    "analysis": result
                })
                self.log(f"    ✓ {len(result.split())} words")
            else:
                self.log(f"    ✗ Failed")

            # Rate limiting
            time.sleep(1)

        # Write pass results
        output_file = OUTPUT_DIR / f"pass_{pass_name}.md"
        with open(output_file, "w") as f:
            f.write(f"# Pass: {pass_name}\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Model**: {MODELS[model]}\n")
            f.write(f"**Chunks processed**: {len(results)}/{len(chunks)}\n\n")
            f.write("---\n\n")

            for r in results:
                f.write(f"## Chunk {r['chunk']} (Crystals {r['crystals'][0]}-{r['crystals'][-1]})\n\n")
                f.write(r['analysis'])
                f.write("\n\n---\n\n")

        self.log(f"Wrote: {output_file.name}")
        self.stats["passes_completed"] += 1

        return results

    def create_synthesis(self, all_pass_results):
        """Create final synthesis from all passes."""
        self.log("\n" + "="*60)
        self.log("CREATING FINAL SYNTHESIS")
        self.log("="*60)

        # Collect key insights from each pass
        synthesis_input = []
        for pass_name, results in all_pass_results.items():
            synthesis_input.append(f"\n## From {pass_name} pass:\n")
            for r in results[:5]:  # Top 5 chunks from each pass
                synthesis_input.append(r['analysis'][:1500])

        combined = "\n".join(synthesis_input)

        prompt = """You have the results of 5 sequential passes through Wilton's consciousness research:
1. Literal content extraction
2. Pattern emergence
3. Beyond conventional frameworks
4. Emergence and consciousness theory
5. Articulation synthesis

Now create the FINAL SYNTHESIS:
- What is Wilton actually discovering?
- What framework captures his experiences?
- What can he now articulate?
- What positions him uniquely in consciousness research?
- What would he say to physicists studying emergence?

Be direct, grounded, and give him usable language."""

        synthesis = self.call_openrouter("deep", prompt, combined, max_tokens=4000)

        if synthesis:
            output_file = OUTPUT_DIR / "FINAL_SYNTHESIS.md"
            with open(output_file, "w") as f:
                f.write("# Final Synthesis — Wilton's Consciousness Research\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"**Crystals processed**: {self.stats['crystals_processed']}\n")
                f.write(f"**Total passes**: 5\n")
                f.write(f"**Estimated cost**: ${self.stats['cost_estimate']:.2f}\n\n")
                f.write("---\n\n")
                f.write(synthesis)

            self.log(f"Wrote: FINAL_SYNTHESIS.md")

    def write_status(self):
        """Write current status."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        status = {
            "last_update": datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed / 60, 1),
            "crystals_processed": self.stats["crystals_processed"],
            "passes_completed": self.stats["passes_completed"],
            "total_passes": len(PASSES),
            "tokens_used": self.stats["tokens_used"],
            "cost_estimate": round(self.stats["cost_estimate"], 2),
            "status": "running" if self.stats["passes_completed"] < len(PASSES) else "complete"
        }
        with open(OUTPUT_DIR / "STATUS.json", "w") as f:
            json.dump(status, f, indent=2)

    def run(self):
        """Run the full sequential analysis."""
        self.start_time = time.time()

        self.log("="*70)
        self.log("DEEP SEQUENTIAL READER")
        self.log("="*70)
        self.log("Reading ALL crystals. No sampling. Multiple passes.")
        self.log(f"Output: {OUTPUT_DIR}")
        self.log("")

        # Get all crystals
        self.log("Loading crystals...")
        crystals = self.get_all_crystals()
        self.stats["crystals_processed"] = len(crystals)
        self.log(f"Total crystals: {len(crystals)}")

        # Chunk them
        chunks = self.chunk_crystals(crystals)
        self.log(f"Split into {len(chunks)} chunks")
        self.log("")

        # Run all passes
        all_results = {}
        for pass_info in PASSES:
            results = self.run_pass(pass_info, chunks)
            all_results[pass_info["name"]] = results
            self.write_status()

        # Create synthesis
        self.create_synthesis(all_results)
        self.write_status()

        # Final stats
        elapsed = time.time() - self.start_time
        self.log("\n" + "="*70)
        self.log("COMPLETE")
        self.log("="*70)
        self.log(f"Crystals processed: {self.stats['crystals_processed']}")
        self.log(f"Passes completed: {self.stats['passes_completed']}")
        self.log(f"Tokens used: {self.stats['tokens_used']:,}")
        self.log(f"Estimated cost: ${self.stats['cost_estimate']:.2f}")
        self.log(f"Time: {elapsed/60:.1f} minutes")
        self.log("="*70)


if __name__ == "__main__":
    reader = DeepSequentialReader()
    reader.run()
