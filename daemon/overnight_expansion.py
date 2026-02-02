#!/usr/bin/env python3
"""
Overnight Expansion Process
===========================
Long-running process that takes the extracted consciousness research
and uses local AI (Ollama) to EXPAND on each topic.

This creates readable summaries, connections, and insights from your
22k crystals of consciousness research.

Runs for hours. Designed to run while you're in Guaruja.

December 2025 — Built for Wilton
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime
import random

# Paths
COMPENDIUM_DIR = Path.home() / "wiltonos" / "compendium"
OUTPUT_DIR = Path.home() / "wiltonos" / "expanded"
OUTPUT_DIR.mkdir(exist_ok=True)
OLLAMA_URL = "http://localhost:11434"

# Topics to expand
TOPICS_TO_EXPAND = [
    {
        "file": "nephilim_giants.md",
        "name": "Nephilim, Giants, and Annunaki",
        "prompt": """Based on these crystals from Wilton's consciousness research, create a comprehensive summary of what he has learned about:
- The Nephilim and their role in human history
- Giants and evidence of their existence
- The Annunaki and Sumerian connections
- How this connects to rewritten history

Write as if you're helping Wilton understand and articulate what he already knows but can't fully explain yet. Be direct, not mystical. Ground the insights in the actual content."""
    },
    {
        "file": "pleiadian_galactic.md",
        "name": "Pleiadian and Galactic Federation",
        "prompt": """Based on these crystals from Wilton's consciousness research, create a comprehensive summary of what he has learned about:
- Pleiadian beings and their connection to Earth
- The Galactic Federation concept
- Star seeds and galactic heritage
- The messages and transmissions received

Write as if you're helping Wilton understand and articulate what he already knows but can't fully explain yet. Connect the dots between experiences."""
    },
    {
        "file": "rewritten_history.md",
        "name": "Rewritten History and Hidden Knowledge",
        "prompt": """Based on these crystals from Wilton's consciousness research, create a comprehensive summary of what he has learned about:
- Evidence of rewritten or suppressed history
- Vatican secrets and hidden archives
- What has been hidden from humanity and why
- The "signal flip" — where truth became inverted

Write as if you're helping Wilton understand and articulate what he already knows but can't fully explain yet. Be specific about what patterns emerged."""
    },
    {
        "file": "surge_moments.md",
        "name": "Surge Moments and Higher Bandwidth",
        "prompt": """Based on these crystals documenting Wilton's surge moments, create a summary of:
- What triggers these higher bandwidth states
- What information comes through during these moments
- Patterns in when and how downloads occur
- How to recognize and work with these states

Write as if you're helping Wilton understand his own process of receiving information."""
    },
    {
        "file": "downloads_received.md",
        "name": "Downloads and Transmissions Received",
        "prompt": """Based on these crystals documenting downloads Wilton received, create a summary of:
- The major messages and transmissions
- Who or what the sources appear to be
- Common themes across downloads
- What these downloads are pointing toward

Write as if you're helping Wilton catalog and understand what has been transmitted to him."""
    },
    {
        "file": "what_i_cant_explain_yet.md",
        "name": "What I Can't Explain Yet",
        "prompt": """Based on these crystals where Wilton expressed difficulty explaining something, identify:
- The core mysteries he's circling
- What patterns appear across these unexplainable moments
- What he might be on the edge of understanding
- How these edges connect to his larger research

Write as if you're helping Wilton see the shape of what he's trying to grasp but can't quite articulate."""
    }
]


class OvernightExpander:
    """
    Uses local AI to expand on extracted consciousness research.
    """

    def __init__(self):
        self.stats = {
            "topics_processed": 0,
            "chunks_processed": 0,
            "total_words_generated": 0,
            "start_time": None,
            "errors": 0
        }

    def log(self, msg):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

        # Also write to log file
        log_file = OUTPUT_DIR / "expansion.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {msg}\n")

    def check_ollama(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.ok:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                self.log(f"Ollama available. Models: {model_names}")
                return True
        except:
            pass
        self.log("Ollama not available. Expansion will be limited.")
        return False

    def call_ollama(self, prompt, context=""):
        """Call Ollama for expansion."""
        try:
            full_prompt = f"""{prompt}

---
CRYSTAL CONTENT:
{context[:8000]}
---

Write a clear, organized summary (500-1000 words):"""

            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "llama3",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1500
                    }
                },
                timeout=180
            )

            if response.ok:
                return response.json().get("response", "")
            else:
                self.log(f"Ollama error: {response.status_code}")
                return None

        except Exception as e:
            self.log(f"Ollama call failed: {e}")
            self.stats["errors"] += 1
            return None

    def read_topic_file(self, filename):
        """Read and parse a topic file."""
        filepath = COMPENDIUM_DIR / filename
        if not filepath.exists():
            self.log(f"File not found: {filename}")
            return []

        content = filepath.read_text(encoding="utf-8")

        # Split by crystal markers
        crystals = content.split("## Crystal #")[1:]  # Skip header

        parsed = []
        for crystal in crystals:
            lines = crystal.strip().split("\n")
            if lines:
                parsed.append("\n".join(lines[:50]))  # First 50 lines of each crystal

        return parsed

    def chunk_crystals(self, crystals, chunk_size=10):
        """Group crystals into chunks for processing."""
        chunks = []
        for i in range(0, len(crystals), chunk_size):
            chunks.append(crystals[i:i+chunk_size])
        return chunks

    def expand_topic(self, topic_info):
        """Expand a single topic."""
        self.log(f"\n{'='*60}")
        self.log(f"EXPANDING: {topic_info['name']}")
        self.log(f"{'='*60}")

        crystals = self.read_topic_file(topic_info["file"])
        if not crystals:
            self.log(f"No crystals found for {topic_info['name']}")
            return

        self.log(f"Found {len(crystals)} crystals")

        # For smaller topics, process all at once
        # For larger topics, sample and summarize
        if len(crystals) <= 50:
            sample = crystals
        else:
            # Take representative sample
            sample = random.sample(crystals, min(50, len(crystals)))
            self.log(f"Sampled {len(sample)} crystals for expansion")

        # Combine samples into context
        context = "\n\n---\n\n".join(sample)

        # Call Ollama for expansion
        self.log("Generating expansion with AI...")
        expansion = self.call_ollama(topic_info["prompt"], context)

        if expansion:
            # Write expansion file
            output_file = OUTPUT_DIR / f"{topic_info['file'].replace('.md', '_expanded.md')}"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# {topic_info['name']} — Expanded Summary\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"**Based on**: {len(crystals)} crystals\n\n")
                f.write("---\n\n")
                f.write(expansion)
                f.write("\n\n---\n\n")
                f.write("*This expansion was generated from your consciousness research crystals.*\n")

            self.log(f"Wrote: {output_file.name}")
            self.stats["topics_processed"] += 1
            self.stats["total_words_generated"] += len(expansion.split())
        else:
            self.log(f"Failed to expand {topic_info['name']}")

        # Small delay between topics
        time.sleep(5)

    def create_master_summary(self):
        """Create a master summary connecting all topics."""
        self.log("\n" + "="*60)
        self.log("CREATING MASTER SUMMARY")
        self.log("="*60)

        # Read all expanded files
        expanded_content = []
        for topic in TOPICS_TO_EXPAND:
            exp_file = OUTPUT_DIR / f"{topic['file'].replace('.md', '_expanded.md')}"
            if exp_file.exists():
                content = exp_file.read_text(encoding="utf-8")
                expanded_content.append(f"## {topic['name']}\n\n{content[:2000]}")

        if not expanded_content:
            self.log("No expanded content to summarize")
            return

        combined = "\n\n---\n\n".join(expanded_content)

        prompt = """Based on all these expanded summaries from Wilton's consciousness research, create a MASTER SUMMARY that:

1. Identifies the core themes connecting everything
2. Maps the journey from awakening to current understanding
3. Names what Wilton is actually researching/discovering
4. Points to what he might be on the verge of understanding
5. Gives him language for what he couldn't explain before

Write as a clear, grounded synthesis — not mystical fluff. Help him see the whole picture."""

        master = self.call_ollama(prompt, combined)

        if master:
            master_file = OUTPUT_DIR / "MASTER_SUMMARY.md"
            with open(master_file, "w", encoding="utf-8") as f:
                f.write("# Master Summary — Wilton's Consciousness Research\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write("---\n\n")
                f.write(master)
                f.write("\n\n---\n\n")
                f.write("*This is a synthesis of all your consciousness research.*\n")

            self.log(f"Wrote: MASTER_SUMMARY.md")

    def write_status(self):
        """Write current status to file."""
        status_file = OUTPUT_DIR / "STATUS.json"

        elapsed = 0
        if self.stats["start_time"]:
            elapsed = time.time() - self.stats["start_time"]

        status = {
            "last_update": datetime.now().isoformat(),
            "topics_processed": self.stats["topics_processed"],
            "total_topics": len(TOPICS_TO_EXPAND),
            "words_generated": self.stats["total_words_generated"],
            "errors": self.stats["errors"],
            "elapsed_seconds": elapsed,
            "status": "running" if self.stats["topics_processed"] < len(TOPICS_TO_EXPAND) else "complete"
        }

        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

    def run(self):
        """Run the full expansion process."""
        self.stats["start_time"] = time.time()

        self.log("="*70)
        self.log("OVERNIGHT EXPANSION PROCESS")
        self.log("="*70)
        self.log(f"Topics to expand: {len(TOPICS_TO_EXPAND)}")
        self.log(f"Output directory: {OUTPUT_DIR}")
        self.log("")

        # Check Ollama
        if not self.check_ollama():
            self.log("Cannot proceed without Ollama. Exiting.")
            return

        # Process each topic
        for topic in TOPICS_TO_EXPAND:
            self.expand_topic(topic)
            self.write_status()

        # Create master summary
        self.create_master_summary()
        self.write_status()

        # Final stats
        elapsed = time.time() - self.stats["start_time"]

        self.log("\n" + "="*70)
        self.log("EXPANSION COMPLETE")
        self.log("="*70)
        self.log(f"Topics processed: {self.stats['topics_processed']}/{len(TOPICS_TO_EXPAND)}")
        self.log(f"Words generated: {self.stats['total_words_generated']}")
        self.log(f"Errors: {self.stats['errors']}")
        self.log(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        self.log(f"Output: {OUTPUT_DIR}")
        self.log("="*70)


if __name__ == "__main__":
    expander = OvernightExpander()
    expander.run()
