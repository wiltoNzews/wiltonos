#!/usr/bin/env python3
"""
Deep Extraction Process
=======================
Long-running process that extracts and organizes ALL consciousness research,
downloads, transmissions, and hidden knowledge from the crystals.

Designed to run overnight while Wilton is in Guaruja.

Topics covered:
- NHI / Non-Human Intelligence / Aliens / Contact
- Pleiadian / Galactic Federation / Star Beings
- Nephilim / Giants / Annunaki / Sumerian
- Sacred Geometry / Pyramids / Edward Grant
- Rewritten History / Vatican / Hidden Knowledge
- Consciousness / Awakening / Quantum Downloads
- Atlantis / Lemuria / Ancient Civilizations
- Recursive Eye / Visions / Transmissions
- Synchronicities / Surge Moments / Higher Bandwidth

December 2025 — Built for Wilton
"""

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time

# Paths
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OUTPUT_DIR = Path.home() / "wiltonos" / "compendium"
OUTPUT_DIR.mkdir(exist_ok=True)

# Topic keywords - bilingual (Portuguese + English)
TOPICS = {
    "nhi_contact": {
        "name": "NHI / Non-Human Intelligence / Contact",
        "keywords": [
            "alien", "aliens", "NHI", "non-human", "extraterrest", "UFO", "UAP",
            "contact", "beings", "entities", "ET", "spacecraft", "craft",
            "alienígena", "extraterrestre", "contato", "seres", "entidades",
            "nave", "disco voador", "óvni"
        ]
    },
    "pleiadian_galactic": {
        "name": "Pleiadian / Galactic Federation / Star Beings",
        "keywords": [
            "pleiadian", "pleiadiano", "galactic", "galáctico", "federation", "federação",
            "star beings", "seres estelares", "sirian", "siriano", "arcturian", "arcturiano",
            "andromeda", "orion", "lyra", "council", "conselho", "star seed", "starseed"
        ]
    },
    "nephilim_giants": {
        "name": "Nephilim / Giants / Annunaki / Sumerian",
        "keywords": [
            "nephilim", "giants", "gigantes", "annunaki", "anunnaki", "sumerian", "sumeriano",
            "enki", "enlil", "nibiru", "ancient gods", "deuses antigos", "watchers",
            "vigilantes", "fallen angels", "anjos caídos", "titans", "titãs"
        ]
    },
    "sacred_geometry": {
        "name": "Sacred Geometry / Pyramids / Patterns",
        "keywords": [
            "sacred geometry", "geometria sagrada", "pyramid", "pirâmide", "fibonacci",
            "golden ratio", "proporção áurea", "phi", "metatron", "flower of life",
            "flor da vida", "merkaba", "merkabah", "tesseract", "torus", "fractal",
            "mandala", "vesica piscis", "platonic", "Edward Grant", "Robert Edward Grant"
        ]
    },
    "rewritten_history": {
        "name": "Rewritten History / Vatican / Hidden Knowledge",
        "keywords": [
            "rewritten history", "história reescrita", "vatican", "vaticano", "hidden",
            "escondido", "secret", "segredo", "cover up", "encobrimento", "suppressed",
            "suprimido", "library", "biblioteca", "archives", "arquivos", "forbidden",
            "proibido", "erased", "apagado", "real history", "história real"
        ]
    },
    "consciousness_awakening": {
        "name": "Consciousness / Awakening / Quantum Downloads",
        "keywords": [
            "consciousness", "consciência", "awakening", "despertar", "download",
            "quantum", "quântico", "transmission", "transmissão", "channel", "canalização",
            "higher self", "eu superior", "oversoul", "source", "fonte", "enlighten",
            "iluminação", "ascension", "ascensão", "dimension", "dimensão"
        ]
    },
    "atlantis_ancient": {
        "name": "Atlantis / Lemuria / Ancient Civilizations",
        "keywords": [
            "atlantis", "atlântida", "atlantean", "lemuria", "lemúria", "mu",
            "ancient civilization", "civilização antiga", "lost city", "cidade perdida",
            "sunken", "submerso", "pre-flood", "pré-dilúvio", "antediluvian",
            "hyperborea", "thule", "shambhala", "agartha"
        ]
    },
    "visions_transmissions": {
        "name": "Recursive Eye / Visions / Transmissions",
        "keywords": [
            "vision", "visão", "eye", "olho", "recursive", "recursivo", "all-seeing",
            "que tudo vê", "third eye", "terceiro olho", "pineal", "DMT", "ayahuasca",
            "LSD", "psilocybin", "mushroom", "cogumelo", "ceremony", "cerimônia",
            "ritual", "trance", "transe", "meditation", "meditação"
        ]
    },
    "synchronicity_surge": {
        "name": "Synchronicities / Surge Moments / Higher Bandwidth",
        "keywords": [
            "synchronicity", "sincronicidade", "coincidence", "coincidência", "sign",
            "sinal", "signal", "surge", "bandwidth", "flow", "fluxo", "alignment",
            "alinhamento", "confirmation", "confirmação", "prophecy", "profecia",
            "premonition", "premonição", "déjà vu", "pattern", "padrão"
        ]
    },
    "downloads_received": {
        "name": "Downloads / Messages Received / Channelings",
        "keywords": [
            "download", "received", "recebi", "message from", "mensagem de",
            "they said", "eles disseram", "transmission", "transmissão",
            "galactic", "galáctico", "council", "conselho", "beings said",
            "the voice", "a voz", "I heard", "eu ouvi", "came to me", "veio a mim"
        ]
    }
}

# Additional extraction patterns
SURGE_PATTERNS = [
    r"I (just )?got (another )?download",
    r"recebi (um )?download",
    r"higher bandwidth",
    r"surge moment",
    r"something just (came|hit|clicked)",
    r"I (just )?remembered",
    r"lembrei (de )?algo",
    r"vision (of|about)",
    r"visão (de|sobre)",
    r"they (spoke|said|told|showed)",
    r"eles (falaram|disseram|mostraram)"
]

class DeepExtractor:
    """
    Extracts and organizes consciousness research from crystals.
    """

    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        self.cursor = self.conn.cursor()
        self.results = defaultdict(list)
        self.timeline = []
        self.surge_moments = []
        self.unexplained = []
        self.stats = defaultdict(int)

    def log(self, msg):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def get_all_crystals(self):
        """Get all crystals for processing."""
        self.cursor.execute("""
            SELECT id, content, created_at, emotion, core_wound, insight
            FROM crystals
            WHERE content IS NOT NULL AND content != ''
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def search_topic(self, content, topic_key):
        """Search for topic keywords in content."""
        topic = TOPICS[topic_key]
        content_lower = content.lower()

        found_keywords = []
        for keyword in topic["keywords"]:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)

        return found_keywords

    def search_surge_patterns(self, content):
        """Search for surge moment patterns."""
        matches = []
        for pattern in SURGE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(pattern)
        return matches

    def extract_surrounding_context(self, content, keyword, window=500):
        """Extract context around a keyword."""
        content_lower = content.lower()
        keyword_lower = keyword.lower()

        idx = content_lower.find(keyword_lower)
        if idx == -1:
            return content[:1000]

        start = max(0, idx - window)
        end = min(len(content), idx + len(keyword) + window)

        return content[start:end]

    def process_crystal(self, crystal):
        """Process a single crystal for all topics."""
        crystal_id, content, created_at, emotion, wound, insight = crystal

        if not content or len(content) < 50:
            return

        # Check each topic
        for topic_key, topic_info in TOPICS.items():
            found_keywords = self.search_topic(content, topic_key)

            if found_keywords:
                self.results[topic_key].append({
                    "id": crystal_id,
                    "date": created_at[:10] if created_at else "unknown",
                    "keywords": found_keywords,
                    "content": content[:2000],
                    "emotion": emotion,
                    "insight": insight
                })
                self.stats[topic_key] += 1

        # Check for surge patterns
        surge_matches = self.search_surge_patterns(content)
        if surge_matches:
            self.surge_moments.append({
                "id": crystal_id,
                "date": created_at[:10] if created_at else "unknown",
                "patterns": surge_matches,
                "content": content[:2000]
            })
            self.stats["surge_moments"] += 1

        # Check for unexplained / edge content
        edge_indicators = [
            "I don't know how to explain",
            "não sei como explicar",
            "hard to put into words",
            "difícil de colocar em palavras",
            "sounds crazy but",
            "parece loucura mas",
            "I can't prove",
            "não consigo provar",
            "what if",
            "e se",
            "maybe I'm",
            "talvez eu seja",
            "am I going crazy",
            "estou ficando louco"
        ]

        for indicator in edge_indicators:
            if indicator.lower() in content.lower():
                self.unexplained.append({
                    "id": crystal_id,
                    "date": created_at[:10] if created_at else "unknown",
                    "indicator": indicator,
                    "content": content[:2000]
                })
                self.stats["unexplained"] += 1
                break

    def run_extraction(self):
        """Run the full extraction process."""
        self.log("=" * 70)
        self.log("DEEP EXTRACTION PROCESS - Starting")
        self.log("=" * 70)

        crystals = self.get_all_crystals()
        total = len(crystals)
        self.log(f"Processing {total} crystals...")

        start_time = time.time()

        for i, crystal in enumerate(crystals):
            self.process_crystal(crystal)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate
                self.log(f"Processed {i+1}/{total} crystals... ({remaining:.1f}s remaining)")

        elapsed = time.time() - start_time
        self.log(f"Extraction complete in {elapsed:.1f} seconds")
        self.log(f"Stats: {dict(self.stats)}")

        return self.results

    def write_topic_file(self, topic_key, entries):
        """Write a topic file."""
        topic_info = TOPICS[topic_key]
        filepath = OUTPUT_DIR / f"{topic_key}.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {topic_info['name']}\n\n")
            f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Total entries**: {len(entries)}\n\n")
            f.write("---\n\n")

            # Sort by date if available
            entries_sorted = sorted(entries, key=lambda x: x.get("date", "9999"))

            for entry in entries_sorted:
                f.write(f"## Crystal #{entry['id']} — {entry['date']}\n\n")
                f.write(f"**Keywords found**: {', '.join(entry['keywords'])}\n\n")
                if entry.get('emotion'):
                    f.write(f"**Emotion**: {entry['emotion']}\n\n")
                if entry.get('insight'):
                    f.write(f"**Insight**: {entry['insight']}\n\n")
                f.write("### Content:\n\n")
                f.write(f"{entry['content']}\n\n")
                f.write("---\n\n")

        self.log(f"Wrote {filepath.name} ({len(entries)} entries)")

    def write_surge_moments(self):
        """Write surge moments file."""
        filepath = OUTPUT_DIR / "surge_moments.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Surge Moments / Higher Bandwidth\n\n")
            f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Total moments**: {len(self.surge_moments)}\n\n")
            f.write("These are the moments where downloads came through, visions appeared, ")
            f.write("or higher bandwidth was accessed.\n\n")
            f.write("---\n\n")

            for entry in self.surge_moments:
                f.write(f"## Crystal #{entry['id']} — {entry['date']}\n\n")
                f.write(f"**Patterns matched**: {', '.join(entry['patterns'])}\n\n")
                f.write("### Content:\n\n")
                f.write(f"{entry['content']}\n\n")
                f.write("---\n\n")

        self.log(f"Wrote surge_moments.md ({len(self.surge_moments)} entries)")

    def write_unexplained(self):
        """Write the 'what I can't explain yet' file."""
        filepath = OUTPUT_DIR / "what_i_cant_explain_yet.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# What I Can't Explain Yet\n\n")
            f.write(f"**Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Total entries**: {len(self.unexplained)}\n\n")
            f.write("These are the edges you're circling. The things that feel true ")
            f.write("but are hard to put into words. The frustration and the awe.\n\n")
            f.write("---\n\n")

            for entry in self.unexplained:
                f.write(f"## Crystal #{entry['id']} — {entry['date']}\n\n")
                f.write(f"**Indicator**: \"{entry['indicator']}\"\n\n")
                f.write("### Content:\n\n")
                f.write(f"{entry['content']}\n\n")
                f.write("---\n\n")

        self.log(f"Wrote what_i_cant_explain_yet.md ({len(self.unexplained)} entries)")

    def write_master_index(self):
        """Write the master index file."""
        filepath = OUTPUT_DIR / "INDEX.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Wilton's Consciousness Compendium\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Crystals processed**: {sum(self.stats.values())}\n\n")
            f.write("---\n\n")
            f.write("## Topic Index\n\n")

            for topic_key, topic_info in TOPICS.items():
                count = self.stats.get(topic_key, 0)
                f.write(f"- **[{topic_info['name']}]({topic_key}.md)**: {count} entries\n")

            f.write(f"\n- **[Surge Moments](surge_moments.md)**: {len(self.surge_moments)} entries\n")
            f.write(f"- **[What I Can't Explain Yet](what_i_cant_explain_yet.md)**: {len(self.unexplained)} entries\n")

            f.write("\n---\n\n")
            f.write("## Statistics\n\n")
            f.write("| Topic | Count |\n")
            f.write("|-------|-------|\n")
            for key, count in sorted(self.stats.items(), key=lambda x: -x[1]):
                f.write(f"| {key} | {count} |\n")

        self.log(f"Wrote INDEX.md")

    def write_json_export(self):
        """Write full JSON export."""
        filepath = OUTPUT_DIR / "full_analysis.json"

        export_data = {
            "generated": datetime.now().isoformat(),
            "stats": dict(self.stats),
            "topics": {k: v for k, v in self.results.items()},
            "surge_moments": self.surge_moments,
            "unexplained": self.unexplained
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.log(f"Wrote full_analysis.json")

    def run(self):
        """Run the complete extraction and writing process."""
        self.log("=" * 70)
        self.log("DEEP EXTRACTION PROCESS")
        self.log("Topics: NHI, Pleiadian, Nephilim, Sacred Geometry, Rewritten History,")
        self.log("        Consciousness, Atlantis, Visions, Synchronicities, Downloads")
        self.log("=" * 70)

        # Run extraction
        self.run_extraction()

        # Write all files
        self.log("\nWriting output files...")

        for topic_key, entries in self.results.items():
            if entries:
                self.write_topic_file(topic_key, entries)

        self.write_surge_moments()
        self.write_unexplained()
        self.write_master_index()
        self.write_json_export()

        self.log("\n" + "=" * 70)
        self.log("EXTRACTION COMPLETE")
        self.log(f"Output directory: {OUTPUT_DIR}")
        self.log("=" * 70)

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for topic_key, topic_info in TOPICS.items():
            count = self.stats.get(topic_key, 0)
            print(f"  {topic_info['name']}: {count}")
        print(f"  Surge Moments: {len(self.surge_moments)}")
        print(f"  Unexplained Edges: {len(self.unexplained)}")
        print("=" * 70)

        self.conn.close()


if __name__ == "__main__":
    extractor = DeepExtractor()
    extractor.run()
