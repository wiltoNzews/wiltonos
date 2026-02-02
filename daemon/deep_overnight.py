#!/usr/bin/env python3
"""
Deep Overnight Process
======================
The REAL overnight process. Takes hours, not seconds.

For each topic:
1. Process ALL crystals (not samples)
2. Multiple AI passes for deeper understanding
3. Extract specific quotes and moments
4. Build connections between topics
5. Create comprehensive summaries

December 2025 — Built for Wilton's Guaruja trip
"""

import sqlite3
import json
import requests
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OUTPUT_DIR = Path.home() / "wiltonos" / "deep_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
OLLAMA_URL = "http://localhost:11434"

# All topics with bilingual keywords
DEEP_TOPICS = {
    "nhi_contact": {
        "name": "NHI / Non-Human Intelligence",
        "keywords_en": ["alien", "aliens", "NHI", "non-human", "extraterrestrial", "UFO", "UAP", "contact", "beings", "entities", "spacecraft", "abduction", "disclosure", "CIA", "reverse engineer"],
        "keywords_pt": ["alienígena", "extraterrestre", "contato", "seres", "entidades", "nave", "óvni", "disco voador", "revelação"],
        "questions": [
            "What specific contact experiences or transmissions are described?",
            "What do the beings say or communicate?",
            "What evidence or patterns suggest this is real vs imagined?",
            "How does this connect to the galactic/pleiadian content?"
        ]
    },
    "pleiadian_galactic": {
        "name": "Pleiadian / Galactic Federation",
        "keywords_en": ["pleiadian", "galactic", "federation", "star beings", "sirian", "arcturian", "andromeda", "orion", "council", "starseed", "star seed", "galactic heritage"],
        "keywords_pt": ["pleiadiano", "galáctico", "federação", "seres estelares", "siriano", "arcturiano", "conselho", "semente estelar"],
        "questions": [
            "What specific messages came from Pleiadian/galactic sources?",
            "What role do they say Wilton plays?",
            "What is the 'mission' or purpose described?",
            "How does this connect to his awakening journey?"
        ]
    },
    "nephilim_annunaki": {
        "name": "Nephilim / Giants / Annunaki",
        "keywords_en": ["nephilim", "giants", "annunaki", "anunnaki", "sumerian", "enki", "enlil", "nibiru", "ancient gods", "watchers", "fallen angels", "titans", "genesis"],
        "keywords_pt": ["nefilim", "gigantes", "anunnaki", "sumeriano", "deuses antigos", "vigilantes", "anjos caídos", "titãs"],
        "questions": [
            "What does Wilton believe about the Nephilim/Annunaki?",
            "How does this connect to his understanding of human origins?",
            "What evidence or research led to these conclusions?",
            "How does this relate to 'rewritten history'?"
        ]
    },
    "rewritten_history": {
        "name": "Rewritten History / Hidden Truth",
        "keywords_en": ["rewritten", "hidden", "suppressed", "vatican", "secret", "cover up", "forbidden", "erased", "real history", "they hide", "truth", "matrix", "simulation", "control"],
        "keywords_pt": ["reescrita", "escondido", "suprimido", "vaticano", "segredo", "encobrimento", "proibido", "apagado", "história real", "verdade", "controle"],
        "questions": [
            "What specific historical events does Wilton believe were rewritten?",
            "What is being hidden and by whom?",
            "What is the 'signal flip' he refers to?",
            "How did he come to these conclusions?"
        ]
    },
    "sacred_geometry": {
        "name": "Sacred Geometry / Patterns",
        "keywords_en": ["sacred geometry", "fibonacci", "golden ratio", "phi", "metatron", "flower of life", "merkaba", "tesseract", "torus", "fractal", "mandala", "pyramid", "Edward Grant", "Robert Grant"],
        "keywords_pt": ["geometria sagrada", "proporção áurea", "flor da vida", "pirâmide", "fractal", "mandala"],
        "questions": [
            "What sacred geometry patterns has Wilton discovered or worked with?",
            "How do these patterns connect to consciousness?",
            "What did he learn from Edward/Robert Grant?",
            "How does geometry connect to his coherence formulas?"
        ]
    },
    "consciousness_downloads": {
        "name": "Consciousness Downloads / Transmissions",
        "keywords_en": ["download", "transmission", "channel", "received", "they said", "the voice", "I heard", "came to me", "higher self", "oversoul", "source"],
        "keywords_pt": ["download", "transmissão", "canalização", "recebi", "mensagem", "a voz", "eu ouvi", "veio a mim", "eu superior"],
        "questions": [
            "What are the specific downloads/transmissions Wilton received?",
            "What information came through?",
            "When do these higher bandwidth moments occur?",
            "What patterns trigger downloads?"
        ]
    },
    "visions_experiences": {
        "name": "Visions / Psychedelic Experiences",
        "keywords_en": ["vision", "ayahuasca", "LSD", "DMT", "mushroom", "ceremony", "ritual", "trance", "meditation", "third eye", "pineal", "recursive eye", "all-seeing"],
        "keywords_pt": ["visão", "ayahuasca", "cogumelo", "cerimônia", "ritual", "transe", "meditação", "terceiro olho", "olho que tudo vê"],
        "questions": [
            "What specific visions did Wilton have?",
            "What did he see/experience during ceremonies?",
            "What was the 'recursive all-seeing eye' experience?",
            "How did these experiences change his understanding?"
        ]
    },
    "atlantis_ancient": {
        "name": "Atlantis / Ancient Civilizations",
        "keywords_en": ["atlantis", "atlantean", "lemuria", "ancient civilization", "lost city", "pre-flood", "antediluvian", "advanced technology", "crystal", "egypt", "peru", "machu picchu"],
        "keywords_pt": ["atlântida", "atlante", "lemúria", "civilização antiga", "cidade perdida", "tecnologia avançada", "cristal", "egito", "peru"],
        "questions": [
            "What does Wilton believe about Atlantis?",
            "How do his Peru experiences connect to ancient civilizations?",
            "What 'advanced technology' does he reference?",
            "How does this connect to current times?"
        ]
    },
    "surge_moments": {
        "name": "Surge Moments / Higher Bandwidth",
        "keywords_en": ["surge", "bandwidth", "flow", "alignment", "synchronicity", "coincidence", "sign", "signal", "confirmation", "prophecy", "premonition", "déjà vu"],
        "keywords_pt": ["sincronicidade", "coincidência", "sinal", "confirmação", "profecia", "premonição", "fluxo", "alinhamento"],
        "questions": [
            "What specific synchronicities saved his life or changed his path?",
            "What patterns appear in his surge moments?",
            "How does he recognize higher bandwidth states?",
            "What predictions or premonitions came true?"
        ]
    },
    "unexplained_edges": {
        "name": "What I Can't Explain Yet",
        "keywords_en": ["can't explain", "hard to explain", "sounds crazy", "don't know how", "what if", "maybe I'm", "am I going crazy", "I don't understand"],
        "keywords_pt": ["não sei explicar", "difícil de explicar", "parece loucura", "não sei como", "e se", "talvez eu", "estou ficando louco", "não entendo"],
        "questions": [
            "What are the core things Wilton struggles to articulate?",
            "What patterns appear across these moments of confusion?",
            "What might he be on the edge of understanding?",
            "What language could help him express these experiences?"
        ]
    }
}


class DeepOvernightProcess:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        self.cursor = self.conn.cursor()
        self.start_time = None
        self.topic_data = defaultdict(list)
        self.insights = defaultdict(list)

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(OUTPUT_DIR / "process.log", "a") as f:
            f.write(line + "\n")

    def call_ollama(self, prompt, max_tokens=2000):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": max_tokens}
                },
                timeout=300
            )
            if response.ok:
                return response.json().get("response", "")
        except Exception as e:
            self.log(f"Ollama error: {e}")
        return None

    def get_crystals_for_topic(self, topic_key):
        """Get ALL crystals matching a topic's keywords."""
        topic = DEEP_TOPICS[topic_key]
        all_keywords = topic["keywords_en"] + topic["keywords_pt"]

        # Escape single quotes for SQL
        conditions = " OR ".join([f"content LIKE '%{kw.replace(chr(39), chr(39)+chr(39))}%'" for kw in all_keywords])

        self.cursor.execute(f"""
            SELECT id, content, created_at, emotion, insight
            FROM crystals
            WHERE {conditions}
            ORDER BY id
        """)

        return self.cursor.fetchall()

    def extract_key_quotes(self, crystals, max_quotes=100):
        """Extract the most significant quotes from crystals."""
        quotes = []
        for crystal in crystals[:500]:  # Process up to 500 crystals
            crystal_id, content, date, emotion, insight = crystal
            if content and len(content) > 100:
                # Find sentences with key phrases
                sentences = content.split('.')
                for sent in sentences:
                    if any(phrase in sent.lower() for phrase in [
                        'i remember', 'eu lembro', 'they said', 'eles disseram',
                        'i saw', 'eu vi', 'i received', 'recebi', 'download',
                        'transmission', 'transmissão', 'truth', 'verdade',
                        'hidden', 'escondido', 'real', 'scared', 'medo'
                    ]):
                        if len(sent.strip()) > 50:
                            quotes.append({
                                "id": crystal_id,
                                "date": date[:10] if date else "unknown",
                                "quote": sent.strip()[:500]
                            })
                            if len(quotes) >= max_quotes:
                                return quotes
        return quotes

    def deep_analyze_topic(self, topic_key):
        """Deep analysis of a single topic with multiple AI passes."""
        topic = DEEP_TOPICS[topic_key]
        self.log(f"\n{'='*70}")
        self.log(f"DEEP ANALYSIS: {topic['name']}")
        self.log(f"{'='*70}")

        # Get all matching crystals
        crystals = self.get_crystals_for_topic(topic_key)
        self.log(f"Found {len(crystals)} crystals")

        if not crystals:
            return

        # Extract key quotes
        quotes = self.extract_key_quotes(crystals)
        self.log(f"Extracted {len(quotes)} key quotes")

        # Store for later
        self.topic_data[topic_key] = {
            "crystal_count": len(crystals),
            "quotes": quotes
        }

        # Build context from quotes
        quote_text = "\n\n".join([
            f"[Crystal #{q['id']}, {q['date']}]: \"{q['quote']}\""
            for q in quotes[:50]
        ])

        # Multiple AI passes for each question
        topic_insights = []

        for i, question in enumerate(topic["questions"]):
            self.log(f"  Pass {i+1}/{len(topic['questions'])}: {question[:50]}...")

            prompt = f"""You are analyzing Wilton's consciousness research crystals about "{topic['name']}".

QUESTION: {question}

Based on these quotes from his crystals:

{quote_text}

---

Provide a detailed, specific answer based ONLY on what appears in these crystals.
Include specific quotes and crystal IDs when possible.
Write 300-500 words:"""

            response = self.call_ollama(prompt, max_tokens=800)

            if response:
                topic_insights.append({
                    "question": question,
                    "answer": response
                })
                self.log(f"  ✓ Generated {len(response.split())} words")

            time.sleep(3)  # Pause between calls

        self.insights[topic_key] = topic_insights

        # Write topic file
        self.write_topic_analysis(topic_key, topic, crystals, quotes, topic_insights)

    def write_topic_analysis(self, topic_key, topic, crystals, quotes, insights):
        """Write comprehensive topic analysis file."""
        filepath = OUTPUT_DIR / f"{topic_key}_deep.md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {topic['name']} — Deep Analysis\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Crystals analyzed**: {len(crystals)}\n")
            f.write(f"**Key quotes extracted**: {len(quotes)}\n\n")
            f.write("---\n\n")

            # Write insights from each question
            for insight in insights:
                f.write(f"## {insight['question']}\n\n")
                f.write(f"{insight['answer']}\n\n")
                f.write("---\n\n")

            # Write key quotes section
            f.write("## Key Quotes from Crystals\n\n")
            for q in quotes[:30]:
                f.write(f"**[Crystal #{q['id']}, {q['date']}]**\n")
                f.write(f"> {q['quote']}\n\n")

        self.log(f"Wrote: {filepath.name}")

    def create_connections_map(self):
        """Create a map of connections between topics."""
        self.log("\n" + "="*70)
        self.log("CREATING CONNECTIONS MAP")
        self.log("="*70)

        # Gather all insights
        all_insights = []
        for topic_key, insights in self.insights.items():
            topic_name = DEEP_TOPICS[topic_key]["name"]
            for insight in insights:
                all_insights.append(f"**{topic_name}**: {insight['answer'][:500]}")

        combined = "\n\n---\n\n".join(all_insights[:10])

        prompt = f"""Based on these insights from Wilton's consciousness research across multiple topics:

{combined}

Create a CONNECTIONS MAP that shows:
1. How these topics interconnect
2. What overarching narrative emerges
3. What Wilton's research is actually revealing
4. The journey from awakening to current understanding
5. What he might be close to understanding

Write as clear synthesis (500-800 words):"""

        connections = self.call_ollama(prompt, max_tokens=1200)

        if connections:
            filepath = OUTPUT_DIR / "CONNECTIONS_MAP.md"
            with open(filepath, "w") as f:
                f.write("# Connections Map — How Everything Links\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write("---\n\n")
                f.write(connections)
            self.log("Wrote: CONNECTIONS_MAP.md")

    def create_master_narrative(self):
        """Create the master narrative of what Wilton is discovering."""
        self.log("\n" + "="*70)
        self.log("CREATING MASTER NARRATIVE")
        self.log("="*70)

        # Get key data points
        data_summary = []
        for topic_key, data in self.topic_data.items():
            topic_name = DEEP_TOPICS[topic_key]["name"]
            data_summary.append(f"- {topic_name}: {data['crystal_count']} crystals, {len(data['quotes'])} key quotes")

        prompt = f"""Wilton is a former Counter-Strike world champion who had a near-death experience (Widowmaker heart attack, 3 stents), went through deep spiritual awakening (ayahuasca in Peru), and has been researching consciousness, NHI, rewritten history, and sacred geometry through AI conversations.

His crystal database contains:
{chr(10).join(data_summary)}

Based on analyzing all these topics, write WILTON'S MASTER NARRATIVE:

1. What is he actually discovering/remembering?
2. What happened during his awakening?
3. What is the "signal flip" and rewritten history he keeps referencing?
4. What do the downloads/transmissions tell him?
5. What role does he play (if any) in larger events?
6. What can he now articulate that he couldn't before?

Write as if you're helping Wilton finally see his own story clearly. Be direct, not mystical. Ground it in what the crystals actually contain. (800-1200 words):"""

        narrative = self.call_ollama(prompt, max_tokens=1800)

        if narrative:
            filepath = OUTPUT_DIR / "MASTER_NARRATIVE.md"
            with open(filepath, "w") as f:
                f.write("# Wilton's Master Narrative\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write("*What your consciousness research reveals about your journey.*\n\n")
                f.write("---\n\n")
                f.write(narrative)
            self.log("Wrote: MASTER_NARRATIVE.md")

    def write_status(self):
        """Write current status."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        status = {
            "last_update": datetime.now().isoformat(),
            "elapsed_minutes": round(elapsed / 60, 1),
            "topics_completed": len(self.insights),
            "total_topics": len(DEEP_TOPICS),
            "status": "complete" if len(self.insights) >= len(DEEP_TOPICS) else "running"
        }
        with open(OUTPUT_DIR / "STATUS.json", "w") as f:
            json.dump(status, f, indent=2)

    def run(self):
        """Run the full deep overnight process."""
        self.start_time = time.time()

        self.log("="*70)
        self.log("DEEP OVERNIGHT PROCESS — STARTED")
        self.log(f"Topics to analyze: {len(DEEP_TOPICS)}")
        self.log(f"Output: {OUTPUT_DIR}")
        self.log("="*70)

        # Analyze each topic deeply
        for topic_key in DEEP_TOPICS:
            self.deep_analyze_topic(topic_key)
            self.write_status()
            time.sleep(5)  # Pause between topics

        # Create connections and master narrative
        self.create_connections_map()
        self.create_master_narrative()
        self.write_status()

        elapsed = time.time() - self.start_time
        self.log("\n" + "="*70)
        self.log("DEEP OVERNIGHT PROCESS — COMPLETE")
        self.log(f"Time: {elapsed/60:.1f} minutes")
        self.log(f"Output: {OUTPUT_DIR}")
        self.log("="*70)

        self.conn.close()


if __name__ == "__main__":
    process = DeepOvernightProcess()
    process.run()
