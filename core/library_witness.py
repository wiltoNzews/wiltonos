#!/usr/bin/env python3
"""
Library Witness - Deep Understanding of Indexed Knowledge
=========================================================
Applies the deep reading pattern to library content.

Not just "make it searchable" but "understand what's there."

For each library chunk, extracts:
- domain: theory/implementation/architecture/reference
- significance: foundation/building/advanced/reference
- concepts_introduced: What terms/ideas appear here
- depends_on: What must be understood first
- learning: One-sentence extracted wisdom

Follows the pattern of deep_witness.py but for library, not crystals.

Usage:
    python library_witness.py                    # Process unwitnessed chunks
    python library_witness.py --limit 100        # Process 100 chunks
    python library_witness.py --category replit_gold  # Only replit gold
    python library_witness.py --stats            # Show progress stats
    python library_witness.py --inventory        # Generate knowledge inventory

January 2026 - The system understanding itself
"""

import sys
import json
import sqlite3
import argparse
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:latest"  # Fast general model
REASONING_MODEL = "deepseek-r1:32b"  # For complex analysis

# Output paths
OUTPUT_DIR = Path.home() / "wiltonos" / "data" / "library_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class KnowledgeExtraction:
    """Extracted understanding from a library chunk."""
    domain: str  # theory/implementation/architecture/reference
    significance: str  # foundation/building/advanced/reference
    concepts_introduced: List[str]
    depends_on: List[str]
    learning: str  # One sentence

    def to_dict(self) -> Dict:
        return asdict(self)


EXTRACTION_PROMPT = """Extract info from this text. Return ONLY valid JSON, no other text.

TEXT:
{content}

Required JSON format:
{{"domain":"<theory|implementation|architecture|reference>","significance":"<foundation|building|advanced|reference>","concepts":["term1","term2","term3"],"insight":"<the actual principle/technique/wisdom - what you would REMEMBER from this>"}}"""


class LibraryWitness:
    """
    Deep reads library chunks to extract structured understanding.
    """

    def __init__(self, db_path: Path = DB_PATH, model: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.model = model
        self.stats = {
            'processed': 0,
            'errors': 0,
            'concepts_found': set(),
            'domain_counts': {},
            'significance_counts': {}
        }

    def get_unwitnessed_chunks(
        self,
        limit: int = 100,
        category: str = None,
        file_type: str = None
    ) -> List[Dict]:
        """Get library chunks that haven't been deep-read yet."""
        # Use timeout and WAL mode for concurrent access
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        conditions = ["witnessed_at IS NULL"]
        params = []

        if category:
            conditions.append("category LIKE ?")
            params.append(f"%{category}%")

        if file_type:
            conditions.append("file_type = ?")
            params.append(file_type)

        where_clause = " AND ".join(conditions)

        c.execute(f"""
            SELECT id, file_path, file_type, chunk_index, content, category, tags
            FROM library
            WHERE {where_clause}
            ORDER BY
                CASE
                    WHEN category LIKE '%tecnologias%' THEN 1
                    WHEN category LIKE '%o4%' THEN 2
                    WHEN category LIKE '%codex%' THEN 3
                    WHEN category LIKE '%replit_gold%' THEN 4
                    ELSE 5
                END,
                id
            LIMIT ?
        """, params + [limit])

        rows = c.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def extract_knowledge(self, chunk: Dict) -> Optional[KnowledgeExtraction]:
        """Use LLM to extract structured understanding from a chunk."""
        content = chunk.get('content', '')
        if len(content) < 50:
            return None

        # Truncate very long content
        if len(content) > 3000:
            content = content[:3000] + "\n..."

        prompt = EXTRACTION_PROMPT.format(content=content)

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            response.raise_for_status()

            result_text = response.json().get("response", "")

            # Extract JSON from response - handle nested arrays
            # Find the outermost { ... } block
            brace_count = 0
            start_idx = None
            end_idx = None
            for i, char in enumerate(result_text):
                if char == '{':
                    if start_idx is None:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx is not None:
                        end_idx = i + 1
                        break

            if start_idx is not None and end_idx is not None:
                json_str = result_text[start_idx:end_idx]
                data = json.loads(json_str)

                # Handle both old and new field names
                concepts = data.get('concepts', data.get('concepts_introduced', []))

                # Validate domain (must be one of allowed values)
                valid_domains = {'theory', 'implementation', 'architecture', 'reference'}
                domain = data.get('domain', 'reference').lower().strip()
                if domain not in valid_domains:
                    domain = 'reference'

                # Validate significance
                valid_significance = {'foundation', 'building', 'advanced', 'reference'}
                significance = data.get('significance', 'reference').lower().strip()
                if significance not in valid_significance:
                    significance = 'reference'

                # Filter out template literals from concepts
                template_literals = {'key', 'terms', 'max', '5', 'term1', 'term2', 'term3'}
                if isinstance(concepts, list):
                    concepts = [c for c in concepts if isinstance(c, str) and c.lower() not in template_literals]

                # Get insight/learning (check both field names)
                insight = data.get('insight', data.get('learning', ''))
                # Filter out meta-descriptions and placeholders
                if isinstance(insight, str):
                    insight_lower = insight.lower().strip()
                    bad_starts = ('this document', 'this section', 'describes how', 'the document', '<the actual', 'the actual')
                    if insight_lower.startswith(bad_starts) or 'what you would remember' in insight_lower:
                        insight = ''  # Will be counted as missing

                return KnowledgeExtraction(
                    domain=domain,
                    significance=significance,
                    concepts_introduced=concepts[:10] if isinstance(concepts, list) else [],
                    depends_on=data.get('depends_on', [])[:5],
                    learning=insight[:500] if isinstance(insight, str) else ''
                )

        except Exception as e:
            logger.debug(f"Extraction error: {e}")
            return None

        return None

    def update_chunk(self, chunk_id: int, knowledge: KnowledgeExtraction) -> bool:
        """Update library chunk with extracted knowledge."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        c = conn.cursor()

        try:
            c.execute("""
                UPDATE library SET
                    significance = ?,
                    domain = ?,
                    concepts_introduced = ?,
                    depends_on = ?,
                    learning = ?,
                    witnessed_at = ?
                WHERE id = ?
            """, (
                knowledge.significance,
                knowledge.domain,
                json.dumps(knowledge.concepts_introduced),
                json.dumps(knowledge.depends_on),
                knowledge.learning,
                datetime.now().isoformat(),
                chunk_id
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
            conn.close()
            return False

    def process_batch(
        self,
        limit: int = 100,
        category: str = None,
        file_type: str = None
    ) -> Dict:
        """Process a batch of unwitnessed chunks."""
        chunks = self.get_unwitnessed_chunks(limit, category, file_type)

        if not chunks:
            logger.info("No unwitnessed chunks found.")
            return self.stats

        logger.info(f"Processing {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            file_name = Path(chunk['file_path']).name
            logger.info(f"[{i+1}/{len(chunks)}] {file_name} (chunk {chunk['chunk_index']})")

            knowledge = self.extract_knowledge(chunk)

            if knowledge:
                if self.update_chunk(chunk['id'], knowledge):
                    self.stats['processed'] += 1

                    # Track stats
                    for concept in knowledge.concepts_introduced:
                        # Handle case where LLM returns dict/nested instead of string
                        if isinstance(concept, str):
                            self.stats['concepts_found'].add(concept.lower())
                        elif isinstance(concept, dict):
                            # Try to extract a string value from the dict
                            for v in concept.values():
                                if isinstance(v, str):
                                    self.stats['concepts_found'].add(v.lower())
                                    break

                    self.stats['domain_counts'][knowledge.domain] = \
                        self.stats['domain_counts'].get(knowledge.domain, 0) + 1

                    self.stats['significance_counts'][knowledge.significance] = \
                        self.stats['significance_counts'].get(knowledge.significance, 0) + 1
            else:
                self.stats['errors'] += 1

            # Progress every 20
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {self.stats['processed']} processed, {self.stats['errors']} errors")

        return self.stats

    def get_progress_stats(self) -> Dict:
        """Get witnessing progress stats."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        c = conn.cursor()

        stats = {}

        c.execute("SELECT COUNT(*) FROM library")
        stats['total_chunks'] = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM library WHERE witnessed_at IS NOT NULL")
        stats['witnessed'] = c.fetchone()[0]

        stats['unwitnessed'] = stats['total_chunks'] - stats['witnessed']
        stats['progress_pct'] = round(100 * stats['witnessed'] / max(1, stats['total_chunks']), 1)

        # Domain distribution (of witnessed)
        c.execute("""
            SELECT domain, COUNT(*) FROM library
            WHERE witnessed_at IS NOT NULL
            GROUP BY domain
        """)
        stats['by_domain'] = dict(c.fetchall())

        # Significance distribution
        c.execute("""
            SELECT significance, COUNT(*) FROM library
            WHERE witnessed_at IS NOT NULL
            GROUP BY significance
        """)
        stats['by_significance'] = dict(c.fetchall())

        # Top concepts
        c.execute("""
            SELECT concepts_introduced FROM library
            WHERE witnessed_at IS NOT NULL AND concepts_introduced IS NOT NULL
        """)
        concept_counts = {}
        for row in c.fetchall():
            try:
                concepts = json.loads(row[0])
                for c_name in concepts:
                    c_lower = c_name.lower()
                    concept_counts[c_lower] = concept_counts.get(c_lower, 0) + 1
            except:
                pass

        stats['top_concepts'] = sorted(
            concept_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:30]

        conn.close()
        return stats

    def generate_inventory(self) -> str:
        """Generate a knowledge inventory report."""
        stats = self.get_progress_stats()

        lines = []
        lines.append("# Knowledge Inventory")
        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        lines.append(f"\n## Progress")
        lines.append(f"- Total chunks: {stats['total_chunks']}")
        lines.append(f"- Witnessed: {stats['witnessed']} ({stats['progress_pct']}%)")
        lines.append(f"- Remaining: {stats['unwitnessed']}")

        lines.append(f"\n## Domain Distribution")
        for domain, count in sorted(stats['by_domain'].items(), key=lambda x: -x[1]):
            lines.append(f"- {domain}: {count}")

        lines.append(f"\n## Significance Distribution")
        for sig, count in sorted(stats['by_significance'].items(), key=lambda x: -x[1]):
            lines.append(f"- {sig}: {count}")

        lines.append(f"\n## Top Concepts (by frequency)")
        for concept, count in stats['top_concepts']:
            lines.append(f"- {concept}: {count}")

        # Get some example learnings
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            SELECT file_path, learning FROM library
            WHERE witnessed_at IS NOT NULL
              AND learning IS NOT NULL
              AND significance = 'foundation'
            ORDER BY RANDOM()
            LIMIT 10
        """)
        foundation_learnings = c.fetchall()
        conn.close()

        if foundation_learnings:
            lines.append(f"\n## Foundation Learnings (sample)")
            for path, learning in foundation_learnings:
                file_name = Path(path).name
                lines.append(f"\n**{file_name}**")
                lines.append(f"> {learning}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Deep read library content to extract understanding"
    )

    parser.add_argument("--limit", type=int, default=100,
                        help="Number of chunks to process")
    parser.add_argument("--category", type=str,
                        help="Filter by category (e.g. replit_gold)")
    parser.add_argument("--file-type", type=str,
                        help="Filter by file type (e.g. .md)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Ollama model to use")
    parser.add_argument("--stats", action="store_true",
                        help="Show progress statistics")
    parser.add_argument("--inventory", action="store_true",
                        help="Generate knowledge inventory")

    args = parser.parse_args()

    witness = LibraryWitness(model=args.model)

    if args.stats:
        stats = witness.get_progress_stats()
        print("\n" + "=" * 60)
        print("LIBRARY WITNESS PROGRESS")
        print("=" * 60)
        print(f"Total chunks:  {stats['total_chunks']}")
        print(f"Witnessed:     {stats['witnessed']} ({stats['progress_pct']}%)")
        print(f"Remaining:     {stats['unwitnessed']}")
        print("\nBy domain:")
        for d, c in stats['by_domain'].items():
            print(f"  {d}: {c}")
        print("\nBy significance:")
        for s, c in stats['by_significance'].items():
            print(f"  {s}: {c}")
        print("\nTop concepts:")
        for concept, count in stats['top_concepts'][:15]:
            print(f"  {concept}: {count}")
        return

    if args.inventory:
        inventory = witness.generate_inventory()
        print(inventory)

        # Also save to file
        inv_path = OUTPUT_DIR / "knowledge_inventory.md"
        inv_path.write_text(inventory)
        print(f"\nSaved to: {inv_path}")
        return

    # Process batch
    stats = witness.process_batch(
        limit=args.limit,
        category=args.category,
        file_type=args.file_type
    )

    print("\n" + "=" * 60)
    print("WITNESS COMPLETE")
    print("=" * 60)
    print(f"Processed:     {stats['processed']}")
    print(f"Errors:        {stats['errors']}")
    print(f"Unique concepts: {len(stats['concepts_found'])}")
    print("\nBy domain:")
    for d, c in stats['domain_counts'].items():
        print(f"  {d}: {c}")
    print("\nBy significance:")
    for s, c in stats['significance_counts'].items():
        print(f"  {s}: {c}")
    print("=" * 60)


if __name__ == "__main__":
    main()
