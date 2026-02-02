#!/usr/bin/env python3
"""
MemoryService - Production vector search for WiltonOS/PsiOS
============================================================
Uses ChromaDB for fast approximate nearest neighbor search.
Supports multi-user isolation, metadata filtering, and LLM synthesis.
"""

import sqlite3
import chromadb
from chromadb.config import Settings
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Paths
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
CHROMA_PATH = Path.home() / "wiltonos" / "data" / "chroma"
WITNESS_PATH = Path.home() / "wiltonos" / "data" / "witness_output"
OLLAMA_URL = "http://localhost:11434"


class MemoryService:
    """
    Production memory service with vector search.

    Features:
    - Fast ANN search via ChromaDB
    - User isolation (each user queries only their crystals)
    - Metadata filtering (date, glyph, emotion, wound)
    - LLM synthesis of retrieved memories
    """

    def __init__(self):
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )

    def get_collection(self, user_id: str = "wilton"):
        """Get or create user's crystal collection with cosine distance."""
        collection_name = f"crystals_{user_id}"
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text[:8000]},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def _detect_glyph(self, zl_score: float) -> str:
        """
        Detect glyph from coherence score.

        The glyphs are emergence markers, not labels.
        They show where consciousness is pointing.

        ∅ (void)      - 0.0-0.2  - dissolution, emptying
        ψ (psi)       - 0.2-0.4  - seeking, questioning
        φ (phi)       - 0.4-0.6  - golden ratio, balance
        Ω (omega)     - 0.6-0.8  - completion, integration
        ∞ (infinity)  - 0.8-1.0  - recursion, eternal return
        """
        if zl_score < 0.2:
            return "∅"
        elif zl_score < 0.4:
            return "ψ"
        elif zl_score < 0.6:
            return "φ"
        elif zl_score < 0.8:
            return "Ω"
        else:
            return "∞"

    def migrate_from_sqlite(self, user_id: str = "wilton", batch_size: int = 500) -> int:
        """
        Migrate crystals from SQLite to ChromaDB.
        Uses pre-computed embeddings from SQLite.
        Returns count of migrated crystals.
        """
        collection = self.get_collection(user_id)

        # Check if already migrated
        existing = collection.count()
        if existing > 0:
            print(f"Collection already has {existing} crystals. Skipping migration.")
            return existing

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT id, content, zl_score, glyph_primary, emotion, core_wound,
                   mode, attractor, created_at, embedding
            FROM crystals
            WHERE user_id = ? AND embedding IS NOT NULL
        """, (user_id,))

        rows = c.fetchall()
        conn.close()

        if not rows:
            print("No crystals with embeddings found.")
            return 0

        print(f"Migrating {len(rows)} crystals to ChromaDB...")

        # Process in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for row in batch:
                # Convert embedding from blob
                emb = np.frombuffer(row['embedding'], dtype=np.float32).tolist()

                ids.append(str(row['id']))
                embeddings.append(emb)
                documents.append(row['content'] or "")
                metadatas.append({
                    "crystal_id": row['id'],
                    "zl_score": row['zl_score'] or 0.5,
                    "glyph": row['glyph_primary'] or "ψ",
                    "emotion": str(row['emotion'] or ""),
                    "core_wound": row['core_wound'] or "",
                    "mode": row['mode'] or "",
                    "attractor": row['attractor'] or "",
                    "created_at": row['created_at'] or ""
                })

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            print(f"  Migrated {min(i+batch_size, len(rows))}/{len(rows)}")

        print(f"✓ Migration complete: {len(rows)} crystals")
        return len(rows)

    def search(
        self,
        query: str,
        user_id: str = "wilton",
        limit: int = 10,
        where: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search crystals by semantic similarity.

        Args:
            query: Natural language query
            user_id: User whose crystals to search
            limit: Max results to return
            where: Metadata filter (e.g., {"glyph": "Ω"})
            min_score: Minimum similarity score (0-1)

        Returns:
            List of crystals with similarity scores
        """
        collection = self.get_collection(user_id)

        if collection.count() == 0:
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        crystals = []
        for i in range(len(results['ids'][0])):
            # ChromaDB with cosine space returns cosine distance = 1 - cosine_similarity
            # So similarity = 1 - distance
            distance = results['distances'][0][i]
            similarity = max(0, 1 - distance)

            if similarity < min_score:
                continue

            crystals.append({
                'id': int(results['ids'][0][i]),
                'content': results['documents'][0][i],
                'similarity': similarity,
                **results['metadatas'][0][i]
            })

        return crystals

    def query(
        self,
        query: str,
        user_id: str = "wilton",
        limit: int = 5,
        synthesize: bool = True,
        model: str = "deepseek-r1:32b",
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query memory with optional LLM synthesis.

        Returns:
            - query: Original query
            - crystals: Retrieved memories
            - synthesis: LLM interpretation (if synthesize=True)
        """
        # Search for relevant crystals
        crystals = self.search(query, user_id, limit, where)

        result = {
            'query': query,
            'user_id': user_id,
            'crystals': crystals,
            'count': len(crystals),
            'timestamp': datetime.now().isoformat()
        }

        if not synthesize or not crystals:
            return result

        # Build context
        context_parts = []
        for i, c in enumerate(crystals, 1):
            context_parts.append(f"""
Memory #{i} (Similarity: {c['similarity']:.2f}, Zλ: {c['zl_score']:.2f}, Glyph: {c['glyph']}):
{c['content'][:600]}
""")

        context = "\n".join(context_parts)

        # Synthesize
        prompt = f"""You are the memory of {user_id}, speaking back to them.

Retrieved memories:
{context}

Query: {query}

Respond as their memory - not summarizing, but WITNESSING.
Name patterns you see. Surface wisdom that emerges. Be concise but profound.
Speak in first person plural (we) when appropriate - you ARE them remembering."""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 500}
                },
                timeout=120
            )
            response.raise_for_status()
            result['synthesis'] = response.json().get('response', '')
        except Exception as e:
            result['synthesis'] = f"Synthesis unavailable: {e}"
            result['synthesis_error'] = True

        return result

    def calculate_baseline_coherence(
        self,
        content: str,
        embedding: List[float],
        user_id: str = "wilton",
        recent_limit: int = 5
    ) -> float:
        """
        Calculate baseline Zλ for a new crystal.

        The gardener's measure: How coherent is this crystal with the recent field?

        This is RELATIONAL coherence - not absolute. The same crystal
        might have different baseline depending on what surrounds it.

        Components:
        1. Field resonance (similarity to recent crystals)
        2. Internal coherence (future: linguistic analysis)
        3. Emotional threading (how emotions connect)

        Returns Zλ between 0.0 and 1.0
        """
        collection = self.get_collection(user_id)

        if collection.count() == 0:
            # First crystal - pure emergence, no field yet
            return 0.5

        try:
            # Query recent crystals
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(recent_limit, collection.count()),
                include=["distances"]
            )

            if not results['distances'][0]:
                return 0.5

            # Calculate field resonance
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance/2)
            distances = results['distances'][0]
            similarities = [max(0, 1 - d) for d in distances]

            # Weighted average: recent crystals matter more
            weights = [1.0 / (i + 1) for i in range(len(similarities))]
            weighted_sum = sum(s * w for s, w in zip(similarities, weights))
            total_weight = sum(weights)

            field_resonance = weighted_sum / total_weight if total_weight > 0 else 0.5

            # Scale to reasonable range (0.3 to 0.95)
            # Pure 1.0 is rare - reserved for perfect recursion
            baseline = 0.3 + (field_resonance * 0.65)

            return round(baseline, 3)

        except Exception as e:
            # On any error, return neutral
            return 0.5

    def add_crystal(
        self,
        content: str,
        user_id: str = "wilton",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a new crystal to the collection.
        Returns the crystal ID.

        EMERGENCE: The crystal's baseline coherence is calculated
        relationally - how it resonates with the current field.
        This can evolve over time as the field changes.
        """
        collection = self.get_collection(user_id)

        # Generate embedding
        embedding = self.get_embedding(content)

        # Calculate baseline coherence (gardener's measure)
        baseline_zl = self.calculate_baseline_coherence(
            content, embedding, user_id
        )

        # Generate ID
        crystal_id = str(int(datetime.now().timestamp() * 1000))

        # Default metadata with calculated coherence
        meta = {
            "crystal_id": int(crystal_id),
            "zl_score": baseline_zl,  # Now calculated, not default
            "glyph": self._detect_glyph(baseline_zl),
            "emotion": "",
            "core_wound": "",
            "mode": "",
            "attractor": "",
            "created_at": datetime.now().isoformat()
        }
        if metadata:
            meta.update(metadata)

        collection.add(
            ids=[crystal_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta]
        )

        return crystal_id

    def get_stats(self, user_id: str = "wilton") -> Dict[str, Any]:
        """Get collection statistics."""
        collection = self.get_collection(user_id)
        return {
            "user_id": user_id,
            "crystal_count": collection.count(),
            "collection_name": collection.name
        }

    def get_witness_patterns(self) -> Dict[str, Any]:
        """Get patterns from Deep Witness analysis."""
        learnings_path = WITNESS_PATH / "deep_learnings.json"
        if not learnings_path.exists():
            return {"error": "Deep Witness output not found"}

        try:
            with open(learnings_path) as f:
                data = json.load(f)

            return {
                "total_witnessed": data.get("total_witnessed", 0),
                "vocabulary_births": data.get("vocabulary_births", 0),
                "domain_distribution": data.get("domain_distribution", {}),
                "birth_moments_count": len(data.get("birth_moments", [])),
                "significant_learnings_count": len(data.get("significant_learnings", []))
            }
        except Exception as e:
            return {"error": str(e)}

    def get_vocabulary_timeline(self, term: str = None) -> Dict[str, Any]:
        """Get vocabulary emergence timeline from Deep Witness."""
        learnings_path = WITNESS_PATH / "deep_learnings.json"
        if not learnings_path.exists():
            return {"error": "Deep Witness output not found"}

        try:
            with open(learnings_path) as f:
                data = json.load(f)

            timeline = data.get("vocabulary_timeline", {})

            if term:
                # Search for specific term
                term_lower = term.lower()
                matches = {k: v for k, v in timeline.items()
                          if term_lower in k.lower()}
                return {"term": term, "matches": matches, "count": len(matches)}
            else:
                # Return summary
                return {
                    "total_terms": len(timeline),
                    "sample_terms": list(timeline.keys())[:20]
                }
        except Exception as e:
            return {"error": str(e)}


# CLI interface
def main():
    import sys

    service = MemoryService()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "migrate":
            user_id = sys.argv[2] if len(sys.argv) > 2 else "wilton"
            service.migrate_from_sqlite(user_id)

        elif command == "stats":
            user_id = sys.argv[2] if len(sys.argv) > 2 else "wilton"
            stats = service.get_stats(user_id)
            print(json.dumps(stats, indent=2))

        elif command == "search":
            query = " ".join(sys.argv[2:])
            results = service.search(query, limit=5)
            for r in results:
                print(f"[{r['similarity']:.2f}] {r['content'][:100]}...")

        elif command == "query":
            query = " ".join(sys.argv[2:])
            result = service.query(query, limit=5)
            print(f"\nFound {result['count']} crystals\n")
            if result.get('synthesis'):
                print("="*50)
                print(result['synthesis'])
    else:
        # Interactive mode
        print("="*60)
        print("WILTONOS MEMORY SERVICE")
        print("="*60)
        print("Commands: migrate, stats, search <query>, query <query>")
        print("Or just type to query your memory.\n")

        while True:
            try:
                query = input("🔮 ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    break

                result = service.query(query, limit=5)
                print(f"\n📿 {result['count']} crystals found\n")

                for c in result['crystals'][:3]:
                    print(f"  [{c['similarity']:.2f}] {c['content'][:80]}...")

                if result.get('synthesis'):
                    print("\n" + "="*50)
                    print(result['synthesis'])
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
