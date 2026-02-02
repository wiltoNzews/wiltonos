"""
Hybrid Router
=============
Routes queries between local (Ollama) and API (OpenRouter) models
based on complexity, coherence state, and cost optimization.

Architecture:
    Query → Complexity Score → Route Decision
        Low (0-0.3)    → Local Fast (llama3)
        Medium (0.3-0.6) → Local Smart (qwen3:32b)
        High (0.6-0.8)   → Local Deep (deepseek-r1:32b)
        Critical (0.8+)  → API (Claude Opus via OpenRouter)

Also supports:
- Coherence-based escalation (high Zλ moments get better models)
- Semantic caching to reduce API costs
- Fallback chains when models fail

Usage:
    from hybrid_router import HybridRouter, ModelTier

    router = HybridRouter()
    response = router.query("Your question here")

    # Or with coherence context
    response = router.query("Deep question", user_coherence=0.85)
"""

import os
import json
import hashlib
import time
import requests
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
import sqlite3


class ModelTier(Enum):
    LOCAL_FAST = "local_fast"
    LOCAL_SMART = "local_smart"
    LOCAL_DEEP = "local_deep"
    API_QUALITY = "api_quality"
    API_BEST = "api_best"


@dataclass
class ModelConfig:
    tier: ModelTier
    name: str  # Display name
    model_id: str  # Actual model identifier
    provider: str  # "ollama" or "openrouter"
    cost_per_1k_tokens: float  # Approximate cost
    max_tokens: int = 4096
    temperature: float = 0.7


# Model configurations
MODELS = {
    ModelTier.LOCAL_FAST: ModelConfig(
        tier=ModelTier.LOCAL_FAST,
        name="Llama 3 (Fast)",
        model_id="llama3:latest",
        provider="ollama",
        cost_per_1k_tokens=0.0,
    ),
    ModelTier.LOCAL_SMART: ModelConfig(
        tier=ModelTier.LOCAL_SMART,
        name="Qwen3 32B (Smart)",
        model_id="qwen3:32b",
        provider="ollama",
        cost_per_1k_tokens=0.0,
    ),
    ModelTier.LOCAL_DEEP: ModelConfig(
        tier=ModelTier.LOCAL_DEEP,
        name="DeepSeek R1 32B (Deep)",
        model_id="deepseek-r1:32b",
        provider="ollama",
        cost_per_1k_tokens=0.0,
    ),
    ModelTier.API_QUALITY: ModelConfig(
        tier=ModelTier.API_QUALITY,
        name="Claude Sonnet (Quality)",
        model_id="anthropic/claude-sonnet-4",
        provider="openrouter",
        cost_per_1k_tokens=0.015,  # $3 input + $15 output averaged
    ),
    ModelTier.API_BEST: ModelConfig(
        tier=ModelTier.API_BEST,
        name="Claude Opus (Best)",
        model_id="anthropic/claude-opus-4",
        provider="openrouter",
        cost_per_1k_tokens=0.025,  # $5 input + $25 output averaged
    ),
}


class HybridRouter:
    """
    Intelligent router between local and API models.
    """

    # Complexity indicators
    COMPLEX_PATTERNS = [
        # Analytical
        "explain why", "analyze", "compare", "debug", "design",
        "optimize", "relationship between", "step by step",
        "prove", "derive", "synthesize", "integrate",
        # Emotional/introspective
        "what do you think", "how do you feel", "what does this mean",
        "pattern", "notice", "reflect", "deeper", "underneath",
        "when responding", "in yourself", "about yourself",
        # Philosophical/consciousness
        "consciousness", "paradox", "awareness", "meaning", "purpose",
        "identity", "self", "mirror", "truth", "coherence",
        # Emotional content
        "relationship", "feel lighter", "uncertain", "ended", "lost",
        "afraid", "love", "grief", "trauma", "healing",
    ]

    SIMPLE_PATTERNS = [
        "what is", "define", "list", "format", "convert",
        "translate", "summarize", "when was", "who is",
        "how many", "yes or no", "true or false", "hello", "hi"
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        openrouter_key: Optional[str] = None,
        db_path: Optional[str] = None,
        enable_cache: bool = True,
        cache_threshold: float = 0.88,  # Cosine similarity for cache hits
        force_local: bool = False,  # Never use API
        force_api: bool = False,  # Always use API (for testing)
    ):
        self.ollama_url = ollama_url
        self.openrouter_key = openrouter_key or self._load_openrouter_key()
        self.db_path = db_path or str(Path.home() / "wiltonos/data/crystals_unified.db")
        self.enable_cache = enable_cache
        self.cache_threshold = cache_threshold
        self.force_local = force_local
        self.force_api = force_api

        # Initialize cache table
        if enable_cache:
            self._init_cache_table()

        # Track usage
        self.session_stats = {
            "queries": 0,
            "cache_hits": 0,
            "local_calls": 0,
            "api_calls": 0,
            "total_cost": 0.0,
        }

    def _load_openrouter_key(self) -> Optional[str]:
        """Load OpenRouter API key from file"""
        key_file = Path.home() / ".openrouter_key"
        if key_file.exists():
            return key_file.read_text().strip()
        return os.environ.get("OPENROUTER_API_KEY")

    def _init_cache_table(self):
        """Create semantic cache table"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE,
                query_text TEXT,
                response TEXT,
                model_tier TEXT,
                created_at TEXT,
                hit_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def _hash_query(self, query: str) -> str:
        """Create hash for cache lookup"""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _check_cache(self, query: str) -> Optional[str]:
        """Check if we have a cached response"""
        if not self.enable_cache:
            return None

        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT response FROM response_cache
            WHERE query_hash = ?
        """, (query_hash,))

        row = cur.fetchone()
        if row:
            # Update hit count
            cur.execute("""
                UPDATE response_cache SET hit_count = hit_count + 1
                WHERE query_hash = ?
            """, (query_hash,))
            conn.commit()
            conn.close()
            self.session_stats["cache_hits"] += 1
            return row[0]

        conn.close()
        return None

    def _save_to_cache(self, query: str, response: str, tier: ModelTier):
        """Save response to cache"""
        if not self.enable_cache:
            return

        query_hash = self._hash_query(query)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT OR REPLACE INTO response_cache
                (query_hash, query_text, response, model_tier, created_at, hit_count)
                VALUES (?, ?, ?, ?, datetime('now'), 0)
            """, (query_hash, query[:500], response, tier.value))
            conn.commit()
        except Exception as e:
            print(f"[Router] Cache save error: {e}")
        finally:
            conn.close()

    def score_complexity(self, query: str) -> float:
        """
        Score query complexity from 0 (simple) to 1 (complex).
        """
        query_lower = query.lower()
        words = query.split()

        # Pattern matching
        complex_hits = sum(1 for p in self.COMPLEX_PATTERNS if p in query_lower)
        simple_hits = sum(1 for p in self.SIMPLE_PATTERNS if p in query_lower)

        # Length factor (longer queries often more complex)
        length_score = min(len(words) / 50, 1.0) * 0.3

        # Question depth (multiple questions = more complex)
        question_count = query.count("?")
        question_score = min(question_count / 3, 1.0) * 0.2

        # Base score from patterns
        pattern_score = (complex_hits - simple_hits + 2) / 5
        pattern_score = np.clip(pattern_score, 0, 1) * 0.5

        return np.clip(pattern_score + length_score + question_score, 0, 1)

    def select_tier(
        self,
        query: str,
        user_coherence: float = 0.5,
        force_tier: Optional[ModelTier] = None
    ) -> ModelTier:
        """
        Select model tier based on query complexity and user state.
        """
        if force_tier:
            return force_tier

        if self.force_api:
            return ModelTier.API_QUALITY

        if self.force_local:
            complexity = self.score_complexity(query)
            if complexity < 0.5:
                return ModelTier.LOCAL_FAST
            elif complexity < 0.75:
                return ModelTier.LOCAL_SMART
            else:
                return ModelTier.LOCAL_DEEP

        # Normal routing
        complexity = self.score_complexity(query)

        # Coherence boost: high coherence moments deserve better models
        coherence_boost = 0.0
        if user_coherence > 0.85:
            coherence_boost = 0.25
        elif user_coherence > 0.7:
            coherence_boost = 0.15

        effective_complexity = min(complexity + coherence_boost, 1.0)

        # Route decision
        if effective_complexity < 0.3:
            return ModelTier.LOCAL_FAST
        elif effective_complexity < 0.55:
            return ModelTier.LOCAL_SMART
        elif effective_complexity < 0.75:
            return ModelTier.LOCAL_DEEP
        elif effective_complexity < 0.9:
            return ModelTier.API_QUALITY
        else:
            return ModelTier.API_BEST

    def _call_ollama(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model_id,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            return f"[Ollama Error: {e}]"

    def _call_openrouter(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> str:
        """Call OpenRouter API"""
        if not self.openrouter_key:
            return "[Error: No OpenRouter API key]"

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"[OpenRouter Error: {e}]"

    def query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        user_coherence: float = 0.5,
        force_tier: Optional[ModelTier] = None,
        skip_cache: bool = False,
    ) -> Tuple[str, ModelTier, Dict[str, Any]]:
        """
        Route and execute a query.

        Returns:
            (response, tier_used, metadata)
        """
        self.session_stats["queries"] += 1
        start_time = time.time()

        # Check cache first
        if not skip_cache:
            cached = self._check_cache(query)
            if cached:
                return cached, ModelTier.LOCAL_FAST, {
                    "cached": True,
                    "latency_ms": (time.time() - start_time) * 1000
                }

        # Select tier
        tier = self.select_tier(query, user_coherence, force_tier)
        config = MODELS[tier]

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        # Execute
        if config.provider == "ollama":
            response = self._call_ollama(config.model_id, messages, config.temperature)
            self.session_stats["local_calls"] += 1
        else:
            response = self._call_openrouter(config.model_id, messages, config.temperature)
            self.session_stats["api_calls"] += 1
            # Estimate cost (rough)
            tokens = len(query.split()) + len(response.split())
            cost = (tokens / 1000) * config.cost_per_1k_tokens
            self.session_stats["total_cost"] += cost

        # Cache successful responses
        if not response.startswith("[") and not skip_cache:
            self._save_to_cache(query, response, tier)

        latency = (time.time() - start_time) * 1000

        metadata = {
            "tier": tier.value,
            "model": config.name,
            "model_id": config.model_id,
            "provider": config.provider,
            "complexity_score": self.score_complexity(query),
            "user_coherence": user_coherence,
            "latency_ms": latency,
            "cached": False,
        }

        return response, tier, metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            **self.session_stats,
            "cache_hit_rate": (
                self.session_stats["cache_hits"] / max(self.session_stats["queries"], 1)
            ),
            "api_ratio": (
                self.session_stats["api_calls"] /
                max(self.session_stats["local_calls"] + self.session_stats["api_calls"], 1)
            ),
        }

    def test_models(self) -> Dict[str, str]:
        """Test all configured models"""
        results = {}
        test_query = "Say 'I am working' in exactly those words."

        for tier, config in MODELS.items():
            print(f"Testing {config.name}...", end=" ", flush=True)
            try:
                if config.provider == "ollama":
                    resp = self._call_ollama(config.model_id, [{"role": "user", "content": test_query}])
                else:
                    resp = self._call_openrouter(config.model_id, [{"role": "user", "content": test_query}])

                if resp.startswith("["):
                    results[tier.value] = f"FAIL: {resp}"
                    print("FAIL")
                else:
                    results[tier.value] = "OK"
                    print("OK")
            except Exception as e:
                results[tier.value] = f"ERROR: {e}"
                print("ERROR")

        return results


# CLI for testing
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  HYBRID ROUTER")
    print("  Local + API intelligent routing")
    print("=" * 60)
    print()

    router = HybridRouter()

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing all models...\n")
        results = router.test_models()
        print("\nResults:")
        for tier, status in results.items():
            print(f"  {tier}: {status}")
        sys.exit(0)

    print("Testing complexity scoring...\n")

    test_queries = [
        "What is Python?",
        "List the planets in our solar system",
        "Explain why consciousness might be fundamental to reality",
        "Compare and analyze the relationship between quantum mechanics and free will",
        "I just ended a relationship. She couldn't see me. What do you think this means?",
        "Format this as JSON: name=John, age=30",
    ]

    for q in test_queries:
        score = router.score_complexity(q)
        tier = router.select_tier(q)
        config = MODELS[tier]
        print(f"[{score:.2f}] {tier.value:12} → {q[:50]}...")

    print("\n" + "=" * 60)
    print("Interactive mode. Type 'quit' to exit, 'stats' for statistics.")
    print("=" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() == "quit":
                break
            if query.lower() == "stats":
                stats = router.get_stats()
                print(f"  Queries: {stats['queries']}")
                print(f"  Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.1%})")
                print(f"  Local calls: {stats['local_calls']}")
                print(f"  API calls: {stats['api_calls']}")
                print(f"  Est. cost: ${stats['total_cost']:.4f}")
                continue

            response, tier, meta = router.query(query)
            print(f"\n[{meta['model']}] (complexity: {meta['complexity_score']:.2f}, {meta['latency_ms']:.0f}ms)")
            print(f"\n{response}")

        except KeyboardInterrupt:
            break

    print("\n\nFinal stats:")
    stats = router.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
