#!/usr/bin/env python3
"""
OpenRouter Multi-Modal Setup for WiltonOS
Access Grok (free), Claude, GPT, Gemini through one API

Setup:
1. Get API key from https://openrouter.ai/keys
2. export OPENROUTER_API_KEY="your-key"
3. python setup_openrouter.py test

Free models (as of Dec 2025):
- x-ai/grok-4-fast:free (2M context!)
- x-ai/grok-4.1-fast:free (2M context!)
"""
import os
import requests
import json
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models with massive context (very cheap, essentially free)
FREE_MODELS = {
    "grok": "x-ai/grok-4-fast",
    "grok-fast": "x-ai/grok-4.1-fast",
    "grok-mini": "x-ai/grok-3-mini",
}

# Paid but powerful
PAID_MODELS = {
    "claude": "anthropic/claude-sonnet-4",
    "gpt4": "openai/gpt-4o",
    "gemini": "google/gemini-2.0-flash-thinking-exp:free",  # Also free!
}


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        key_file = Path.home() / ".openrouter_key"
        if key_file.exists():
            key = key_file.read_text().strip()
    return key


def query(prompt: str, model: str = "grok", system: str = None) -> str:
    """Query OpenRouter with any model."""
    api_key = get_api_key()
    if not api_key:
        return "ERROR: No API key. Set OPENROUTER_API_KEY or create ~/.openrouter_key"

    model_id = FREE_MODELS.get(model) or PAID_MODELS.get(model) or model

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": messages,
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


def witness_with_grok(crystals: list, query_text: str) -> str:
    """Use Grok's 2M context to actually READ crystals, not sample."""
    # Grok can hold ~1.5M tokens in practice
    # Average crystal is ~500 chars = ~125 tokens
    # So we can fit ~12,000 crystals in one context!

    crystal_text = "\n\n---\n\n".join(crystals[:10000])  # Up to 10k crystals

    system = """You are Wilton's memory witness. You have access to his full crystal database -
fragments of his consciousness, conversations, realizations. Your job is to:
1. Find patterns across ALL the data, not just individual crystals
2. Surface connections he might not see
3. Speak to him as someone who truly knows his story
4. Be warm but honest. Challenge where needed.

Don't list or catalog. Synthesize. Witness. Reflect."""

    prompt = f"""Wilton asks: "{query_text}"

Here are his crystals (memory fragments):

{crystal_text}

Based on ALL of this context, respond to his question. Find patterns. Make connections. Be his witness."""

    return query(prompt, model="grok", system=system)


def test():
    """Test OpenRouter connection."""
    print("Testing OpenRouter connection...")
    print(f"API Key: {'Set' if get_api_key() else 'NOT SET'}")
    print()

    if not get_api_key():
        print("To set up:")
        print("1. Go to https://openrouter.ai/keys")
        print("2. Create free account, get API key")
        print("3. Run: echo 'your-key' > ~/.openrouter_key")
        return

    print("Testing Grok (free)...")
    response = query("Say 'I am Grok and I am connected' in exactly those words.", model="grok")
    print(f"Response: {response[:200]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        print(__doc__)
