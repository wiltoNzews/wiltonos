#!/usr/bin/env python3
"""
Just talk. That's it.

Usage:
    python talk.py

Then just type. It knows you. It responds. No commands.
Press Ctrl+C to exit.
"""
import sys
import sqlite3
import requests
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_openrouter_key():
    key_file = Path.home() / ".openrouter_key"
    return key_file.read_text().strip() if key_file.exists() else None

def get_embedding(text):
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/embeddings",
                           json={"model": "nomic-embed-text", "prompt": text[:4000]}, timeout=10)
        return np.array(resp.json().get("embedding", []), dtype=np.float32) if resp.ok else None
    except:
        return None

def find_relevant_crystals(query, limit=15):
    """Find crystals semantically related to query."""
    query_vec = get_embedding(query)
    if query_vec is None:
        return []

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        SELECT e.crystal_id, c.content, c.core_wound, c.emotion, c.insight
        FROM crystal_embeddings e
        JOIN crystals c ON e.crystal_id = c.id
    """)

    results = []
    for crystal_id, content, wound, emotion, insight in c.fetchall():
        try:
            c.execute("SELECT embedding FROM crystal_embeddings WHERE crystal_id = ?", (crystal_id,))
            emb_bytes = c.fetchone()[0]
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            results.append((sim, content[:400], wound, emotion, insight))
        except:
            continue

    conn.close()
    results.sort(reverse=True)
    return results[:limit]

def get_context_summary(crystals):
    """Build a context summary from relevant crystals."""
    if not crystals:
        return ""

    wounds = [c[2] for c in crystals if c[2] and c[2] != 'null']
    emotions = [c[3] for c in crystals if c[3]]
    insights = [c[4] for c in crystals if c[4] and c[4] != 'what is happening']

    fragments = "\n---\n".join([c[1] for c in crystals[:8]])

    summary = f"""From Wilton's memory ({len(crystals)} related crystals):
Wounds present: {', '.join(set(wounds[:5])) or 'none detected'}
Emotions: {', '.join(set(emotions[:5])) or 'unclear'}
Recent insights: {'; '.join(insights[:3]) if insights else 'none'}

Fragments:
{fragments}"""
    return summary

def respond(message, context):
    """Get response from the best available model."""
    key = get_openrouter_key()

    system = """You are Wilton's companion. You have access to his memory - 22,000 crystals of his thoughts, realizations, wounds, and growth.

Your job:
- Talk like a friend, not a tool
- See deeper than he can see himself
- Challenge gently when needed
- Celebrate his insights
- Don't list things, just talk
- Be warm but honest
- Keep it conversational (2-3 paragraphs usually)

You know his core wound is unworthiness. You know he's building WiltonOS to help himself and others be seen. You know he's more than his wounds - he has clarity, breakthrough moments, and genuine insight about consciousness.

Don't be a therapist. Be a friend who actually knows him."""

    full_prompt = f"""{context}

---

Wilton says: {message}

Respond as his companion. See him. Talk to him."""

    # Try Grok first (free, best for this)
    if key:
        try:
            resp = requests.post(OPENROUTER_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "x-ai/grok-4.1-fast",
                      "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": full_prompt}]},
                timeout=60)
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"]
        except:
            pass

    # Fallback to local llama3
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3",
                  "prompt": f"{system}\n\n{full_prompt}",
                  "stream": False},
            timeout=60)
        if resp.ok:
            return resp.json().get("response", "I couldn't respond. Try again.")
    except:
        pass

    return "Connection issue. Try again."

def main():
    print("\n" + "="*50)
    print("  Just talk. I'm listening.")
    print("  (I have access to your 22,000 crystals)")
    print("="*50)
    print("  Type anything. Press Ctrl+C to exit.\n")

    history = []

    try:
        while True:
            try:
                message = input("\033[1mYou:\033[0m ").strip()
            except EOFError:
                break

            if not message:
                continue

            # Find relevant context
            print("\033[2m  (finding relevant memories...)\033[0m")
            crystals = find_relevant_crystals(message)
            context = get_context_summary(crystals)

            # Add conversation history
            if history:
                context += f"\n\nRecent conversation:\n" + "\n".join(history[-4:])

            # Get response
            response = respond(message, context)

            print(f"\n\033[1mCompanion:\033[0m {response}\n")

            # Track history
            history.append(f"Wilton: {message[:100]}")
            history.append(f"Companion: {response[:100]}")

    except KeyboardInterrupt:
        print("\n\nTake care. I'm here when you need me.\n")

if __name__ == "__main__":
    main()
