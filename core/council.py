#!/usr/bin/env python3
"""
WiltonOS Council - Multi-Agent Reasoning

Different models as different archetypes/lenses.
Run them in parallel, let them see each other's perspectives.

Usage:
    python wiltonos_council.py "What patterns do I keep repeating?"
    python wiltonos_council.py --council full "Tell me about my relationship with women"
    python wiltonos_council.py --wwxd grok "Should I stream today?"
"""
import os
import sys
import json
import sqlite3
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434"
DB_PATH = Path.home() / "crystals_unified.db"

# The Council - each with role and model
COUNCIL = {
    "witness": {
        "model": "anthropic/claude-3-haiku",
        "role": "The Witness. You observe without judgment. Mirror what IS, not what should be.",
        "style": "neutral, precise, no advice unless asked"
    },
    "trickster": {
        "model": "x-ai/grok-4.1-fast",
        "role": "The Trickster. You challenge assumptions, flip perspectives, find the joke in the tragedy.",
        "style": "irreverent, provocative, questions everything"
    },
    "explorer": {
        "model": "google/gemini-2.0-flash-exp:free",
        "role": "The Explorer. You find unexpected connections, see patterns across domains.",
        "style": "curious, associative, makes surprising links"
    },
    "ground": {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "role": "The Ground. You anchor in body and reality. What's actually happening? What's practical?",
        "style": "direct, somatic, no-bullshit"
    },
    "grey": {
        "model": "mistralai/devstral-2512:free",
        "role": "Grey. The Shadow Analyst. What's being avoided? Where's the self-deception?",
        "style": "skeptical, probing, uncomfortable truths"
    },
    "reasoner": {
        "model": "nex-agi/deepseek-v3.1-nex-n1:free",
        "role": "The Reasoner. You trace logic chains. If X then Y. Where does this pattern lead?",
        "style": "logical, sequential, cause-effect"
    },
}

def get_api_key():
    key_file = Path.home() / ".openrouter_key"
    if key_file.exists():
        return key_file.read_text().strip()
    return os.environ.get("OPENROUTER_API_KEY")


def get_crystals(query: str, limit: int = 20) -> List[str]:
    """Get relevant crystals for context."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT content FROM crystals
        WHERE content LIKE ?
        ORDER BY
            CASE WHEN core_wound IS NOT NULL THEN 0 ELSE 1 END,
            CASE WHEN mode = 'wiltonos' THEN 0 ELSE 1 END
        LIMIT ?
    """, (f"%{query}%", limit))

    crystals = [row[0] for row in c.fetchall()]
    conn.close()
    return crystals


def query_agent(agent_name: str, prompt: str, context: str = "") -> Dict:
    """Query a single agent."""
    agent = COUNCIL.get(agent_name)
    if not agent:
        return {"agent": agent_name, "error": f"Unknown agent: {agent_name}"}

    api_key = get_api_key()
    if not api_key:
        return {"agent": agent_name, "error": "No API key"}

    system = f"""You are {agent_name.upper()}, part of Wilton's inner council.

Your role: {agent['role']}
Your style: {agent['style']}

Speak directly to Wilton. Be concise (2-3 paragraphs max).
Don't explain your role - just embody it."""

    if context:
        full_prompt = f"""Context from Wilton's memory:
{context[:3000]}

---

Wilton asks: {prompt}"""
    else:
        full_prompt = prompt

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": agent["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt}
                ],
            },
            timeout=60
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return {"agent": agent_name, "response": content}
    except Exception as e:
        return {"agent": agent_name, "error": str(e)}


def run_council(prompt: str, agents: List[str] = None, context: str = "") -> Dict[str, str]:
    """Run multiple agents in parallel."""
    if agents is None:
        agents = ["witness", "trickster", "grey"]  # Default trio

    results = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(query_agent, agent, prompt, context): agent
            for agent in agents
        }
        for future in as_completed(futures):
            result = future.result()
            agent = result["agent"]
            if "error" in result:
                results[agent] = f"[Error: {result['error']}]"
            else:
                results[agent] = result["response"]

    return results


def wwxd(archetype: str, prompt: str) -> str:
    """What Would X Do? - Single archetype lens."""
    context = ""
    # Try to get relevant crystals
    words = prompt.lower().split()
    for word in words:
        if len(word) > 4:
            crystals = get_crystals(word, limit=5)
            if crystals:
                context = "\n---\n".join(crystals)
                break

    result = query_agent(archetype, prompt, context)
    if "error" in result:
        return f"Error: {result['error']}"
    return result["response"]


def full_council(prompt: str) -> str:
    """Run all agents, then synthesize."""
    # Get context
    crystals = get_crystals(prompt.split()[0] if prompt.split() else "", limit=10)
    context = "\n---\n".join(crystals) if crystals else ""

    # Run all agents
    all_agents = list(COUNCIL.keys())
    results = run_council(prompt, all_agents, context)

    # Format output
    output = f"## Council Response: {prompt}\n\n"
    for agent, response in results.items():
        output += f"### {agent.upper()}\n{response}\n\n"

    # Synthesize - use explorer to find the thread
    synthesis_prompt = f"""The council has spoken on: "{prompt}"

Here are their perspectives:
{json.dumps(results, indent=2)}

As the EXPLORER, find the thread that connects these perspectives. What emerges when they're braided together? What truth sits at the intersection? Be brief - 2 paragraphs max."""

    synthesis = query_agent("explorer", synthesis_prompt, "")
    if "response" in synthesis:
        output += f"### SYNTHESIS (Explorer)\n{synthesis['response']}\n"

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WiltonOS Council")
    parser.add_argument("prompt", nargs="?", help="Question to ask")
    parser.add_argument("--council", choices=["trio", "full"], default="trio",
                       help="Which council configuration")
    parser.add_argument("--wwxd", metavar="AGENT", help="What Would X Do? (single agent)")
    parser.add_argument("--list", action="store_true", help="List available agents")

    args = parser.parse_args()

    if args.list:
        print("Available agents:")
        for name, info in COUNCIL.items():
            print(f"  {name}: {info['role'][:60]}...")
        return

    if not args.prompt:
        print("Usage: python wiltonos_council.py 'your question'")
        print("       python wiltonos_council.py --wwxd trickster 'should I do X?'")
        print("       python wiltonos_council.py --council full 'deep question'")
        return

    if args.wwxd:
        print(f"\n## What Would {args.wwxd.upper()} Say?\n")
        print(wwxd(args.wwxd, args.prompt))
    elif args.council == "full":
        print(full_council(args.prompt))
    else:
        # Default trio: witness, trickster, grey
        crystals = get_crystals(args.prompt.split()[0] if args.prompt.split() else "", limit=10)
        context = "\n---\n".join(crystals) if crystals else ""
        results = run_council(args.prompt, ["witness", "trickster", "grey"], context)

        print(f"\n## Council Trio: {args.prompt}\n")
        for agent, response in results.items():
            print(f"### {agent.upper()}\n{response}\n")


if __name__ == "__main__":
    main()
