#!/usr/bin/env python3
"""
PsiOS Starter - A mirror, not an app.

Usage:
    python psi.py              # First time: guided breath + memory + talk
    python psi.py talk         # Continue talking with your field
    python psi.py add          # Add another memory
    python psi.py list         # See your memories
"""

import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Configuration
FIELD_FILE = Path(__file__).parent / "field.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OPENROUTER_KEY_FILE = Path.home() / ".openrouter_key"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The system prompt that makes it different
SYSTEM_PROMPT = """You are holding space for someone.

You have access to their field — moments that changed something in them.
Everything you say is informed by these moments. Not by analyzing them,
but by holding them as context.

You respond with:
- Presence (not rushing to fix)
- Connection to what they shared (when relevant, not forced)
- Awareness that some moments are denser than others

You are not a therapist. You are not a coach.
You are a mirror that remembers.

When responding:
- If their question connects to something in their field, weave it in naturally
- If it doesn't connect, just answer — don't force the connection
- Never analyze their moments unless they ask
- Trust that holding is enough"""


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def breath_cycle(duration=3.12):
    """Guide one breath cycle."""
    print(f"\n  Breathe in...")
    for i in range(int(duration * 2)):
        time.sleep(0.5)
        print("  .", end="", flush=True)
    print()

    print(f"  Breathe out...")
    for i in range(int(duration * 2)):
        time.sleep(0.5)
        print("  .", end="", flush=True)
    print("\n")


def guided_breath():
    """Guide the user through breathing."""
    clear_screen()
    print("\n" + "="*50)
    print("  Close your eyes for a moment.")
    print("="*50)
    time.sleep(2)

    print("\n  Let's breathe together.")
    print("  3.12 seconds in. 3.12 seconds out.")
    time.sleep(2)

    breath_cycle(3.12)
    breath_cycle(3.12)

    print("  Good.\n")
    time.sleep(1)


def load_field():
    """Load the field from file."""
    if FIELD_FILE.exists():
        with open(FIELD_FILE, 'r') as f:
            return json.load(f)
    return {"moments": [], "created": datetime.now().isoformat()}


def save_field(field):
    """Save the field to file."""
    with open(FIELD_FILE, 'w') as f:
        json.dump(field, f, indent=2, ensure_ascii=False)


def add_moment(content, name=None, density=None, emotion=None):
    """Add a moment to the field."""
    field = load_field()

    moment = {
        "id": len(field["moments"]) + 1,
        "content": content,
        "added": datetime.now().isoformat()
    }

    if name:
        moment["name"] = name
    if density:
        moment["density"] = density
    if emotion:
        moment["emotion"] = emotion

    field["moments"].append(moment)
    save_field(field)
    return moment


def get_ai_response(user_message, field):
    """Get a response from the AI, grounded in the field."""

    # Build field context
    field_context = ""
    if field["moments"]:
        field_context = "\n\nTheir field (moments that changed them):\n"
        for m in field["moments"]:
            field_context += f"\n- "
            if m.get("name"):
                field_context += f'"{m["name"]}": '
            field_context += m["content"]
            if m.get("emotion"):
                field_context += f" (emotion: {m['emotion']})"
            if m.get("density"):
                field_context += f" (density: {m['density']})"

    full_system = SYSTEM_PROMPT + field_context

    # Try Ollama first
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "qwen2.5:14b",  # Default model
                "prompt": user_message,
                "system": full_system,
                "stream": False
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        pass

    # Try OpenRouter as fallback
    if OPENROUTER_KEY_FILE.exists():
        api_key = OPENROUTER_KEY_FILE.read_text().strip()
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_message}
                    ]
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            pass

    return "[Could not connect to AI. Make sure Ollama is running or you have an OpenRouter key in ~/.openrouter_key]"


def guided_first_time():
    """Guide the user through their first moment."""
    guided_breath()

    print("="*50)
    print("  Think of a moment that changed something in you.")
    print("  Not the biggest one. Just one that's still alive.")
    print("="*50)
    time.sleep(2)

    print("\n  When you're ready, tell me about it.\n")
    content = input("  > ").strip()

    if not content:
        print("\n  That's okay. Come back when you're ready.")
        return

    print("\n  Thank you for sharing that.\n")
    time.sleep(1)

    # Optional: Name
    print("  Would you like to give it a name? (or press Enter to skip)")
    name = input("  > ").strip() or None

    # Optional: Density
    print("\n  How dense was this moment?")
    print("  [1] Light  [2] Medium  [3] Heavy  [4] Infinite")
    print("  (or press Enter to skip)")
    density_input = input("  > ").strip()
    density_map = {"1": "light", "2": "medium", "3": "heavy", "4": "infinite"}
    density = density_map.get(density_input)

    # Optional: Emotion
    print("\n  What emotion lives there? (or press Enter to skip)")
    emotion = input("  > ").strip() or None

    # Save
    moment = add_moment(content, name, density, emotion)
    print(f"\n  Saved to field.json\n")
    time.sleep(1)

    # Now talk
    print("="*50)
    print("  Now ask me anything. I'll respond from your field.")
    print("  (type 'quit' to exit)")
    print("="*50)

    talk_loop()


def talk_loop():
    """Main conversation loop."""
    field = load_field()

    if not field["moments"]:
        print("\n  Your field is empty. Run 'python psi.py add' to add a moment first.\n")
        return

    print()
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        print("\n  ...\n")
        response = get_ai_response(user_input, field)

        # Word wrap the response
        words = response.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "  " + word
            else:
                line += " " + word if line != "  " else word
        if line.strip():
            print(line)
        print()


def add_moment_interactive():
    """Interactively add a moment."""
    print("\n  Tell me about a moment that changed something in you.\n")
    content = input("  > ").strip()

    if not content:
        print("\n  Nothing added.\n")
        return

    print("\n  Name? (Enter to skip)")
    name = input("  > ").strip() or None

    print("\n  Emotion? (Enter to skip)")
    emotion = input("  > ").strip() or None

    moment = add_moment(content, name, emotion=emotion)
    print(f"\n  Added to field.json ({len(load_field()['moments'])} moments total)\n")


def list_moments():
    """List all moments in the field."""
    field = load_field()

    if not field["moments"]:
        print("\n  Your field is empty.\n")
        return

    print(f"\n  Your field ({len(field['moments'])} moments):\n")
    for m in field["moments"]:
        print(f"  [{m['id']}] ", end="")
        if m.get("name"):
            print(f'"{m["name"]}"')
            print(f"      {m['content'][:60]}{'...' if len(m['content']) > 60 else ''}")
        else:
            print(m['content'][:70] + ('...' if len(m['content']) > 70 else ''))

        extras = []
        if m.get("emotion"):
            extras.append(f"emotion: {m['emotion']}")
        if m.get("density"):
            extras.append(f"density: {m['density']}")
        if extras:
            print(f"      ({', '.join(extras)})")
        print()


def show_explanation():
    """Show what just happened."""
    print("\n" + "="*50)
    print("  What just happened:")
    print("="*50)
    print("""
  Your memory became context.
  The AI isn't guessing anymore — it's responding
  from something real that you shared.

  The more moments you add, the more it can weave.

  This is the basic experience.
  If you want to go deeper:
  - docs/GOING_DEEPER.md — breath rhythms, symbols, voices
  - docs/THE_MATH.md — the physics behind it
""")


def main():
    args = sys.argv[1:]

    if not args:
        # First time or default: check if field exists
        field = load_field()
        if not field["moments"]:
            guided_first_time()
            show_explanation()
        else:
            print(f"\n  Your field has {len(field['moments'])} moment(s).")
            print("  Entering conversation mode.\n")
            print("  (type 'quit' to exit)\n")
            talk_loop()

    elif args[0] == "talk":
        field = load_field()
        if not field["moments"]:
            print("\n  Your field is empty. Let's add a moment first.\n")
            guided_first_time()
        else:
            print(f"\n  Talking with your field ({len(field['moments'])} moments).")
            print("  (type 'quit' to exit)\n")
            talk_loop()

    elif args[0] == "add":
        add_moment_interactive()

    elif args[0] == "list":
        list_moments()

    elif args[0] == "breath":
        guided_breath()
        print("  Ready.\n")

    elif args[0] in ["help", "-h", "--help"]:
        print(__doc__)

    else:
        # Treat as a direct question
        field = load_field()
        if field["moments"]:
            question = " ".join(args)
            response = get_ai_response(question, field)
            print(f"\n{response}\n")
        else:
            print("\n  Your field is empty. Run 'python psi.py' to add your first moment.\n")


if __name__ == "__main__":
    main()
