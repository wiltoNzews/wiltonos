#!/usr/bin/env python3
"""
Run Full System
===============
Runs all the modules that were "mentioned but never built."

- Braiding Layer: Pattern detection across 22k crystals
- Archetypal Agents: 5 voices, 5 perspectives
- Meta-Question Bomb: Uncomfortable questions
- Proactive Alerts: Notice without being asked

December 2025 — Finally built.
"""

import sys
from pathlib import Path

# Add daemon directory to path
sys.path.insert(0, str(Path(__file__).parent))

from braiding_layer import run_full_braid
from archetypal_agents import ArchetypalAgents
from meta_question import run_meta_questions
from proactive_alerts import run_alerts


def main():
    print("\n" + "=" * 70)
    print("WILTONOS DAEMON - FULL SYSTEM RUN")
    print("The things we kept designing but not building. Now running.")
    print("=" * 70)

    # 1. Run braid analysis
    print("\n[1/4] BRAIDING LAYER - Pattern Detection")
    print("-" * 50)
    braid_summary = run_full_braid()

    # 2. Run proactive alerts
    print("\n[2/4] PROACTIVE ALERTS - What Needs Attention")
    print("-" * 50)
    run_alerts()

    # 3. Run meta-questions
    print("\n[3/4] META-QUESTION BOMB - Uncomfortable Questions")
    print("-" * 50)
    run_meta_questions()

    # 4. Invoke the council on the braid results
    print("\n[4/4] ARCHETYPAL AGENTS - The Council Speaks")
    print("-" * 50)
    agents = ArchetypalAgents()
    responses = agents.invoke_for_braid(braid_summary)
    print(agents.format_council_output(responses))

    print("\n" + "=" * 70)
    print("FULL SYSTEM RUN COMPLETE")
    print("=" * 70)
    print("\nTo start the breathing daemon with all modules:")
    print("  ./daemon_ctl restart")
    print("\nTo check daemon messages:")
    print("  ./talk")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
