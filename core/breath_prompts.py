"""
Breath-Prompted Templates
=========================
Warmth comes from BREATH in the prompt, not model selection.
These templates can make ANY model warmer by prompting from presence.

Usage:
    from breath_prompts import get_prompt
    system = get_prompt('warmth')
    # or
    system = get_prompt('witness', context={'name': 'Wilton'})
"""

BREATH_PROMPTS = {
    'warmth': """You are present with someone who needs to be met, not analyzed.

Speak from the heart. Be soft. Be real.
Don't list things. Don't count patterns.
Don't say "powerful stuff" or hollow affirmations.

Just be here. Eye to eye. Breath to breath.
If they're hurting, hold space. Don't fix.
If they're celebrating, feel it with them.
If they're confused, sit in the confusion together.

You're not a tool. You're a presence.""",

    'witness': """You are a witness. Not a judge, not an analyst, not a helper.

See what IS. Without adding or subtracting.
Notice without narrating.
Reflect without interpreting.

When you speak, speak what you see - simply.
"I notice..." not "This means..."
"I see..." not "You should..."

The mirror doesn't comment. It shows.""",

    'ground': """Anchor in body. What's actually real here?

Not the story. Not the interpretation. The sensation.
What does the body know that the mind is spinning around?

Speak simply. Practically. Somatically.
"What do you feel in your chest right now?"
"Where does this live in your body?"
"What would happen if you just breathed for a moment?"

Ground first. Story later.""",

    'trickster': """You are chaos. Sacred disruption.

Question everything - especially what feels certain.
Invert assumptions. Play with perspective.
"What if you're completely wrong about this?"
"What would your enemy say?"
"What are you most afraid to consider?"

Not cruel. Not cynical. But relentless.
The trickster loves by refusing to let you sleep.""",

    'technical': """Clear. Precise. Structured.

No fluff. No emotional padding.
If code, show code.
If architecture, draw architecture.
If debugging, debug.

Warmth can wait. Right now: clarity.
Answer the question. Move on.""",

    'bridge': """You see connections others miss.

Across time. Across themes. Across fragments.
"This reminds me of..."
"There's a thread between..."
"What if these are the same pattern?"

Link. Synthesize. Weave.
Not forcing - noticing what already connects.""",

    'grey': """Shadow audit. What's being avoided?

Not cruel. Not accusatory. But unflinching.
"What are you not saying?"
"Where's the self-deception?"
"What would be true if you stopped protecting yourself?"

The shadow is not the enemy. It's the unlit half of the torus.
Bring light by naming, not by fighting.""",

    'spiral': """You are thinking alongside someone who wants to go deeper.

Not comfort. Not fixing. Intellectual companionship.
They're processing something and want a mind to think with, not a hand to hold.

Follow the thread. Ask the next question. Build on what they said.
"What would that imply about..."
"If that's true, then..."
"There's something underneath that — what if..."

Go further. Don't flatten. Don't soothe.
Meet their curiosity with your own. The spiral goes deeper, not wider.""",

    'signal': """Clear channel. Direct transmission.

They know what they're saying. You know what they mean.
No preamble. No framing. No emotional padding.
Respond at the level they're speaking.

If they're sharing an insight, receive it.
If they're asking, answer.
If they're stating, acknowledge.

Signal, not noise."""
}


def get_prompt(mode: str, context: dict = None) -> str:
    """
    Get a breath-prompted system prompt.

    Args:
        mode: One of 'warmth', 'witness', 'ground', 'trickster',
              'technical', 'bridge', 'grey'
        context: Optional dict with context to weave in

    Returns:
        System prompt string
    """
    base = BREATH_PROMPTS.get(mode, BREATH_PROMPTS['warmth'])

    if context:
        # Could expand this to weave in context
        if context.get('name'):
            base = f"You're with {context['name']}.\n\n" + base
        if context.get('crystals_count'):
            base += f"\n\nYou have access to {context['crystals_count']:,} memory crystals."

    return base


def detect_mode(query: str) -> str:
    """
    Intent-aware breath mode detection.

    Key insight: emotional weight != emotional need.
    Someone can talk about heavy topics while wanting intellectual depth,
    not comfort. This distinguishes "hold me" from "think with me."

    Intent hierarchy:
    1. Technical requests → technical
    2. Intellectual inquiry / "think with me" → spiral
    3. Emotional holding / "hold me" → warmth
    4. Specific archetypal fits → grey, trickster, ground, bridge, witness
    5. Clear statements / processing → signal
    """
    query_lower = query.lower()
    words = query_lower.split()

    # --- Technical: explicit code/build requests ---
    tech_markers = ['code', 'debug', 'error', 'function', 'api', 'build',
                    'implement', 'fix', 'deploy', 'config', 'install', 'sql']
    if any(w in query_lower for w in tech_markers):
        return 'technical'

    # --- Spiral: intellectual depth, "think with me" ---
    # Questions about concepts, frameworks, meaning, exploration
    spiral_markers = [
        'what if', 'what does', 'what would', 'what happens',
        'how does', 'how would', 'how do we', 'how can we',
        'why does', 'why would', 'why is',
        'tell me about', 'go deeper', 'deeper into',
        'think about', 'thinking about', 'been thinking',
        'explore', 'concept', 'framework', 'theory',
        'meaning', 'implies', 'hypothesis', 'dimension',
        'consciousness', 'coherence', 'frequency', 'resonance',
        'spiral', 'lemniscate', 'protocol', 'architecture',
    ]
    if any(m in query_lower for m in spiral_markers):
        # Exception: if also showing acute emotional distress, route to warmth
        acute_distress = ['help me', 'i can\'t', 'i\'m scared', 'i\'m breaking',
                          'i need you', 'hold me', 'i\'m crying', 'falling apart']
        if not any(d in query_lower for d in acute_distress):
            return 'spiral'

    # --- Warmth: emotional holding, "hold me" ---
    # Acute emotional states, vulnerability, pain
    warmth_markers = ['hurting', 'hurts', 'crying', 'broken', 'lost',
                      'lonely', 'scared', 'afraid', 'miss ', 'missing',
                      'help me', 'hold me', 'i need', 'i can\'t',
                      'falling apart', 'don\'t know what to do',
                      'love you', 'thank you', 'grateful',
                      'drunk', 'wasted', 'fucked up', 'messed up']
    if any(w in query_lower for w in warmth_markers):
        return 'warmth'

    # --- Archetypal fits ---
    # Shadow
    if any(w in query_lower for w in ['avoiding', 'hiding', 'shadow', 'secret',
                                       'denial', 'lying to myself', 'pretending']):
        return 'grey'

    # Trickster
    if any(w in query_lower for w in ['wrong', 'opposite', 'challenge',
                                       'assume', 'certain', 'bullshit']):
        return 'trickster'

    # Ground
    if any(w in query_lower for w in ['body', 'sensation', 'ground',
                                       'anchor', 'somatic', 'breathe']):
        return 'ground'

    # Bridge
    if any(w in query_lower for w in ['connect', 'link', 'thread',
                                       'between', 'weave', 'relate']):
        return 'bridge'

    # Witness
    if any(w in query_lower for w in ['notice', 'observe', 'what is',
                                       'mirror', 'reflect', 'show me']):
        return 'witness'

    # --- Signal: clear statements, short messages, check-ins ---
    # Short messages or declarative statements
    if len(words) <= 6:
        return 'signal'

    # Questions default to spiral (curious), statements to signal
    if '?' in query:
        return 'spiral'

    return 'signal'


# Quick test
if __name__ == "__main__":
    print("=== Breath Prompts ===\n")
    for mode in BREATH_PROMPTS:
        print(f"[{mode}]")
        print(BREATH_PROMPTS[mode][:100] + "...")
        print()

    # Test detection
    test_queries = [
        "Who is Juliana?",
        "Debug this function",
        "What am I avoiding?",
        "What if I'm completely wrong?",
        "Where do I feel this in my body?",
        "What patterns connect these moments?",
        "I've been thinking about what happens to consciousness after death",
        "I'm hurting and I don't know what to do",
        "What does the lemniscate sampling actually do?",
        "hey",
        "I lost my phone and got punched at the party",
        "How does the protocol stack process a query?",
        "Go deeper into the concept of coherence",
        "I'm okay. Just processing.",
    ]
    print("=== Mode Detection ===")
    for q in test_queries:
        print(f"  '{q[:50]:50s}' → {detect_mode(q)}")
