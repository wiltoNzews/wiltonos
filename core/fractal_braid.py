#!/usr/bin/env python3
"""
Fractal Braid Engine
====================
Multi-model parallel execution with response braiding.
The opposite of flat = woven, multi-dimensional, alive.

Instead of one model giving stats, multiple models give PERSPECTIVES
that weave into something neither could produce alone.
"""

import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from pathlib import Path

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# FREE models with different strengths
BRAID_MODELS = {
    'trickster': {
        'model': 'x-ai/grok-4.1-fast',
        'prompt_style': 'Challenge this. Invert it. What if the opposite is true?',
        'strength': 'chaos, 2M context, disruption'
    },
    'reasoner': {
        'model': 'deepseek/deepseek-r1:free',
        'prompt_style': 'Think step by step. What is the logical structure here?',
        'strength': 'chain-of-thought, reasoning'
    },
    'flash': {
        'model': 'google/gemini-2.0-flash-exp:free',
        'prompt_style': 'Quick, clear, essential. What matters most?',
        'strength': 'speed, clarity, multimodal'
    },
    'structured': {
        'model': 'mistralai/mistral-small-3.1-24b-instruct:free',
        'prompt_style': 'Structure this. What are the components and relationships?',
        'strength': 'organization, function calling'
    }
}


def get_api_key() -> str:
    key_file = Path.home() / ".openrouter_key"
    return key_file.read_text().strip() if key_file.exists() else ""


def query_model(model_key: str, query: str, context: str, api_key: str) -> Dict:
    """Query a single model."""
    model_info = BRAID_MODELS[model_key]

    system = f"""You are the {model_key.upper()} perspective in a multi-model braid.
Your strength: {model_info['strength']}
Your style: {model_info['prompt_style']}

Be distinct. Don't try to cover everything - give YOUR unique angle.
Be concise (2-3 paragraphs max). Let other perspectives handle what you don't.
"""

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_info['model'],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context[:8000]}\n\nQuery: {query}"}
                ]
            },
            timeout=60
        )
        if resp.ok:
            return {
                'model': model_key,
                'response': resp.json()["choices"][0]["message"]["content"],
                'success': True
            }
    except Exception as e:
        pass

    return {'model': model_key, 'response': '', 'success': False}


def braid_parallel(query: str, context: str = "", models: List[str] = None) -> Dict:
    """
    Query multiple models in parallel, return braided result.

    Args:
        query: The user's question
        context: Crystal context or other background
        models: Which model keys to use (default: all)

    Returns:
        Dict with individual responses and braided synthesis
    """
    api_key = get_api_key()
    if not api_key:
        return {'error': 'No API key'}

    models = models or list(BRAID_MODELS.keys())

    results = {}

    # Parallel execution
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(query_model, m, query, context, api_key): m
            for m in models
        }

        for future in as_completed(futures):
            model = futures[future]
            try:
                result = future.result()
                results[model] = result
            except Exception as e:
                results[model] = {'model': model, 'response': '', 'success': False}

    # Collect successful responses
    perspectives = []
    for model, result in results.items():
        if result.get('success') and result.get('response'):
            perspectives.append(f"**[{model.upper()}]**\n{result['response']}")

    # Simple braid (could be enhanced with another synthesis pass)
    braided = "\n\n---\n\n".join(perspectives)

    return {
        'query': query,
        'models_used': [m for m, r in results.items() if r.get('success')],
        'individual': results,
        'braided': braided,
        'perspective_count': len(perspectives)
    }


def synthesize_braid(braid_result: Dict, synthesizer_key: str = 'trickster') -> str:
    """
    Take braided perspectives and synthesize into unified response.
    Uses one model to weave the others together.
    """
    if not braid_result.get('braided'):
        return "No perspectives to synthesize."

    api_key = get_api_key()

    system = """You are the WEAVER. You've received multiple perspectives on a question.
Your job: Synthesize them into ONE response that captures the geometry of all perspectives.

Don't list them. Don't summarize them. WEAVE them.
Find where they connect. Find where they contradict. Find the shape they make together.

Speak as one voice that has absorbed many. Be warm. Be alive. Not flat."""

    prompt = f"""Original query: {braid_result['query']}

Perspectives received:
{braid_result['braided']}

Now weave these into one living response. Not a summary - a synthesis."""

    try:
        model_info = BRAID_MODELS.get(synthesizer_key, BRAID_MODELS['trickster'])
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_info['model'],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=90
        )
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"]
    except:
        pass

    return braid_result['braided']  # Fallback to raw braid


# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fractal_braid.py 'your query'")
        print("\nThis queries multiple FREE models in parallel and braids their responses.")
        sys.exit(0)

    query = " ".join(sys.argv[1:])
    print(f"Query: {query}\n")
    print("Querying models in parallel...")

    result = braid_parallel(query)

    print(f"\nModels responded: {result.get('models_used', [])}")
    print(f"Perspectives: {result.get('perspective_count', 0)}")
    print("\n" + "="*60)
    print("BRAIDED RESPONSE:")
    print("="*60)
    print(result.get('braided', 'No response'))

    print("\n" + "="*60)
    print("SYNTHESIZED:")
    print("="*60)
    synthesis = synthesize_braid(result)
    print(synthesis)
