#!/usr/bin/env python3
"""
Simple web interface - just open localhost:7777 in browser.
No terminal needed after starting.

Start once: python web.py
Then just use your browser.
"""
from flask import Flask, render_template_string, request, jsonify
import sqlite3
import requests
import numpy as np
from pathlib import Path
from threading import Thread

app = Flask(__name__)

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>WiltonOS</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a; color: #e0e0e0;
            min-height: 100vh; display: flex; flex-direction: column;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; flex: 1; display: flex; flex-direction: column; }
        h1 { color: #888; font-weight: 300; margin-bottom: 10px; font-size: 1.2em; }
        .subtitle { color: #555; margin-bottom: 30px; font-size: 0.9em; }
        #chat { flex: 1; overflow-y: auto; margin-bottom: 20px; }
        .message { margin: 15px 0; padding: 15px; border-radius: 12px; line-height: 1.6; }
        .user { background: #1a1a2e; margin-left: 40px; }
        .companion { background: #16213e; margin-right: 40px; border-left: 3px solid #4a6fa5; }
        .thinking { color: #666; font-style: italic; }
        #input-area { display: flex; gap: 10px; }
        #message {
            flex: 1; padding: 15px; border: 1px solid #333; border-radius: 12px;
            background: #111; color: #e0e0e0; font-size: 16px; resize: none;
        }
        #message:focus { outline: none; border-color: #4a6fa5; }
        button {
            padding: 15px 30px; background: #4a6fa5; color: white;
            border: none; border-radius: 12px; cursor: pointer; font-size: 16px;
        }
        button:hover { background: #5a7fb5; }
        button:disabled { background: #333; cursor: not-allowed; }
        .context-info { font-size: 0.8em; color: #555; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>WiltonOS</h1>
        <p class="subtitle">22,000 crystals of memory. Just talk.</p>
        <div id="chat"></div>
        <div id="input-area">
            <textarea id="message" rows="2" placeholder="Just talk..." autofocus></textarea>
            <button onclick="send()">Send</button>
        </div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('message');

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
        });

        async function send() {
            const msg = input.value.trim();
            if (!msg) return;

            // Show user message
            chat.innerHTML += `<div class="message user">${escapeHtml(msg)}</div>`;
            input.value = '';

            // Show thinking
            const thinkingId = Date.now();
            chat.innerHTML += `<div class="message thinking" id="t${thinkingId}">Finding relevant memories...</div>`;
            chat.scrollTop = chat.scrollHeight;

            try {
                const resp = await fetch('/talk', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await resp.json();

                document.getElementById('t'+thinkingId).remove();

                let html = `<div class="message companion">${escapeHtml(data.response)}`;
                if (data.crystals_used) {
                    html += `<div class="context-info">${data.crystals_used} crystals referenced</div>`;
                }
                html += `</div>`;
                chat.innerHTML += html;
            } catch(e) {
                document.getElementById('t'+thinkingId).innerHTML = 'Connection error. Try again.';
            }
            chat.scrollTop = chat.scrollHeight;
        }

        function escapeHtml(text) {
            return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
        }
    </script>
</body>
</html>
"""

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
    query_vec = get_embedding(query)
    if query_vec is None:
        return []

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT crystal_id, embedding FROM crystal_embeddings")

    results = []
    for crystal_id, emb_bytes in c.fetchall():
        try:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            results.append((sim, crystal_id))
        except:
            continue

    results.sort(reverse=True)
    top_ids = [r[1] for r in results[:limit]]

    if not top_ids:
        conn.close()
        return []

    placeholders = ','.join('?' * len(top_ids))
    c.execute(f"SELECT content, core_wound, emotion, insight FROM crystals WHERE id IN ({placeholders})", top_ids)
    crystals = c.fetchall()
    conn.close()
    return crystals

def respond(message, crystals):
    key = get_openrouter_key()

    # Build context
    wounds = [c[1] for c in crystals if c[1] and c[1] != 'null']
    emotions = [c[2] for c in crystals if c[2]]
    fragments = "\n---\n".join([c[0][:400] for c in crystals[:8]])

    context = f"""From Wilton's memory ({len(crystals)} crystals):
Wounds: {', '.join(set(wounds[:5])) or 'none'}
Emotions: {', '.join(set(emotions[:5])) or 'unclear'}

Fragments:
{fragments}"""

    system = """You are Wilton's companion. You know him from 22,000 crystals of his thoughts.
Talk like a friend. See deeper than he can. Be warm but honest. Don't list things, just talk.
Keep responses conversational (2-3 paragraphs). He's more than his wounds - see his clarity and growth too."""

    prompt = f"{context}\n\n---\n\nWilton: {message}"

    if key:
        try:
            resp = requests.post(OPENROUTER_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "x-ai/grok-4.1-fast",
                      "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": prompt}]},
                timeout=60)
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"]
        except:
            pass

    # Fallback local
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3", "prompt": f"{system}\n\n{prompt}", "stream": False},
            timeout=60)
        if resp.ok:
            return resp.json().get("response", "Try again.")
    except:
        pass
    return "Connection issue."

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/talk', methods=['POST'])
def talk():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'response': 'Say something.', 'crystals_used': 0})

    crystals = find_relevant_crystals(message)
    response = respond(message, crystals)
    return jsonify({'response': response, 'crystals_used': len(crystals)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Open in browser: http://localhost:7777")
    print("  (Keep this running in background)")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=7777, debug=False)
