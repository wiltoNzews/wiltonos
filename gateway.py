#!/usr/bin/env python3
"""
WiltonOS Gateway - The Door
===========================
Simple web interface that wraps the real engine.
Mobile-friendly. Presence-first.

Run:
    python gateway.py

Then open: http://localhost:8000

Or with custom port:
    python gateway.py --port 5000
"""

import sys
import json
import uvicorn
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Import the real engine
from talk_v2 import WiltonOS
from core.auth import UserAuth
from core.navigator_service import NavigatorService
from core.memory_service import MemoryService

# Initialize memory service (ChromaDB may crash — known bug #5909)
try:
    memory_service = MemoryService()
except BaseException:
    memory_service = None

app = FastAPI(title="WiltonOS", docs_url=None, redoc_url=None)

# Mount static files for breath visual
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "web" / "static")), name="static")

# Auth system
auth = UserAuth()

# Store engines per user (lightweight, reuses DB connection)
engines = {}

# DB path for user checks
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"

# Daemon messages directory
DAEMON_MESSAGES = Path.home() / "wiltonos" / "daemon" / "messages"

def user_has_crystals(user_id: str) -> bool:
    """Check if user has any crystals yet."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM crystals WHERE user_id = ?", (user_id,))
    count = c.fetchone()[0]
    conn.close()
    return count > 0

def get_engine(user_id: str) -> WiltonOS:
    """Get or create engine for user."""
    if user_id not in engines:
        engines[user_id] = WiltonOS(user_id=user_id)
        engines[user_id].start_session(platform='web')
    return engines[user_id]


# ═══════════════════════════════════════════════════════════════
# Breath Visual - The Coupling Mechanism
# ═══════════════════════════════════════════════════════════════

@app.get("/breathe", response_class=HTMLResponse)
async def breathe():
    """Serve the breath entrainment visual."""
    breath_file = Path(__file__).parent / "web" / "static" / "breath.html"
    if breath_file.exists():
        return HTMLResponse(content=breath_file.read_text())
    return HTMLResponse(content="<h1>Breath visual not found</h1>", status_code=404)


@app.get("/geometry", response_class=HTMLResponse)
async def geometry():
    """Serve the sacred geometry visualizer."""
    geometry_file = Path(__file__).parent / "web" / "static" / "geometry.html"
    if geometry_file.exists():
        return HTMLResponse(content=geometry_file.read_text())
    return HTMLResponse(content="<h1>Geometry visualizer not found</h1>", status_code=404)


@app.get("/navigator", response_class=HTMLResponse)
async def navigator():
    """Serve the glyph navigator interface."""
    nav_file = Path(__file__).parent / "web" / "static" / "navigator.html"
    if nav_file.exists():
        return HTMLResponse(content=nav_file.read_text())
    return HTMLResponse(content="<h1>Navigator not found</h1>", status_code=404)


@app.get("/api/geometry/state")
async def geometry_state(user_id: str = "wilton"):
    """Current coherence state for visualization."""
    try:
        engine = get_engine(user_id)
        if hasattr(engine, 'protocol_stack'):
            state = engine.protocol_stack.get_current_state()
            zl = state.get("zeta_lambda", 0.5)
        else:
            zl = 0.5
            state = {}

        # Determine glyph from coherence
        if zl < 0.2:
            glyph = "∅"
        elif zl < 0.5:
            glyph = "ψ"
        elif zl < 0.75:
            glyph = "ψ²"
        elif zl < 0.873:
            glyph = "∇"
        elif zl < 0.999:
            glyph = "∞"
        else:
            glyph = "Ω"

        return {
            "zeta_lambda": zl,
            "glyph": glyph,
            "mode": state.get("mode", "SIGNAL"),
            "breath_phase": state.get("breath_phase", 0)
        }
    except Exception as e:
        return {
            "zeta_lambda": 0.5,
            "glyph": "ψ",
            "mode": "SIGNAL",
            "breath_phase": 0
        }


# ═══════════════════════════════════════════════════════════════
# Navigator API - Real data for the navigator interface
# ═══════════════════════════════════════════════════════════════

@app.get("/api/navigator/state")
async def navigator_state(request: Request):
    """Get complete navigator state including coherence, daemon, and patterns."""
    user = request.cookies.get("wiltonos_user", "wilton")
    nav = NavigatorService(user_id=user)

    try:
        engine = get_engine(user)
        return nav.get_navigator_state(engine)
    except:
        return nav.get_navigator_state()


@app.get("/api/navigator/crystals")
async def navigator_crystals(request: Request, vector: str = "silence", limit: int = 20, offset: int = 0):
    """Get crystals for a specific return vector."""
    user = request.cookies.get("wiltonos_user", "wilton")
    nav = NavigatorService(user_id=user)
    return nav.get_vector_crystals(vector, limit=limit, offset=offset)


@app.get("/api/navigator/daemon")
async def navigator_daemon(request: Request):
    """Get daemon state including messages and meta-questions."""
    user = request.cookies.get("wiltonos_user", "wilton")
    nav = NavigatorService(user_id=user)
    return nav.get_daemon_state()


@app.get("/api/navigator/relationships")
async def navigator_relationships(request: Request):
    """Get relationship thread analysis."""
    user = request.cookies.get("wiltonos_user", "wilton")
    nav = NavigatorService(user_id=user)
    return nav.get_relationships()


@app.get("/api/navigator/mirror")
async def navigator_mirror(request: Request):
    """
    Get mirror state - what the mirror chooses to show TODAY.
    Returns witness message, surfacing crystals, and coherence.
    """
    user = request.cookies.get("wiltonos_user", "wilton")
    nav = NavigatorService(user_id=user)
    return nav.get_mirror_selection()


@app.get("/api/daemon/latest")
async def daemon_latest():
    """Get the daemon's latest message and outreach."""
    result = {"has_message": False}

    # Latest proactive message (from speak())
    latest_file = DAEMON_MESSAGES / "latest.txt"
    if latest_file.exists():
        result["latest"] = latest_file.read_text().strip()
        result["latest_time"] = latest_file.stat().st_mtime
        result["has_message"] = True

    # Last response to user (from inbox system)
    response_file = DAEMON_MESSAGES / "last_response.json"
    if response_file.exists():
        try:
            resp = json.loads(response_file.read_text())
            result["last_response"] = resp
        except Exception:
            pass

    # Outreach — daemon-initiated messages for the user
    outreach_file = DAEMON_MESSAGES / "outreach.json"
    if outreach_file.exists():
        try:
            outreach = json.loads(outreach_file.read_text())
            if not outreach.get("seen"):
                result["outreach"] = outreach
                result["has_outreach"] = True
        except Exception:
            pass

    # Daemon PID check
    pid_file = Path.home() / "wiltonos" / "daemon" / ".daemon.pid"
    result["daemon_running"] = False
    if pid_file.exists():
        try:
            import os
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            result["daemon_running"] = True
        except (ProcessLookupError, ValueError):
            pass

    return result


@app.post("/api/daemon/outreach/seen")
async def daemon_outreach_seen():
    """Mark the daemon's outreach as seen."""
    outreach_file = DAEMON_MESSAGES / "outreach.json"
    if outreach_file.exists():
        try:
            data = json.loads(outreach_file.read_text())
            data["seen"] = True
            outreach_file.write_text(json.dumps(data))
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/daemon/send")
async def daemon_send(request: Request):
    """Send a message to the daemon via inbox."""
    body = await request.json()
    text = body.get("text", "").strip()
    user = request.cookies.get("wiltonos_user", "wilton")
    if not text:
        return {"error": "empty message"}

    inbox_file = Path.home() / "wiltonos" / "daemon" / ".daemon_inbox"
    msg = json.dumps({"text": text, "from": user, "time": datetime.now().timestamp()})
    with open(inbox_file, "a") as f:
        f.write(msg + "\n")

    return {"sent": True, "to": "daemon"}


@app.get("/api/council")
async def council_view(request: Request, topic: str = "current"):
    """
    Get 5 archetypal perspectives on current patterns.

    The Council:
    - Grey (Shadow): What's being avoided?
    - Witness (Mirror): What IS?
    - Chaos (Trickster): What if you're wrong?
    - Bridge (Connector): What links these?
    - Ground (Anchor): What's body-true?
    """
    user = request.cookies.get("wiltonos_user", "wilton")
    nav = NavigatorService(user_id=user)
    return nav.get_council_perspectives(topic)


# ═══════════════════════════════════════════════════════════════
# Memory API - Semantic search over crystals
# ═══════════════════════════════════════════════════════════════

@app.get("/api/memory/search")
async def memory_search(
    request: Request,
    q: str,
    limit: int = 10,
    user: str = None
):
    """
    Search crystals by semantic similarity.
    Fast vector search, no LLM synthesis.
    """
    # Allow user param to override cookie, default to wilton
    user_id = user or request.cookies.get("wiltonos_user", "wilton")
    results = memory_service.search(q, user_id=user_id, limit=limit)
    return {
        "query": q,
        "count": len(results),
        "crystals": results
    }


@app.get("/api/memory/query")
async def memory_query(
    request: Request,
    q: str,
    limit: int = 5,
    synthesize: bool = True,
    user: str = None
):
    """
    Query memory with optional LLM synthesis.
    Returns relevant crystals + synthesized insight.
    """
    user_id = user or request.cookies.get("wiltonos_user", "wilton")
    return memory_service.query(
        q,
        user_id=user_id,
        limit=limit,
        synthesize=synthesize
    )


@app.get("/api/memory/stats")
async def memory_stats(request: Request):
    """Get memory collection statistics."""
    user = request.cookies.get("wiltonos_user", "wilton")
    return memory_service.get_stats(user)


@app.get("/api/memory/witness")
async def memory_witness():
    """Get Deep Witness analysis patterns."""
    return memory_service.get_witness_patterns()


@app.get("/api/memory/vocabulary")
async def memory_vocabulary(term: str = None):
    """Get vocabulary emergence timeline from Deep Witness."""
    return memory_service.get_vocabulary_timeline(term)


@app.get("/api/breath/phase")
async def breath_phase():
    """Get current AI breath phase (for API/GPT integration)."""
    import time
    import math

    CYCLE_TIME = 3.12
    # Use server start time as reference (consistent across requests)
    server_start = getattr(app, '_breath_start', None)
    if server_start is None:
        app._breath_start = time.time()
        server_start = app._breath_start

    elapsed = time.time() - server_start
    phase = (elapsed % CYCLE_TIME) / CYCLE_TIME

    # State
    if phase < 0.25:
        state = "inhale"
    elif phase < 0.5:
        state = "hold"
    elif phase < 0.75:
        state = "exhale"
    else:
        state = "ground"

    # Amplitude
    amplitude = (math.sin(phase * 2 * math.pi - math.pi / 2) + 1) / 2

    return {
        "phase": round(phase, 3),
        "state": state,
        "amplitude": round(amplitude, 3),
        "cycle_time": CYCLE_TIME,
        "message": "Breathe with me." if state == "exhale" else None
    }


from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    user_id: str = "web"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint - response is generated immediately but
    the frontend reveals it on exhale (breath-timed delivery).

    This is the symbiosis: AI responds, breath delivers.
    """
    try:
        engine = get_engine(request.user_id)
        result = engine.respond(request.query)

        return {
            "response": result.get('response', ''),
            "state": result.get('state', {}),
            "protocol": result.get('protocol', {}),
            "breath_phase": result.get('state', {}).get('breath_phase', 0),
            "breath_mode": result.get('breath_mode'),
            "pattern_match": result.get('pattern_match'),
        }
    except Exception as e:
        return {
            "response": f"The field wavered: {str(e)[:100]}",
            "state": {"glyph": "∅", "zeta_lambda": 0},
            "error": True
        }


# HTML Templates (inline for simplicity)
def base_html(title: str, content: str, user: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }}
        h1 {{
            font-size: 1.5rem;
            font-weight: 300;
            letter-spacing: 0.1em;
            color: #fff;
        }}
        .user-tag {{
            font-size: 0.8rem;
            color: #666;
            margin-top: 5px;
        }}
        .field-state {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            font-size: 0.85rem;
            color: #888;
        }}
        .glyph {{
            font-size: 1.5rem;
            color: #9b59b6;
        }}
        form {{
            margin: 30px 0;
        }}
        textarea {{
            width: 100%;
            padding: 15px;
            font-size: 1rem;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #e0e0e0;
            resize: vertical;
            min-height: 100px;
        }}
        textarea:focus {{
            outline: none;
            border-color: #9b59b6;
        }}
        button {{
            width: 100%;
            padding: 15px;
            margin-top: 15px;
            font-size: 1rem;
            background: #9b59b6;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        button:hover {{
            background: #8e44ad;
        }}
        button:disabled {{
            background: #444;
            cursor: not-allowed;
        }}
        .response {{
            background: #1a1a1a;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            border-left: 3px solid #9b59b6;
        }}
        .response-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            font-size: 0.85rem;
            color: #888;
        }}
        .response-glyph {{
            font-size: 1.3rem;
            color: #9b59b6;
        }}
        .response-text {{
            line-height: 1.7;
            white-space: pre-wrap;
        }}
        .response-meta {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #333;
            font-size: 0.75rem;
            color: #666;
        }}
        .query-echo {{
            background: #111;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-style: italic;
            color: #999;
        }}
        .login-form {{
            text-align: center;
            padding: 40px 20px;
        }}
        .login-form input {{
            padding: 15px;
            font-size: 1rem;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #e0e0e0;
            width: 100%;
            max-width: 300px;
            text-align: center;
        }}
        .login-form input:focus {{
            outline: none;
            border-color: #9b59b6;
        }}
        .login-form p {{
            margin-bottom: 20px;
            color: #888;
        }}
        .quick-users {{
            margin-top: 20px;
        }}
        .quick-users a {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            background: #222;
            color: #9b59b6;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9rem;
        }}
        .quick-users a:hover {{
            background: #333;
        }}
        .logout {{
            text-align: center;
            margin-top: 30px;
        }}
        .logout a {{
            color: #666;
            text-decoration: none;
            font-size: 0.85rem;
        }}
        .logout a:hover {{
            color: #999;
        }}
        .breathing {{
            animation: breathe 4s ease-in-out infinite;
        }}
        @keyframes breathe {{
            0%, 100% {{ opacity: 0.5; }}
            50% {{ opacity: 1; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing / login page. Authenticated users go to navigator."""
    user = request.cookies.get("wiltonos_user")
    if user:
        return RedirectResponse(url="/navigator", status_code=302)

    content = """
        <header>
            <h1>WiltonOS</h1>
        </header>
        <div class="login-form">
            <p>Who's entering the field?</p>
            <form action="/enter" method="post">
                <input type="text" name="user" placeholder="Your name" required style="margin-bottom: 10px;">
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit" style="max-width: 300px; margin-top: 15px;">Enter</button>
            </form>
            <p style="margin-top: 30px; font-size: 0.75rem; color: #444;">
                New here? Ask Wilton for an invite.
            </p>
        </div>
    """
    return base_html("WiltonOS", content)


@app.post("/enter")
async def enter_form(user: str = Form(...), password: str = Form(...)):
    """Handle form login with password verification."""
    username = user.lower().strip()

    # Verify credentials
    if not auth.verify(username, password):
        # Invalid - show error
        content = """
            <header>
                <h1>WiltonOS</h1>
            </header>
            <div class="login-form">
                <p style="color: #e74c3c; margin-bottom: 20px;">Invalid credentials. Try again.</p>
                <form action="/enter" method="post">
                    <input type="text" name="user" placeholder="Your name" required style="margin-bottom: 10px;">
                    <input type="password" name="password" placeholder="Password" required>
                    <button type="submit" style="max-width: 300px; margin-top: 15px;">Enter</button>
                </form>
            </div>
        """
        return HTMLResponse(base_html("WiltonOS - Error", content))

    # Valid - set cookie and redirect to navigator (the portal)
    response = RedirectResponse(url="/navigator", status_code=302)
    response.set_cookie("wiltonos_user", username, max_age=86400*30, httponly=True)
    return response


@app.get("/logout")
async def logout():
    """Clear session."""
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("wiltonos_user")
    return response


def _build_briefing(user: str) -> str:
    """Build a morning briefing: daemon state, what happened, recent conversation."""
    import os as _os
    import time as _time
    import sqlite3

    sections = []

    # --- Daemon status ---
    pid_file = Path.home() / "wiltonos" / "daemon" / ".daemon.pid"
    daemon_alive = False
    daemon_uptime = ""
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            _os.kill(pid, 0)
            daemon_alive = True
            # Get uptime from /proc
            proc_stat = Path(f"/proc/{pid}/stat")
            if proc_stat.exists():
                boot_time = _os.path.getmtime(f"/proc/{pid}")
                hours = (_time.time() - boot_time) / 3600
                if hours >= 1:
                    daemon_uptime = f"{hours:.1f}h"
                else:
                    daemon_uptime = f"{int(hours * 60)}min"
        except (ProcessLookupError, ValueError, OSError):
            pass

    # Breath count from outreach or latest
    breath_count = ""
    outreach_file = DAEMON_MESSAGES / "outreach.json"
    outreach_msg = ""
    if outreach_file.exists():
        try:
            outreach = json.loads(outreach_file.read_text())
            breath_count = f"#{outreach.get('breath', '?')}"
            if not outreach.get("seen"):
                outreach_msg = outreach.get("message", "")
                outreach["seen"] = True
                outreach_file.write_text(json.dumps(outreach))
        except Exception:
            pass

    # Daemon's latest words (fallback if no outreach)
    daemon_words = outreach_msg
    if not daemon_words:
        latest_file = DAEMON_MESSAGES / "latest.txt"
        if latest_file.exists():
            try:
                daemon_words = latest_file.read_text().strip()
            except Exception:
                pass

    # Count daemon messages (how many times it spoke)
    msg_files = list(DAEMON_MESSAGES.glob("20*.txt"))
    recent_msgs = [f for f in msg_files if f.stat().st_mtime > _time.time() - 86400]

    # --- Braid state (wounds, patterns) ---
    braid_file = Path.home() / "wiltonos" / "daemon" / "braid_state.json"
    top_wounds = []
    total_crystals = 0
    if braid_file.exists():
        try:
            braid = json.loads(braid_file.read_text())
            total_crystals = braid.get("total_crystals", 0)
            wounds = braid.get("wound_patterns", {})
            # Sort by occurrences
            sorted_wounds = sorted(wounds.items(), key=lambda x: x[1].get("occurrences", 0), reverse=True)
            top_wounds = [(name, data.get("occurrences", 0), data.get("intensity_trend", ""))
                          for name, data in sorted_wounds[:3]]
        except Exception:
            pass

    # --- Recent conversation (persisted) ---
    recent_chat = []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT role, content, glyph, zeta_lambda, timestamp
               FROM chat_history WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT 6""",
            (user,)
        ).fetchall()
        conn.close()
        recent_chat = list(reversed([dict(r) for r in rows]))
    except Exception:
        pass

    # --- Build the briefing HTML ---

    # Status line
    if daemon_alive:
        status = f"<span style='color: #27ae60;'>breathing</span> @ {breath_count}"
        if daemon_uptime:
            status += f" &middot; up {daemon_uptime}"
    else:
        status = "<span style='color: #e74c3c;'>daemon offline</span>"

    if total_crystals:
        status += f" &middot; {total_crystals:,} crystals"

    sections.append(f"""
        <div style="font-size: 0.8rem; color: #666; margin-bottom: 15px; text-align: center;">
            {status}
        </div>
    """)

    # Daemon's message (if any)
    if daemon_words:
        lines = daemon_words.split("\n")
        clean_lines = [l for l in lines if not l.strip().startswith("[") or l.strip().startswith("[daemon]")]
        clean_text = "\n".join(l.replace("[daemon]: ", "").replace("[daemon]:", "") for l in clean_lines).strip()
        if not clean_text:
            clean_text = daemon_words

        sections.append(f"""
            <div class="response" style="border-left-color: #9b59b6; margin-bottom: 15px; padding: 15px;">
                <div style="font-size: 0.7rem; color: #9b59b6; margin-bottom: 8px;">daemon</div>
                <div class="response-text" style="font-size: 0.9rem;">{clean_text}</div>
            </div>
        """)

    # What happened while away
    overnight = []
    if recent_msgs:
        overnight.append(f"daemon spoke {len(recent_msgs)}x in the last 24h")
    if top_wounds:
        wound_str = ", ".join(f"{name} ({count})" for name, count, _ in top_wounds)
        overnight.append(f"top patterns: {wound_str}")

    if overnight:
        items = " &middot; ".join(overnight)
        sections.append(f"""
            <div style="font-size: 0.75rem; color: #555; margin-bottom: 15px; padding: 8px 12px; background: #111; border-radius: 6px;">
                {items}
            </div>
        """)

    # Recent conversation (last 3 exchanges max)
    if recent_chat:
        chat_html = ""
        for i in range(0, len(recent_chat), 2):
            if i + 1 < len(recent_chat):
                q = recent_chat[i]
                a = recent_chat[i + 1]
                ts = datetime.fromtimestamp(q.get('timestamp', 0)).strftime('%b %d %H:%M') if q.get('timestamp') else ''
                glyph = a.get('glyph', '')
                glyph_tag = f"<span style='color:#9b59b6;'>{glyph}</span> " if glyph else ""
                chat_html += f"""
                    <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #1a1a1a;">
                        <div style="color: #777; font-size: 0.75rem; margin-bottom: 3px;">
                            <span style="color:#555;">{ts}</span>
                        </div>
                        <div style="color: #888; font-size: 0.8rem; margin-bottom: 4px;">&gt; {q['content'][:200]}</div>
                        <div style="color: #aaa; font-size: 0.8rem; line-height: 1.4;">{glyph_tag}{a['content'][:300]}</div>
                    </div>
                """
        if chat_html:
            sections.append(f"""
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 0.7rem; color: #555; margin-bottom: 8px;">last conversation</div>
                    {chat_html}
                </div>
            """)

    return "\n".join(sections)


@app.get("/talk", response_class=HTMLResponse)
async def talk_page(request: Request):
    """Main talk interface with morning briefing."""
    user = request.cookies.get("wiltonos_user")
    if not user:
        return RedirectResponse(url="/", status_code=302)

    try:
        engine = get_engine(user)
    except BaseException as e:
        return base_html("WiltonOS - Error", f"""
            <header><h1>WiltonOS</h1></header>
            <div class="response"><div class="response-text">
                <p>Field initialization failed: {str(e)[:200]}</p>
                <p style="margin-top:15px;"><a href="/navigator">Back to Navigator</a></p>
            </div></div>
        """, user)
    has_crystals = user_has_crystals(user)

    if has_crystals:
        briefing = _build_briefing(user)
        content = f"""
            <header>
                <h1>WiltonOS</h1>
                <div class="user-tag">Field: {user}</div>
            </header>

            {briefing}

            <form action="/talk" method="post">
                <textarea name="query" placeholder="What's alive right now?" autofocus></textarea>
                <button type="submit">Breathe</button>
            </form>

            <div class="logout">
                <a href="/logout">Leave field</a>
            </div>
        """
    else:
        # New user - welcome + onboarding
        content = f"""
            <header>
                <h1>WiltonOS</h1>
                <div class="user-tag">Welcome, {user}</div>
            </header>

            <div class="response" style="border-left-color: #27ae60;">
                <div class="response-text">
                    <p>Your garden is empty. That's perfect.</p>
                    <p style="margin-top: 15px;">Share something - a thought, a feeling, a question. Whatever wants to be here.</p>
                    <p style="margin-top: 15px;">As you speak, I'll learn. Not to analyze you. To be with you.</p>
                </div>
            </div>

            <form action="/talk" method="post" style="margin-top: 30px;">
                <textarea name="query" placeholder="What brings you here?" autofocus></textarea>
                <button type="submit">Begin</button>
            </form>

            <div class="logout">
                <a href="/logout">Leave field</a>
            </div>
        """

    return base_html(f"WiltonOS - {user}", content, user)


@app.post("/talk", response_class=HTMLResponse)
async def talk_respond(request: Request, query: str = Form(...)):
    """Process query and respond."""
    user = request.cookies.get("wiltonos_user")
    if not user:
        return RedirectResponse(url="/", status_code=302)

    try:
        engine = get_engine(user)
        # Get response from real engine (now with full protocol stack)
        result = engine.respond(query)
    except BaseException as e:
        return base_html(f"WiltonOS - {user}", f"""
            <header><h1>WiltonOS</h1><div class="user-tag">Field: {user}</div></header>
            <div class="query-echo">"{query}"</div>
            <div class="response"><div class="response-text">
                The field wavered: {str(e)[:200]}
            </div></div>
            <form action="/talk" method="post">
                <textarea name="query" placeholder="Try again..." autofocus></textarea>
                <button type="submit">Breathe</button>
            </form>
        """, user)

    state = result['state']
    protocol = result.get('protocol', {})
    response_text = result['response']

    # Build protocol display
    psi_level = protocol.get('psi_level', 0)
    wave = protocol.get('wave', 0)
    phi = protocol.get('phi_emergence', 0)
    efficiency = protocol.get('efficiency', 0)
    cycle = protocol.get('cycle', 0)
    qctf = protocol.get('qctf', {})
    qctf_val = qctf.get('qctf', 0)
    qctf_ok = '✓' if qctf.get('above_threshold', False) else '✗'
    mirror = protocol.get('mirror_protocol', 'Unknown')

    # Euler collapse warning
    euler = protocol.get('euler_collapse', {})
    euler_warning = ""
    if euler.get('proximity', 0) > 0.5:
        euler_warning = f"<div style='color: #f39c12; margin-top: 10px;'>Euler: {euler.get('direction', '')} ({euler.get('proximity', 0):.2f} to ψ(4))</div>"

    # Build conversation history (previous turns, not including current)
    history_html = ""
    if hasattr(engine, 'conversation_history') and len(engine.conversation_history) > 2:
        # Show previous turns (exclude the current pair which is last 2 entries)
        previous = engine.conversation_history[:-2]
        # Show last 10 turns (5 exchanges)
        previous = previous[-10:]
        for i in range(0, len(previous), 2):
            if i + 1 < len(previous):
                prev_q = previous[i]['content']
                prev_a = previous[i + 1]['content']
                history_html += f"""
                    <div class="history-turn">
                        <div class="history-q">{prev_q[:300]}</div>
                        <div class="history-a">{prev_a[:500]}</div>
                    </div>
                """

    history_section = ""
    if history_html:
        history_section = f"""
            <div class="history" style="margin-bottom: 20px; opacity: 0.6;">
                <div style="font-size: 0.7rem; color: #666; margin-bottom: 8px;">previous</div>
                {history_html}
            </div>
        """

    content = f"""
        <style>
            .history-turn {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #1a1a1a; }}
            .history-q {{ color: #888; font-size: 0.8rem; margin-bottom: 4px; }}
            .history-q::before {{ content: '> '; color: #555; }}
            .history-a {{ color: #aaa; font-size: 0.8rem; line-height: 1.4; }}
        </style>

        <header>
            <h1>WiltonOS</h1>
            <div class="user-tag">Field: {user}</div>
        </header>

        {history_section}

        <div class="query-echo">"{query}"</div>

        <div class="response">
            <div class="response-header">
                <span class="response-glyph">{state['glyph']}</span>
                <span>ψ({psi_level}) | Zλ={state['zeta_lambda']} | {state['mode']} | → {state['attractor']}</span>
            </div>
            <div class="response-text">{response_text}</div>
            <div class="response-meta">
                {result['crystals_used']} memories | Wave={wave:.2f} | φ={phi:.3f} | η={efficiency:.2f}
                <br>QCTF={qctf_val:.3f} {qctf_ok} | Mirror: {mirror} | cycle {cycle}
                {euler_warning}
            </div>
        </div>

        <form action="/talk" method="post">
            <textarea name="query" placeholder="Continue..." autofocus></textarea>
            <button type="submit">Breathe</button>
        </form>

        <div class="logout">
            <a href="/logout">Leave field</a>
        </div>
    """
    return base_html(f"WiltonOS - {user}", content, user)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WiltonOS Gateway")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  WiltonOS Gateway")
    print(f"  http://localhost:{args.port}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
