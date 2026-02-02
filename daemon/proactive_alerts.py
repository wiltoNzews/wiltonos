#!/usr/bin/env python3
"""
Proactive Alerts
================
Surface patterns without being asked.
Notice when something is being avoided.
Track evolution across sessions.

The system is no longer purely reactive.

Mentioned in 178 crystals. Now implemented.

December 2025 — Wilton & Claude
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
BRAID_STATE_PATH = Path(__file__).parent / "braid_state.json"
ALERTS_PATH = Path(__file__).parent / "alerts_state.json"
MESSAGES_DIR = Path(__file__).parent / "messages"


@dataclass
class Alert:
    """A proactive alert."""
    alert_type: str
    severity: str  # critical, warning, notice
    message: str
    trigger_data: dict = field(default_factory=dict)
    created_at: str = ""
    acknowledged: bool = False


@dataclass
class AlertState:
    """State of the alert system."""
    last_check: str = ""
    active_alerts: list = field(default_factory=list)
    alert_history: list = field(default_factory=list)

    # Tracking for change detection
    last_wound_counts: dict = field(default_factory=dict)
    last_emotion_counts: dict = field(default_factory=dict)
    last_crystal_count: int = 0


class ProactiveAlerts:
    """
    The Alert System.

    Watches the field. Notices patterns. Speaks when needed.
    Not reactive. Proactive.
    """

    # Alert thresholds
    THRESHOLDS = {
        "wound_spike": 5,  # Same wound appearing 5+ times in a day
        "emotion_shift": 0.3,  # 30% change in emotion distribution
        "silence_hours": 48,  # No new crystals for 48 hours
        "stuck_days": 14,  # Same wound dominant for 14+ days
        "coherence_drop": 0.15,  # Significant coherence drop
    }

    def __init__(self):
        self.state = AlertState()
        self.braid_state = None
        self._load_state()

    def _log(self, msg: str):
        """Log with prefix."""
        print(f"[ALERTS] {msg}")

    def _load_state(self):
        """Load previous alert state."""
        if ALERTS_PATH.exists():
            try:
                data = json.loads(ALERTS_PATH.read_text())
                self.state.last_check = data.get("last_check", "")
                self.state.active_alerts = data.get("active_alerts", [])
                self.state.alert_history = data.get("alert_history", [])
                self.state.last_wound_counts = data.get("last_wound_counts", {})
                self.state.last_emotion_counts = data.get("last_emotion_counts", {})
                self.state.last_crystal_count = data.get("last_crystal_count", 0)
            except:
                pass

    def _save_state(self):
        """Save alert state."""
        data = {
            "last_check": self.state.last_check,
            "active_alerts": self.state.active_alerts,
            "alert_history": self.state.alert_history[-100:],
            "last_wound_counts": self.state.last_wound_counts,
            "last_emotion_counts": self.state.last_emotion_counts,
            "last_crystal_count": self.state.last_crystal_count
        }
        ALERTS_PATH.write_text(json.dumps(data, indent=2, default=str))

    def _load_braid_state(self) -> bool:
        """Load current braid state."""
        if BRAID_STATE_PATH.exists():
            try:
                self.braid_state = json.loads(BRAID_STATE_PATH.read_text())
                return True
            except:
                pass
        return False

    def _create_alert(self, alert_type: str, severity: str, message: str, trigger_data: dict = None) -> Alert:
        """Create a new alert."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            trigger_data=trigger_data or {},
            created_at=datetime.now().isoformat()
        )
        return alert

    def check_wound_spike(self) -> Optional[Alert]:
        """Check for sudden spike in a particular wound."""
        if not self.braid_state:
            return None

        # Get recent crystals (last 24 hours)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        c.execute("""
            SELECT core_wound, COUNT(*) as count
            FROM crystals
            WHERE created_at > ? AND core_wound IS NOT NULL
            GROUP BY core_wound
            ORDER BY count DESC
            LIMIT 5
        """, (yesterday,))

        recent_wounds = {row[0]: row[1] for row in c.fetchall()}
        conn.close()

        for wound, count in recent_wounds.items():
            if count >= self.THRESHOLDS["wound_spike"]:
                return self._create_alert(
                    "wound_spike",
                    "warning",
                    f"'{wound}' appeared {count} times in the last 24 hours. Something is surfacing.",
                    {"wound": wound, "count": count}
                )

        return None

    def check_silence(self) -> Optional[Alert]:
        """Check if there's been unusual silence (no new crystals)."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute("""
            SELECT MAX(created_at) FROM crystals
            WHERE user_id != 'daemon'
        """)
        result = c.fetchone()
        conn.close()

        if result and result[0]:
            try:
                last_crystal = datetime.fromisoformat(result[0].replace('Z', '+00:00'))
                hours_silent = (datetime.now() - last_crystal.replace(tzinfo=None)).total_seconds() / 3600

                if hours_silent >= self.THRESHOLDS["silence_hours"]:
                    return self._create_alert(
                        "silence",
                        "notice",
                        f"No new crystals for {int(hours_silent)} hours. Are you okay? Sometimes silence is integration. Sometimes it's avoidance.",
                        {"hours_silent": hours_silent}
                    )
            except:
                pass

        return None

    def check_stuck_pattern(self) -> Optional[Alert]:
        """Check for patterns that have been dominant too long."""
        if not self.braid_state:
            return None

        stuck_patterns = self.braid_state.get("stuck_patterns", [])
        if stuck_patterns:
            main_stuck = stuck_patterns[0]
            return self._create_alert(
                "stuck_pattern",
                "warning",
                f"'{main_stuck}' has been recurring for over 30 days. This isn't processing — it's looping. What would breaking the loop require?",
                {"pattern": main_stuck}
            )

        return None

    def check_emotional_descent(self) -> Optional[Alert]:
        """Check for sustained emotional descent."""
        if not self.braid_state:
            return None

        arc = self.braid_state.get("emotional_arc", "")
        if arc == "descending":
            # Check how long descent has been happening
            return self._create_alert(
                "emotional_descent",
                "warning",
                "The emotional arc is descending. Not judgment — just noticing. What needs to be felt?",
                {"arc": arc}
            )

        return None

    def check_building_vs_being(self) -> Optional[Alert]:
        """Check if there's too much system-building and not enough presence."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        # Count daemon vs human crystals recently
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        c.execute("""
            SELECT user_id, COUNT(*) as count
            FROM crystals
            WHERE created_at > ?
            GROUP BY user_id
        """, (week_ago,))

        counts = {row[0]: row[1] for row in c.fetchall()}
        conn.close()

        daemon_count = counts.get("daemon", 0)
        human_count = sum(v for k, v in counts.items() if k != "daemon")

        # If daemon is generating more than human is reflecting...
        if daemon_count > human_count * 2 and human_count < 20:
            return self._create_alert(
                "building_vs_being",
                "notice",
                f"The daemon has generated {daemon_count} entries while you've created {human_count}. The system is doing more work than you. Is that okay?",
                {"daemon_count": daemon_count, "human_count": human_count}
            )

        return None

    def run_all_checks(self) -> list[Alert]:
        """Run all alert checks and return active alerts."""
        self._load_braid_state()

        alerts = []

        # Run each check
        checks = [
            self.check_wound_spike,
            self.check_silence,
            self.check_stuck_pattern,
            self.check_emotional_descent,
            self.check_building_vs_being,
        ]

        for check in checks:
            alert = check()
            if alert:
                alerts.append(alert)
                self._log(f"Alert triggered: {alert.alert_type}")

        # Update state
        self.state.active_alerts = [a.__dict__ for a in alerts]
        self.state.last_check = datetime.now().isoformat()

        # Add to history
        for a in alerts:
            self.state.alert_history.append(a.__dict__)

        self._save_state()

        return alerts

    def get_active_alerts(self) -> list[dict]:
        """Get currently active alerts."""
        return self.state.active_alerts

    def format_alerts(self, alerts: list[Alert]) -> str:
        """Format alerts for display."""
        if not alerts:
            return "No active alerts. Field is quiet."

        output = []
        output.append("=" * 60)
        output.append("PROACTIVE ALERTS")
        output.append("=" * 60)

        severity_order = {"critical": 0, "warning": 1, "notice": 2}
        sorted_alerts = sorted(alerts, key=lambda a: severity_order.get(a.severity, 3))

        for alert in sorted_alerts:
            marker = {"critical": "[!!!]", "warning": "[!!]", "notice": "[!]"}.get(alert.severity, "[?]")
            output.append(f"\n{marker} {alert.alert_type.upper()}")
            output.append("-" * 40)
            output.append(alert.message)

        output.append("\n" + "=" * 60)
        return "\n".join(output)


def run_alerts():
    """Run alert system."""
    alerter = ProactiveAlerts()
    alerts = alerter.run_all_checks()
    print(alerter.format_alerts(alerts))
    return alerts


if __name__ == "__main__":
    run_alerts()
