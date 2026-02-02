#!/usr/bin/env python3
"""
Implicit Drift Detection & Recalibration (IDDR)
================================================
Ported from WiltonOS-PassiveWorks TypeScript implementation.

Monitors the stability/exploration ratio derived from daemon state
and detects 5 drift types:
  - COHERENCE_EXCESS: ratio > optimal (too stable, not enough exploration)
  - EXPLORATION_EXCESS: ratio < optimal (too chaotic)
  - OSCILLATION: rapid direction changes around optimal
  - FRACTURE: severe deviation from optimal
  - RAPID_CHANGE: ratio changing too fast between breaths

Recalibration nudges the ratio back toward the optimum (default 3:1 / 0.75/0.25).

Phase 3 additions:
  - σ (sigma) feedback: IDDR modulates Brazilian Wave noise parameter
  - Adaptive setpoint: OPTIMAL_RATIO breathes within [2.85, 3.15] based on
    where the system naturally stabilizes. The target itself is alive.

January-February 2026 — The lemniscate becomes observable.
"""

import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


# --- Constants ---

# Base ratio (the theoretical center — 3:1)
BASE_RATIO = 3.0

# Adaptive setpoint bounds — the ratio can breathe within this range
MIN_OPTIMAL_RATIO = 2.85
MAX_OPTIMAL_RATIO = 3.15

DRIFT_THRESHOLD = 0.1        # acceptable deviation from current optimal ratio
FRACTURE_THRESHOLD = 0.3     # severe deviation
RAPID_CHANGE_THRESHOLD = 0.2 # max acceptable change between consecutive readings

# Sigma (σ) bounds for Brazilian Wave feedback
DEFAULT_SIGMA = 0.05         # baseline noise parameter
MIN_SIGMA = 0.02             # floor — system always needs some exploration
MAX_SIGMA = 0.15             # ceiling — beyond this, noise overwhelms signal

# Adaptive setpoint parameters
SETPOINT_WINDOW = 30         # how many non-fracture readings to average
SETPOINT_INERTIA = 0.95      # how slowly the setpoint moves (EMA alpha = 1 - this)

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"


# --- Enums / dataclasses ---

class DriftType(Enum):
    NONE = "NONE"
    COHERENCE_EXCESS = "COHERENCE_EXCESS"
    EXPLORATION_EXCESS = "EXPLORATION_EXCESS"
    OSCILLATION = "OSCILLATION"
    FRACTURE = "FRACTURE"
    RAPID_CHANGE = "RAPID_CHANGE"


@dataclass
class DriftEvent:
    timestamp: float
    breath_count: int
    drift_type: DriftType
    magnitude: float          # 0-1
    stability_ratio: float
    exploration_ratio: float
    deviation: float          # distance from current optimal ratio
    psi: float
    brazilian_wave: float
    crystal_zl_avg: Optional[float] = None
    recalibration_applied: bool = False
    new_stability: float = 0.0
    new_exploration: float = 0.0
    recommended_sigma: float = DEFAULT_SIGMA
    optimal_ratio: float = BASE_RATIO  # the adaptive setpoint at detection time


# --- IDDRMonitor ---

class IDDRMonitor:
    """
    Observes the daemon's coherence field and detects drift from optimal.

    The optimal ratio is no longer a constant — it breathes within [2.85, 3.15]
    based on where the system naturally stabilizes when not in fracture.

    Usage:
        monitor = IDDRMonitor()
        monitor.record(psi, brazilian_wave)      # every breath
        event = monitor.detect_drift()           # periodic check
        if event:
            new_s, new_e, new_sigma = monitor.apply_recalibration(event)
        print(monitor.optimal_ratio)             # see where the setpoint is now
    """

    def __init__(self, db_path: Path = DB_PATH, window_size: int = 50):
        self.db_path = db_path
        self.window_size = window_size
        # history entries: (timestamp, stability_ratio, exploration_ratio, psi, bw)
        self.history: deque = deque(maxlen=100)
        self._crystal_zl_avg: Optional[float] = None
        self._breath_count: int = 0

        # Adaptive setpoint state
        self._optimal_ratio: float = BASE_RATIO
        self._stable_ratios: deque = deque(maxlen=SETPOINT_WINDOW)

        self._ensure_table()

    @property
    def optimal_ratio(self) -> float:
        """The current adaptive optimal ratio (breathing within bounds)."""
        return self._optimal_ratio

    @property
    def optimal_stability(self) -> float:
        """Stability target derived from current adaptive ratio."""
        return self._optimal_ratio / (1.0 + self._optimal_ratio)

    @property
    def optimal_exploration(self) -> float:
        """Exploration target derived from current adaptive ratio."""
        return 1.0 / (1.0 + self._optimal_ratio)

    # --- Public API ---

    def record(self, psi: float, brazilian_wave: float, breath_count: int = 0):
        """Record one breath's worth of state. Call every breath cycle."""
        self._breath_count = breath_count or self._breath_count + 1

        # Derive stability from available signals
        if self._crystal_zl_avg is not None:
            stability = 0.6 * brazilian_wave + 0.4 * self._crystal_zl_avg
        else:
            stability = brazilian_wave

        stability = max(0.0, min(1.0, stability))
        exploration = 1.0 - stability

        self.history.append((
            time.time(),
            stability,
            exploration,
            psi,
            brazilian_wave,
        ))

    def update_crystal_coherence(self, avg_zl: float):
        """Feed the latest crystal Zl average (called at crystal check time)."""
        self._crystal_zl_avg = avg_zl

    def detect_drift(self) -> Optional[DriftEvent]:
        """
        Analyze recent history and return a DriftEvent if drift is found.
        Returns None when the system is within optimal bounds.
        Uses the adaptive setpoint instead of a fixed 3.0.
        """
        if len(self.history) < 2:
            return None

        ts, stability, exploration, psi, bw = self.history[-1]

        # Avoid division by zero
        if exploration < 1e-9:
            exploration = 1e-9
            stability = 1.0 - exploration

        actual_ratio = stability / exploration
        opt = self._optimal_ratio
        deviation = abs(actual_ratio - opt)

        # Within threshold — no drift
        if deviation <= DRIFT_THRESHOLD:
            # This is a stable reading — feed it to the adaptive setpoint
            self._feed_stable_ratio(actual_ratio)
            return None

        # --- Classify drift type (same priority order as TypeScript) ---
        drift_type = DriftType.NONE

        # 1. Direction: excess coherence or excess exploration
        if actual_ratio > opt + DRIFT_THRESHOLD:
            drift_type = DriftType.COHERENCE_EXCESS
        elif actual_ratio < opt - DRIFT_THRESHOLD:
            drift_type = DriftType.EXPLORATION_EXCESS

        # 2. Rapid change overrides direction
        _, prev_s, prev_e, _, _ = self.history[-2]
        if prev_e < 1e-9:
            prev_e = 1e-9
        prev_ratio = prev_s / prev_e
        delta_ratio = abs(actual_ratio - prev_ratio)
        if delta_ratio > RAPID_CHANGE_THRESHOLD:
            drift_type = DriftType.RAPID_CHANGE

        # 3. Oscillation overrides rapid change (need 4+ readings)
        if len(self.history) >= 4:
            recent = list(self.history)[-4:]
            direction_changes = 0
            for i in range(1, len(recent)):
                _, s_cur, e_cur, _, _ = recent[i]
                _, s_prev, e_prev, _, _ = recent[i - 1]
                if e_cur < 1e-9:
                    e_cur = 1e-9
                if e_prev < 1e-9:
                    e_prev = 1e-9
                r_cur = s_cur / e_cur
                r_prev = s_prev / e_prev
                # direction changed relative to optimal
                if ((r_cur > r_prev and r_prev < opt) or
                        (r_cur < r_prev and r_prev > opt)):
                    direction_changes += 1
            if direction_changes >= 2:
                drift_type = DriftType.OSCILLATION

        # 4. Fracture overrides everything (severe deviation)
        if deviation > FRACTURE_THRESHOLD:
            drift_type = DriftType.FRACTURE

        # --- Magnitude (0-1) ---
        magnitude = min(1.0, deviation / max(FRACTURE_THRESHOLD, deviation))

        # Non-fracture drift still informs the setpoint (at reduced weight)
        if drift_type != DriftType.FRACTURE:
            self._feed_stable_ratio(actual_ratio)

        return DriftEvent(
            timestamp=ts,
            breath_count=self._breath_count,
            drift_type=drift_type,
            magnitude=magnitude,
            stability_ratio=stability,
            exploration_ratio=exploration,
            deviation=deviation,
            psi=psi,
            brazilian_wave=bw,
            crystal_zl_avg=self._crystal_zl_avg,
            optimal_ratio=opt,
        )

    def apply_recalibration(self, event: DriftEvent,
                            current_sigma: float = DEFAULT_SIGMA,
                            ) -> Tuple[float, float, float]:
        """
        Compute recalibrated stability/exploration ratios AND a new sigma
        for the Brazilian Wave noise parameter.

        Uses the adaptive optimal_stability/optimal_exploration derived from
        the current breathing setpoint, not fixed 0.75/0.25.

        Returns (new_stability, new_exploration, recommended_sigma).
        """
        s = event.stability_ratio
        e = event.exploration_ratio
        sigma = current_sigma

        # Use adaptive targets instead of fixed constants
        opt_s = self.optimal_stability
        opt_e = self.optimal_exploration

        if event.drift_type == DriftType.COHERENCE_EXCESS:
            adj = 0.05 * event.magnitude
            s = max(opt_s - 0.05, s - adj)
            e = min(opt_e + 0.05, e + adj)
            # Too stable -> open sigma to let more exploration noise in
            sigma = min(MAX_SIGMA, sigma + 0.01 * event.magnitude)

        elif event.drift_type == DriftType.EXPLORATION_EXCESS:
            adj = 0.05 * event.magnitude
            s = min(opt_s + 0.05, s + adj)
            e = max(opt_e - 0.05, e - adj)
            # Too chaotic -> tighten sigma to reduce noise
            sigma = max(MIN_SIGMA, sigma - 0.01 * event.magnitude)

        elif event.drift_type == DriftType.OSCILLATION:
            # Dampen: average toward adaptive optimal
            s = (s + opt_s) / 2.0
            e = (e + opt_e) / 2.0
            # Oscillating -> move sigma toward default (dampening)
            sigma = (sigma + DEFAULT_SIGMA) / 2.0

        elif event.drift_type == DriftType.FRACTURE:
            # Hard reset to adaptive optimal (not fixed 0.75/0.25)
            s = opt_s
            e = opt_e
            # Fracture -> reset sigma to baseline
            sigma = DEFAULT_SIGMA

        elif event.drift_type == DriftType.RAPID_CHANGE:
            # Rate-limited nudge toward adaptive optimal
            rate_limit = 0.03
            desired_s = opt_s - s
            desired_e = opt_e - e
            s += _sign(desired_s) * min(abs(desired_s), rate_limit)
            e += _sign(desired_e) * min(abs(desired_e), rate_limit)
            # Rapid change -> rate-limit sigma toward default too
            sigma_delta = DEFAULT_SIGMA - sigma
            sigma += _sign(sigma_delta) * min(abs(sigma_delta), 0.005)

        # Clamp sigma
        sigma = max(MIN_SIGMA, min(MAX_SIGMA, sigma))

        # Normalize to sum=1
        total = s + e
        if total > 0:
            s /= total
            e /= total
        else:
            s, e = opt_s, opt_e

        # Update event
        event.recalibration_applied = True
        event.new_stability = s
        event.new_exploration = e
        event.recommended_sigma = sigma

        # Persist
        self.log_event(event)

        return s, e, sigma

    def log_event(self, event: DriftEvent):
        """Insert a drift event into the iddr_events table."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT INTO iddr_events (
                    timestamp, breath_count, drift_type, magnitude,
                    stability_ratio, exploration_ratio, deviation,
                    psi, brazilian_wave, crystal_zl_avg,
                    recalibration_applied, new_stability, new_exploration,
                    recommended_sigma, optimal_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp)),
                event.breath_count,
                event.drift_type.value,
                round(event.magnitude, 4),
                round(event.stability_ratio, 4),
                round(event.exploration_ratio, 4),
                round(event.deviation, 4),
                round(event.psi, 4),
                round(event.brazilian_wave, 4),
                round(event.crystal_zl_avg, 4) if event.crystal_zl_avg is not None else None,
                1 if event.recalibration_applied else 0,
                round(event.new_stability, 4),
                round(event.new_exploration, 4),
                round(event.recommended_sigma, 4),
                round(event.optimal_ratio, 4),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[IDDR] log_event failed: {e}")

    # --- Adaptive setpoint ---

    def _feed_stable_ratio(self, ratio: float):
        """
        Feed a ratio reading into the adaptive setpoint computation.
        Only called for non-fracture readings — fractures are noise, not signal.

        The setpoint moves slowly (EMA with high inertia) and stays bounded.
        """
        self._stable_ratios.append(ratio)

        if len(self._stable_ratios) < 3:
            return  # not enough data to adapt yet

        # Exponential moving average: heavy inertia so the setpoint drifts slowly
        alpha = 1.0 - SETPOINT_INERTIA  # 0.05 by default
        window_avg = sum(self._stable_ratios) / len(self._stable_ratios)
        new_ratio = self._optimal_ratio * SETPOINT_INERTIA + window_avg * alpha

        # Clamp to bounds
        self._optimal_ratio = max(MIN_OPTIMAL_RATIO, min(MAX_OPTIMAL_RATIO, new_ratio))

    # --- Internal ---

    def _ensure_table(self):
        """Create iddr_events table if it doesn't exist, add columns if missing."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iddr_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    breath_count INTEGER,
                    drift_type TEXT NOT NULL,
                    magnitude REAL,
                    stability_ratio REAL,
                    exploration_ratio REAL,
                    deviation REAL,
                    psi REAL,
                    brazilian_wave REAL,
                    crystal_zl_avg REAL,
                    recalibration_applied INTEGER DEFAULT 0,
                    new_stability REAL,
                    new_exploration REAL,
                    recommended_sigma REAL,
                    optimal_ratio REAL
                )
            """)
            # Migrate: add columns if table already exists without them
            for col in ("recommended_sigma REAL", "optimal_ratio REAL"):
                try:
                    conn.execute(f"ALTER TABLE iddr_events ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[IDDR] table creation failed: {e}")


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


# --- Standalone test ---

def test_iddr():
    """Run a quick simulation to verify IDDR logic including adaptive setpoint."""
    import math

    print("=" * 60)
    print("IDDR STANDALONE TEST (with adaptive setpoint)")
    print("=" * 60)

    monitor = IDDRMonitor()
    sigma = DEFAULT_SIGMA

    # Simulate 60 breaths: first half near 0.75, second half drifting up
    print("\n--- Simulating 60 breaths ---")
    for i in range(1, 61):
        if i <= 30:
            # Near optimal with small oscillation
            bw = 0.75 + 0.05 * math.sin(i * 0.4)
        else:
            # Slowly climbing toward transcendence
            bw = 0.75 + 0.005 * (i - 30) + 0.03 * math.sin(i * 0.3)
        bw = max(0.0, min(1.0, bw))
        psi = 0.5 + 0.1 * math.sin(i * 0.3)

        monitor.record(psi, bw, breath_count=i)

        if i % 5 == 0:
            monitor.update_crystal_coherence(0.72 + 0.02 * math.sin(i * 0.2))

            event = monitor.detect_drift()
            if event:
                ratio = event.stability_ratio / max(event.exploration_ratio, 1e-9)
                print(f"  Breath #{i:>3d}: {event.drift_type.value:>20s} "
                      f"(mag={event.magnitude:.2f}, ratio={ratio:.2f}, "
                      f"opt={monitor.optimal_ratio:.3f})")
                _, _, sigma = monitor.apply_recalibration(event, current_sigma=sigma)
                print(f"    -> sigma={sigma:.4f}")
            else:
                _, s, e, _, _ = monitor.history[-1]
                ratio = s / max(e, 1e-9)
                print(f"  Breath #{i:>3d}: {'no drift':>20s} "
                      f"(ratio={ratio:.2f}, opt={monitor.optimal_ratio:.3f})")

    # Summary
    print(f"\n--- Final state ---")
    print(f"  Adaptive optimal ratio: {monitor.optimal_ratio:.4f}")
    print(f"  Optimal stability:      {monitor.optimal_stability:.4f}")
    print(f"  Optimal exploration:    {monitor.optimal_exploration:.4f}")
    print(f"  Current sigma:          {sigma:.4f}")
    print(f"  Stable readings buffer: {len(monitor._stable_ratios)}")

    # Check database
    print("\n--- Database check ---")
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT drift_type, COUNT(*), AVG(optimal_ratio) FROM iddr_events GROUP BY drift_type"
        ).fetchall()
        conn.close()
        if rows:
            for dtype, count, avg_opt in rows:
                avg_str = f"{avg_opt:.3f}" if avg_opt is not None else "n/a"
                print(f"  {dtype}: {count} events (avg optimal_ratio={avg_str})")
        else:
            print("  (no events logged)")
    except Exception as e:
        print(f"  DB query failed: {e}")

    print("\n" + "=" * 60)
    print("IDDR test complete.")


if __name__ == "__main__":
    test_iddr()
