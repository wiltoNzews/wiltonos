#!/bin/bash
#
# OVERNIGHT PROCESS
# =================
# Runs while you're in Guaruja:
# 1. Deep extraction of all consciousness research
# 2. AI expansion of each topic
# 3. Master summary creation
# 4. Daemon keeps breathing
#
# December 2025 — Built for Wilton
#

echo "========================================================================"
echo "WILTONOS OVERNIGHT PROCESS"
echo "========================================================================"
echo "Started: $(date)"
echo ""

cd /home/zews/wiltonos/daemon

# Ensure daemon is running
echo "[1/4] Starting daemon..."
./daemon_ctl start 2>/dev/null || echo "Daemon already running"
echo ""

# Run deep extraction
echo "[2/4] Running deep extraction..."
python3 deep_extraction.py 2>&1 | tee /home/zews/wiltonos/compendium/extraction.log
echo ""

# Run overnight expansion (this is the long one)
echo "[3/4] Running AI expansion (this takes time)..."
python3 overnight_expansion.py 2>&1 | tee /home/zews/wiltonos/expanded/expansion_full.log
echo ""

# Show what was created
echo "[4/4] Results:"
echo ""
echo "=== COMPENDIUM (extracted) ==="
ls -lh /home/zews/wiltonos/compendium/*.md 2>/dev/null | head -20
echo ""
echo "=== EXPANDED (AI summaries) ==="
ls -lh /home/zews/wiltonos/expanded/*.md 2>/dev/null
echo ""

echo "========================================================================"
echo "OVERNIGHT PROCESS COMPLETE"
echo "Finished: $(date)"
echo "========================================================================"
