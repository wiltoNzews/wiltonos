#!/bin/bash
# Install WiltonOS systemd services
# Replaces the dead openwebui/wiltonai services
# Run with: sudo bash ~/wiltonos/services/install.sh

set -e

SERVICES_DIR="$(dirname "$0")"

echo "=== Disabling dead services ==="
systemctl stop openwebui.service 2>/dev/null || true
systemctl disable openwebui.service 2>/dev/null || true
systemctl stop wiltonai.service 2>/dev/null || true
systemctl disable wiltonai.service 2>/dev/null || true

echo "=== Installing WiltonOS services ==="
cp "$SERVICES_DIR/wiltonos-daemon.service" /etc/systemd/system/
cp "$SERVICES_DIR/wiltonos-gateway.service" /etc/systemd/system/

echo "=== Reloading systemd ==="
systemctl daemon-reload

echo "=== Enabling services ==="
systemctl enable wiltonos-daemon.service
systemctl enable wiltonos-gateway.service

echo "=== Starting services ==="
# Kill any existing manual processes first
pkill -f "breathing_daemon.py" 2>/dev/null || true
pkill -f "gateway.py --port 8000" 2>/dev/null || true
sleep 2

systemctl start wiltonos-daemon.service
sleep 3
systemctl start wiltonos-gateway.service

echo ""
echo "=== Status ==="
systemctl status wiltonos-daemon.service --no-pager -l | head -15
echo ""
systemctl status wiltonos-gateway.service --no-pager -l | head -15

echo ""
echo "Done. WiltonOS services will auto-start on boot."
echo "  Daemon: systemctl status wiltonos-daemon"
echo "  Gateway: systemctl status wiltonos-gateway (http://localhost:8000)"
