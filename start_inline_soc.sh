#!/bin/bash

cd ~/5g_ml_project || exit 1

echo "[*] Opening inline SOC dashboard..."

gnome-terminal -- bash -c "cd ~/5g_ml_project && sudo -E env HOME=$HOME PYTHONPATH=$HOME/.local/lib/python3.10/site-packages python3 scripts/inline_soc_nfqueue.py; exec bash"

sleep 3

echo "[*] Applying AI firewall rules..."
./scripts/setup_ai_firewall.sh

echo "[*] True inline SOC is now armed."
