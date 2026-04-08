#!/bin/bash

echo "[*] Clearing AI firewall rules..."

sudo iptables -D FORWARD -j AI_FIREWALL 2>/dev/null || true
sudo iptables -F AI_FIREWALL 2>/dev/null || true
sudo iptables -X AI_FIREWALL 2>/dev/null || true

sudo iptables -P FORWARD ACCEPT

echo "[*] Firewall cleared."
sudo iptables -L FORWARD -v -n
