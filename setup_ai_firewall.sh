#!/bin/bash
set -e

QUEUE_NUM=0
IFACE="br-free5gc"

echo "[*] Configuring AI firewall chain..."

sudo iptables -P FORWARD ACCEPT

sudo iptables -N AI_FIREWALL 2>/dev/null || true

sudo iptables -C FORWARD -j AI_FIREWALL 2>/dev/null || sudo iptables -I FORWARD 1 -j AI_FIREWALL

sudo iptables -F AI_FIREWALL

sudo iptables -A AI_FIREWALL -i "$IFACE" -j NFQUEUE --queue-num "$QUEUE_NUM"
sudo iptables -A AI_FIREWALL -j ACCEPT

echo "[*] AI firewall ready."
sudo iptables -L AI_FIREWALL -v -n --line-numbers
