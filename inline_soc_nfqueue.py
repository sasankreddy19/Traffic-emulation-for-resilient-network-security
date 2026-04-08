from pathlib import Path
from datetime import datetime
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import joblib
import numpy as np
import pandas as pd
import subprocess

from netfilterqueue import NetfilterQueue
from scapy.all import IP
from collections import defaultdict, deque
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.rule import Rule
from rich import box
from tensorflow.keras.models import load_model

console = Console()

QUEUE_NUM = 0
INTERFACE = "br-free5gc"

# Mini-flow behavior
MIN_PACKETS = 4
IDLE_TIMEOUT = 1.0

# Blocking policy
BLOCK_AFTER_N_ATTACK_FLOWS = 1
HIGH_SCORE_BLOCK_THRESHOLD = 0.90

# Temporary blocking
BLOCK_DURATION = 60  # seconds

# Attack memory
RECENT_ATTACK_MEMORY_DURATION = 120  # seconds
RECENT_ATTACK_SCORE_THRESHOLD = 0.20

MODEL_NAME = "AE + MLP"
MODE_NAME = "TRUE INLINE NFQUEUE"

# Load models
ae_model = load_model("models/autoencoder_model.h5", compile=False)
mlp = joblib.load("models/ae_mlp_classifier.pkl")
scaler = joblib.load("models/ae_scaler.pkl")
feature_columns = joblib.load("models/ae_feature_columns.pkl")

# Live state
flows = defaultdict(lambda: {
    "sizes": [],
    "times": [],
    "proto": 0,
    "src": None,
    "dst": None,
    "last_seen": 0.0
})

blocked_ips = set()
blocked_ip_times = {}
recent_blocked_memory = {}

source_attack_counts = defaultdict(int)
recent_events = deque(maxlen=16)

seen_packets = 0
accepted_packets = 0
dropped_packets = 0

finalized_flows = 0
allowed_flows = 0
suspect_flows = 0
blocked_flows = 0

last_src = "-"
last_dst = "-"
last_packets = 0
last_score = 0.0
last_decision = "-"
start_time = time.time()

live = None

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "inline_events.log"


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def log_event(event_type, src="-", dst="-", pkt_count="-", score="-", extra=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{timestamp} | {event_type} | SRC={src} | DST={dst} | "
        f"PKTS={pkt_count} | SCORE={score} {extra}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)


def add_block_rule(ip: str):
    check = subprocess.run(
        ["iptables", "-C", "AI_FIREWALL", "-s", ip, "-j", "DROP"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if check.returncode != 0:
        run_cmd(["iptables", "-I", "AI_FIREWALL", "1", "-s", ip, "-j", "DROP"])

    blocked_ips.add(ip)
    blocked_ip_times[ip] = time.time()


def unblock_expired_ips():
    now = time.time()
    expired = []

    for ip, t in list(blocked_ip_times.items()):
        if now - t > BLOCK_DURATION:
            expired.append(ip)

    for ip in expired:
        subprocess.run(
            ["iptables", "-D", "AI_FIREWALL", "-s", ip, "-j", "DROP"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        blocked_ips.discard(ip)
        blocked_ip_times.pop(ip, None)

        recent_events.appendleft(f"[cyan]UNBLOCK SRC={ip} (timeout)[/cyan]")
        log_event("UNBLOCK", ip)


def cleanup_recent_attack_memory():
    now = time.time()
    expired = []

    for ip, t in list(recent_blocked_memory.items()):
        if now - t > RECENT_ATTACK_MEMORY_DURATION:
            expired.append(ip)

    for ip in expired:
        recent_blocked_memory.pop(ip, None)


def extract_features(flow):
    sizes = np.array(flow["sizes"], dtype=float)
    times = np.array(flow["times"], dtype=float)

    if len(times) < 2:
        return None

    inter = np.diff(times)

    return {
        "proto_id": int(flow["proto"]),
        "packet_size_mean": float(np.mean(sizes)),
        "packet_size_std": float(np.std(sizes)),
        "inter_arrival_mean": float(np.mean(inter)),
        "inter_arrival_std": float(np.std(inter)),
    }


def classify_features(features):
    X_df = pd.DataFrame(
        [[features[c] for c in feature_columns]],
        columns=feature_columns
    )

    X_scaled = scaler.transform(X_df)
    recon = ae_model.predict(X_scaled, verbose=0)
    error = np.mean((X_scaled - recon) ** 2, axis=1).reshape(-1, 1)

    X_final = np.hstack([X_scaled, error])
    pred = int(mlp.predict(X_final)[0])
    score = float(mlp.predict_proba(X_final)[0][1])

    return pred, score


def classify_flow(flow):
    features = extract_features(flow)
    if features is None:
        return None, None
    return classify_features(features)


def mark_blocked(src, dst, pkt_count, score):
    global blocked_flows, last_src, last_dst, last_packets, last_score, last_decision

    blocked_flows += 1
    last_src = src
    last_dst = dst
    last_packets = pkt_count
    last_score = score
    last_decision = "BLOCKED"

    add_block_rule(src)
    recent_blocked_memory[src] = time.time()

    recent_events.appendleft(
        f"[bold red]BLOCK[/bold red] SRC={src} -> DST={dst} | "
        f"PKTS={pkt_count} | SCORE={score:.3f}"
    )
    log_event("BLOCK", src, dst, pkt_count, f"{score:.3f}")


def mark_suspect(src, dst, pkt_count, score):
    global suspect_flows, last_src, last_dst, last_packets, last_score, last_decision

    suspect_flows += 1
    last_src = src
    last_dst = dst
    last_packets = pkt_count
    last_score = score
    last_decision = "SUSPECT"

    recent_events.appendleft(
        f"[yellow]SUSPECT[/yellow] SRC={src} -> DST={dst} | "
        f"PKTS={pkt_count} | SCORE={score:.3f}"
    )
    log_event("SUSPECT", src, dst, pkt_count, f"{score:.3f}")


def mark_allowed(src, dst, pkt_count, score):
    global allowed_flows, last_src, last_dst, last_packets, last_score, last_decision

    allowed_flows += 1
    last_src = src
    last_dst = dst
    last_packets = pkt_count
    last_score = score
    last_decision = "ALLOWED"

    recent_events.appendleft(
        f"[green]ALLOW[/green] SRC={src} -> DST={dst} | "
        f"PKTS={pkt_count} | SCORE={score:.3f}"
    )
    log_event("ALLOW", src, dst, pkt_count, f"{score:.3f}")


def flush_idle_flows():
    global finalized_flows

    now = time.time()
    stale_keys = []

    for key, flow in list(flows.items()):
        if (now - flow["last_seen"]) > IDLE_TIMEOUT:
            stale_keys.append(key)

    for key in stale_keys:
        flow = flows.get(key)
        if not flow:
            continue

        pkt_count = len(flow["sizes"])
        if pkt_count < MIN_PACKETS:
            del flows[key]
            continue

        pred, score = classify_flow(flow)
        if pred is None:
            del flows[key]
            continue

        src = flow["src"]
        dst = flow["dst"]
        finalized_flows += 1

        recently_blocked = src in recent_blocked_memory
        is_attack_like = pred == 1 or score >= HIGH_SCORE_BLOCK_THRESHOLD
        is_repeat_offender = recently_blocked and score >= RECENT_ATTACK_SCORE_THRESHOLD

        if is_attack_like or is_repeat_offender:
            source_attack_counts[src] += 1

            if (
                source_attack_counts[src] >= BLOCK_AFTER_N_ATTACK_FLOWS
                or score >= HIGH_SCORE_BLOCK_THRESHOLD
                or is_repeat_offender
            ):
                mark_blocked(src, dst, pkt_count, score)
            else:
                mark_suspect(src, dst, pkt_count, score)
        else:
            mark_allowed(src, dst, pkt_count, score)

        del flows[key]


def build_overview_table():
    table = Table(box=box.SQUARE, expand=True, title="[bold cyan]Traffic Overview[/bold cyan]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right", style="bold green")

    table.add_row("Packets Seen", str(seen_packets))
    table.add_row("Packets Accepted", str(accepted_packets))
    table.add_row("Packets Dropped", f"[red]{dropped_packets}[/red]")
    table.add_row("Flows Finalized", str(finalized_flows))
    table.add_row("Active Flows", str(len(flows)))

    return table


def build_threat_table():
    ratio = blocked_flows / max(1, allowed_flows + blocked_flows + suspect_flows)

    if ratio > 0.40:
        threat = "[bold red blink]ELEVATED[/bold red blink]"
    elif ratio > 0.15:
        threat = "[bold yellow]MEDIUM[/bold yellow]"
    else:
        threat = "[bold green]NORMAL[/bold green]"

    uptime = int(time.time() - start_time)
    attack_rate = (blocked_flows / max(1, finalized_flows)) * 100

    table = Table(box=box.SQUARE, expand=True, title="[bold magenta]Threat Intelligence[/bold magenta]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right", style="bold cyan")

    table.add_row("Threat Level", threat)
    table.add_row("Model", MODEL_NAME)
    table.add_row("Mode", MODE_NAME)
    table.add_row("Queue", str(QUEUE_NUM))
    table.add_row("Interface", INTERFACE)
    table.add_row("Blocked IPs", str(len(blocked_ips)))
    table.add_row("Attack Rate", f"{attack_rate:.2f}%")
    table.add_row("Uptime (s)", str(uptime))

    return table


def build_decision_table():
    decision_style = (
        "[bold green]ALLOWED[/bold green]" if last_decision == "ALLOWED"
        else "[bold red]BLOCKED[/bold red]" if last_decision == "BLOCKED"
        else "[bold yellow]SUSPECT[/bold yellow]" if last_decision == "SUSPECT"
        else "-"
    )

    table = Table(box=box.SQUARE, expand=True, title="[bold red]Latest Decision[/bold red]")
    table.add_column("Field", style="bold white")
    table.add_column("Value", justify="right", style="bold white")

    table.add_row("Source IP", last_src)
    table.add_row("Destination IP", last_dst)
    table.add_row("Packets", str(last_packets))
    table.add_row("Score", f"{last_score:.3f}")
    table.add_row("Decision", decision_style)

    return table


def build_stats_table():
    table = Table(box=box.SQUARE, expand=True, title="[bold green]Decision Stats[/bold green]")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right", style="bold white")

    table.add_row("Allowed Flows", f"[green]{allowed_flows}[/green]")
    table.add_row("Suspect Flows", f"[yellow]{suspect_flows}[/yellow]")
    table.add_row("Blocked Flows", f"[red]{blocked_flows}[/red]")

    if blocked_ips:
        blocked_list = []
        for ip in blocked_ips:
            remaining = int(max(0, BLOCK_DURATION - (time.time() - blocked_ip_times.get(ip, time.time()))))
            blocked_list.append(f"{ip} ({remaining}s)")
        blocked_preview = ", ".join(blocked_list[:3])
        if len(blocked_ips) > 3:
            blocked_preview += " ..."
    else:
        blocked_preview = "-"

    table.add_row("Blocked Sources", blocked_preview)
    return table


def build_events_panel():
    if recent_events:
        event_lines = "\n".join(recent_events)
    else:
        event_lines = "[dim]No events yet[/dim]"

    return Panel(
        event_lines,
        title="[bold white]Recent Security Events[/bold white]",
        border_style="bright_blue",
        box=box.SQUARE,
        padding=(1, 2)
    )


def make_dashboard():
    header = Panel(
        Align.center(
            Text("5G AI TRUE INLINE SECURITY OPERATIONS CENTER", style="bold bright_cyan"),
            vertical="middle"
        ),
        border_style="bright_blue",
        box=box.SQUARE
    )

    top_row = Columns(
        [build_overview_table(), build_threat_table(), build_decision_table(), build_stats_table()],
        equal=True,
        expand=True
    )

    body = Group(
        header,
        Rule(style="bright_blue"),
        top_row,
        Rule(style="bright_blue"),
        build_events_panel()
    )

    return Panel(body, border_style="bright_magenta", box=box.DOUBLE)


def update_dashboard():
    if live is not None:
        live.update(make_dashboard())


def process_packet(packet):
    global seen_packets, accepted_packets, dropped_packets, finalized_flows

    seen_packets += 1
    unblock_expired_ips()
    cleanup_recent_attack_memory()

    try:
        payload = packet.get_payload()
        ip = IP(payload)
    except Exception:
        accepted_packets += 1
        packet.accept()
        update_dashboard()
        return

    src = ip.src
    dst = ip.dst
    proto = int(ip.proto)
    size = len(ip)
    now = time.time()

    if src in blocked_ips:
        dropped_packets += 1
        recent_events.appendleft(
            f"[bold red]DROP[/bold red] SRC={src} -> DST={dst} | reason=blocked_source"
        )
        log_event("DROP", src, dst, "-", "-", "reason=blocked_source")
        packet.drop()
        update_dashboard()
        return

    key = (src, dst, proto)
    flow = flows[key]
    flow["src"] = src
    flow["dst"] = dst
    flow["proto"] = proto
    flow["sizes"].append(size)
    flow["times"].append(now)
    flow["last_seen"] = now

    flush_idle_flows()

    if len(flow["sizes"]) >= MIN_PACKETS:
        pred, score = classify_flow(flow)

        if pred is None:
            accepted_packets += 1
            packet.accept()
            del flows[key]
            update_dashboard()
            return

        pkt_count = len(flow["sizes"])
        finalized_flows += 1

        recently_blocked = src in recent_blocked_memory
        is_attack_like = pred == 1 or score >= HIGH_SCORE_BLOCK_THRESHOLD
        is_repeat_offender = recently_blocked and score >= RECENT_ATTACK_SCORE_THRESHOLD

        if is_attack_like or is_repeat_offender:
            source_attack_counts[src] += 1

            if (
                source_attack_counts[src] >= BLOCK_AFTER_N_ATTACK_FLOWS
                or score >= HIGH_SCORE_BLOCK_THRESHOLD
                or is_repeat_offender
            ):
                mark_blocked(src, dst, pkt_count, score)
                dropped_packets += 1
                packet.drop()
            else:
                mark_suspect(src, dst, pkt_count, score)
                accepted_packets += 1
                packet.accept()
        else:
            mark_allowed(src, dst, pkt_count, score)
            accepted_packets += 1
            packet.accept()

        del flows[key]
        update_dashboard()
        return

    accepted_packets += 1
    packet.accept()
    update_dashboard()


if __name__ == "__main__":
    nfqueue = NetfilterQueue()

    with Live(make_dashboard(), console=console, refresh_per_second=6, screen=True) as live_obj:
        live = live_obj
        recent_events.appendleft("[cyan]NFQUEUE armed. Waiting for packets...[/cyan]")
        log_event("ENGINE_START", "-", "-", "-", "-", "NFQUEUE armed")
        update_dashboard()

        nfqueue.bind(QUEUE_NUM, process_packet)

        try:
            nfqueue.run()
        except KeyboardInterrupt:
            recent_events.appendleft("[yellow]Stopping inline engine...[/yellow]")
            update_dashboard()
        finally:
            try:
                nfqueue.unbind()
            except Exception:
                pass
