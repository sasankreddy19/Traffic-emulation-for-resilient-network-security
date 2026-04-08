import pyshark
import pandas as pd
import numpy as np
from collections import defaultdict

IDLE_TIMEOUT = 2.0  # seconds

def canonical_flow_key(src_ip, src_port, dst_ip, dst_port, proto):
    a = (str(src_ip), int(src_port))
    b = (str(dst_ip), int(dst_port))
    if a <= b:
        return (a[0], a[1], b[0], b[1], int(proto))
    return (b[0], b[1], a[0], a[1], int(proto))

def extract_flows(pcap_file, label):
    cap = pyshark.FileCapture(
        pcap_file,
        keep_packets=False,
        use_json=True
    )

    last_seen = {}
    subflow_idx = defaultdict(int)

    flows = defaultdict(lambda: {
        "src_ip": None,
        "packet_sizes": [],
        "timestamps": [],
        "proto_id": None
    })

    for pkt in cap:
        try:
            if hasattr(pkt, "ip"):
                src = pkt.ip.src
                dst = pkt.ip.dst
                proto_id = int(pkt.ip.proto)
            elif hasattr(pkt, "ipv6"):
                src = pkt.ipv6.src
                dst = pkt.ipv6.dst
                proto_id = 0
            else:
                continue

            proto = pkt.transport_layer
            if proto is None:
                continue

            sport = int(getattr(pkt[proto], "srcport", 0))
            dport = int(getattr(pkt[proto], "dstport", 0))
            length = int(pkt.length)
            timestamp = float(pkt.sniff_timestamp)

            base_key = canonical_flow_key(src, sport, dst, dport, proto_id)

            if base_key in last_seen and (timestamp - last_seen[base_key]) > IDLE_TIMEOUT:
                subflow_idx[base_key] += 1

            current_idx = subflow_idx[base_key]
            flow_key = base_key + (current_idx,)

            flow = flows[flow_key]
            if flow["src_ip"] is None:
                flow["src_ip"] = src
                flow["proto_id"] = proto_id

            flow["packet_sizes"].append(length)
            flow["timestamps"].append(timestamp)

            last_seen[base_key] = timestamp

        except Exception:
            continue

    cap.close()

    records = []
    for _, data in flows.items():
        sizes = np.array(data["packet_sizes"], dtype=float)
        times = np.array(data["timestamps"], dtype=float)

        if len(times) < 2:
            continue

        duration = float(times[-1] - times[0])
        inter_arrivals = np.diff(times)

        records.append({
            "src_ip": data["src_ip"],
            "flow_start_time": float(times[0]),
            "proto_id": int(data["proto_id"]),
            "packet_count": int(len(sizes)),
            "byte_count": float(np.sum(sizes)),
            "duration": duration,
            "packets_per_sec": float(len(sizes) / duration) if duration > 0 else 0.0,
            "bytes_per_sec": float(np.sum(sizes) / duration) if duration > 0 else 0.0,
            "packet_size_mean": float(np.mean(sizes)),
            "packet_size_std": float(np.std(sizes)),
            "inter_arrival_mean": float(np.mean(inter_arrivals)),
            "inter_arrival_std": float(np.std(inter_arrivals)),
            "min_inter_arrival": float(np.min(inter_arrivals)),
            "max_inter_arrival": float(np.max(inter_arrivals)),
            "label": label
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    print("Processing attack traffic...")
    attack_df = extract_flows("pcaps/attack_2p9m.pcap", 1)
    print("Attack flows extracted:", len(attack_df))
    attack_df.to_csv("datasets/attack_flows.csv", index=False)

    print("Processing normal traffic...")
    normal_df = extract_flows("pcaps/5g_normal_inline.pcap", 0)
    print("Normal flows extracted:", len(normal_df))
    normal_df.to_csv("datasets/normal_flows.csv", index=False)

    print("Flow extraction completed.")
