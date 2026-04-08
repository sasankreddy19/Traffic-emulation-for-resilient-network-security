# Traffic Emulation and Anomaly Detection for Resilient Network Security

## Project Overview

Modern communication networks such as **5G and IoT systems** generate extremely large volumes of network traffic. Traditional rule-based security systems struggle to detect unknown or evolving cyber threats.

This project presents a **traffic emulation and anomaly detection framework** designed to improve security in modern network infrastructures. The system simulates realistic **5G network traffic**, captures and analyzes packet flows, and applies **machine learning techniques** to identify abnormal behavior or cyber attacks.

The framework integrates **network simulation, packet monitoring, feature extraction, and machine learning-based anomaly detection**, enabling intelligent and adaptive security monitoring.

---

## Objectives

- Simulate realistic **5G network traffic**
- Capture and analyze network packets
- Extract meaningful features from traffic flows
- Train a **machine learning model** to learn normal network behavior
- Detect abnormal or malicious traffic patterns
- Automatically block suspicious network activity using inline prevention

---

## System Architecture

The system consists of four major components:

1. **5G Network Simulation** (Open5GS + UERANSIM)
2. **Traffic Generation** (Normal & Attack traffic)
3. **Packet Capture and Feature Extraction** (tcpdump + Scapy)
4. **Machine Learning Based Anomaly Detection** (Autoencoder + MLP Classifier)

### High-Level Workflow

1. Set up and start the simulated 5G core network
2. Launch gNB and UE simulators
3. Generate normal and malicious network traffic
4. Capture packets using tcpdump
5. Extract features from captured traffic
6. Train the anomaly detection model
7. Run real-time anomaly detection with inline prevention (NFQUEUE + iptables)

---

## System Requirements

### Hardware Requirements

| Component     | Minimum Requirement                  | Recommended          |
|---------------|--------------------------------------|----------------------|
| Processor     | Intel i5 / i7 or equivalent          | Intel i7 or better   |
| RAM           | 8 GB                                 | 16 GB or more        |
| Storage       | 100 GB free disk space               | 200 GB+              |
| Network       | Stable internet connection           | -                    |

### Software Requirements

| Software              | Purpose                                      |
|-----------------------|----------------------------------------------|
| Ubuntu Linux (20.04 or later) | Operating System                        |
| Open5GS               | 5G Core Network implementation               |
| UERANSIM              | UE and gNB simulator                         |
| Python 3.x            | Scripting and Machine Learning               |
| Zeek (optional)       | Advanced network traffic monitoring          |
| tcpdump               | Packet capture                               |
| Scapy                 | Packet manipulation and analysis             |
| scikit-learn          | Machine Learning library                     |
| pandas, numpy         | Data processing                              |
| matplotlib, seaborn   | Visualization                                |

---

## Installation

### 1. Install Python and Required Libraries

```bash
sudo apt update
sudo apt install python3 python3-pip tcpdump
pip3 install pandas numpy scikit-learn matplotlib seaborn scapy
