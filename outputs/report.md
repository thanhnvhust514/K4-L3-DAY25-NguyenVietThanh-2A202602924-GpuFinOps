# NimbusAI — GPU Cost Optimization Report

> **Executive Summary:** Comprehensive FinOps assessment of GPU infrastructure spend, inference economics, capacity procurement, and carbon sustainability.

- **Period:** monthly
- **Baseline spend:** $27,133
- **Optimized spend:** $14,626
- **Projected savings:** $12,507 (**46%** cut)

---

## 1. Savings by Lever

| FinOps Lever | Monthly Savings (USD) | Share of Savings | Primary Mechanism |
|---|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9.7% | Cascade to small model + 90% prompt cache discount + 50% batch discount |
| Purchasing (spot/reserved) | $10,040 | 80.3% | Checkpointable spot instances + 1yr/3yr reserved commitments (>55% duty cycle) |
| Right-size util-lies | $655 | 5.2% | Downgrade memory/kernel-stalled GPUs (H100 -> A100 / A10G -> L4) |
| Kill idle GPUs | $600 | 4.8% | Automate overnight teardown for instances with <10% utilization |

---

## 2. Root Cause Analysis: The 'GPU-Util Lie'

- **What is the GPU-Util Lie?** `nvidia-smi` reports `gpu_util_pct` based on the fraction of time the GPU kernel or clock is active, **not** the mathematical throughput (FLOPs) delivered.
- **The Culprit (`gpu-h100-4`):** Telemetry shows **98.2% GPU-Util** but **MFU is only ~0.194** (19.4%). The cluster pays full H100 hourly rates ($2.50/hr) while receiving less than 1/5th of rated computational throughput.
- **Underlying Mechanism:** Severe memory stalls, host-to-device I/O bottlenecks, or un-batched kernel launch overhead. The GPU cores are waiting on memory controllers rather than computing tensor operations.
- **Financial Impact:** Paying $1,800/month for an H100 that provides only A100-level compute output leaks over $500/month per instance.

---

## 3. Prioritized Action Plan for NimbusAI

1. **P0 (Immediate - Week 1): Kill Idle GPUs & Enable Automated Off-Hours Shutdown**
   - *Action:* Terminate `gpu-h100-5` during overnight idle windows (00:00 - 08:00).
   - *Impact:* Immediate saving of **$600/month** with zero application risk.
2. **P0 (Immediate - Week 1): Inference Cascade & Prompt Caching**
   - *Action:* Route easy queries (80%) to small model tier; turn on prompt caching for Chat/RAG system prompts.
   - *Impact:* Cuts unit cost from **$6.49 to $1.13 per 1M tokens** (saving **$1,212/month**, 82.6% reduction).
3. **P1 (Near Term - Month 1): Purchasing Optimization (Spot + Reserved)**
   - *Action:* Migrate interruptible training workloads to Spot with periodic checkpointing; commit 24/7 inference to 1-Year Reserved.
   - *Impact:* Largest dollar impact, saving **$10,040/month** (39.1% hardware procurement reduction).
4. **P2 (Medium Term - Quarter 1): Right-size Memory-Bound & Util-Lie GPUs**
   - *Action:* Downgrade `gpu-h100-4` and under-utilized A100 inference nodes to A10G/L4 based on achieved memory bandwidth.
   - *Impact:* Additional **$655 - $1,282/month** sustained hardware savings.

---

## 4. Sustainability & Carbon Economics

- **Energy per query:** 0.24 Wh (Median query base: ~0.24 Wh)
- **Carbon per query:** 0.091 gCO2e (in baseline `us-east-1`)
- **Cheapest + Cleanest Region:** `europe-north1`

### Regional Grid Comparison
| Cloud Region | Power Grid Profile | Carbon Intensity (gCO2/kWh) | Electricity Rate ($/kWh) | Environmental Suitability |
|---|---|---|---|---|
| `europe-north1` | Hydro / Renewable (Norway) | 30 | $0.090 | **Cleanest** (92.1% CO2 reduction) |
| `us-east-wa` | Hydro (Washington State) | 90 | $0.055 | **Lowest Cost** (54.2% power cost reduction) |
| `us-west-2` | Hydro / Wind mix (Oregon) | 120 | $0.070 | Balanced west-coast hub |
| `us-east-1` | Natural gas / Coal (Virginia) | 380 | $0.120 | Baseline default cloud region |
| `europe-central2` | Coal-dominated (Poland) | 660 | $0.180 | **Avoid** (Dirtiest grid & highest electricity price) |

> **Latency vs. Carbon Trade-Off:** Training and batch eval jobs are asynchronous, offline workloads. Routing all 4,227 kWh/month of interruptible training compute to `europe-north1` eliminates **1,479.5 kg CO2e/month** with zero latency degradation for real-time inference users.

---

## 5. Deep-Dive Extensions ('Your Turn')

### Extension 1 — Purchasing Tier Decision Matrix
- Workloads evaluated along multi-factor matrix: GPU architecture risk, duty cycle, and technology obsolescence.
- Fast-evolving hardware (H100, B200) recommended for **1-Year Reserved** or **Spot** rather than 3-Year locks to hedge hardware refresh cycles.

### Extension 2 — Right-Sizing by MBU & VRAM Economics
- Memory-bound decode workloads require memory bandwidth (`peak_bw_tbs`) rather than raw TFLOPs.
- Analysis of `$/GB-VRAM`: A100 ($0.0224/GB-hr) under-utilized in inference replaced by A10G ($1.00/hr) saves 44% with zero throughput penalty.

### Extension 3 — Prompt Cache Economics (`cache_is_worth_it`)
- Break-even reads per prompt prefix: **1.39 reads**.
- Observed average reuse in chat/RAG system prompts: **6.5 reads**.
- Prompt caching verified mathematically profitable before billing discount application.

### Extension 4 — Reasoning Budget Governance
- **Traffic vs. Energy Disparity:** Reasoning queries represent only **8.4% of total traffic** but generate **16.5% of inference costs** and consume **94.0% of all inference electrical energy**.
- **Recommended Governance Policy:** Enforce prompt classifier gating. Only invoke reasoning models when task complexity score >= 0.75 or confidence < 0.60.
- **Simulated 50% Cap Savings:** **$15/month** and **441.2 kWh/month**.

### Extension 5 — Carbon-Aware Scheduling
- Schedulable training compute: **4,227 kWh/month**.
- Net carbon abatement: **1,479.5 kg CO2e/month** via zero-impact geographic workload placement.

---

_Figures are June-2026 as-of snapshots; re-baseline before committing production reservations._