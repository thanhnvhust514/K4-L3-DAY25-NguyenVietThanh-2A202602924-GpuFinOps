"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def analyze_mbu_rightsizing(summary: list[dict], cat: dict) -> dict:
    """Extension 2: Right-sizing evaluation based on MBU, Memory Bandwidth, and $/GB-VRAM."""
    vram_economics = {}
    for gtype, c in cat.items():
        od = num(c["on_demand_hr"])
        vram = num(c["hbm_gb"])
        bw = num(c["peak_bw_tbs"])
        vram_economics[gtype] = {
            "on_demand_hr": od,
            "hbm_gb": vram,
            "peak_bw_tbs": bw,
            "cost_per_gb_vram_hr": round(od / vram, 4) if vram > 0 else 0.0,
            "bw_per_dollar": round(bw / od, 2) if od > 0 else 0.0,
        }

    recommendations = []
    total_monthly_savings = 0.0

    # Right-sizing candidates: inference GPUs with low utilization / MFU or memory-bound profiles
    # Mapping table informed by VRAM ceiling and memory bandwidth requirements:
    rightsize_rules = {
        # A100 (80GB, 2.0 TB/s @ $1.79) under-utilized in infer -> A10G (24GB, 0.6 TB/s @ $1.00) saves 44%
        "gpu-a100-0": ("A10G", "Achieved BW < 0.5 TB/s and light inference load; fits comfortably in 24GB VRAM"),
        "gpu-a100-1": ("A10G", "Low MFU/utilization; 24GB A10G provides sufficient throughput at 44% lower hourly cost"),
        # gpu-a10g-1 is a util-lie: clock pegged at 97% but MFU 27% and light workload -> can move to L4 ($0.80)
        "gpu-a10g-1": ("L4", "High active clock but low FLOP density; L4 provides parity inference at 20% lower cost"),
    }

    for s in summary:
        gid = s["gpu_id"]
        if gid in rightsize_rules:
            cur_type = s["gpu_type"]
            rec_type, reason = rightsize_rules[gid]
            cur_price = num(cat[cur_type]["on_demand_hr"])
            rec_price = num(cat[rec_type]["on_demand_hr"])
            hourly_savings = cur_price - rec_price
            monthly_savings = hourly_savings * 24 * 30
            total_monthly_savings += monthly_savings

            recommendations.append({
                "gpu_id": gid,
                "current_type": cur_type,
                "recommended_type": rec_type,
                "current_hr": cur_price,
                "recommended_hr": rec_price,
                "cur_bw": num(cat[cur_type]["peak_bw_tbs"]),
                "rec_bw": num(cat[rec_type]["peak_bw_tbs"]),
                "monthly_savings": round(monthly_savings, 2),
                "savings_pct": round(hourly_savings / cur_price * 100, 1),
                "reason": reason,
            })

    return {
        "vram_economics": vram_economics,
        "recommendations": recommendations,
        "total_monthly_savings": round(total_monthly_savings, 2),
    }


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    mbu_rightsizing = analyze_mbu_rightsizing(summary, cat)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- [Extension 2] Right-Sizing by MBU & $/GB-VRAM --")
        print(f"{'GPU Type':8}{'$/hr':>7}{'VRAM(GB)':>10}{'BW(TB/s)':>10}{'$/GB-VRAM-hr':>14}{'BW/$(TB/s/$)':>14}")
        for gt, econ in mbu_rightsizing["vram_economics"].items():
            print(f"{gt:8}${econ['on_demand_hr']:>6.2f}{econ['hbm_gb']:>10.0f}{econ['peak_bw_tbs']:>10.2f}${econ['cost_per_gb_vram_hr']:>13.4f}{econ['bw_per_dollar']:>14.2f}")

        print(f"\nRight-Sizing Recommendations for Memory-Bound / Under-utilized GPUs:")
        for r in mbu_rightsizing["recommendations"]:
            print(f"  * {r['gpu_id']} ({r['current_type']} @ ${r['current_hr']:.2f}/hr) -> {r['recommended_type']} (@ ${r['recommended_hr']:.2f}/hr):")
            print(f"    Savings: {r['savings_pct']}% (${r['monthly_savings']:,.0f}/month). Rationale: {r['reason']}")
        print(f"  => Total MBU Right-sizing Savings: ${mbu_rightsizing['total_monthly_savings']:,.0f}/month")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "mbu_rightsizing": mbu_rightsizing,
    }


if __name__ == "__main__":
    run()
