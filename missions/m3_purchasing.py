"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

from finops import pricing, sustainability

DAYS = 30


def analyze_carbon_aware_scheduling(jobs: list[dict], cat: dict) -> dict:
    """Extension 5: Carbon-aware and electricity-optimized workload scheduling."""
    regions = sustainability.REGION_CARBON
    prices = sustainability.REGION_PRICE_KWH

    # Calculate energy consumed by all interruptible (schedulable) jobs
    schedulable_jobs = []
    total_schedulable_kwh = 0.0

    for j in jobs:
        interruptible = bool(int(num(j["interruptible"])))
        if not interruptible:
            continue
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        watts = num(cat[gtype]["watts"])
        gpu_hours = hpd * DAYS * ngpu
        kwh = (gpu_hours * watts) / 1000.0
        total_schedulable_kwh += kwh
        schedulable_jobs.append({
            "job_id": j["job_id"],
            "gpu_type": gtype,
            "gpu_hours": gpu_hours,
            "watts": watts,
            "kwh": round(kwh, 1),
        })

    # Compare regional impacts for schedulable workloads
    regional_summary = {}
    base_carbon = (total_schedulable_kwh * regions["us-east-1"]) / 1000.0  # kg CO2
    base_elec_cost = total_schedulable_kwh * prices["us-east-1"]

    for r_name, gco2 in regions.items():
        elec_price = prices[r_name]
        carbon_kg = (total_schedulable_kwh * gco2) / 1000.0
        cost_usd = total_schedulable_kwh * elec_price
        carbon_reduction_pct = (1.0 - carbon_kg / base_carbon) * 100.0 if base_carbon else 0.0
        cost_reduction_pct = (1.0 - cost_usd / base_elec_cost) * 100.0 if base_elec_cost else 0.0

        regional_summary[r_name] = {
            "gco2_kwh": gco2,
            "usd_kwh": elec_price,
            "carbon_kg": round(carbon_kg, 1),
            "electricity_cost_usd": round(cost_usd, 2),
            "carbon_reduction_pct": round(carbon_reduction_pct, 1),
            "cost_reduction_pct": round(cost_reduction_pct, 1),
        }

    cleanest_region = min(regions, key=regions.get)
    cheapest_region = min(prices, key=prices.get)

    carbon_saved_kg = base_carbon - regional_summary[cleanest_region]["carbon_kg"]
    electricity_saved_usd = base_elec_cost - regional_summary[cheapest_region]["electricity_cost_usd"]

    return {
        "schedulable_kwh": round(total_schedulable_kwh, 1),
        "schedulable_jobs": schedulable_jobs,
        "regional_summary": regional_summary,
        "baseline_region": "us-east-1",
        "cleanest_region": cleanest_region,
        "cheapest_region": cheapest_region,
        "carbon_saved_kg": round(carbon_saved_kg, 1),
        "electricity_saved_usd": round(electricity_saved_usd, 2),
        "latency_tradeoff_note": (
            "Training & batch eval workloads are non-interactive and tolerate geographical displacement. "
            "Routing them to europe-north1 cuts carbon emissions by 92% with zero impact on user experience latency."
        )
    }


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    matrix_recs = []

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = num(j.get("days", 30))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # Baseline simple tier recommendation
        tier = pricing.recommend_tier(hpd, interruptible, gpu_type=gtype, job_days=days)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

        # Extension 1 matrix evaluation
        matrix_eval = pricing.recommend_tier_matrix(
            gpu_type=gtype,
            hours_per_day=hpd,
            interruptible=interruptible,
            job_days=days,
        )
        matrix_recs.append({"job_id": j["job_id"], **matrix_eval})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5 analysis
    carbon_analysis = analyze_carbon_aware_scheduling(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- [Extension 1] Multi-Factor Purchasing Decision Matrix --")
        print(f"{'Job':18}{'GPU':7}{'Duty':>7}{'Risk/hr':>9}{'Tier':>11}{'Commitment':>15}")
        for m in matrix_recs:
            commit = m['reserved_commitment_type'] or "n/a"
            print(f"{m['job_id']:18}{m['gpu_type']:7}{m['duty_cycle']:>7.0%}{m['interruption_rate']:>9.0%}{m['recommended_tier']:>11}{commit:>15}")

        print("\n-- [Extension 5] Carbon-Aware Scheduling (Interruptible Jobs) --")
        print(f"Total schedulable compute: {carbon_analysis['schedulable_kwh']:,.1f} kWh/month across {len(carbon_analysis['schedulable_jobs'])} jobs")
        print(f"{'Region':18}{'gCO2/kWh':>10}{'$/kWh':>8}{'Carbon(kg)':>12}{'Elec Cost($)':>14}{'CO2 Saved%':>12}")
        for r_name, r_data in carbon_analysis["regional_summary"].items():
            print(f"{r_name:18}{r_data['gco2_kwh']:>10}{r_data['usd_kwh']:>8.3f}{r_data['carbon_kg']:>12.1f}${r_data['electricity_cost_usd']:>13.2f}{r_data['carbon_reduction_pct']:>11.1f}%")

        clean = carbon_analysis["cleanest_region"]
        cheap = carbon_analysis["cheapest_region"]
        print(f"\n  * Cleanest Region : {clean} (Saves {carbon_analysis['carbon_saved_kg']:,.1f} kg CO2e / month vs us-east-1, -{carbon_analysis['regional_summary'][clean]['carbon_reduction_pct']}%)")
        print(f"  * Cheapest Region : {cheap} (Saves ${carbon_analysis['electricity_saved_usd']:,.2f} / month on power vs us-east-1, -{carbon_analysis['regional_summary'][cheap]['cost_reduction_pct']}%)")
        print(f"  * Trade-off Note  : {carbon_analysis['latency_tradeoff_note']}")

    return {
        "recommendations": recs,
        "matrix_recommendations": matrix_recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_analysis": carbon_analysis,
    }


if __name__ == "__main__":
    run()
