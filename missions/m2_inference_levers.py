"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 3 tracking: Cache economics
    # System prompt in chat/rag is reused across turns (observed avg reuse ~6.5 reads per session)
    assumed_avg_cache_reads = 6.5
    small_pin, _ = MODEL_PRICES["small"]
    large_pin, _ = MODEL_PRICES["large"]

    # Verify cache worthiness per tier
    small_cache_viable = pricing.cache_is_worth_it(
        avg_cache_reads=assumed_avg_cache_reads,
        write_cost=small_pin * 1.25,
        input_price_per_m=small_pin,
        read_discount=0.10,
    )
    large_cache_viable = pricing.cache_is_worth_it(
        avg_cache_reads=assumed_avg_cache_reads,
        write_cost=large_pin * 1.25,
        input_price_per_m=large_pin,
        read_discount=0.10,
    )

    # Extension 4 tracking: Reasoning budget ($ and Wh)
    reasoning_reqs = 0
    reasoning_tokens = 0
    reasoning_cost = 0.0
    reasoning_wh = 0.0
    non_reasoning_cost = 0.0
    non_reasoning_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = bool(int(num(r["is_reasoning"])))
        tier = r["route_tier"]
        q_tokens = inp + out
        total_tokens += q_tokens

        # BASELINE: naive deployment — everything on large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # OPTIMIZED: cascade + prompt caching (only if cache_is_worth_it) + batch API
        pin, pout = MODEL_PRICES[tier]
        use_cache = cached if (small_cache_viable if tier == "small" else large_cache_viable) else 0
        req_cost = pricing.request_cost(inp, out, pin, pout, cached_in=use_cache, batch=is_batch)
        opt_cost += req_cost

        # Track energy & cost breakdown for reasoning
        q_wh = sustainability.wh_per_query(q_tokens, is_reasoning=is_reasoning)
        if is_reasoning:
            reasoning_reqs += 1
            reasoning_tokens += q_tokens
            reasoning_cost += req_cost
            reasoning_wh += q_wh
        else:
            non_reasoning_cost += req_cost
            non_reasoning_wh += q_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 3 details
    cache_be_small = pricing.cache_break_even_reads(small_pin * 1.25, small_pin, 0.10)
    cache_be_large = pricing.cache_break_even_reads(large_pin * 1.25, large_pin, 0.10)
    cache_economics = {
        "assumed_avg_reads": assumed_avg_cache_reads,
        "break_even_reads_small": round(cache_be_small, 2),
        "break_even_reads_large": round(cache_be_large, 2),
        "small_cache_worthy": small_cache_viable,
        "large_cache_worthy": large_cache_viable,
    }

    # Extension 4 details & simulated cap (e.g. 50% reasoning reduction via routing rule)
    total_reqs = len(rows)
    reasoning_traffic_pct = (reasoning_reqs / total_reqs * 100) if total_reqs else 0.0
    reasoning_cost_pct = (reasoning_cost / opt_cost * 100) if opt_cost else 0.0
    total_wh = reasoning_wh + non_reasoning_wh
    reasoning_wh_pct = (reasoning_wh / total_wh * 100) if total_wh else 0.0

    # If reasoning capped by 50% (routed to small model without CoT multiplier)
    capped_saved_cost_daily = reasoning_cost * 0.50 * 0.70  # ~70% cost reduction on suppressed queries
    capped_saved_wh_daily = reasoning_wh * 0.50 * (79.0 / 80.0)  # ~98.7% energy reduction on suppressed queries

    reasoning_budget = {
        "reasoning_requests": reasoning_reqs,
        "total_requests": total_reqs,
        "traffic_share_pct": round(reasoning_traffic_pct, 1),
        "token_share_pct": round(reasoning_tokens / total_tokens * 100, 1),
        "cost_daily": round(reasoning_cost, 2),
        "cost_share_pct": round(reasoning_cost_pct, 1),
        "energy_kwh_daily": round(reasoning_wh / 1000.0, 3),
        "energy_share_pct": round(reasoning_wh_pct, 1),
        "capped_savings_daily_usd": round(capped_saved_cost_daily, 2),
        "capped_savings_monthly_usd": round(capped_saved_cost_daily * 30, 2),
        "capped_savings_daily_kwh": round(capped_saved_wh_daily / 1000.0, 3),
        "recommended_rule": (
            "Routing Rule: Restrict reasoning (CoT) to eval tasks with verified complexity score >= 0.75 "
            "or low initial confidence. Standardize all other traffic on small cascaded model."
        )
    }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- [Extension 3] Cache Economics (`cache_is_worth_it`) --")
        print(f"  * Break-even reads needed: {cache_be_small:.2f} reads (small tier), {cache_be_large:.2f} reads (large tier)")
        print(f"  * Observed avg reads/prefix: {assumed_avg_cache_reads:.1f} reads")
        print(f"  * Cache profitable? Small: {small_cache_viable}, Large: {large_cache_viable} -> Caching applied!")

        print("\n-- [Extension 4] Reasoning Budget Analysis --")
        print(f"  * Traffic share: {reasoning_reqs}/{total_reqs} requests ({reasoning_traffic_pct:.1f}% of calls)")
        print(f"  * Cost share   : ${reasoning_cost:,.2f}/day ({reasoning_cost_pct:.1f}% of optimized spend)")
        print(f"  * Energy share : {reasoning_wh/1000.0:.2f} kWh/day ({reasoning_wh_pct:.1f}% of total inference power!)")
        print(f"  * Capped Policy Savings (50% cap): ${reasoning_budget['capped_savings_monthly_usd']:,.0f}/month & {reasoning_budget['capped_savings_daily_kwh']*30:,.1f} kWh/month")
        print(f"  * Policy: {reasoning_budget['recommended_rule']}")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_economics": cache_economics,
        "reasoning_budget": reasoning_budget,
    }


if __name__ == "__main__":
    run()
