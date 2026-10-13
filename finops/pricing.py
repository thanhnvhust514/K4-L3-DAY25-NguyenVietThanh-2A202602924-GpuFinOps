"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Interruption risk baseline by GPU architecture (commodity nodes preempted more)
GPU_INTERRUPT_RATES = {
    "H100": 0.03,
    "H200": 0.04,
    "A100": 0.05,
    "A10G": 0.08,
    "L4":   0.08,
    "B200": 0.03,
    "MI300X": 0.05,
}


def cache_break_even_reads(
    write_cost_per_m: float,
    input_price_per_m: float = 3.00,
    cache_discount: float = 0.10,
) -> float:
    """Minimum average reads per cached prefix for prompt caching to break even.

    Each cached read saves `input_price_per_m * (1.0 - cache_discount)`.
    Break-even reads = write_cost / savings_per_read.
    """
    unit_savings = input_price_per_m * (1.0 - cache_discount)
    if unit_savings <= 0:
        return float("inf")
    return write_cost_per_m / unit_savings


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost: float,
    input_price_per_m: float = 3.00,
    read_discount: float = 0.10,
) -> bool:
    """Determine whether prompt caching yields net dollar savings.

    Break-even reads = write_cost / (input_price_per_m * (1.0 - read_discount)).
    Returns True if avg_cache_reads > break-even.
    """
    be = cache_break_even_reads(write_cost, input_price_per_m, read_discount)
    return avg_cache_reads > be


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    interruption_rate: float | None = None,
    shared_pool: bool = True,
) -> str:
    """Pick a purchasing tier from workload duty cycle, interruptibility, and risk.

    Enhanced policy (Extension 1):
      - interruptible & not 24/7:
          Checks interruption risk. If rework overhead is acceptable (<40% of savings),
          recommends 'spot'.
      - Dedicated short-run workload without pool sharing:
          If not shared and annual duration is brief (<120 days), avoids long-term reserved commit.
      - duty cycle >= break-even:
          recommends 'reserved' (1yr or 3yr based on commitment horizon).
      - otherwise:
          recommends 'on_demand'.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)

    if interruptible and hours_per_day < 24:
        irate = interruption_rate or (GPU_INTERRUPT_RATES.get(gpu_type, 0.05) if gpu_type else 0.05)
        # Extreme interruption rates (>25%) destroy spot checkpoint savings
        if irate < 0.25:
            return "spot"

    # If unshared dedicated job with very short lifespan, avoid multi-year lock-in
    if not shared_pool and job_days is not None and job_days < 60 and duty < 0.9:
        return "on_demand"

    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_matrix(
    gpu_type: str,
    hours_per_day: float,
    interruptible: bool,
    job_days: float = 30.0,
    annual_horizon_days: float = 365.0,
) -> dict:
    """Comprehensive multi-factor purchasing decision matrix (Extension 1).

    Evaluates:
      1. Spot viability with GPU-specific preemption risk
      2. 1-Year Reserved vs. 3-Year Reserved (obsolescence vs savings)
      3. Duty cycle vs. break-even threshold
    """
    duty = max(0.0, hours_per_day) / 24.0
    irate = GPU_INTERRUPT_RATES.get(gpu_type, 0.05)
    be_3yr = break_even_utilization(0.45)  # 55%
    be_1yr = break_even_utilization(0.20)  # 80%

    tier = recommend_tier(hours_per_day, interruptible, gpu_type=gpu_type, job_days=job_days)

    reserved_recommendation = "reserved_3yr"
    if tier == "reserved":
        # AI hardware lifecycle: H100/H200 evolve fast; 1yr preferred for agile roadmaps
        if job_days < 180 or gpu_type in ("H100", "B200"):
            reserved_recommendation = "reserved_1yr"
        else:
            reserved_recommendation = "reserved_3yr"

    return {
        "gpu_type": gpu_type,
        "duty_cycle": round(duty, 3),
        "interruptible": interruptible,
        "interruption_rate": irate,
        "break_even_3yr": be_3yr,
        "break_even_1yr": be_1yr,
        "recommended_tier": tier,
        "reserved_commitment_type": reserved_recommendation if tier == "reserved" else None,
        "rationale": (
            f"Duty cycle {duty:.0%} vs 3yr break-even {be_3yr:.0%}. "
            f"Preemption risk: {irate:.0%}/hr. "
            f"{'Safe for spot with checkpointing' if tier == 'spot' else 'Committed utilization justifies reserved' if tier == 'reserved' else 'Low utilization calls for on-demand'}."
        )
    }


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
