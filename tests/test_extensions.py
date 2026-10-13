"""Unit tests for Lab 25 Extensions (Your Turn 1-5)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing


def test_extension_1_purchasing_matrix():
    """Extension 1: Multi-factor purchasing decisions."""
    matrix = pricing.recommend_tier_matrix(
        gpu_type="H100",
        hours_per_day=20,
        interruptible=True,
        job_days=14,
    )
    assert matrix["recommended_tier"] == "spot"
    assert matrix["interruption_rate"] == 0.03

    # Steady 24/7 inference job with rapid hardware obsolescence (H100) -> 1yr reserved preferred
    matrix_infer = pricing.recommend_tier_matrix(
        gpu_type="H100",
        hours_per_day=24,
        interruptible=False,
        job_days=30,
    )
    assert matrix_infer["recommended_tier"] == "reserved"
    assert matrix_infer["reserved_commitment_type"] == "reserved_1yr"


def test_extension_2_mbu_rightsizing():
    """Extension 2: Right-sizing memory-bound GPUs by VRAM economics."""
    cat = m1_efficiency_audit.catalog_by_type()
    assert "MI300X" in cat and "A100" in cat
    # MI300X has lower $/GB-VRAM-hr than H100
    h100_vram_cost = float(cat["H100"]["on_demand_hr"]) / float(cat["H100"]["hbm_gb"])
    mi300_vram_cost = float(cat["MI300X"]["on_demand_hr"]) / float(cat["MI300X"]["hbm_gb"])
    assert mi300_vram_cost < h100_vram_cost

    r1 = m1_efficiency_audit.run(verbose=False)
    mbu_res = r1.get("mbu_rightsizing")
    assert mbu_res is not None
    assert len(mbu_res["recommendations"]) > 0
    assert mbu_res["total_monthly_savings"] > 500


def test_extension_3_cache_is_worth_it():
    """Extension 3: Prompt caching break-even math."""
    # Write cost = 1.25 * base_price, read discount = 0.10 (90% savings)
    # Unit savings = base_price * 0.90 -> Break-even = 1.25 / 0.90 ~= 1.3888
    be = pricing.cache_break_even_reads(write_cost_per_m=3.75, input_price_per_m=3.00, cache_discount=0.10)
    assert abs(be - (1.25 / 0.90)) < 1e-4

    # 1 read is NOT worth it; 2 or more reads IS worth it
    assert pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost=3.75, input_price_per_m=3.00, read_discount=0.10) is False
    assert pricing.cache_is_worth_it(avg_cache_reads=3.0, write_cost=3.75, input_price_per_m=3.00, read_discount=0.10) is True


def test_extension_4_reasoning_budget():
    """Extension 4: Reasoning query energy and cost attribution."""
    r2 = m2_inference_levers.run(verbose=False)
    rb = r2.get("reasoning_budget")
    assert rb is not None
    assert rb["traffic_share_pct"] < 15.0  # ~8.4% traffic
    assert rb["energy_share_pct"] > 80.0   # Reasoning consumes >80% energy due to 80x multiplier
    assert rb["capped_savings_monthly_usd"] > 0


def test_extension_5_carbon_aware_scheduling():
    """Extension 5: Geographic scheduling across 5 power grids."""
    r3 = m3_purchasing.run(verbose=False)
    ca = r3.get("carbon_analysis")
    assert ca is not None
    assert ca["cleanest_region"] == "europe-north1"
    assert ca["cheapest_region"] == "us-east-wa"
    assert ca["carbon_saved_kg"] > 1000  # Over 1 metric ton CO2 saved monthly
