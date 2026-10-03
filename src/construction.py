"""Client mandate + recommended product family -> concrete ProductSpec geometries.

This module answers one question only:
    "Which contractual geometry of this recommended product should we test?"

The mandate FIXES the underlying, currency, notional and maturity.
Construction varies only product terms that a structurer would still choose
(strike, barriers, range width, etc.). Each geometry is then priced separately.
"""
import numpy as np
from .schemas import ProductSpec


def _linspace(center, half_width, n=3, lo=None, hi=None):
    vals = np.linspace(center-half_width, center+half_width, n)
    if lo is not None: vals = np.maximum(vals, lo)
    if hi is not None: vals = np.minimum(vals, hi)
    return sorted(set(round(float(x), 4) for x in vals))


def generate_specs(mandate, recommendation, max_geometries=20):
    label = recommendation.product_family
    fam = "Phoenix" if label == "Memory Phoenix" else label
    if len(mandate.underlyings) != 1:
        raise ValueError("construction.py supports one underlying only.")

    underlying = [mandate.underlyings[0]]
    T = float(mandate.maturity_years or 2.0)
    notional = float(mandate.notional or 1_000_000)
    ccy = "USD"
    downside = mandate.downside_tolerance if mandate.downside_tolerance is not None else 0.35
    protection = max(0.50, min(0.85, 1.0-downside))
    specs = []

    if fam == "Phoenix":
        for ac in _linspace(1.0, 0.05, 3, 0.90, 1.10):
            for cb in _linspace(max(protection, 0.65), 0.05, 3, 0.55, 0.90):
                for pb in _linspace(protection, 0.05, 3, 0.50, 0.85):
                    specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                        "autocall_barrier": ac, "coupon_barrier": cb, "protection_barrier": pb,
                        "coupon": None, "memory": label == "Memory Phoenix", "frequency": 4,
                        "display_label": label,
                    }))

    elif fam == "Athena":
        for ac in _linspace(1.0, 0.05, 3, 0.90, 1.10):
            for pb in _linspace(protection, 0.05, 3, 0.50, 0.85):
                specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                    "autocall_barrier": ac, "protection_barrier": pb,
                    "coupon": None, "frequency": 4, "display_label": label,
                }))

    elif fam in {"Reverse Convertible", "ELN"}:
        for strike in _linspace(0.90, 0.10, 5, 0.70, 1.00):
            specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                "strike_pct": strike, "coupon": None, "display_label": label,
            }))

    elif fam == "Vanilla":
        for strike in _linspace(1.0, 0.10, 5, 0.80, 1.20):
            specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                "strike_pct": strike, "option_type": "call", "display_label": label,
            }))

    elif fam == "Digital":
        for strike in _linspace(1.0, 0.10, 5, 0.80, 1.20):
            specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                "strike_pct": strike, "payout": 0.10, "display_label": label,
            }))

    elif fam == "Barrier":
        for b in _linspace(protection, 0.10, 5, 0.50, 0.90):
            specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                "barrier_pct": b, "strike_pct": 1.0, "display_label": label,
            }))

    elif fam == "Range Accrual":
        for width in [0.15, 0.20, 0.25, 0.30]:
            specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
                "lower": 1-width, "upper": 1+width,
                "accrual_rate": 0.10, "frequency": 12, "display_label": label,
            }))

    elif fam == "Strip of Digitals":
        specs.append(ProductSpec(fam, underlying, T, notional, ccy, {
            "strikes_pct": [0.9, 1.0, 1.1], "payout_each": 0.04, "display_label": label,
        }))

    return specs[:max_geometries]
