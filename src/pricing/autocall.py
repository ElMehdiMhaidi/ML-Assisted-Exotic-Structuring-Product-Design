"""Single-asset Athena / Phoenix Monte Carlo pricer under risk-neutral measure Q."""
import numpy as np
from .base import BasePricer
from ..market import nearest_rate
from ..schemas import PricingResult


def simulate_paths(spec, market, paths, seed):
    rng = np.random.default_rng(seed)
    ticker = spec.underlyings[0]
    freq = int(spec.params.get("frequency", 4))
    steps = max(1, int(round(spec.maturity * freq)))
    dt = spec.maturity / steps
    r = nearest_rate(market, spec.maturity)
    q = market.dividends[ticker]
    vol = market.vols[ticker]
    ref = spec.params.get("reference_spots", {ticker: market.spots[ticker]})
    s_ref = float(ref.get(ticker, market.spots[ticker]))
    rel = np.full(paths, market.spots[ticker] / s_ref, dtype=float)
    history = np.empty((steps, paths), dtype=float)
    for i in range(steps):
        z = rng.standard_normal(paths)
        rel *= np.exp((r-q-0.5*vol*vol)*dt + vol*np.sqrt(dt)*z)
        history[i] = rel
    return history, r, dt


class AutocallPricer(BasePricer):
    def _price_given_coupon(self, spec, market, coupon, paths=50000, seed=42, phoenix=False):
        hist, r, dt = simulate_paths(spec, market, paths, seed)
        ac = spec.params.get("autocall_barrier", 1.0)
        cb = spec.params.get("coupon_barrier", 0.7)
        pb = spec.params.get("protection_barrier", 0.65)
        memory = bool(spec.params.get("memory", False))
        npaths = hist.shape[1]
        pv = np.zeros(npaths)
        alive = np.ones(npaths, dtype=bool)
        missed = np.zeros(npaths)

        for i in range(hist.shape[0]):
            t = (i+1)*dt
            spot_rel = hist[i]
            if phoenix:
                coupon_due = (spot_rel >= cb) & alive
                if memory:
                    missed[alive & ~coupon_due] += coupon
                    cflow = np.where(coupon_due, coupon + missed, 0.0)
                    missed[coupon_due] = 0.0
                else:
                    cflow = np.where(coupon_due, coupon, 0.0)
            else:
                cflow = np.where(alive, coupon, 0.0)

            pv += np.exp(-r*t) * cflow * 100.0
            call = (spot_rel >= ac) & alive
            pv += np.exp(-r*t) * 100.0 * call
            alive[call] = False

        final_rel = hist[-1]
        redemption = np.where(final_rel >= pb, 100.0, 100.0*final_rel)
        pv += np.exp(-r*spec.maturity) * redemption * alive
        return float(pv.mean()), float(pv.std(ddof=1)/np.sqrt(npaths))

    def price(self, spec, market, paths=50000, seed=42):
        phoenix = spec.product_family == "Phoenix"
        coupon = spec.params.get("coupon")
        if coupon is None:
            base, _ = self._price_given_coupon(spec, market, 0.0, paths, seed, phoenix)
            unit, _ = self._price_given_coupon(spec, market, 0.01, paths, seed, phoenix)
            slope = (unit-base)/0.01
            coupon = max(0.0, (100.0-base)/slope) if slope > 1e-10 else 0.0
            fair, se = self._price_given_coupon(spec, market, coupon, paths, seed, phoenix)
            return PricingResult(float(fair), "coupon", float(coupon),
                                 "Single-asset Monte Carlo under Q",
                                 {"mc_se": float(se), "base_pv": float(base), "coupon_unit_slope": float(slope)})
        fair, se = self._price_given_coupon(spec, market, float(coupon), paths, seed, phoenix)
        return PricingResult(float(fair), None, None, "Single-asset Monte Carlo under Q", {"mc_se": float(se)})
