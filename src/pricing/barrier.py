"""Single-asset down-and-out call using conditional Monte Carlo under Q.

A terminal GBM draw is combined with the Brownian-bridge survival probability
for a continuously monitored lower barrier. The implementation uses conditional survival
without introducing a large time-step simulation.
"""
import numpy as np
from .base import BasePricer
from ..market import nearest_rate
from ..schemas import PricingResult

class BarrierPricer(BasePricer):
    def price(self, spec, market, paths=50000, seed=42):
        t = spec.underlyings[0]
        S = market.spots[t]; T = spec.maturity
        r = nearest_rate(market, T); q = market.dividends[t]; vol = market.vols[t]
        ref = spec.params.get("reference_spots", {t:S}); S0 = float(ref.get(t,S))
        K = S0*spec.params.get("strike_pct",1.0)
        H = S0*spec.params.get("barrier_pct",0.7)
        if S <= H:
            return PricingResult(0.0, pricing_method="Conditional Monte Carlo down-and-out call under Q")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(int(paths))
        ST = S*np.exp((r-q-0.5*vol*vol)*T + vol*np.sqrt(T)*z)
        alive_endpoint = ST > H
        log1 = np.log(S/H)
        log2 = np.where(alive_endpoint, np.log(np.maximum(ST,H*(1+1e-12))/H), 0.0)
        hit_prob = np.exp(-2.0*log1*log2/(vol*vol*T))
        survival = np.where(alive_endpoint, np.clip(1.0-hit_prob,0.0,1.0), 0.0)
        payoff = np.maximum(ST-K,0.0)*survival
        disc = np.exp(-r*T)*payoff
        price_points = float(disc.mean()/S0*100.0)
        se = float(disc.std(ddof=1)/np.sqrt(paths)/S0*100.0)
        return PricingResult(price_points, pricing_method="Conditional Monte Carlo down-and-out call under Q", diagnostics={"mc_se":se})
