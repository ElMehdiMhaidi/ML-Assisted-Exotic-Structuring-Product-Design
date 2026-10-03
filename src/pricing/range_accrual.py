import numpy as np
from .base import BasePricer
from ..market import nearest_rate
from ..schemas import PricingResult

class RangeAccrualPricer(BasePricer):
    def price(self,spec,market,paths=50000,seed=42):
        """
        Equity range accrual:
        coupon accrues on each observation date if the underlying is between
        fixed lower/upper contractual levels defined versus inception spot.
        """
        t=spec.underlyings[0]
        S=market.spots[t]
        vol=market.vols[t]
        q=market.dividends[t]
        r=nearest_rate(market,spec.maturity)

        ref_spots=spec.params.get("reference_spots",{t:S})
        S_ref=float(ref_spots.get(t,S))

        lower_level=S_ref*spec.params.get("lower",0.8)
        upper_level=S_ref*spec.params.get("upper",1.2)

        freq=int(spec.params.get("frequency",12))
        steps=max(1,int(round(spec.maturity*freq)))
        dt=spec.maturity/steps

        rng=np.random.default_rng(seed)
        spot_paths=np.full(paths,S,dtype=float)
        inside_count=np.zeros(paths)

        for _ in range(steps):
            z=rng.standard_normal(paths)
            spot_paths *= np.exp((r-q-0.5*vol*vol)*dt + vol*np.sqrt(dt)*z)
            inside_count += ((spot_paths>=lower_level)&(spot_paths<=upper_level))

        accrual_fraction=inside_count/steps
        accrual_rate=spec.params.get("accrual_rate",0.10)

        payoff=100.0 + 100.0*accrual_rate*spec.maturity*accrual_fraction
        discounted=payoff*np.exp(-r*spec.maturity)

        return PricingResult(
            fair_value=float(discounted.mean()),
            pricing_method="Monte Carlo range accrual",
            diagnostics={
                "mc_se":float(discounted.std(ddof=1)/np.sqrt(paths)),
                "lower_level":float(lower_level),
                "upper_level":float(upper_level),
            }
        )
