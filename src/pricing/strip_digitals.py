from .digital import digital_cash_price
from ..market import nearest_rate
from ..schemas import PricingResult
from .base import BasePricer

class StripDigitalsPricer(BasePricer):
    def price(self,spec,market,paths=50000,seed=42):
        t=spec.underlyings[0]
        S=market.spots[t]
        T=spec.maturity
        r=nearest_rate(market,T)
        q=market.dividends[t]
        vol=market.vols[t]
        ref_spots=spec.params.get("reference_spots",{t:S})
        S_ref=float(ref_spots.get(t,S))
        total=0.0
        for k_pct in spec.params.get("strikes_pct",[0.9,1.0,1.1]):
            total += digital_cash_price(
                S,S_ref*k_pct,T,r,q,vol,
                spec.params.get("payout_each",0.04)*spec.notional
            )
        return PricingResult(total/spec.notional*100,pricing_method="Sum of Black-Scholes digitals")
