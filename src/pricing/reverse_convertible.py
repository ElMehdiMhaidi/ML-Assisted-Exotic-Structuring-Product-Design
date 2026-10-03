from .base import BasePricer
from .vanilla import bs_put
from ..market import nearest_rate
from ..schemas import PricingResult
import math

class ReverseConvertiblePricer(BasePricer):
    def price(self,spec,market,paths=50000,seed=42):
        t=spec.underlyings[0]
        S=market.spots[t]
        T=spec.maturity
        r=nearest_rate(market,T)
        q=market.dividends[t]
        vol=market.vols[t]

        # Contract strike is fixed at inception, not recomputed from shocked spot.
        ref_spots=spec.params.get("reference_spots", {t:S})
        S_ref=float(ref_spots.get(t,S))
        K=S_ref*spec.params.get("strike_pct",1.0)

        put=bs_put(S,K,T,r,q,vol)
        principal_pv=100.0*math.exp(-r*T)
        option_cost=put/S_ref*100.0
        annuity=sum(math.exp(-r*(i/4)) for i in range(1,int(round(T*4))+1))/4

        coupon=spec.params.get("coupon")
        if coupon is None:
            coupon=max(0.0,(100.0-principal_pv+option_cost)/(100.0*annuity))
            fair_value=100.0
            solved_name="coupon"
            solved_value=coupon
        else:
            # Risk revaluation: contractual coupon is held fixed.
            fair_value=principal_pv-option_cost+coupon*100.0*annuity
            solved_name=None
            solved_value=None

        return PricingResult(
            fair_value=float(fair_value),
            solved_term_name=solved_name,
            solved_term_value=solved_value,
            pricing_method="Bond + short put decomposition",
            diagnostics={"embedded_put_value_pct":float(option_cost)}
        )
