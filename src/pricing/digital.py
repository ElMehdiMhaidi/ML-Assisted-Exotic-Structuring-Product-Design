import math
from scipy.stats import norm
from .base import BasePricer
from ..market import nearest_rate
from ..schemas import PricingResult

def digital_cash_price(S,K,T,r,q,sigma,payout=1.0):
    if T<=0 or sigma<=0:
        return payout*math.exp(-r*T)*(1.0 if S>=K else 0.0)
    d2=(math.log(S/K)+(r-q-0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    return payout*math.exp(-r*T)*norm.cdf(d2)

def digital_cash_greeks(S,K,T,r,q,sigma,payout=1.0):
    if T<=0 or sigma<=0:
        return {"delta":0.0,"gamma":0.0,"vega":0.0}
    sqrtT=math.sqrt(T)
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*sqrtT)
    d2=d1-sigma*sqrtT
    disc=math.exp(-r*T)
    phi=norm.pdf(d2)

    delta = payout*disc*phi/(S*sigma*sqrtT)

    dd2_dS = 1.0/(S*sigma*sqrtT)
    gamma = payout*disc * (
        (-d2*phi)*(dd2_dS)/(S*sigma*sqrtT)
        - phi/(S*S*sigma*sqrtT)
    )

    dd2_dsigma = -(math.log(S/K)+(r-q)*T)/(sigma*sigma*sqrtT) - 0.5*sqrtT
    vega = payout*disc*phi*dd2_dsigma

    return {"delta":delta,"gamma":gamma,"vega":vega}

class DigitalPricer(BasePricer):
    def price(self,spec,market,paths=50000,seed=42):
        t=spec.underlyings[0]
        S=market.spots[t]
        T=spec.maturity
        r=nearest_rate(market,T)
        q=market.dividends[t]
        vol=market.vols[t]
        ref_spots=spec.params.get("reference_spots",{t:S})
        S_ref=float(ref_spots.get(t,S))
        K=S_ref*spec.params.get("strike_pct",1.0)
        payout=spec.params.get("payout",0.1)*spec.notional
        pv=digital_cash_price(S,K,T,r,q,vol,payout)
        return PricingResult(pv/spec.notional*100,pricing_method="Black-Scholes digital")
