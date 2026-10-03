import math
from scipy.stats import norm
from .base import BasePricer
from ..market import nearest_rate
from ..schemas import PricingResult

def bs_d1_d2(S,K,T,r,q,sigma):
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    return d1,d2

def bs_call(S,K,T,r,q,sigma):
    if T<=0 or sigma<=0:
        return max(S-K,0.0)
    d1,d2=bs_d1_d2(S,K,T,r,q,sigma)
    return S*math.exp(-q*T)*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)

def bs_put(S,K,T,r,q,sigma):
    if T<=0 or sigma<=0:
        return max(K-S,0.0)
    d1,d2=bs_d1_d2(S,K,T,r,q,sigma)
    return K*math.exp(-r*T)*norm.cdf(-d2)-S*math.exp(-q*T)*norm.cdf(-d1)

def bs_greeks(S,K,T,r,q,sigma,option_type="call"):
    if T<=0 or sigma<=0:
        return {"delta":0.0,"gamma":0.0,"vega":0.0}
    d1,_=bs_d1_d2(S,K,T,r,q,sigma)
    disc_q=math.exp(-q*T)
    pdf=norm.pdf(d1)

    if option_type=="call":
        delta=disc_q*norm.cdf(d1)
    else:
        delta=disc_q*(norm.cdf(d1)-1)

    gamma=disc_q*pdf/(S*sigma*math.sqrt(T))
    vega=S*disc_q*pdf*math.sqrt(T)
    return {"delta":delta,"gamma":gamma,"vega":vega}

class VanillaPricer(BasePricer):
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
        typ=spec.params.get("option_type","call")
        px=bs_call(S,K,T,r,q,vol) if typ=="call" else bs_put(S,K,T,r,q,vol)
        return PricingResult(px/S_ref*100,pricing_method="Black-Scholes")
