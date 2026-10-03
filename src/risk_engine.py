"""Single-asset desk risk engine: Delta, Gamma, Vega and preset MTM stresses.

Closed-form Greeks are used where available (Vanilla, Digital).
Other products use central bump-and-reprice with common random numbers.
No correlation sensitivity is included because the project scope has no baskets.
"""
import copy, numpy as np
from .schemas import RiskResult
from .pricing.router import get_pricer
from .market import nearest_rate
from .pricing.vanilla import bs_greeks
from .pricing.digital import digital_cash_greeks

ANALYTIC_FAMILIES={"Vanilla","Digital"}


def _bumped_market(market,spot_mult=1.0,vol_add=0.0):
    m=copy.deepcopy(market)
    m.spots={k:v*spot_mult for k,v in m.spots.items()}
    m.vols={k:max(0.01,v+vol_add) for k,v in m.vols.items()}
    return m


def _analytic_greeks(spec,market):
    fam=spec.product_family; t=spec.underlyings[0]; S=market.spots[t]; T=spec.maturity
    r=nearest_rate(market,T); q=market.dividends[t]; vol=market.vols[t]
    ref=spec.params.get("reference_spots",{t:S}); Sref=float(ref.get(t,S)); scale=100.0/Sref
    if fam=="Vanilla":
        K=Sref*spec.params.get("strike_pct",1.0)
        g=bs_greeks(S,K,T,r,q,vol,spec.params.get("option_type","call"))
        return g["delta"]*S*scale, g["gamma"]*S*S*scale, g["vega"]*scale
    if fam=="Digital":
        K=Sref*spec.params.get("strike_pct",1.0); payout=spec.params.get("payout",0.10)*spec.notional
        g=digital_cash_greeks(S,K,T,r,q,vol,payout); pscale=100.0/spec.notional
        return g["delta"]*S*pscale, g["gamma"]*S*S*pscale, g["vega"]*pscale
    return None


def _bump_greeks(spec,market,paths,seed,spot_bump=0.02,vol_bump=0.01):
    pricer=get_pricer(spec.product_family)
    base=pricer.price(spec,market,paths,seed).fair_value
    up=pricer.price(spec,_bumped_market(market,1+spot_bump),paths,seed).fair_value
    dn=pricer.price(spec,_bumped_market(market,1-spot_bump),paths,seed).fair_value
    delta=(up-dn)/(2*spot_bump); gamma=(up-2*base+dn)/(spot_bump**2)
    vu=pricer.price(spec,_bumped_market(market,vol_add=vol_bump),paths,seed).fair_value
    vd=pricer.price(spec,_bumped_market(market,vol_add=-vol_bump),paths,seed).fair_value
    vega=(vu-vd)/(2*vol_bump)
    return base,delta,gamma,vega


def run(spec,market,paths=50000,seed=42):
    paths=max(50000,int(paths)); pricer=get_pricer(spec.product_family)
    analytic=_analytic_greeks(spec,market)
    if analytic is not None:
        base=pricer.price(spec,market,paths,seed).fair_value; delta,gamma,vega=analytic; method="Analytic Greeks"
    else:
        base,delta,gamma,vega=_bump_greeks(spec,market,paths,seed); method="Central bump-and-reprice with common random numbers"
    scenarios={"Spot -20%":(-0.20,0.0),"Spot -10%":(-0.10,0.0),"Vol +10 pts":(0.0,0.10),"Spot -20% + Vol +10 pts":(-0.20,0.10)}
    stresses={}
    for name,(s,v) in scenarios.items():
        px=pricer.price(spec,_bumped_market(market,1+s,v),paths,seed).fair_value
        stresses[name]={"price":float(px),"pnl":float(px-base)}
    profile=[]
    for mult in np.linspace(0.60,1.40,17):
        lm=_bumped_market(market,float(mult)); lp=pricer.price(spec,lm,paths,seed).fair_value
        la=_analytic_greeks(spec,lm)
        if la is not None: d,g,v=la
        else: _,d,g,v=_bump_greeks(spec,lm,paths,seed)
        profile.append({"spot_pct":float(mult*100),"price":float(lp),"delta":float(d),"gamma":float(g),"vega":float(v)})
    return RiskResult(float(delta),float(gamma),float(vega),stresses,profile,method)
