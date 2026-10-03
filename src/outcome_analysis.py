"""Evaluate client outcomes under physical measure P (not used for pricing).

Pricing is performed under Q in src/pricing/.
This module separately simulates the client's payoff distribution under P with:
    physical drift = risk-free rate + 4% equity risk premium
and calculates the three agreed evaluation indicators:
    Sharpe Ratio, Expected Shortfall 95%, Probability of Loss.
Expected Return is retained because it is needed to compute Sharpe and is useful for display.
"""
import numpy as np
from .market import nearest_rate
from .schemas import OutcomeResult

EQUITY_RISK_PREMIUM = 0.04


def _terminal_rel(spec, market, scenarios, rng):
    t=spec.underlyings[0]; T=spec.maturity; rf=nearest_rate(market,T)
    q=market.dividends[t]; vol=market.vols[t]; mu=rf+EQUITY_RISK_PREMIUM
    z=rng.standard_normal(scenarios)
    return np.exp((mu-q-0.5*vol*vol)*T + vol*np.sqrt(T)*z)


def _path_rel(spec, market, scenarios, rng, frequency):
    t=spec.underlyings[0]; T=spec.maturity; rf=nearest_rate(market,T)
    q=market.dividends[t]; vol=market.vols[t]; mu=rf+EQUITY_RISK_PREMIUM
    steps=max(1,int(round(T*frequency))); dt=T/steps
    rel=np.ones(scenarios); hist=np.empty((steps,scenarios))
    for i in range(steps):
        z=rng.standard_normal(scenarios)
        rel *= np.exp((mu-q-0.5*vol*vol)*dt + vol*np.sqrt(dt)*z)
        hist[i]=rel
    return hist


def _payoff_price_points(spec, market, pricing, scenarios, seed):
    rng=np.random.default_rng(seed); fam=spec.product_family; p=spec.params
    coupon=float(p.get("coupon") or pricing.solved_term_value or 0.0)

    if fam in {"Phoenix","Athena"}:
        freq=int(p.get("frequency",4)); hist=_path_rel(spec,market,scenarios,rng,freq)
        ac=p.get("autocall_barrier",1.0); cb=p.get("coupon_barrier",0.70); pb=p.get("protection_barrier",0.65)
        phoenix=fam=="Phoenix"; memory=bool(p.get("memory",False))
        alive=np.ones(scenarios,dtype=bool); cash=np.zeros(scenarios); missed=np.zeros(scenarios)
        for i in range(hist.shape[0]):
            x=hist[i]
            if phoenix:
                due=(x>=cb)&alive
                if memory:
                    missed[alive & ~due]+=coupon
                    cflow=np.where(due,coupon+missed,0.0); missed[due]=0.0
                else:
                    cflow=np.where(due,coupon,0.0)
            else:
                cflow=np.where(alive,coupon,0.0)
            cash += 100.0*cflow
            call=(x>=ac)&alive
            cash += 100.0*call
            alive[call]=False
        xT=hist[-1]
        cash += np.where(xT>=pb,100.0,100.0*xT)*alive
        return cash

    if fam=="Barrier":
        hist=_path_rel(spec,market,scenarios,rng,52)
        k=p.get("strike_pct",1.0); b=p.get("barrier_pct",0.7)
        alive=hist.min(axis=0)>b
        return 100.0*np.maximum(hist[-1]-k,0.0)*alive

    if fam=="Range Accrual":
        freq=int(p.get("frequency",12)); hist=_path_rel(spec,market,scenarios,rng,freq)
        inside=((hist>=p.get("lower",0.8))&(hist<=p.get("upper",1.2))).mean(axis=0)
        return 100.0*(1.0+p.get("accrual_rate",0.10)*spec.maturity*inside)

    x=_terminal_rel(spec,market,scenarios,rng)
    if fam in {"Reverse Convertible","ELN"}:
        k=p.get("strike_pct",1.0)
        redemption=np.maximum(0.0,100.0-100.0*np.maximum(k-x,0.0))
        return redemption + 100.0*coupon*spec.maturity
    if fam=="Digital":
        return np.where(x>=p.get("strike_pct",1.0),100.0*p.get("payout",0.10),0.0)
    if fam=="Strip of Digitals":
        payoff=np.zeros(scenarios)
        for k in p.get("strikes_pct",[0.9,1.0,1.1]):
            payoff += np.where(x>=k,100.0*p.get("payout_each",0.04),0.0)
        return payoff
    if fam=="Vanilla":
        k=p.get("strike_pct",1.0)
        if p.get("option_type","call")=="put": return 100.0*np.maximum(k-x,0.0)
        return 100.0*np.maximum(x-k,0.0)
    return 100.0*x


def analyze(spec, market, pricing, scenarios=50000, seed=123):
    payoff=_payoff_price_points(spec,market,pricing,int(scenarios),seed)
    initial=max(1e-8,float(pricing.fair_value))
    returns=payoff/initial-1.0
    expected=float(returns.mean())
    std=float(returns.std(ddof=1))
    rf=nearest_rate(market,spec.maturity)
    sharpe=(expected-rf*spec.maturity)/std if std>1e-12 else 0.0
    q05=float(np.quantile(returns,0.05)); tail=returns[returns<=q05]
    es=max(0.0,-float(tail.mean())) if len(tail) else 0.0
    prob_loss=float((returns<0.0).mean())
    return OutcomeResult(expected, float(sharpe), float(es), prob_loss)
