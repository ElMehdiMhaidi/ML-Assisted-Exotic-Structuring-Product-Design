"""Terminal payoff illustration for the supported single-asset product families."""
import numpy as np

def payoff_profile(spec,pricing,points=161):
    x=np.linspace(0.4,1.6,points); fam=spec.product_family; p=spec.params
    coupon=p.get("coupon")
    if coupon is None and pricing.solved_term_name=="coupon": coupon=pricing.solved_term_value
    coupon=float(coupon or 0.0)
    if fam=="Vanilla":
        k=float(p.get("strike_pct",1.0)); y=np.maximum(x-k,0.0)
    elif fam in {"Reverse Convertible","ELN"}:
        k=float(p.get("strike_pct",1.0)); y=np.maximum(0.0,1.0-np.maximum(k-x,0.0))+coupon*spec.maturity
    elif fam=="Digital":
        k=float(p.get("strike_pct",1.0)); y=np.where(x>=k,float(p.get("payout",0.10)),0.0)
    elif fam=="Barrier":
        k=float(p.get("strike_pct",1.0)); b=float(p.get("barrier_pct",0.7)); y=np.where(x>b,np.maximum(x-k,0.0),0.0)
    elif fam in {"Phoenix","Athena"}:
        pb=float(p.get("protection_barrier",0.65)); ac=float(p.get("autocall_barrier",1.0)); cb=float(p.get("coupon_barrier",0.70))
        y=np.where(x>=ac,1.0+coupon,np.where(x>=cb,1.0+coupon,np.where(x>=pb,1.0,x)))
    elif fam=="Range Accrual":
        lo=float(p.get("lower",0.8)); hi=float(p.get("upper",1.2)); a=float(p.get("accrual_rate",0.10)); y=1.0+np.where((x>=lo)&(x<=hi),a*spec.maturity,0.0)
    elif fam=="Strip of Digitals":
        y=np.zeros_like(x)
        for k in p.get("strikes_pct",[0.9,1.0,1.1]): y += np.where(x>=float(k),float(p.get("payout_each",0.04)),0.0)
    else: y=x
    barriers=[]
    for key,label in [("strike_pct","Strike"),("barrier_pct","Barrier"),("autocall_barrier","Autocall barrier"),("coupon_barrier","Coupon barrier"),("protection_barrier","Protection barrier"),("lower","Lower range"),("upper","Upper range")]:
        if key in p and p[key] is not None: barriers.append({"x":100*float(p[key]),"label":label})
    return [{"spot_pct":float(100*s),"payoff_pct":float(100*v)} for s,v in zip(x,y)],barriers
