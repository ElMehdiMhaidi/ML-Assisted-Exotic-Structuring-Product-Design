"""Human-readable indicative product terms and markdown export."""
from datetime import date

def _pct(x,digits=2): return "—" if x is None else f"{100*x:.{digits}f}%"
def _num(x,digits=2): return "—" if x is None else f"{x:.{digits}f}"

def product_terms(spec,pricing):
    p=spec.params; terms=[("Product",p.get("display_label",spec.product_family)),("Underlying",spec.underlyings[0]),("Currency",spec.currency),("Notional",f"{spec.currency} {spec.notional:,.0f}"),("Maturity",f"{spec.maturity:.2f}Y")]
    for key,label,kind in [("strike_pct","Strike","pct"),("autocall_barrier","Autocall Barrier","pct"),("coupon_barrier","Coupon Barrier","pct"),("protection_barrier","Protection Barrier","pct"),("barrier_pct","Barrier","pct"),("lower","Lower Range","pct"),("upper","Upper Range","pct"),("frequency","Observation Frequency","raw"),("memory","Memory Coupon","bool"),("payout","Digital Payout","pct"),("payout_each","Digital Payout / Leg","pct")]:
        if key not in p or p[key] is None: continue
        v=p[key]; display=_pct(float(v)) if kind=="pct" else ("Yes" if bool(v) else "No") if kind=="bool" else str(v)
        terms.append((label,display))
    coupon=p.get("coupon")
    if coupon is None and pricing.solved_term_name=="coupon": coupon=pricing.solved_term_value
    if coupon is not None: terms.append(("Coupon",_pct(float(coupon))))
    return terms

def build_markdown_term_sheet(mandate,candidate,candidate_average=None,final_score=None):
    spec=candidate.spec; pricing=candidate.pricing; outcome=candidate.outcome; risk=candidate.risk
    lines=["# Indicative Structured Product Term Sheet","",f"**Date:** {date.today().isoformat()}","","> Indicative / educational project output only. Not a tradable quote or investment recommendation.","","## Client Mandate","",f"- Client: {mandate.client_name} ({mandate.client_id})",f"- Client Type: {mandate.client_type}",f"- Request: {mandate.request_raw}","","## Product Terms",""]
    for label,value in product_terms(spec,pricing): lines.append(f"- **{label}:** {value}")
    lines += ["","## Economics & Ranking","",f"- **Fair Value:** {_num(pricing.fair_value,2)}",f"- **Expected Return (P):** {_pct(outcome.expected_return)}",f"- **Sharpe Ratio:** {_num(outcome.sharpe_ratio,3)}",f"- **Expected Shortfall 95%:** {_pct(outcome.expected_shortfall_95)}",f"- **Probability of Loss:** {_pct(outcome.probability_of_loss)}",f"- **Recommendation Score:** {_pct(candidate.recommendation.score,1)}"]
    if final_score is not None: lines.append(f"- **Final Score:** {_pct(final_score,1)}")
    if candidate_average:
        lines += [f"- **Candidate Average Sharpe:** {_num(candidate_average.get('sharpe_ratio'),3)}",f"- **Candidate Average ES95:** {_pct(candidate_average.get('expected_shortfall_95'))}",f"- **Candidate Average Probability of Loss:** {_pct(candidate_average.get('probability_of_loss'))}"]
    lines += ["","## Current Risk Sensitivities","",f"- **Delta:** {_num(risk.delta,4)}",f"- **Gamma:** {_num(risk.gamma,4)}",f"- **Vega:** {_num(risk.vega,4)}",f"- **Greek Method:** {risk.method}","","## Preset Stress Scenarios",""]
    for name,res in risk.stresses.items(): lines.append(f"- **{name}:** Price {_num(res['price'],2)} | P&L {_num(res['pnl'],2)} price points")
    lines += ["","## Pricing Method","",f"- {pricing.pricing_method}",""]
    return "\n".join(lines)
