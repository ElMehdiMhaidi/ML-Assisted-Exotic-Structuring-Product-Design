"""End-to-end orchestration; contains no product-specific financial logic.

Client request -> NLP mandate -> Random Forest Top 5 -> construction geometries ->
pricing under Q -> outcome metrics under P -> equal-weight normalized ranking ->
Top 3 risk analysis.
"""
import os, dataclasses, pandas as pd
from .nlp import extract_mandate
from .recommendation_model import load_model,predict_top5,train_and_save,MODEL_PATH
from .construction import generate_specs
from .market import load_market_snapshot
from .pricing.router import get_pricer
from .outcome_analysis import analyze
from .risk_engine import run as run_risk
from .schemas import CandidateResult
from .ranking import rank_candidates,averages,select_best_geometry


def _mandate_to_row(m): return dataclasses.asdict(m)


def run_for_client(client_id,use_yfinance=True,mc_paths=50000):
    # Run BERT only for the selected request; do not re-process the full client file
    # every time the Streamlit user changes a client.
    clients = pd.read_csv("data/raw/client_requests.csv")
    matches = clients[clients["client_id"].astype(str) == str(client_id)]
    if matches.empty:
        raise ValueError(f"Unknown client_id={client_id}")
    mandate = extract_mandate(matches.iloc[0])

    os.makedirs("data/nlp_check",exist_ok=True)
    pd.DataFrame([_mandate_to_row(mandate)]).to_csv(
        "data/nlp_check/client_mandates.csv", index=False
    )
    if len(mandate.underlyings)!=1:
        raise ValueError(f"Project scope requires exactly one underlying; parsed {mandate.underlyings} for {client_id}")
    if mandate.currency!="USD":
        raise ValueError("Project scope assumes USD product currency and USD underlying currency; FX/quanto is out of scope.")

    if not os.path.exists(MODEL_PATH): train_and_save()
    model=load_model(); recommendations=predict_top5(model,mandate,top_k=5)
    market=load_market_snapshot(mandate.underlyings,mandate.maturity_years or 2.0,use_yfinance=use_yfinance)

    candidates=[]
    for recommendation in recommendations:
        local=[]
        for spec in generate_specs(mandate,recommendation,max_geometries=20):
            spec.params["reference_spots"]={spec.underlyings[0]:float(market.spots[spec.underlyings[0]])}
            pricer=get_pricer(spec.product_family); pricing=pricer.price(spec,market,mc_paths,42)
            if pricing.solved_term_name and pricing.solved_term_value is not None:
                spec.params[pricing.solved_term_name]=pricing.solved_term_value
            outcome=analyze(spec,market,pricing,scenarios=mc_paths,seed=123)
            local.append((spec,pricing,outcome))
        best=select_best_geometry(local)
        if best is None: continue
        spec,pricing,outcome=best
        candidates.append(CandidateResult(recommendation,spec,pricing,outcome,None))

    ranked=rank_candidates(candidates,k=3)
    top3=[x[0] for x in ranked]
    for c in top3:
        c.risk=run_risk(c.spec,market,paths=mc_paths,seed=42)
    ranking_meta={c.recommendation.product_family:{"combined_score":score,**meta} for c,score,meta in ranked}
    return mandate,market,recommendations,candidates,top3,averages(candidates),ranking_meta
