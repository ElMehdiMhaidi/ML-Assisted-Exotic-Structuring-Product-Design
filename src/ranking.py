"""Normalize and combine the four agreed candidate-selection criteria.

Higher is better for Recommendation Score and Sharpe.
Lower is better for Expected Shortfall and Probability of Loss.
All four are min-max normalized across the compared set and equally weighted (25% each).
"""

def _minmax_high(values):
    values=list(values)
    if not values: return []
    lo,hi=min(values),max(values)
    if abs(hi-lo)<1e-12: return [0.5]*len(values)
    return [(v-lo)/(hi-lo) for v in values]


def _minmax_low(values):
    return [1.0-x for x in _minmax_high(values)]


def _scores(candidates):
    reco=_minmax_high([c.recommendation.score for c in candidates])
    sharpe=_minmax_high([c.outcome.sharpe_ratio for c in candidates])
    es=_minmax_low([c.outcome.expected_shortfall_95 for c in candidates])
    pol=_minmax_low([c.outcome.probability_of_loss for c in candidates])
    return reco,sharpe,es,pol


def rank_candidates(candidates,k=3):
    if not candidates: return []
    reco,sharpe,es,pol=_scores(candidates)
    scored=[]
    for c,a,b,d,e in zip(candidates,reco,sharpe,es,pol):
        final=(a+b+d+e)/4.0
        meta={"normalized_recommendation":a,"normalized_sharpe":b,"normalized_es":d,"normalized_probability_of_loss":e}
        scored.append((c,final,meta))
    scored.sort(key=lambda x:x[1],reverse=True)
    return scored[:k]


def select_best_geometry(local_results):
    """Choose the best geometry inside one product family using Sharpe, ES and Probability of Loss only."""
    if not local_results: return None
    outcomes=[x[2] for x in local_results]
    s=_minmax_high([o.sharpe_ratio for o in outcomes])
    e=_minmax_low([o.expected_shortfall_95 for o in outcomes])
    p=_minmax_low([o.probability_of_loss for o in outcomes])
    scores=[(a+b+c)/3.0 for a,b,c in zip(s,e,p)]
    return local_results[max(range(len(scores)),key=lambda i:scores[i])]


def averages(candidates):
    if not candidates: return {}
    n=len(candidates)
    return {
        "expected_return":sum(c.outcome.expected_return for c in candidates)/n,
        "sharpe_ratio":sum(c.outcome.sharpe_ratio for c in candidates)/n,
        "expected_shortfall_95":sum(c.outcome.expected_shortfall_95 for c in candidates)/n,
        "probability_of_loss":sum(c.outcome.probability_of_loss for c in candidates)/n,
        "recommendation_score":sum(c.recommendation.score for c in candidates)/n,
    }
