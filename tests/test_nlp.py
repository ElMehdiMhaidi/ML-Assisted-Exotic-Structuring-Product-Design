import pandas as pd

from src.nlp import mandate_from_predictions, normalize_maturity, normalize_notional


def test_financial_normalization():
    assert normalize_maturity("18 months") == 1.5
    assert normalize_notional("5m") == 5_000_000


def test_predictions_to_mandate_without_loading_bert():
    row = pd.Series({"client_name": "X", "client_id": "1", "client_type": "AM"})
    text = "Need a digital option on Amazon. Maturity 1 year. Ticket USD 2m."
    spans = [
        {"label": "PRODUCT", "text": "digital option", "start": 7, "end": 21},
        {"label": "UNDERLYING", "text": "Amazon", "start": 25, "end": 31},
        {"label": "MATURITY", "text": "1 year", "start": 42, "end": 48},
        {"label": "CURRENCY", "text": "USD", "start": 57, "end": 60},
        {"label": "NOTIONAL", "text": "2m", "start": 61, "end": 63},
    ]
    sequence = {
        "directional_view": "bullish",
        "return_type": "binary_payout",
        "income_preference": "no",
        "autocall_acceptance": "not_specified",
        "risk_appetite": "medium",
        "complexity_tolerance": "medium",
        "memory_requested": "not_specified",
    }
    m = mandate_from_predictions(row, text, spans, sequence)
    assert m.underlyings == ["AMZN"]
    assert m.requested_product_family == "Digital"
    assert m.maturity_years == 1.0
    assert m.notional == 2_000_000
    assert m.income_preference is False
    assert m.autocall_acceptance is None
