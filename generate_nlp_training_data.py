"""Generate the supervised NLP dataset used to map client requests to ClientMandate.

The generator is intentionally OUTSIDE src/nlp.py: inference code never fabricates
training data.  Each JSONL record contains:
- text: one free-form single-asset client request
- entities: character spans for token-classification / NER slots
- sequence_labels: request-level labels for the sequence-classification heads

Run:
    python generate_nlp_training_data.py --n 4000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


OUT = Path("data/nlp_training/client_mandate_nlp.jsonl")

UNDERLYINGS = {
    "META": ["META", "Meta", "Meta Platforms"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA"],
    "MSFT": ["MSFT", "Microsoft"],
    "AAPL": ["AAPL", "Apple"],
    "AMZN": ["AMZN", "Amazon"],
    "TSLA": ["TSLA", "Tesla"],
}

THEMES = {
    "META": ("Technology", ["technology", "large-cap technology", "US tech"]),
    "NVDA": ("AI", ["artificial intelligence", "AI", "semiconductor AI"]),
    "MSFT": ("Technology", ["technology", "software", "US tech"]),
    "AAPL": ("Technology", ["technology", "consumer technology", "US tech"]),
    "AMZN": ("Technology", ["technology", "e-commerce technology", "US tech"]),
    "TSLA": ("Growth", ["growth equities", "electric vehicles", "high-beta growth"]),
}

PRODUCT_ALIASES = {
    "Vanilla": ["call option", "vanilla option", "plain-vanilla call"],
    "Digital": ["digital option", "digital payoff", "binary option"],
    "Barrier": ["barrier option", "knock-out call", "knock-in option"],
    "Reverse Convertible": ["reverse convertible", "RC note"],
    "ELN": ["equity-linked note", "ELN"],
    "Athena": ["Athena", "Athena autocall"],
    "Phoenix": ["Phoenix", "Phoenix autocall"],
    "Memory Phoenix": ["Memory Phoenix", "memory-coupon Phoenix"],
    "Range Accrual": ["range accrual", "equity range-accrual note"],
    "Strip of Digitals": ["strip of digitals", "digital strip"],
}

PRODUCT_PROFILE = {
    "Vanilla": dict(return_type="participation", income="no", autocall="not_specified", memory="not_specified", directional="bullish"),
    "Digital": dict(return_type="binary_payout", income="no", autocall="not_specified", memory="not_specified", directional="bullish"),
    "Barrier": dict(return_type="participation", income="no", autocall="not_specified", memory="not_specified", directional="bullish"),
    "Reverse Convertible": dict(return_type="income", income="yes", autocall="reject", memory="no", directional="neutral"),
    "ELN": dict(return_type="income", income="yes", autocall="not_specified", memory="not_specified", directional="bullish"),
    "Athena": dict(return_type="income", income="yes", autocall="accept", memory="no", directional="neutral"),
    "Phoenix": dict(return_type="income", income="yes", autocall="accept", memory="no", directional="neutral"),
    "Memory Phoenix": dict(return_type="income", income="yes", autocall="accept", memory="yes", directional="neutral"),
    "Range Accrual": dict(return_type="range_income", income="yes", autocall="reject", memory="no", directional="neutral"),
    "Strip of Digitals": dict(return_type="binary_payout", income="no", autocall="reject", memory="no", directional="bullish"),
}


@dataclass
class Piece:
    text: str
    label: Optional[str] = None


def render(pieces: List[Piece]) -> Tuple[str, List[Dict]]:
    """Join already-punctuated pieces and return exact character-level entity spans."""
    text = ""
    entities: List[Dict] = []
    for piece in pieces:
        start = len(text)
        text += piece.text
        end = len(text)
        if piece.label:
            entities.append({"start": start, "end": end, "label": piece.label, "text": piece.text})
    return text, entities


def p(text: str, label: Optional[str] = None) -> Piece:
    return Piece(text, label)


def maturity_phrase(rng: random.Random) -> Tuple[str, float]:
    options = [
        ("1 year", 1.0), ("12 months", 1.0), ("18 months", 1.5),
        ("2 years", 2.0), ("24 months", 2.0), ("3 years", 3.0),
    ]
    return rng.choice(options)


def notional_phrase(rng: random.Random) -> Tuple[str, float]:
    value = rng.choice([1, 2, 3, 5, 6, 8, 10, 12, 15, 20])
    style = rng.choice([f"{value}m", f"{value} million", f"{value}mm"])
    return style, float(value * 1_000_000)


def pct_phrase(rng: random.Random, low: int, high: int) -> Tuple[str, float]:
    value = rng.randint(low, high)
    return f"{value}%", value / 100.0


def sequence_phrase(rng: random.Random, kind: str, value: str) -> str:
    bank = {
        ("directional", "bullish"): ["The client has a bullish view.", "The view is constructive on the stock.", "The client expects moderate upside."],
        ("directional", "neutral"): ["The view is broadly neutral.", "The client expects the stock to stay broadly range-bound.", "No strong directional view."],
        ("directional", "bearish"): ["The client is cautious and mildly bearish.", "The view is negative on the stock.", "The client expects some downside."],
        ("income", "yes"): ["Regular income is a priority.", "The client wants coupon income.", "Income generation matters."],
        ("income", "no"): ["No periodic income is needed.", "The client does not need coupons.", "Focus on payoff rather than income."],
        ("income", "not_specified"): [""],
        ("autocall", "accept"): ["Autocall is acceptable.", "The client is comfortable with early redemption.", "Early redemption is fine if economics improve."],
        ("autocall", "reject"): ["Please avoid autocall.", "The client does not want early redemption.", "No autocall feature."],
        ("autocall", "not_specified"): [""],
        ("risk", "low"): ["Risk appetite is conservative.", "The client has low risk tolerance.", "Capital preservation is important."],
        ("risk", "medium"): ["Risk appetite is balanced.", "The client has medium risk tolerance.", "A moderate risk profile is acceptable."],
        ("risk", "high"): ["The client is comfortable with high risk.", "Risk appetite is aggressive.", "The client accepts substantial equity risk."],
        ("complexity", "low"): ["Keep the structure simple.", "Low product complexity is preferred.", "The client wants a straightforward payoff."],
        ("complexity", "medium"): ["Moderate product complexity is acceptable.", "The client is comfortable with a standard structured payoff."],
        ("complexity", "high"): ["The client is comfortable with exotic features.", "Higher payoff complexity is acceptable.", "The mandate allows a more complex structure."],
        ("memory", "yes"): ["Missed coupons should be recoverable later.", "A memory coupon feature is requested.", "The client wants unpaid coupons to accumulate."],
        ("memory", "no"): ["No memory coupon feature is needed.", "The client does not want coupon memory."],
        ("memory", "not_specified"): [""],
        ("return_type", "income"): ["The objective is conditional income."],
        ("return_type", "participation"): ["The objective is upside participation."],
        ("return_type", "binary_payout"): ["The client wants a fixed payout if a threshold is met."],
        ("return_type", "range_income"): ["The objective is to monetize a range-bound view."],
        ("return_type", "capital_protection"): ["The objective is capital preservation with some upside."],
        ("return_type", "unspecified"): [""],
    }
    return rng.choice(bank[(kind, value)])


def generate_one(i: int, rng: random.Random) -> Dict:
    ticker = rng.choice(list(UNDERLYINGS))
    underlying_text = rng.choice(UNDERLYINGS[ticker])
    theme_value, theme_aliases = THEMES[ticker]
    product = rng.choice(list(PRODUCT_ALIASES))
    profile = dict(PRODUCT_PROFILE[product])

    # Inject controlled diversity in request-level labels while keeping the request coherent.
    if product in {"Vanilla", "Digital", "Barrier", "ELN", "Strip of Digitals"} and rng.random() < 0.18:
        profile["directional"] = "bearish"

    # Some requests simply do not state income or return-shape preferences.
    if rng.random() < 0.12:
        profile["income"] = "not_specified"
    if rng.random() < 0.10:
        profile["return_type"] = "unspecified"
    elif product in {"ELN", "Vanilla"} and rng.random() < 0.22:
        profile["return_type"] = "capital_protection"
        if profile["income"] == "yes":
            profile["income"] = "not_specified"

    # Even for an autocallable product, a client may name the structure without explicitly
    # stating an autocall preference; preserve that distinction in the labels.
    if product in {"Athena", "Phoenix", "Memory Phoenix"} and rng.random() < 0.12:
        profile["autocall"] = "not_specified"

    risk = rng.choices(["low", "medium", "high"], weights=[0.25, 0.5, 0.25])[0]
    complexity_default = "high" if product in {"Athena", "Phoenix", "Memory Phoenix", "Range Accrual", "Strip of Digitals", "Barrier"} else "medium"
    complexity = complexity_default if rng.random() < 0.65 else rng.choice(["low", "medium", "high"])

    maturity_text, maturity = maturity_phrase(rng)
    notional_text, notional = notional_phrase(rng)
    target_text, target = pct_phrase(rng, 5, 14)
    downside_text, downside = pct_phrase(rng, 15, 45)
    protection_text, protection = pct_phrase(rng, 70, 100)

    explicit_product = rng.random() < 0.72
    explicit_theme = rng.random() < 0.45
    include_target = rng.random() < (0.78 if profile["income"] == "yes" else 0.35)
    include_downside = rng.random() < 0.45
    include_protection = (not include_downside) and (profile["return_type"] == "capital_protection" or rng.random() < 0.25)

    # Intro block with NER spans.
    intro: List[Piece] = [p(rng.choice(["Client wants ", "Looking for ", "Need ", "The mandate is for "]))]
    if explicit_product:
        intro += [p("a "), p(rng.choice(PRODUCT_ALIASES[product]), "PRODUCT"), p(" on ")]
    else:
        intro += [p("single-name exposure on ")]
    intro += [p(underlying_text, "UNDERLYING"), p(". ")]

    blocks: List[List[Piece]] = [intro]
    blocks.append([p("Maturity: "), p(maturity_text, "MATURITY"), p(". ")])
    blocks.append([p("Ticket: "), p("USD", "CURRENCY"), p(" "), p(notional_text, "NOTIONAL"), p(". ")])

    if explicit_theme:
        blocks.append([p("The exposure is in "), p(rng.choice(theme_aliases), "THEME"), p(". ")])
    if include_target:
        blocks.append([p(rng.choice(["Targeting around ", "Target return is ", "Looking for roughly "])), p(target_text, "TARGET_RETURN"), p(" per year. ")])
    if include_downside:
        blocks.append([p(rng.choice(["Comfortable with ", "Can tolerate roughly ", "Downside tolerance is "])), p(downside_text, "DOWNSIDE_TOLERANCE"), p(" downside. ")])
    if include_protection:
        blocks.append([p(rng.choice(["Needs ", "Requests ", "Wants "])), p(protection_text, "CAPITAL_PROTECTION"), p(" capital protection. ")])

    # Request-level intents are written as natural-language clauses, not entity spans.
    seq_phrases = [
        sequence_phrase(rng, "directional", profile["directional"]),
        sequence_phrase(rng, "return_type", profile["return_type"]),
        sequence_phrase(rng, "income", profile["income"]),
        sequence_phrase(rng, "autocall", profile["autocall"]),
        sequence_phrase(rng, "risk", risk),
        sequence_phrase(rng, "complexity", complexity),
        sequence_phrase(rng, "memory", profile["memory"]),
    ]
    for phrase in seq_phrases:
        if phrase:
            blocks.append([p(phrase + " ")])

    # Keep the intro first, shuffle the rest so the model does not memorize one ordering.
    rest = blocks[1:]
    rng.shuffle(rest)
    pieces = blocks[0] + [piece for block in rest for piece in block]
    text, entities = render(pieces)
    text = text.strip()

    # Trimming the final whitespace does not change entity spans because no entity is last whitespace.
    seq = {
        "directional_view": profile["directional"],
        "return_type": profile["return_type"],
        "income_preference": profile["income"],
        "autocall_acceptance": profile["autocall"],
        "risk_appetite": risk,
        "complexity_tolerance": complexity,
        "memory_requested": profile["memory"],
    }
    expected = {
        "underlyings": [ticker],
        "theme": theme_value,  # explicit NER span or deterministic ticker fallback
        "maturity_years": maturity,
        "target_return": target if include_target else None,
        "notional": notional,
        "currency": "USD",
        "downside_tolerance": downside if include_downside else None,
        "capital_protection": protection if include_protection else None,
        "requested_product_family": product if explicit_product else None,
        **seq,
    }
    return {"id": f"NLP{i:05d}", "text": text, "entities": entities, "sequence_labels": seq, "expected_mandate": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(generate_one(i, rng), ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} examples to {args.out}")


if __name__ == "__main__":
    main()
