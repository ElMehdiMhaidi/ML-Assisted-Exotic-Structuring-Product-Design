"""Transformer NLP pipeline: free-form client request -> standardized ClientMandate.

ROLE
----
This module performs INFERENCE only.  It does not generate training data and it
does not train models.

Architecture
------------
raw request
    -> light cleaning
    -> native BERT subword tokenizer
    -> shared fine-tuned BERT encoder
       -> token-classification head (BIO NER / slot filling)
       -> request-level sequence-classification heads
    -> deterministic financial normalization
    -> ClientMandate

Token head extracts explicit spans:
    UNDERLYING, THEME, TARGET_RETURN, MATURITY, NOTIONAL, CURRENCY,
    DOWNSIDE_TOLERANCE, CAPITAL_PROTECTION, PRODUCT.

Sequence heads classify the whole request:
    directional_view, return_type, income_preference, autocall_acceptance,
    risk_appetite, complexity_tolerance, memory_requested.

The project scope is single-asset and same-currency USD only.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from .nlp_model import MandateNLPModel
from .schemas import ClientMandate


MODEL_DIR = Path("models/nlp_mandate")

TICKER_ALIASES = {
    "meta": "META", "meta platforms": "META", "facebook": "META", "meta platforms inc": "META",
    "nvidia": "NVDA", "nvidia corporation": "NVDA", "nvda": "NVDA",
    "microsoft": "MSFT", "microsoft corporation": "MSFT", "msft": "MSFT",
    "apple": "AAPL", "apple inc": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amazon.com": "AMZN", "amzn": "AMZN",
    "tesla": "TSLA", "tesla inc": "TSLA", "tsla": "TSLA",
}

THEME_ALIASES = {
    "technology": "Technology", "large-cap technology": "Technology", "us tech": "Technology",
    "software": "Technology", "consumer technology": "Technology", "e-commerce technology": "Technology",
    "artificial intelligence": "AI", "ai": "AI", "semiconductor ai": "AI",
    "growth equities": "Growth", "electric vehicles": "Growth", "high-beta growth": "Growth",
}

TICKER_THEME = {
    "META": "Technology", "NVDA": "AI", "MSFT": "Technology", "AAPL": "Technology",
    "AMZN": "Technology", "TSLA": "Growth",
}

PRODUCT_ALIASES = {
    "call option": "Vanilla", "put option": "Vanilla", "vanilla option": "Vanilla", "plain-vanilla call": "Vanilla",
    "digital option": "Digital", "digital payoff": "Digital", "binary option": "Digital", "digital": "Digital",
    "barrier option": "Barrier", "knock-out call": "Barrier", "knock-in option": "Barrier",
    "reverse convertible": "Reverse Convertible", "rc note": "Reverse Convertible",
    "equity-linked note": "ELN", "equity linked note": "ELN", "eln": "ELN",
    "athena": "Athena", "athena autocall": "Athena",
    "phoenix": "Phoenix", "phoenix autocall": "Phoenix",
    "memory phoenix": "Memory Phoenix", "memory-coupon phoenix": "Memory Phoenix",
    "range accrual": "Range Accrual", "equity range-accrual note": "Range Accrual",
    "strip of digitals": "Strip of Digitals", "digital strip": "Strip of Digitals",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "twelve": 12, "eighteen": 18, "twenty-four": 24, "thirty-six": 36,
}


class MandateExtractor:
    """Loads the trained BERT checkpoint once and extracts one mandate at a time."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: Optional[str] = None) -> None:
        if not (model_dir / "model.pt").exists():
            raise RuntimeError(
                f"NLP checkpoint not found at {model_dir}. "
                "Run `python train_nlp.py` once before launching the pipeline."
            )

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Transformer NLP pipeline requires the `transformers` package. "
                "Install project requirements first."
            ) from exc

        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.token_labels: List[str] = metadata["token_labels"]
        self.sequence_labels: Dict[str, List[str]] = metadata["sequence_labels"]
        self.id_to_token = {i: label for i, label in enumerate(self.token_labels)}
        self.max_length = int(metadata.get("max_length", 192))

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", use_fast=True)
        encoder_config = AutoConfig.from_pretrained(model_dir / "encoder_config")
        encoder = AutoModel.from_config(encoder_config)
        self.model = MandateNLPModel(
            encoder=encoder,
            n_token_labels=len(self.token_labels),
            sequence_head_sizes={k: len(v) for k, v in self.sequence_labels.items()},
        )
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, raw_text: str) -> Tuple[str, List[Dict], Dict[str, str]]:
        text = clean_text(raw_text)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        model_inputs = {k: v.to(self.device) for k, v in encoded.items()}
        out = self.model(**model_inputs)

        token_ids = out["token_logits"][0].argmax(-1).detach().cpu().tolist()
        spans = decode_bio_spans(text, offsets, token_ids, self.id_to_token)

        sequence = {}
        for name, logits in out["sequence_logits"].items():
            class_id = int(logits[0].argmax().detach().cpu())
            sequence[name] = self.sequence_labels[name][class_id]
        return text, spans, sequence


_EXTRACTOR: Optional[MandateExtractor] = None


def get_extractor() -> MandateExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = MandateExtractor()
    return _EXTRACTOR


def clean_text(text: str) -> str:
    """Light cleaning only: preserve wording, negation, numbers, %, and punctuation."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def decode_bio_spans(
    text: str,
    offsets: List[List[int]],
    token_ids: List[int],
    id_to_label: Dict[int, str],
) -> List[Dict]:
    """Merge B/I subword predictions back into character spans in the request."""
    spans: List[Dict] = []
    current = None

    for (start, end), token_id in zip(offsets, token_ids):
        if start == end:
            continue
        label = id_to_label[int(token_id)]
        if label == "O":
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                spans.append(current)
                current = None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or current is None or current["label"] != entity_type:
            if current is not None:
                current["text"] = text[current["start"]:current["end"]]
                spans.append(current)
            current = {"start": int(start), "end": int(end), "label": entity_type}
        else:
            current["end"] = int(end)

    if current is not None:
        current["text"] = text[current["start"]:current["end"]]
        spans.append(current)
    return spans


def _first_span(spans: List[Dict], label: str) -> Optional[str]:
    for span in spans:
        if span["label"] == label:
            return span["text"]
    return None


def _all_spans(spans: List[Dict], label: str) -> List[str]:
    return [span["text"] for span in spans if span["label"] == label]


def normalize_underlying(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = clean_text(value).lower().strip(" .,;:")
    if key.upper() in TICKER_THEME:
        return key.upper()
    return TICKER_ALIASES.get(key)


def normalize_theme(value: Optional[str], ticker: Optional[str]) -> str:
    if value:
        key = clean_text(value).lower().strip(" .,;:")
        if key in THEME_ALIASES:
            return THEME_ALIASES[key]
    return TICKER_THEME.get(ticker or "", "")


def normalize_maturity(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    t = clean_text(value).lower()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(year|years|yr|yrs)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(month|months|mo|mos)", t)
    if m:
        return float(m.group(1)) / 12.0
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\s+years?\b", t):
            return float(number)
        if re.search(rf"\b{re.escape(word)}\s+months?\b", t):
            return float(number) / 12.0
    return None


def normalize_notional(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    t = clean_text(value).lower().replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(bn|billion|m|mm|million|k|thousand)?", t)
    if not m:
        return None
    amount = float(m.group(1))
    suffix = m.group(2) or ""
    if suffix in {"bn", "billion"}:
        amount *= 1_000_000_000
    elif suffix in {"m", "mm", "million"}:
        amount *= 1_000_000
    elif suffix in {"k", "thousand"}:
        amount *= 1_000
    return amount


def normalize_percentage(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", clean_text(value))
    return None if not m else float(m.group(1)) / 100.0


def normalize_currency(value: Optional[str]) -> str:
    if value is None:
        return "USD"  # explicit same-currency project assumption
    key = clean_text(value).upper().strip(" .,;:")
    if key in {"USD", "$", "US DOLLAR", "US DOLLARS"}:
        return "USD"
    raise ValueError(f"Project scope supports USD only; NLP extracted currency={value!r}")


def normalize_product(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = clean_text(value).lower().strip(" .,;:")
    if key in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[key]
    # Longest alias match handles spans such as 'a memory Phoenix'.
    for alias in sorted(PRODUCT_ALIASES, key=len, reverse=True):
        if alias in key:
            return PRODUCT_ALIASES[alias]
    return None


def tri_state_bool(label: str, positive: str, negative: str) -> Optional[bool]:
    if label == positive:
        return True
    if label == negative:
        return False
    return None


def mandate_from_predictions(row, text: str, spans: List[Dict], sequence: Dict[str, str]) -> ClientMandate:
    """Convert learned NLP outputs into the exact ClientMandate dataclass."""
    underlying_values = [normalize_underlying(v) for v in _all_spans(spans, "UNDERLYING")]
    underlyings = []
    for ticker in underlying_values:
        if ticker and ticker not in underlyings:
            underlyings.append(ticker)

    if len(underlyings) > 1:
        raise ValueError(f"Project scope is single-asset only; NLP extracted multiple underlyings: {underlyings}")
    ticker = underlyings[0] if underlyings else None

    return ClientMandate(
        client_id=str(row["client_id"]),
        client_name=str(row["client_name"]),
        client_type=str(row["client_type"]),
        request_raw=text,
        underlyings=underlyings,
        theme=normalize_theme(_first_span(spans, "THEME"), ticker),
        directional_view=sequence["directional_view"],
        target_return=normalize_percentage(_first_span(spans, "TARGET_RETURN")),
        return_type=sequence["return_type"],
        maturity_years=normalize_maturity(_first_span(spans, "MATURITY")),
        notional=normalize_notional(_first_span(spans, "NOTIONAL")),
        currency=normalize_currency(_first_span(spans, "CURRENCY")),
        downside_tolerance=normalize_percentage(_first_span(spans, "DOWNSIDE_TOLERANCE")),
        capital_protection=normalize_percentage(_first_span(spans, "CAPITAL_PROTECTION")),
        income_preference=tri_state_bool(sequence["income_preference"], "yes", "no"),
        autocall_acceptance=tri_state_bool(sequence["autocall_acceptance"], "accept", "reject"),
        risk_appetite=sequence["risk_appetite"],
        complexity_tolerance=sequence["complexity_tolerance"],
        requested_product_family=normalize_product(_first_span(spans, "PRODUCT")),
        memory_requested=tri_state_bool(sequence["memory_requested"], "yes", "no"),
    )


def extract_mandate(row, extractor: Optional[MandateExtractor] = None) -> ClientMandate:
    extractor = extractor or get_extractor()
    text, spans, sequence = extractor.predict(row["request"])
    return mandate_from_predictions(row, text, spans, sequence)


def process_csv(path: str) -> List[ClientMandate]:
    df = pd.read_csv(path)
    extractor = get_extractor()  # load BERT only once for the whole file
    return [extract_mandate(row, extractor=extractor) for _, row in df.iterrows()]


def mandates_to_dataframe(mandates: List[ClientMandate]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "client_id": m.client_id,
            "client_name": m.client_name,
            "client_type": m.client_type,
            "request_raw": m.request_raw,
            "underlyings": "|".join(m.underlyings),
            "theme": m.theme,
            "directional_view": m.directional_view,
            "target_return": m.target_return,
            "return_type": m.return_type,
            "maturity_years": m.maturity_years,
            "notional": m.notional,
            "currency": m.currency,
            "downside_tolerance": m.downside_tolerance,
            "capital_protection": m.capital_protection,
            "income_preference": m.income_preference,
            "autocall_acceptance": m.autocall_acceptance,
            "risk_appetite": m.risk_appetite,
            "complexity_tolerance": m.complexity_tolerance,
            "requested_product_family": m.requested_product_family,
            "memory_requested": m.memory_requested,
        }
        for m in mandates
    ])
