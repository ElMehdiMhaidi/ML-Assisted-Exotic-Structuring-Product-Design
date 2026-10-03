"""Core dataclasses shared across the structuring workflow.

Project scope is single-asset and same-currency (USD) only.
The schemas contain only fields that are actually used by the current code.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ClientMandate:
    """Standardized mandate extracted from one natural-language client request.

    Preference booleans are tri-state: True / False / None (not specified).
    """
    client_id: str
    client_name: str
    client_type: str
    request_raw: str

    underlyings: List[str] = field(default_factory=list)
    theme: str = ""
    directional_view: str = "neutral"
    target_return: Optional[float] = None
    return_type: str = "unspecified"
    maturity_years: Optional[float] = None
    notional: Optional[float] = None
    currency: str = "USD"
    downside_tolerance: Optional[float] = None
    capital_protection: Optional[float] = None
    income_preference: Optional[bool] = None
    autocall_acceptance: Optional[bool] = None
    risk_appetite: str = "medium"
    complexity_tolerance: str = "medium"
    requested_product_family: Optional[str] = None
    memory_requested: Optional[bool] = None


@dataclass
class ProductRecommendation:
    """Random-Forest recommendation: product family + model probability score."""
    product_family: str
    score: float


@dataclass
class ProductSpec:
    """One concrete product geometry to be priced."""
    product_family: str
    underlyings: List[str]
    maturity: float
    notional: float
    currency: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    """Single-asset market inputs used by pricing under risk-neutral measure Q."""
    spots: Dict[str, float]
    rates: Dict[float, float]
    vols: Dict[str, float]
    dividends: Dict[str, float]


@dataclass
class PricingResult:
    fair_value: float
    solved_term_name: Optional[str] = None
    solved_term_value: Optional[float] = None
    pricing_method: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeResult:
    """Physical-measure P outcome metrics used for candidate comparison."""
    expected_return: float
    sharpe_ratio: float
    expected_shortfall_95: float
    probability_of_loss: float


@dataclass
class RiskResult:
    """Desk risk sensitivities and preset mark-to-market stresses."""
    delta: float
    gamma: float
    vega: float
    stresses: Dict[str, Dict[str, float]]
    spot_profile: List[Dict[str, float]]
    method: str = ""


@dataclass
class CandidateResult:
    recommendation: ProductRecommendation
    spec: ProductSpec
    pricing: PricingResult
    outcome: OutcomeResult
    risk: Optional[RiskResult] = None
