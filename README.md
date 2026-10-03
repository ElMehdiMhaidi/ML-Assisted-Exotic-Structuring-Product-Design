# ML-Assisted Exotic Structuring & Product Design

A single-asset equity-derivatives workflow that transforms a natural-language client request into a standardized mandate, recommends compatible products with machine learning, constructs product geometries, prices them, compares their risk/return profiles and produces an indicative structured-product proposal.

```text
CLIENT REQUEST
      ↓
NLP → STANDARDIZED CLIENT MANDATE
      ↓
RANDOM-FOREST PRODUCT RECOMMENDATION
      ↓
TOP 5 PRODUCT FAMILIES
      ↓
PRODUCT CONSTRUCTION
      ↓
PRICING UNDER Q + TERM SOLVING
      ↓
OUTCOME ANALYSIS UNDER P
      ↓
NORMALIZED MULTI-METRIC RANKING
      ↓
TOP 3
      ↓
GREEKS + STRESS TESTS + PRODUCT RISKS
      ↓
INDICATIVE TERM SHEET
```

## Project Scope

This repository represents the **V1** of the project, intentionally kept focused on a controlled and fully defensible equity-derivatives universe.

- Equity structured products on **one underlying at a time**.
- Same-currency transactions only; the current demo scope uses **USD**.
- No basket, worst-of, cross-asset correlation, FX or quanto modelling in V1.
- Supported demo underlyings: **META, NVDA, AAPL, MSFT, AMZN, TSLA and S&P 500**.
- Supported products: **Vanilla, Digital, Barrier, Reverse Convertible, ELN, Athena, Phoenix, Memory Phoenix, Range Accrual and Strip of Digitals**.

The objective of V1 is to keep the pricing, construction and risk workflow **simple, consistent and fully explainable**, rather than expanding the product universe before the underlying market-data and modelling assumptions are robust.

A **V2 is currently being prepared** to extend the framework to:
- **multi-asset structures**, including baskets and worst-of payoffs;
- **cross-asset correlation modelling**;
- **multiple currencies**;
- **FX and quanto effects**;
- a broader catalogue of supported structured products.

## Natural-Language Processing pipeline

The NLP layer converts free-form client language into the structured fields required by the recommendation and product-design modules.

```text
RAW CLIENT REQUEST
        ↓
Light text cleaning
        ↓
BERT native subword tokenizer
        ↓
BERT Transformer Encoder
        ↓
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
TOKEN CLASSIFICATION HEAD              SEQUENCE CLASSIFICATION HEADS
(NER / Slot Filling)                   (Request-level intent)
        │                                      │
        ├─ UNDERLYING                          ├─ DIRECTIONAL_VIEW
        ├─ THEME                               ├─ RETURN_TYPE
        ├─ TARGET_RETURN                       ├─ INCOME_PREFERENCE
        ├─ MATURITY                            ├─ AUTOCALL_ACCEPTANCE
        ├─ NOTIONAL                            ├─ RISK_APPETITE
        ├─ CURRENCY                            ├─ COMPLEXITY_TOLERANCE
        ├─ DOWNSIDE_TOLERANCE                  └─ MEMORY_REQUESTED
        ├─ CAPITAL_PROTECTION
        └─ PRODUCT
                    │
                    ▼
        DETERMINISTIC NORMALIZATION
                    │
        Amazon → AMZN
        "two years" → 2.0
        "USD 5m" → USD + 5,000,000
        "10%" → 0.10
        "digital option" → Digital
        "tech" → Technology
                    │
                    ▼
              CLIENT MANDATE
```

`src/nlp_model.py` defines the BERT encoder and task-specific heads. `src/nlp.py` performs inference, normalization and `ClientMandate` construction.

## Product recommendation

The recommendation layer uses a fixed `RandomForestClassifier` trained on pairwise **client mandate × candidate product** observations stored in:

```text
data/model_training/recommender_training.csv
```

For each supported product, the model estimates a suitability score and returns the Top 5 candidates:

```text
Recommendation Score = P(selected = 1 | mandate, product features)
```

The operational input remains `data/raw/client_requests.csv`; it is separate from the model-training dataset.

## Product construction

`src/construction.py` answers one specific question:

> **Which contractual geometry of the recommended product should be tested?**

The mandate fixes the underlying, maturity, notional and currency. The construction layer generates a small set of feasible product terms around that mandate, such as coupon barriers, autocall barriers, protection levels or strikes depending on the product family.

## Pricing

Pricing is performed under the risk-neutral measure **Q** with product-specific methods:

- closed-form Black-Scholes where appropriate;
- decomposition for notes such as Reverse Convertibles and ELNs;
- Monte Carlo for path-dependent autocallable structures and other non-closed-form payoffs.

`src/market.py` builds the market snapshot used by every pricer: spot, USD rate, implied volatility and dividend yield.

## Outcome analysis and final ranking

`src/outcome_analysis.py` simulates the client payoff distribution under the physical measure **P**. The scenario assumption uses:

```text
μ = r_f + 4%
```

The comparison uses four indicators:

1. **Recommendation Score** — higher is better;
2. **Sharpe Ratio** — higher is better;
3. **Expected Shortfall 95%** — lower is better;
4. **Probability of Loss** — lower is better.

The four metrics are min-max normalized across the priced candidates. Expected Shortfall and Probability of Loss are inverted after normalization, then all four scores receive equal weight:

```text
Final Score = 25% Recommendation
            + 25% Sharpe
            + 25% Inverted ES95
            + 25% Inverted Probability of Loss
```

The Sharpe Ratio is retained as a familiar risk-adjusted metric while ES95 and Probability of Loss provide additional information for asymmetric structured-product payoff distributions.

## Risk analysis

The final candidates are analyzed through:

- Delta;
- Gamma;
- Vega;
- spot stress scenarios;
- volatility stress scenarios;
- product-specific qualitative risks such as barrier, gap/hedging, skew, autocall/path-dependency and model risk.

## Market data

Spot can be retrieved from Yahoo Finance. Rates, implied-volatility and dividend-yield inputs are intentionally stored as **synthetic market data** under:

```text
data/market_data/synthetic/
```

This keeps the current project focused on the structuring workflow rather than pretending to provide production-grade calibration. The folder contains its own README documenting the assumptions and the planned market-data V2 scope.

## Repository map

| Path | Purpose |
|---|---|
| `data/raw/client_requests.csv` | Operational natural-language client requests. |
| `data/nlp_training/` | Annotated data for BERT token and sequence classification. |
| `data/model_training/` | Pairwise training data for the Random-Forest product recommender. |
| `data/market_data/synthetic/` | Rates, IV and dividend assumptions used by the pricing layer. |
| `data/reference/` | Product and qualitative-risk libraries. |
| `src/nlp_model.py` | BERT encoder + token / sequence classification heads. |
| `src/nlp.py` | NLP inference, deterministic normalization and ClientMandate assembly. |
| `src/recommendation_model.py` | Random-Forest product recommendation. |
| `src/construction.py` | Generates feasible contractual geometries for each recommended product. |
| `src/market.py` | Builds the single-asset market snapshot. |
| `src/pricing/` | Product-specific pricing under Q. |
| `src/outcome_analysis.py` | Physical-scenario payoff simulation and outcome metrics. |
| `src/ranking.py` | Metric normalization and final candidate ranking. |
| `src/risk_engine.py` | Greeks and preset MTM stress tests. |
| `src/term_sheet.py` | Product-term summary and indicative markdown term sheet. |
| `src/pipeline.py` | End-to-end orchestration. |
| `app.py` | Streamlit interface. |

## Run

```bash
pip install -r requirements.txt
python train_nlp.py
python train_recommender.py
streamlit run app.py
```

CLI example:

```bash
python runner/run_pipeline.py --client-id CL001 --no-yfinance --mc-paths 50000
```

## Positioning

This repository is a structured-product decision-support project rather than a production bank pricing or risk platform. Its scope is intentionally kept single-asset and same-currency so that product construction, pricing mechanics, client-mandate interpretation and risk analysis remain explicit and auditable.
