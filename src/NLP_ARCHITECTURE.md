# Natural-Language Processing Architecture

The NLP layer converts a free-form client request into the standardized `ClientMandate` consumed by the recommendation and structuring workflow.

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

`src/nlp_model.py` defines the shared BERT encoder and classification heads. `src/nlp.py` loads the trained model, performs inference, normalizes extracted values and builds the final `ClientMandate`.

The Transformer encoder and the task-specific heads are fine-tuned jointly on the annotated NLP training dataset.
