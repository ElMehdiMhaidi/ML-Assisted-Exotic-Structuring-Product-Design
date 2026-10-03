# NLP model training data

`client_mandate_nlp.jsonl` trains the transformation from **Client Request → ClientMandate**.

Each observation contains:

- `text`: free-form single-asset client request;
- `entities`: character spans used by the token-classification / NER head;
- `sequence_labels`: request-level targets used by the sequence-classification heads;
- `expected_mandate`: normalized reference values used to audit the generated examples.

## Token-classification slots

`UNDERLYING`, `THEME`, `TARGET_RETURN`, `MATURITY`, `NOTIONAL`, `CURRENCY`,
`DOWNSIDE_TOLERANCE`, `CAPITAL_PROTECTION`, `PRODUCT`.

The spans are converted to BIO labels during `train_nlp.py`.

## Sequence-classification targets

`directional_view`, `return_type`, `income_preference`, `autocall_acceptance`,
`risk_appetite`, `complexity_tolerance`, `memory_requested`.

Preference labels support `not_specified`, so the absence of a preference is not interpreted as rejection.

The dataset contains 4,000 single-asset, USD examples and can be regenerated with `generate_nlp_training_data.py`.
