# Recommender model training data

`recommender_training.csv` is the labelled pairwise dataset used to train the Random-Forest product recommender.

Each row represents one **client mandate × candidate product** pair. The binary `selected` field indicates whether the candidate product is suitable for that mandate.

The model uses the mandate features and product capability fields already present in the CSV, filters them to the supported single-asset universe, and estimates:

```text
P(product suitable | client mandate, product features)
```

Operational client requests are stored separately in `data/raw/client_requests.csv`.
