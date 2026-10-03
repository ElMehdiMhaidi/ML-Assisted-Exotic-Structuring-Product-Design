"""Train the single-asset Random-Forest recommender.

Input:
    data/model_training/recommender_training.csv

The CSV keeps its existing pairwise schema:
    one row = one (client mandate, candidate product) pair
    selected = 1/0 suitability label

The model filters the file to the single-asset project scope before training.
Training uses the fixed Random-Forest configuration defined in src/recommendation_model.py.
"""

from src.recommendation_model import train_and_save


if __name__ == "__main__":
    _, metrics = train_and_save()
    print(metrics)
