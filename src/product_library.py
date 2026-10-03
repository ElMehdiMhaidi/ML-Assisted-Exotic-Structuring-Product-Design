import pandas as pd

def load_product_library(path="data/reference/product_library.csv"):
    return pd.read_csv(path)

def load_risk_library(path="data/reference/risk_library.csv"):
    return pd.read_csv(path)
