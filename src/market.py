"""Build the single-asset MarketSnapshot used by all pricers.

Project scope:
- one USD equity/index underlying per request
- live/latest spot attempted with yfinance, deterministic fallback if unavailable
- synthetic demo USD rate curve
- synthetic ATM implied volatility selected at the nearest maturity
- synthetic dividend yield by ticker

No baskets, correlations, FX or quanto adjustments are modelled in the current project scope.
"""
import pandas as pd
import numpy as np
from .schemas import MarketSnapshot

SUPPORTED_TICKERS = {"META","NVDA","AAPL","MSFT","AMZN","TSLA","^GSPC"}
FALLBACK_SPOTS = {"META":590.0,"NVDA":180.0,"AAPL":260.0,"MSFT":520.0,"AMZN":230.0,"TSLA":440.0,"^GSPC":6800.0}


def _try_yfinance(ticker):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def load_market_snapshot(underlyings, maturity_years=2.0, use_yfinance=True):
    if len(underlyings or []) != 1:
        raise ValueError("Project scope supports exactly one underlying per client request.")
    ticker = underlyings[0]
    if ticker not in SUPPORTED_TICKERS:
        raise ValueError(f"Unsupported underlying: {ticker}. Supported: {sorted(SUPPORTED_TICKERS)}")

    live_spot = _try_yfinance(ticker) if use_yfinance else None
    spots = {ticker: float(live_spot if live_spot is not None else FALLBACK_SPOTS[ticker])}

    rates_df = pd.read_csv("data/market_data/synthetic/rates_curve.csv")
    rates = {float(r.tenor_years): float(r.rate) for _, r in rates_df.iterrows()}

    vol_df = pd.read_csv("data/market_data/synthetic/implied_vol_surface.csv")
    sub = vol_df[(vol_df["ticker"] == ticker) & (vol_df["moneyness"] == 1.0)].copy()
    if sub.empty:
        raise ValueError(f"No synthetic ATM IV data available for {ticker}")
    sub["distance"] = (sub["maturity_years"] - float(maturity_years)).abs()
    atm_iv = float(sub.sort_values("distance").iloc[0]["iv"])

    div_df = pd.read_csv("data/market_data/synthetic/dividend_yields.csv")
    d = div_df[div_df["ticker"] == ticker]
    if d.empty:
        raise ValueError(f"No dividend-yield input available for {ticker}")
    dividend = float(d.iloc[0]["dividend_yield"])

    return MarketSnapshot(spots=spots, rates=rates, vols={ticker: atm_iv}, dividends={ticker: dividend})


def nearest_rate(snapshot, maturity):
    """Linear interpolation of the demo USD rate curve used by the pricers."""
    tenors = np.array(sorted(snapshot.rates), dtype=float)
    values = np.array([snapshot.rates[t] for t in tenors], dtype=float)
    return float(np.interp(float(maturity), tenors, values))
