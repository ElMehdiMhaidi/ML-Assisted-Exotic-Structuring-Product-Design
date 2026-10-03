# Synthetic market data

The rates, implied-volatility and dividend-yield files in this folder are intentionally synthetic market inputs used to keep the project focused on structured-product construction, pricing, recommendation and risk analysis.

Current inputs:

- `rates_curve.csv` — USD rate term structure;
- `implied_vol_surface.csv` — single-name implied-volatility grid;
- `dividend_yields.csv` — dividend-yield assumptions by underlying.

The application can retrieve the latest spot through Yahoo Finance, while the remaining market parameters come from these controlled files.

This choice is explicit: the current project does not claim production-grade market-data calibration. The planned **market-data V2** will focus on richer external inputs, implied-volatility interpolation / surface construction and more robust rate and dividend term structures, while keeping the single-asset same-currency scope.
