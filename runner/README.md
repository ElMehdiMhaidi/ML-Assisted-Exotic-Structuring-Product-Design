# CLI runner

Run the complete structuring workflow from the project root:

```bash
python runner/run_pipeline.py --client-id CL001 --no-yfinance --mc-paths 50000
```

The runner writes `outputs/reports/<CLIENT_ID>_results.json` containing the parsed mandate, Random-Forest Top 5, final Top 3, ranking components and risk outputs.
