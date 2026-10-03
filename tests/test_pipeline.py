def test_smoke_import():
    from src.pipeline import run_for_client
    assert callable(run_for_client)
