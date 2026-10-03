from src.schemas import ProductSpec
from src.market import load_market_snapshot
from src.pricing.digital import DigitalPricer

def test_digital_price_positive():
    m=load_market_snapshot(["META"],1.0,use_yfinance=False)
    s=ProductSpec("Digital",["META"],1.0,1_000_000,"USD",{"strike_pct":1.0,"payout":0.1,"reference_spots":{"META":m.spots["META"]}})
    assert DigitalPricer().price(s,m,1000,42).fair_value>0
