from src.schemas import ClientMandate,ProductRecommendation
from src.construction import generate_specs

def test_phoenix_generation_single_asset():
    m=ClientMandate("1","x","AM","",underlyings=["META"],maturity_years=2,notional=1e6,downside_tolerance=.35)
    specs=generate_specs(m,ProductRecommendation("Phoenix",.9),20)
    assert 1<=len(specs)<=20 and all(len(s.underlyings)==1 for s in specs)
