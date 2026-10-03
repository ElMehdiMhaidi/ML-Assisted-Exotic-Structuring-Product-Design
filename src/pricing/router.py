"""Route each supported single-asset product family to its reference pricer."""
from .reverse_convertible import ReverseConvertiblePricer
from .eln import ELNPricer
from .digital import DigitalPricer
from .autocall import AutocallPricer
from .range_accrual import RangeAccrualPricer
from .barrier import BarrierPricer
from .strip_digitals import StripDigitalsPricer
from .vanilla import VanillaPricer


def get_pricer(product_family):
    if product_family == "Reverse Convertible": return ReverseConvertiblePricer()
    if product_family == "ELN": return ELNPricer()
    if product_family in {"Athena", "Phoenix"}: return AutocallPricer()
    if product_family == "Vanilla": return VanillaPricer()
    if product_family == "Digital": return DigitalPricer()
    if product_family == "Range Accrual": return RangeAccrualPricer()
    if product_family == "Barrier": return BarrierPricer()
    if product_family == "Strip of Digitals": return StripDigitalsPricer()
    raise ValueError(f"No supported pricer for {product_family}")
