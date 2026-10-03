from .reverse_convertible import ReverseConvertiblePricer

class ELNPricer(ReverseConvertiblePricer):
    """ELN pricing uses the bond + short put decomposition defined for the supported note template."""
    pass
