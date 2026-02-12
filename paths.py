from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"
    RESOURCES = ROOT / "resources"
    RESULTS = ROOT / "results"
    FIGURES = ROOT / "figures"

    DERIVATIVES = DATA / "derivatives"
    DEMOGRAPHICS = DATA / "demographics"

