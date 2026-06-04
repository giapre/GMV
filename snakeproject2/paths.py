from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data"
    DERIVATIVES = "/mnt/nasvep/SCZ_CHINA/paula/derivatives/derivatives/freesurfer"
    RESOURCES = ROOT / "resources"
    RESULTS = ROOT / "results"
    FIGURES = ROOT / "figures"
    SNAKEMAKE = ROOT / "snakeproject2"

    TYPE_OF_SWEEP = "double_rd2_less_ws_less_zd1_for_njdopa"


