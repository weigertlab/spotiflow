# ruff: noqa: F401
from .spots import SpotsDataset, collate_spots


def test_data_hybiss_2d():
    from tifffile import imread
    from pathlib import Path

    return imread(Path(__file__).parent / "_example_data" / "hybiss_2d.tif")
