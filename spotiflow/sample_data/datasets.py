import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ..utils import NotRegisteredError, get_data
from ..utils.get_file import get_file


@dataclass
class RegisteredDataset:
    """
    Dataclass to store information about a registered dataset.

    url: the url of the zipped dataset folder
    md5_hash: the md5 hash of the zipped dataset folder
    """

    url: str
    md5_hash: str
    is_3d: bool


def list_registered():
    return list(_REGISTERED.keys())

def _default_cache_dir():
    default_cache_dir = os.getenv("SPOTIFLOW_CACHE_DIR", None)
    if default_cache_dir is None:
        return Path("~").expanduser() / ".spotiflow" / "datasets"
    default_cache_dir = Path(default_cache_dir)
    if default_cache_dir.stem != "datasets":
        default_cache_dir = default_cache_dir / "datasets"
    return default_cache_dir


def get_training_datasets_path(name: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Downloads and extracts the training dataset with the given name.
    The dataset is downloaded to the given cache_dir. If not given, it
    will be downloaded to ~/.spotiflow/datasets and extracted to ~/.spotiflow/datasets/name.
    """
    if name not in _REGISTERED:
        raise NotRegisteredError(f"No training dataset named {name} found. Available datasets: {','.join(sorted(list_registered()))}")
    dataset = _REGISTERED[name]
    path = Path(
        get_file(
            fname=f"{name}.zip",
            origin=dataset.url,
            file_hash=dataset.md5_hash,
            cache_dir=_default_cache_dir() if cache_dir is None else cache_dir,
            cache_subdir="",
            extract=True,
        )
    )
    return path.parent / name

def load_dataset(name: str, include_test: bool=False, cache_dir: Optional[Union[Path, str]] = None):
    """
    Downloads and extracts the training dataset with the given name.
    The dataset is downloaded to ~/.spotiflow/datasets and extracted to ~/.spotiflow/datasets/name.

    Args:
        name (str): the name of the dataset to load.
        include_test (bool, optional): whether to include the test set in the returned data. Defaults to False.
        cache_dir (Optional[Union[Path, str]], optional): directory to cache the model. Defaults to None. If None, will use the default cache directory (given by the env var SPOTIFLOW_CACHE_DIR if set, otherwise ~/.spotiflow).
    """
    if name not in _REGISTERED:
        raise NotRegisteredError(f"No training dataset named {name} found. Available datasets: {','.join(sorted(list_registered()))}")
    if cache_dir is not None and isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    dataset = _REGISTERED[name]
    path = get_training_datasets_path(name, cache_dir=cache_dir)
    return get_data(path, include_test=include_test, is_3d=dataset.is_3d)


_REGISTERED = {
    "synth_complex": RegisteredDataset(
        url="https://drive.switch.ch/index.php/s/aWdxUHULLkLLtqS/download",
        md5_hash="5f44b03603fe1733ac0f2340a69ae238",
        is_3d=False,
    ),
    "merfish": RegisteredDataset(
        url="https://drive.switch.ch/index.php/s/fsjOypn4ICpSF2w/download",
        md5_hash="17fcdbd12cc71630e4f49652ded837c7",
        is_3d=False,
    ),
    "synth_3d": RegisteredDataset(
        url="https://drive.switch.ch/index.php/s/EemgJK1Bno8c3n4/download",
        md5_hash="f1715515763288362ee3351caca02825",
        is_3d=True,
    ),
}
