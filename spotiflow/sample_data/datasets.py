from pathlib import Path
from ..utils import get_data
from dataclasses import dataclass

from ..utils import get_data, NotRegisteredError
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


def list_registered():
    return list(_REGISTERED.keys())


def _cache_dir():
    return Path("~").expanduser() / ".spotiflow" / "datasets"


def get_training_datasets_path(name: str):
    """
    Downloads and extracts the training dataset with the given name.
    The dataset is downloaded to ~/.spotiflow/datasets and extracted to ~/.spotiflow/datasets/name.
    """
    if name not in _REGISTERED:
        raise NotRegisteredError(f"No training dataset named {name} found. Available datasets: {','.join(sorted(list_registered()))}")
    dataset = _REGISTERED[name]
    path = Path(
        get_file(
            fname=f"{name}.zip",
            origin=dataset.url,
            file_hash=dataset.md5_hash,
            cache_dir=_cache_dir(),
            cache_subdir="",
            extract=True,
        )
    )
    return path.parent / name

def load_dataset(name: str, include_test: bool=False):
    """
    Downloads and extracts the training dataset with the given name.
    The dataset is downloaded to ~/.spotiflow/datasets and extracted to ~/.spotiflow/datasets/name.

    Args:
        name (str): the name of the dataset to load.
        include_test (bool, optional): whether to include the test set in the returned data. Defaults to False.
    """
    path = get_training_datasets_path(name)
    return get_data(path, include_test=include_test)


_REGISTERED = {
    "synth_complex": RegisteredDataset(
        url="https://drive.switch.ch/index.php/s/aWdxUHULLkLLtqS/download",
        md5_hash="5f44b03603fe1733ac0f2340a69ae238",
    ),
    "merfish": RegisteredDataset(
        url="https://drive.switch.ch/index.php/s/fsjOypn4ICpSF2w/download",
        md5_hash="17fcdbd12cc71630e4f49652ded837c7",
    ),
}
