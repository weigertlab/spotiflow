import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils import NotRegisteredError
from ..utils.get_file import get_file


@dataclass
class RegisteredModel:
    """
    Dataclass to store information about a registered model.

    url: the url of the zipped model folder
    md5_hash: the md5 hash of the zipped model folder
    """

    url: str
    md5_hash: str
    is_3d: bool

def list_registered():
    return list(_REGISTERED.keys())

def _default_url_root():
    return "https://github.com/weigertlab/spotiflow-models/releases/download/0.6.0"


def _default_cache_dir():
    default_cache_dir = os.getenv("SPOTIFLOW_CACHE_DIR", None)
    if default_cache_dir is None:
        return Path("~").expanduser() / ".spotiflow" / "models"
    default_cache_dir = Path(default_cache_dir)
    if default_cache_dir.stem != "models":
        default_cache_dir = default_cache_dir / "models"
    return default_cache_dir

def get_pretrained_model_path(name: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Downloads and extracts the pretrained model with the given name.
    The model is downloaded and extracted in the given cache_dir. If it is not given, it will be
    downloaded to ~/.spotiflow and extracted to ~/.spotiflow/models/name.
    """
    if name not in _REGISTERED:
        raise NotRegisteredError(f"No pretrained model named {name} found. Available models: {','.join(sorted(list_registered()))}")
    model = _REGISTERED[name]
    path = Path(
        get_file(
            fname=f"{name}.zip",
            origin=model.url,
            file_hash=model.md5_hash,
            cache_dir=_default_cache_dir() if cache_dir is None else cache_dir,
            cache_subdir="",
            extract=True,
        )
    )
    return path.parent / name


_REGISTERED = {
    "hybiss": RegisteredModel(
        url=f"{_default_url_root()}/hybiss.zip",
        md5_hash="254afa97c137d0bd74fd9c1827f0e323",
        is_3d=False,
    ),
    "general": RegisteredModel(
        url=f"{_default_url_root()}/general.zip",
        md5_hash="9dd31a36b737204e91b040515e3d899e",
        is_3d=False,
    ),
    "synth_complex": RegisteredModel(
        url=f"{_default_url_root()}/synth_complex.zip",
        md5_hash="d692fa21da47e4a50b4c52f49442508b",
        is_3d=False,
    ),
    "synth_3d": RegisteredModel(
        url=f"{_default_url_root()}/synth_3d.zip",
        md5_hash="a031f1284590886fbae37dc583c0270d",
        is_3d=True,
    ),
    "smfish_3d": RegisteredModel(
        url=f"{_default_url_root()}/smfish_3d.zip",
        md5_hash="c5ab30ba3b9ccb07b4c34442d1b5b615",
        is_3d=True,
    ),
    "fluo_live": RegisteredModel(
        url=f"{_default_url_root()}/fluo_live.zip",
        md5_hash="7dc7e633704e3fc0d9a1469bafea9442",
        is_3d=False,
    ),
}
