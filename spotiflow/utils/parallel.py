"""
Parallel processing utilities for inference-only with CNNs.
Wraps csbdeep.internals.predict.tile_iterator (https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349).
DistributedEvalSampler adapted from https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
"""
from typing import Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import torch
import torch.distributed as dist
from csbdeep.internals.predict import tile_iterator as csbd_tile_iterator
from torch.utils.data import Dataset, Sampler

def tile_iterator(x: Union[np.ndarray, da.Array],
                  n_tiles: Tuple[int, ...],
                  block_sizes: Tuple[int, ...],
                  n_block_overlaps: Tuple[int, ...],
                  guarantee: Literal["size", "n_tiles"]="size",
                  rank_id: int=-1,
                  num_replicas: int=None,
                  dataloader_kwargs: Optional[dict]=None) -> torch.utils.data.DataLoader:
    """Wrapper function to create a DataLoader for tiled prediction on multiple GPUs.
       Based on CSBDeep's tile iterator (https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349) as well as
       DistributedEvalSampler (https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py).

    Args:
        x (Union[np.ndarray, da.Array]): input array
        n_tiles (Tuple[int, ...]): see https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349
        block_sizes (Tuple[int, ...]): see https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349
        n_block_overlaps (Tuple[int, ...]): see https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349
        guarantee (str, optional): see https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349. Defaults to "size".
        rank_id (int, optional): rank id to retrieve the iterator for. Defaults to -1.
        num_replicas (int, optional): number of replicas. If None, will set to number of GPUs available. Defaults to None.
        dataloader_kwargs (Optional[dict], optional): additional arguments for the DataLoader. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: disjoint tile iterator for each rank
    """
    assert rank_id >= 0, "rank_id must be >= 0"
    ds = _TiledDataset(
        img=x,
        n_tiles=n_tiles,
        block_sizes=block_sizes,
        n_block_overlaps=n_block_overlaps,
        guarantee=guarantee,
    )
    sampler = _DistributedEvalSampler(ds, num_replicas=num_replicas if num_replicas is not None else torch.cuda.device_count(), rank=rank_id)
    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    
    if dataloader_kwargs.get("batch_size", None) is not None:
        raise ValueError("batch_size dataloader arg must be None")

    if dataloader_kwargs.get("shuffle", False):
        raise ValueError("shuffle dataloader arg must be None")


    num_workers = dataloader_kwargs.get("num_workers", 0)
    pin_memory = dataloader_kwargs.get("pin_memory", True)
    prefetch_factor = dataloader_kwargs.get("prefetch_factor", 2 if num_workers > 0 else None)

    iter_tiles = torch.utils.data.DataLoader(ds, sampler=sampler, shuffle=False, batch_size=None, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return iter_tiles


class _TiledDataset(Dataset):
    """Auxiliary dataset class for parallel tiled prediction on n-d arrays.
       Wraps csbdeep.internals.predict.tile_iterator (https://github.com/CSBDeep/CSBDeep/blob/b4edf699c7a6d2f7ddfcaf973063d15748daf825/csbdeep/internals/predict.py#L349)
       For the arguments, please check the documentation of csbdeep.internals.predict.tile_iterator.
    """
    def __init__(self, img, n_tiles, block_sizes, n_block_overlaps, guarantee):
        self._tile_iterator = tuple(csbd_tile_iterator(img, n_tiles, block_sizes, n_block_overlaps, guarantee))
        self._img = img

    @property
    def img(self):
        return self._img

    @property
    def arr_type(self):
        return self._arr_type

    @property
    def tile_iterator(self):
        return self._tile_iterator

    @img.setter
    def img(self, _):
        raise RuntimeError('Cannot dynamically change class attribute. Create a new instance of TiledDataset instead.')

    @tile_iterator.setter
    def tile_iterator(self, _):
        raise RuntimeError('Cannot dynamically change class attributes. Create a new instance of TiledDataset instead.')
    
    @arr_type.setter
    def arr_type(self, _):
        raise RuntimeError('Cannot dynamically change class attributes. Create a new instance of TiledDataset instead.')

    def __len__(self):
        return len(self._tile_iterator)

    def __getitem__(self, idx: int):
        tile, s_src, s_dst = self._tile_iterator[idx]
        return tile.compute(), tuple(s_src), tuple(s_dst)

class _DistributedEvalSampler(Sampler):
    r"""
    Adapted from https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py

    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
