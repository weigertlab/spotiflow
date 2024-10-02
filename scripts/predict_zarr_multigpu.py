"""
Script to predict spots on a large Zarr volumes using multiple GPUs.
In a nutshell, overlapping tiles are dispatched to different GPUs, which process them.
The results of each GPU are finally gathered and written to a CSV file.

The script is designed to be run with torchrun, and highly suitable for cluster environments.
The distributed environment is automatically setup (but the number of GPUs, which should be specified).

For more info on the arguments, please run `python predict_zarr_multigpu.py -h`.

Example usage:
    torchrun --nproc_per_node=${n_gpus} predict_zarr_multigpu.py --input PATH/TO/ZARR --output PATH/TO/OUTPUT.csv --precomputed-percentiles X Y --dataloader-num-workers 2
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import torch
import torch.distributed as dist
from spotiflow.model import Spotiflow
from spotiflow.model.pretrained import list_registered
from spotiflow.utils import write_coords_csv

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

APPROX_TILE_SIZE = (256, 256, 256)


def get_percentiles(
    x: da.Array,
    pmin: float = 1.0,
    pmax: float = 99.8,
    max_samples: int = 1e5,
):
    n_skip = int(max(1, x.size // max_samples))
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        mi, ma = da.percentile(
            x.ravel()[::n_skip], (pmin, pmax), internal_method="tdigest"
        ).compute()
    return mi, ma


def normalize(x, mi, ma, eps: float = 1e-20):
    return (x - mi) / (ma - mi + eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="smfish_3d", help="Pre-trained model name or path."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path pointing to the input volume. Should be in Zarr format",
    )
    parser.add_argument(
        "--zarr-component",
        type=str,
        default=None,
        help="Zarr component to read from the Zarr file, if necessary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path pointing to the output CSV file where the detected spots will be written to.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Number of workers to use in the dataloader.",
    )
    parser.add_argument(
        "--prob-thresh",
        default=0.4,
        type=float,
        help="Probability threshold for spot detection.",
    )
    parser.add_argument(
        "--min-distance",
        default=1,
        type=int,
        help="Minimum distance between detections.",
    )
    parser.add_argument(
        "--precomputed-percentiles",
        nargs=2,
        default=None,
        help="If given, will use the precomputed percentiles instead of recomputing them.",
    )
    args = parser.parse_args()

    img = da.from_zarr(str(args.input), component=args.zarr_component)

    print(f"Array is {img.shape} (~{img.nbytes/1e9:.2f} GB)")
    print(f"Number of voxels: ~2^{int(np.log2(img.size))}")

    n_tiles = tuple(max(1, s // t) for s, t in zip(img.shape, APPROX_TILE_SIZE))

    distributed = torch.cuda.device_count() > 1
    torch.distributed.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    if gpu_id == 0:
        print("Distributed session successfully initialized")
        t0 = time.monotonic()

    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")

    mi_ma = torch.zeros(2, device=device)

    if gpu_id == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.precomputed_percentiles is None:
            print("Computing percentiles...")
            log.warning(
                "It is highly recommended to precompute percentiles and pass them as an argument to avoid Dask hijacking threads.\nIf the execution seems to halt, please re-run with the --precomputed-percentiles argument set to the percentiles computed here."
            )
            1/0
            t0_p = time.monotonic()
            mi, ma = get_percentiles(img.astype(np.float32), 1, 99.8)
            te_p = time.monotonic()
            print(f"Percentiles ({mi:.2f}, {ma:.2f}) computed in {te_p-t0_p:.2f} s")
        else:
            mi, ma = tuple(float(x) for x in args.precomputed_percentiles)
            print(f"Using precomputed percentiles ({mi:.2f}, {ma:.2f})...")
        mi_ma = torch.tensor([mi, ma], device=device, dtype=torch.float32)

    if args.model not in list_registered():
        args.model = Path(args.model)
        model = Spotiflow.from_folder(args.model)
    else:
        model = Spotiflow.from_pretrained(args.model)
    model.to(device)
    model.eval()
    model = torch.compile(model)

    dist.barrier()
    dist.broadcast(mi_ma, src=0)
    dist.barrier()

    p1, p998 = mi_ma[0].item(), mi_ma[1].item()
    del mi_ma

    spots, _ = model.predict(
        img,
        subpix=True,
        n_tiles=n_tiles,
        min_distance=1,
        prob_thresh=args.prob_thresh,
        device=None,
        normalizer=lambda x: normalize(x, p1, p998),
        distributed_params={
            "gpu_id": gpu_id,
            "num_workers": args.dataloader_num_workers,
            "num_replicas": int(os.environ["WORLD_SIZE"]),
        },
    )

    spots = torch.from_numpy(spots).to(device)
    # Collect shapes
    if gpu_id == 0:
        all_shapes = [spots.shape]  # Start with the root process's own tensor shape
        for src_gpu_id in range(1, dist.get_world_size()):
            shape_tensor = torch.zeros(2, dtype=torch.long, device=device)
            dist.recv(tensor=shape_tensor, src=src_gpu_id)
            all_shapes.append(tuple(shape_tensor.tolist()))
    else:
        # Non-root processes: Send tensor shape to root process
        shape_tensor = torch.tensor(spots.shape, dtype=torch.long, device=device)
        dist.send(tensor=shape_tensor, dst=0)

    # Send based on shape
    if gpu_id == 0:
        all_spots = [spots]  # Start with the root process's own tensor
        for idx, src_gpu_id in enumerate(range(1, dist.get_world_size())):
            recv_tensor = torch.zeros(
                all_shapes[idx + 1], device=device, dtype=spots.dtype
            )  # Use collected shapes
            dist.recv(tensor=recv_tensor, src=src_gpu_id)
            all_spots.append(recv_tensor)
    else:
        # Non-root processes: Send tensor to root process
        dist.send(tensor=spots, dst=0)

    # Concat at root and write
    if gpu_id == 0:
        all_spots = torch.cat(all_spots, dim=0).cpu().numpy()
        print("All spots shape is", all_spots.shape)
        print("Writing...")

        write_coords_csv(
            all_spots,
            str(args.output),
        )
        print("Written!")
        te = time.monotonic()
        print(f"Total ellapsed time: {te-t0:.2f} s")
    dist.barrier()
    sys.exit(0)
