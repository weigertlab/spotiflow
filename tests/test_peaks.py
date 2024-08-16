from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest
from spotiflow.utils import points_from_heatmap_flow, points_to_flow, points_to_prob
from spotiflow.utils.matching import points_matching


def round_trip(points:np.ndarray, grid: Tuple[int], sigma: float=1.5):
    """ Test round trip of points through the flow field and back"""
    points = np.asarray(points)
    ndim = points.shape[1]
    shape = (2*int(points.max()),)*ndim
    # get heatmap and flow 
    heatmap = points_to_prob(points, shape=shape, sigma=sigma, mode="max", grid=grid)
    flow = points_to_flow(points, shape, sigma=sigma, grid=grid)

    points_new = points_from_heatmap_flow(heatmap, flow, sigma=sigma, grid=grid)

    return SimpleNamespace(points=points, points_new=points_new, heatmap=heatmap, flow=flow, sigma=sigma)

@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("grid", (None, (2, 2)))
def test_prob_flow_roundtrip(ndim, grid, debug:bool=False):
    points = np.stack(np.meshgrid(*tuple(np.linspace(10,48,4) for _ in range(ndim)), indexing="ij"), axis=-1).reshape(-1, ndim)
    if ndim == 3 and grid is not None:
        grid = (*grid, 2)

    points = points + np.random.uniform(-1, 1, points.shape)
    if ndim == 2 and (grid is not None or (isinstance(grid, tuple) and any(g > 1 for g in grid))):
        with pytest.raises(NotImplementedError):
            _ = round_trip(points, grid=grid)
    else:
        out = round_trip(points, grid=grid)

        diff = points_matching(out.points, out.points_new).mean_dist

        print(f"Max diff: {diff:4f}")
        if debug: 
            import napari
            v = napari.Viewer()
            v.add_points(out.points, name="points", size=5, face_color="green")
            v.add_points(out.points_new, name="points_new", size=5, face_color="red")
        else: 
            assert diff < 1e-3, f"Max diff: {diff:4f}"        

    

if __name__ == "__main__":

    # works
    test_prob_flow_roundtrip(ndim=3, grid=None)

    # works
    test_prob_flow_roundtrip(ndim=3, grid=(2,2,2))
