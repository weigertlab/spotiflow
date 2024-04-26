import numpy as np
from types import SimpleNamespace
from spotiflow.utils import points_to_prob, points_to_flow, flow_to_vector, local_peaks, points_from_heatmap_flow
from spotiflow.utils.matching import points_matching

def round_trip(points:np.ndarray, grid:list[int], sigma=1.5):
    """ Test round trip of points through the flow field and back"""
    points = np.asarray(points)
    ndim = points.shape[1]
    shape = (2*int(points.max()),)*ndim
    # get heatmap and flow 
    heatmap=points_to_prob(points, shape=shape, sigma=sigma, mode="max", grid=grid)
    flow = points_to_flow(points, shape, sigma=sigma, grid=grid)

    points_new = points_from_heatmap_flow(heatmap, flow, sigma=sigma, grid=grid)

    return SimpleNamespace(points=points, points_new=points_new, heatmap=heatmap, flow=flow, sigma=sigma)


def test_prob_flow_roundtrip(ndim, grid, debug:bool=False):
    points = np.stack(np.meshgrid(*tuple(np.linspace(10,48,4) for _ in range(ndim)), indexing="ij"), axis=-1).reshape(-1, ndim)
    points = points + np.random.uniform(-1, 1, points.shape)
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

    return out
    

if __name__ == "__main__":

    # works
    out = test_prob_flow_roundtrip(ndim=3, grid=None)

    # works
    out = test_prob_flow_roundtrip(ndim=3, grid=(2,2,2))

    # doesnt work
    out = test_prob_flow_roundtrip(ndim=3, grid=(1,2,2))

