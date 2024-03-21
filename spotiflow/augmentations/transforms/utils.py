import torch 

def _filter_points_idx(pts: torch.Tensor, shape: tuple[int]) -> torch.Tensor:
    """returns indices of points that are within the image boundaries"""
    if pts.shape[-1] == len(shape):
        return torch.all(torch.logical_and(pts >= 0, pts < torch.tensor(shape, device=pts.device)), dim=-1)
    elif pts.shape[-1] == len(shape)+1: # Last dimension is class label, ignore it
        # Ignore last dimension, then add it back
        return torch.all(torch.logical_and(pts[..., :-1] >= 0, pts[..., :-1] < torch.tensor(shape, device=pts.device)), dim=-1)
    else:
        raise ValueError(f"Points should have shape (N, {len(shape)}) or (N, {len(shape)+1}) if class labels are given")


def _flatten_axis(ndim, axis=None):
    """Adapted from https://github.com/stardist/augmend/blob/main/augmend/transforms/affine.py not to depend on numpy
    converts axis to a flatten tuple
    e.g.
    flatten_axis(3, axis = None) = (0,1,2)
    flatten_axis(4, axis = (-2,-1)) = (2,3)
    """

    # allow for e.g. axis = -1, axis = None, ...
    all_axis = tuple(range(ndim))

    if axis is None:
        axis = tuple(all_axis)
    else:
        if isinstance(axis, int):
            axis = [axis, ]
        elif isinstance(axis, tuple):
            axis = list(axis)
        if max(axis) > max(all_axis):
            raise ValueError("axis = %s too large" % max(axis))
        axis = tuple([all_axis[i] for i in axis])
    return axis


def _generate_img_from_points(points, shape, sigma=1.):
    """Adapted from https://github.com/weigertlab/spotipy-torch/blob/main/spotipy_torch/utils/utils.py"""
    import numpy as np
    from scipy.spatial.distance import cdist
    import networkx as nx
    import scipy.ndimage as ndi
    def _filter_shape(points, shape, idxr_array=None):
        """  returns all values in "points" that are inside the shape as given by the indexer array
            if the indexer array is None, then the array to be filtered itself is used
        """
        if idxr_array is None:
            idxr_array = points.copy()
        assert idxr_array.ndim==2 and idxr_array.shape[1]==2
        idx = np.all(np.logical_and(idxr_array >= 0, idxr_array < np.array(shape)), axis=1)
        return points[idx]

    x = np.zeros(shape, np.float32)
    points = np.asarray(points).astype(np.int32)
    assert points.ndim==2 and points.shape[1]==2

    points = _filter_shape(points, shape)

    if len(points)==0:
        return x 
    D = cdist(points, points)
    A = D < 8*sigma+1
    np.fill_diagonal(A, False)
    G = nx.from_numpy_array(A)
    x = np.zeros(shape, np.float32)
    while len(G)>0:
        inds = nx.maximal_independent_set(G)
        gauss = np.zeros(shape, np.float32)
        gauss[tuple(points[inds].T)] = 1
        g = ndi.gaussian_filter(gauss, sigma, mode=  "constant")
        g /= np.max(g)
        x = np.maximum(x,g)
        G.remove_nodes_from(inds)   
    return x