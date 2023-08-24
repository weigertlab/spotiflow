import numpy as np
import warnings
from csbdeep.utils import normalize_mi_ma
from skimage.feature import corner_peaks, corner_subpix
import scipy.ndimage as ndi
import networkx as nx
from scipy.spatial.distance import cdist
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

import logging
import os
import wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def read_coords_csv(fname: str): 
    """ parses a csv file and returns correctly ordered points array     
    """
    df = pd.read_csv(fname)
    df = df.rename(columns = str.lower)
    cols = set(df.columns)

    col_candidates = (('axis-0', 'axis-1'), ('y','x'), ("Y", "X"))
    points = None 
    for possible_columns in col_candidates:
        if cols.issuperset(set(possible_columns)):
            points = df[list(possible_columns)].to_numpy()
            break 

    if points is None: 
        raise ValueError(f'could not get points from csv file {fname}')

    return points


def _filter_shape(points, shape, idxr_array=None):
    """  returns all values in "points" that are inside the shape as given by the indexer array
         if the indexer array is None, then the array to be filtered itself is used
    """
    if idxr_array is None:
        idxr_array = points.copy()
    assert idxr_array.ndim==2 and idxr_array.shape[1]==2
    idx = np.all(np.logical_and(idxr_array >= 0, idxr_array < np.array(shape)), axis=1)
    return points[idx]

def points_to_prob(points, shape, sigma = 1.5,  mode = "max"):
    """points are in (y,x) order"""


    x = np.zeros(shape, np.float32)
    points = np.round(points).astype(np.int32)
    assert points.ndim==2 and points.shape[1]==2

    points = _filter_shape(points, shape)

    if len(points)==0:
        return x 
    
    if mode == "max":
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
    else:
        raise ValueError(mode)

    return x


def prob_to_points(prob, prob_thresh=.5, min_distance = 2, subpix:bool=False, mode:str='skimage',exclude_border:bool=True):
    assert prob.ndim==2, "Wrong dimension of prob"
    if mode =='skimage':
        corners = corner_peaks(prob, min_distance = min_distance, threshold_abs = prob_thresh, threshold_rel=0, exclude_border=exclude_border)
        if subpix:
            print("using subpix")
            corners_sub = corner_subpix(prob, corners, window_size=3)
            ind = ~np.isnan(corners_sub[:,0])
            corners[ind] = corners_sub[ind].round().astype(int)
    elif mode=='fast':
        raise NotImplementedError("fast NMS mode not implemented yet")
        # corners = local_peaks(prob, min_distance = min_distance, threshold_abs = prob_thresh, exclude_border=exclude_border)
    else: 
        raise NotImplementedError(f'unknown mode {mode} (supported: "skimage")')
    return corners

def points_matching(p1, p2, cutoff_distance = 3, eps=1e-8):
    """ finds matching that minimizes sum of mean squared distances"""
    
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    if len(p1)==0 or len(p2)==0:
        D = np.zeros((0,0))
    else:
        D = cdist(p1,p2, metric='sqeuclidean')
        

    if D.size>0:
        D[D>cutoff_distance**2] = 1e10*(1+D.max())
    
    i,j = linear_sum_assignment(D)
    valid = D[i,j] <= cutoff_distance**2
    i,j = i[valid], j[valid]

    res = SimpleNamespace()
    
    tp = len(i)
    fp = len(p2)-tp
    fn = len(p1)-tp
    res.tp = tp
    res.fp = fp
    res.fn = fn

    # when there is no tp and we dont predict anything the accuracy should be 1 not 0
    tp_eps = tp+eps 

    res.accuracy  = tp_eps/(tp_eps+fp+fn) if tp_eps > 0 else 0
    res.precision = tp_eps/(tp_eps+fp) if tp_eps > 0 else 0
    res.recall    = tp_eps/(tp_eps+fn) if tp_eps > 0 else 0
    res.f1        = (2*tp_eps)/(2*tp_eps+fp+fn) if tp_eps > 0 else 0
    
    res.dist = np.sqrt(D[i,j])
    res.mean_dist = np.mean(res.dist) if len(res.dist)>0 else 0

    res.false_negatives = tuple(set(range(len(p1))).difference(set(i)))
    res.false_positives = tuple(set(range(len(p2))).difference(set(j)))
    res.matched_pairs = tuple(zip(i,j)) 
    return res

    
def points_matching_dataset(p1s, p2s, cutoff_distance=3, by_image=True, eps=1e-8):
    """ 
    by_image is True -> metrics are computed by image and then averaged
    by_image is False -> TP/FP/FN are aggregated and only then are metrics computed
    """
    stats = tuple(points_matching(p1,p2,cutoff_distance=cutoff_distance, eps=eps) for p1,p2 in zip(p1s,p2s))


    if by_image:
        res = dict()
        for k, v in vars(stats[0]).items():
            if np.isscalar(v):
                res[k] = np.mean([vars(s)[k] for s in stats])
        return SimpleNamespace(**res)
    else:
        res = SimpleNamespace()
        res.tp = 0 
        res.fp = 0 
        res.fn = 0 


        for s in stats: 
            for k in ('tp','fp', 'fn'):
                setattr(res,k, getattr(res,k) + getattr(s, k))

        tp_eps = res.tp+eps
        res.accuracy  = tp_eps/(tp_eps+res.fp+res.fn) if tp_eps>0 else 0
        res.precision = tp_eps/(tp_eps+res.fp) if tp_eps>0 else 0
        res.recall    = tp_eps/(tp_eps+res.fn) if tp_eps>0 else 0
        res.f1        = (2*tp_eps)/(2*tp_eps+res.fp+res.fn) if tp_eps>0 else 0

        return res
        
    

def multiscale_decimate(y, decimate = (4,4), sigma = 1):
    if decimate==(1,1):
        return y
    assert y.ndim==len(decimate)
    from skimage.measure import block_reduce
    y = block_reduce(y, decimate, np.max)
    y = 2*np.pi*sigma**2*ndi.gaussian_filter(y,sigma)
    y = np.clip(y,0,1)
    return y

    
def center_pad(x, shape, mode = "reflect"):
    """ pads x to shape , inverse of center_crop"""
    if x.shape == shape:
        return x, tuple((0,0) for _ in x.shape)
    if not all([s1<=s2 for s1,s2 in zip(x.shape,shape)]):
        raise ValueError(f"shape of x {x.shape} is larger than final shape {shape}")
    diff = np.array(shape)- np.array(x.shape)
    pads = tuple((int(np.ceil(d/2)),d-int(np.ceil(d/2))) if d>0 else (0,0) for d in diff)
    return np.pad(x,pads, mode=mode), pads


def center_crop(x, shape):
    """ crops x to shape, inverse of center_pad 

    y = center_pad(x,shape)
    z = center_crop(y,x.shape)
    np.allclose(x,z)
    """
    if x.shape == shape:
        return x
    if not all([s1>=s2 for s1,s2 in zip(x.shape,shape)]):
        raise ValueError(f"shape of x {x.shape} is smaller than final shape {shape}")
    diff = np.array(x.shape)- np.array(shape)
    ss = tuple(slice(int(np.ceil(d/2)),s-d+int(np.ceil(d/2))) if d>0 else slice(None) for d,s in zip(diff,x.shape))
    return x[ss]


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def str2scalar(dtype):
    def _f(v):
        if v.lower() == "none":
            return None
        else:
            return dtype(v)

    return _f


def normalize(x: np.ndarray, pmin=1, pmax=99.8, subsample:int = 1, clip = False, ignore_val=None):
    """
    normalizes a 2d image with the additional option to ignore a value
    """

    # create subsampled version to compute percentiles
    ss_sample = tuple(slice(None,None, subsample) if s>42*subsample else slice(None,None) for s in x.shape)

    y = x[ss_sample]

    if ignore_val is not None:
        mask = y!=ignore_val
    else: 
        mask = np.ones(y.shape, dtype=bool)

    if not np.any(mask):
        return normalize_mi_ma(x, ignore_val, ignore_val, clip=clip)

    mi, ma = np.percentile(y[mask],(pmin, pmax))
    return normalize_mi_ma(x, mi, ma, clip=clip)    



def normalize_fast2d(x, pmin=1, pmax=99.8, dst_range=(0,1.), clip = False, sub = 4, blocksize=None, order=1, ignore_val=None):
    """
    normalizes a 2d image
    if blocksize is not None (e.g. 512), computes adaptive/blockwise percentiles
    """
    assert x.ndim==2

    out_slice = slice(None),slice(None)
    
    if blocksize is None:
        x_sub = x[::sub,::sub]
        if ignore_val is not None:
            x_sub = x_sub[x_sub!=ignore_val]
        mi, ma = np.percentile(x_sub,(pmin,pmax))#.astype(x.dtype)
        print(f"normalizing_fast with mi = {mi:.2f}, ma = {ma:.2f}")        
    else:
        from csbdeep.internals.predict import tile_iterator_1d
        try:
            import cv2
        except ImportError:
            raise ImportError("normalize_adaptive() needs opencv, which is missing. Please install it via 'pip install opencv-python'")

        if np.isscalar(blocksize):
            blocksize = (blocksize, )*2
            
        if not all(s%b==0 for s,b in zip(x.shape, blocksize)):
            warnings.warn(f"image size {x.shape} not divisible by blocksize {blocksize}")
            pads = tuple(b-s%b for b, s in zip(blocksize, x.shape))
            out_slice = tuple(slice(0,s) for s in x.shape)
            print(f'padding with {pads}')
            x = np.pad(x,tuple((0,p) for p in pads), mode='reflect')

        n_tiles = tuple(max(1,s//b) for s,b in zip(x.shape, blocksize))
        
        print(f"normalizing_fast adaptively with {n_tiles} tiles and order {order}")        
        mi, ma = np.zeros(n_tiles, x.dtype), np.zeros(n_tiles, x.dtype)

        kwargs=dict(block_size=1, n_block_overlap=0, guarantee="n_tiles")

        for i, (itile,is_src,is_dst) in enumerate(tile_iterator_1d(x, axis=0,
                                                                   n_tiles=n_tiles[0], **kwargs)):
            for j, (tile,s_src,s_dst) in enumerate(tile_iterator_1d(itile, axis=1,
                                                                    n_tiles=n_tiles[1], **kwargs)):
                x_sub = tile[::sub,::sub]
                if ignore_val is not None:
                    x_sub = x_sub[x_sub!=ignore_val]
                    x_sub = np.array(0) if len(x_sub)==0 else x_sub
                mi[i,j], ma[i,j] = np.percentile(x_sub,(pmin,pmax)).astype(x.dtype)

        interpolations = {0:cv2.INTER_NEAREST,
                          1:cv2.INTER_LINEAR}

        mi = cv2.resize(mi, x.shape[::-1], interpolation=interpolations[order])
        ma = cv2.resize(ma, x.shape[::-1], interpolation=interpolations[order])

    x = x.astype(np.float32)
    x -= mi
    x *= (dst_range[1]-dst_range[0])
    x /= ma-mi+1e-20
    x = x[out_slice]
    
    x += dst_range[0]
    
    if clip:
        x = np.clip(x,*dst_range)
    return x

def initialize_wandb(options, train_dataset, val_dataset, silent=True):
    if options.get("skip_logging"):
        log.info("Run won't be logged to wandb")
        return None
    else:
        log.info(f"Initializing wandb project for user '{options['wandb_user']}'")
    if silent:
        os.environ["WANDB_SILENT"] = "true"
    try:
        wandb.init(
            project=options["wandb_project"],
            entity=options["wandb_user"],
            name=options["run_name"],
            config=options,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(
            {"n_train_samples": len(train_dataset), "n_val_samples": len(val_dataset)}
        )
        log.info("wandb initialized successfully")
    except KeyError as ke:
        log.warn(f"Skipping logging to wandb due to missing options: {ke}")
        return None
    return None

def write_coords_csv(pts: np.ndarray, fname: Path) -> None:
    """ writes points to csv file
    """
    df = pd.DataFrame(pts, columns=["y", "x"])
    df.to_csv(fname, index=False)
    return

def get_run_name(namespace: SimpleNamespace):
    name = f"{Path(namespace.data_dir).stem}"
    name += f"_{namespace.backbone}"
    name += f"_lvs{int(namespace.levels)}"
    name += f"_fmaps{int(namespace.initial_fmaps)}"
    name += f"_{namespace.mode}"
    name += f"_convperlv{int(namespace.convs_per_level)}"
    name += f"_loss{namespace.loss}"
    name += f"_epochs{int(namespace.num_epochs)}"
    name += f"_lr{namespace.lr}"
    name += f"_bsize{int(namespace.batch_size)}"
    name += f"_ksize{int(namespace.kernel_size)}"
    name += f"_sigma{int(namespace.sigma)}"
    name += f"_crop{int(namespace.crop_size)}"
    name += f"_posweight{int(namespace.pos_weight)}"
    name += f"_seed{int(namespace.seed)}"
    name += "_tormenter" # !
    name += "_skipbgremover" if namespace.skip_bg_remover else ""
    name += "_dry_run" if namespace.dry_run else ""
    name = name.replace(".", "") # Remove dots to avoid confusion with file extensions
    return name