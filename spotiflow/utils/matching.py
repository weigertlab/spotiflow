import numpy as np
from types import SimpleNamespace
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def points_matching(p1, p2, cutoff_distance=3, eps=1e-8):
    """finds matching that minimizes sum of mean squared distances"""

    if len(p1) == 0 or len(p2) == 0:
        D = np.zeros((0, 0))
    else:
        D = cdist(p1, p2, metric="sqeuclidean")

    if D.size > 0:
        D[D > cutoff_distance**2] = 1e10 * (1 + D.max())

    i, j = linear_sum_assignment(D)
    valid = D[i, j] <= cutoff_distance**2

    i, j = i[valid], j[valid]

    res = SimpleNamespace()

    tp = len(i)
    fp = len(p2) - tp
    fn = len(p1) - tp
    res.tp = tp
    res.fp = fp
    res.fn = fn

    # when there is no tp and we dont predict anything the accuracy should be 1 not 0
    tp_eps = tp + eps

    res.accuracy = tp_eps / (tp_eps + fp + fn) if tp_eps > 0 else 0
    res.precision = tp_eps / (tp_eps + fp) if tp_eps > 0 else 0
    res.recall = tp_eps / (tp_eps + fn) if tp_eps > 0 else 0
    res.f1 = (2 * tp_eps) / (2 * tp_eps + fp + fn) if tp_eps > 0 else 0
    res.dist = np.sqrt(D[i, j])
    res.mean_dist = np.mean(res.dist) if len(res.dist) > 0 else 0

    pq_num = np.sum(cutoff_distance-res.dist)/cutoff_distance
    pq_den = tp_eps + fp/2 + fn/2
    res.panoptic_quality = pq_num/pq_den if tp_eps > 0 else 0

    res.false_negatives = tuple(set(range(len(p1))).difference(set(i)))
    res.false_positives = tuple(set(range(len(p2))).difference(set(j)))
    res.matched_pairs = tuple(zip(i, j))
    return res


def points_matching_dataset(p1s, p2s, cutoff_distance=3, by_image=True, eps=1e-8):
    """
    by_image is True -> metrics are computed by image and then averaged
    by_image is False -> TP/FP/FN are aggregated and only then are metrics computed
    """
    stats = tuple(
        points_matching(p1, p2, cutoff_distance=cutoff_distance, eps=eps)
        for p1, p2 in zip(p1s, p2s)
    )

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
            for k in ("tp", "fp", "fn"):
                setattr(res, k, getattr(res, k) + getattr(s, k))
        
        dists = np.concatenate([s.dist for s in stats])

        tp_eps = res.tp + eps
        res.accuracy = tp_eps / (tp_eps + res.fp + res.fn) if tp_eps > 0 else 0
        res.precision = tp_eps / (tp_eps + res.fp) if tp_eps > 0 else 0
        res.recall = tp_eps / (tp_eps + res.fn) if tp_eps > 0 else 0
        res.f1 = (2 * tp_eps) / (2 * tp_eps + res.fp + res.fn) if tp_eps > 0 else 0

        pq_num = np.sum(cutoff_distance-dists)/cutoff_distance
        pq_den = tp_eps + res.fp/2 + res.fn/2

        res.panoptic_quality = pq_num/pq_den if tp_eps > 0 else 0
        res.mean_dist = np.mean(dists) if len(dists) > 0 else 0
        return res
