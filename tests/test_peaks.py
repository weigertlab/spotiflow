import numpy as np
from spotiflow.utils import points_to_prob, points_to_flow, flow_to_vector


if __name__ == "__main__":
    points = np.array([[10.56, 12.4]])

    sigma = 1.5
    u = points_to_prob(points, (32, 32), sigma=sigma)
    f = points_to_flow(points, (32, 32), sigma=sigma)
    v = flow_to_vector(f, sigma=sigma)

    p0 = np.array(np.unravel_index(np.argmax(u), u.shape))
    p = p0 + v[tuple(p0)]

    print("error (L1):", np.abs(points[0] - p).sum())
