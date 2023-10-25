import numpy as np
from spotipy_torch.utils import points_to_prob


def example_data(n_samples: int = 10, size: int = 256, noise: float = 0.2, sigma=2):
    def _single():
        p = np.random.uniform(0, 200, (20, 2))

        p = p.astype(int) + 0.5
        x = points_to_prob(p, (256, 256), sigma=sigma)
        x = x + noise * np.random.normal(0, 1, x.shape)
        return x, p

    X, P = tuple(zip(*tuple(_single() for _ in range(n_samples))))
    X, P = np.stack(X), np.stack(P)
    return X, P
