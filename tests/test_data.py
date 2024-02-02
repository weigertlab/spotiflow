from spotiflow.data import SpotsDataset
from utils import example_data


if __name__ == "__main__":

    imgs, points = example_data()

    data = SpotsDataset(imgs, points, downsample_factors=(1, 2))

    out = data[0]
