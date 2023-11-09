from spotipy_torch.model import Spotipy
from spotipy_torch.utils import normalize
from spotipy_torch.data import test_data_hybiss_2d

if __name__ == "__main__":
    model = Spotipy.from_pretrained("hybiss", map_location="cpu")
    x = test_data_hybiss_2d()
    points, _ = model.predict(normalize(x), device="cpu")
