from spotiflow.model import Spotiflow
from spotiflow.utils import normalize
from spotiflow.sample_data import test_image_hybiss_2d

if __name__ == "__main__":
    model = Spotiflow.from_pretrained("hybiss", map_location="cpu")
    x = test_image_hybiss_2d()
    points, _ = model.predict(normalize(x), device="cpu")
