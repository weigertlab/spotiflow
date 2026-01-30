from spotiflow.model import Spotiflow
from spotiflow.utils import points_matching, points_to_flow, flow_to_vector
from spotiflow.sample_data import test_image_hybiss_2d as _sample_image
from utils import example_data


def test_model():
    model = Spotiflow.from_pretrained("hybiss", map_location="cpu")
    x = _sample_image()
    points, _ = model.predict(x, device="cpu")
    

def test_flow():
    X, P = example_data(64, sigma=3, noise=0.01)

    model = Spotiflow.from_pretrained("hybiss", map_location="cpu")
    p1, details1 = model.predict(X[0], subpix=False, device="cpu")
    p2, details2 = model.predict(X[0], subpix=True, device="cpu")
    p3, details3 = model.predict(X[0], subpix=1, device="cpu")

    s1 = points_matching(P[0], p1)
    s2 = points_matching(P[0], p2)
    s3 = points_matching(P[0], p3)

    f0 = points_to_flow(P[0], sigma=model.config.sigma, shape=X[0].shape)
    flow_to_vector(f0, sigma=model.config.sigma)

    print(f"mean error w/o  subpix    {s1.mean_dist:.4f}")
    print(f"mean error with subpix    {s2.mean_dist:.4f}")
    print(f"mean error with subpix 1  {s3.mean_dist:.4f}")


if __name__ == "__main__":
    # test_model()
    test_flow()
