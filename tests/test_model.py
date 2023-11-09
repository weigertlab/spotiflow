from spotipy_torch.model import Spotipy


if __name__ == "__main__":
    model = Spotipy.from_pretrained("hybiss", map_location="cpu")
