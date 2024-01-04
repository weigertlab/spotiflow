from .datasets import load_dataset

def __abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)

def test_image_hybiss_2d():
    # TODO: proper docstring after paper is published :)
    """ HybISS data from the paper
    ???
    """
    from tifffile import imread
    img = imread(__abspath("images/img_hybiss_2d.tif"))
    return img

def test_image_terra_2d():
    # TODO: proper docstring after paper is published :)
    """ Terra data from the paper
    ???
    """
    from tifffile import imread
    img = imread(__abspath("images/img_terra_2d.tif"))
    return img
