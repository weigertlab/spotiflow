from .datasets import load_dataset

def __abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)

def test_image_hybiss_2d():
    # TODO: proper docstring after paper is published :)
    """ Single test HybISS image from the Spotiflow paper (doi.org/10.1101/2024.02.01.578426)
    """
    from tifffile import imread
    img = imread(__abspath("images/img_hybiss_2d.tif"))
    return img

def test_image_terra_2d():
    # TODO: proper docstring after paper is published :)
    """ Single test Terra frame from the Spotiflow paper (doi.org/10.1101/2024.02.01.578426)
    """
    from tifffile import imread
    img = imread(__abspath("images/img_terra_2d.tif"))
    return img

def test_timelapse_telomeres_2d():
    # TODO: proper docstring after paper is published :)
    """Timelapse of telomeres from the Spotiflow paper (doi.org/10.1101/2024.02.01.578426)
    """
    from tifffile import imread
    img = imread(__abspath("images/timelapse_telomeres_2d.tif"))
    return img

def test_image_synth_3d():
    # TODO: proper docstring after paper is published :)
    """ Single synthetic volumetric stack from the Spotiflow paper (doi.org/10.1101/2024.02.01.578426)
    """
    from tifffile import imread
    img = imread(__abspath("images/img_synth_3d.tif"))
    return img
