import logging
import sys

import torch

from ..model import Spotiflow
from ..sample_data import test_image_hybiss_2d

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

def main():
    img = test_image_hybiss_2d()
    model = Spotiflow.from_pretrained("general")
    try:
        _ = model.predict(img, device="cpu", verbose=False)
        log.info("CPU is available and working")
    except Exception as e:
        log.error("Error occurred while testing CPU: %s", e)

    if torch.cuda.is_available():
        try:
            _ = model.predict(img, device="cuda", verbose=False)
            log.info("CUDA is available and working")
        except Exception as e:
            log.error("Error occurred while testing on CUDA: %s", e)
    else:
        log.info("CUDA is not available on this system.")

    if torch.backends.mps.is_available():
        try:
            _ = model.predict(img, device="mps", verbose=False)
            log.info("MPS is available and working. Note that support in 3D is limited for MPS.")
        except Exception as e:
            log.error("Error occurred while testing on MPS: %s", e)
    else:
        log.info("MPS is not available on this system.")
if __name__ == "__main__":
    sys.exit(main())