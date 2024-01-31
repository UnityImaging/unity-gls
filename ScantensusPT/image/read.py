import logging
import requests_cache

import torch
import torchvision


def read_image_into_t(image_path: str,
                      png_session: requests_cache.CachedSession = None,
                      device="cpu"):

    if image_path.startswith("http://") or image_path.startswith("https://"):
        r = png_session.get(image_path)
        if r.status_code == 200:
            img_bytes = torch.frombuffer(r.content, dtype=torch.uint8)
            image = torchvision.io.decode_png(img_bytes)
            logging.debug(f"{image_path}: Successfully loaded")
        else:
            raise Exception(f"Failed to load {image_path}")
    else:
        image = torchvision.io.read_image(image_path)

    # ensure dim = 3
    if image.ndim == 2:
        image = torch.unsqueeze(image, 0)

    # remove alpha layer
    if image.shape[0] == 4:
        image = image[:3, ...]

    return image
