import logging
import numpy as np
import torch


def get_point(logits: np.ndarray, device="cpu"):
    logits = torch.as_tensor(logits, dtype=torch.float32, device=device)
    clip_constant = 1e-10
    input_dim = logits.dim()

    #This moves 1/4 to second max

    if input_dim == 2:
        batch_size = 1
        channels = 1
        height = logits.shape[0]
        width = logits.shape[1]
    elif input_dim == 4:
        batch_size = logits.shape[0]
        channels = logits.shape[1]
        height = logits.shape[2]
        width = logits.shape[3]
    elif input_dim == 3:
        batch_size = 1
        channels = logits.shape[0]
        height = logits.shape[1]
        width = logits.shape[2]
    else:
        raise Exception

    logits = logits.view(batch_size, channels, height * width)

    val, index = torch.topk(logits, k=2, dim=2)

    y = (index[..., 0] // height).float()
    x = (index[..., 0] % height).float()

    y2 = (index[..., 1] // height).float()
    x2 = (index[..., 1] % height).float()

    lens = torch.sqrt(torch.square(y2-y) + torch.square(x2-x))
    if lens > 10:
        logging.warning(f"point adjust len > 1: {lens}")
    y = y + 0.25 * (y2-y)/lens
    x = x + 0.25 * (x2-x)/lens

    out = torch.stack((y, x, val[..., 0]), dim=2)

    if input_dim == 4 or input_dim == 3:
        out = torch.reshape(out, shape=(batch_size, channels, 3))
    else:
        out = torch.reshape(out, shape=[3])

    out = out.cpu().numpy()

    return out
