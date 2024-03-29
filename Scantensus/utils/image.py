import numpy as np
import torch


def image_logit_overlay_alpha(images=None, logits=None, cols=None, make_bg_bw=False):
    #NHWC

    overlay_intensity = 1.0

    if images is None:
        images = np.zeros_like(logits)

    cols = np.array(cols)

    logits = np.clip(logits, 0, 1)

    if make_bg_bw == True:
        out = images[:, :, :, 0:1]
    else:
        out = images

    cols = np.expand_dims(cols, 0)
    cols = np.expand_dims(cols, 1)
    cols = np.expand_dims(cols, 2)

    alpha = np.expand_dims(logits, 4) * cols
    
    alpha = np.sum(alpha, axis=3)
    alpha = np.clip(alpha, 0, 1)

    out = overlay_intensity * alpha + out * (1.0 - alpha)

    return out


image_logit_overlay_alpha_np = image_logit_overlay_alpha


def nchw_image_logit_overlay_alpha(images=None, logits=None, cols=None, device=None):

    #NCHW

    overlay_intensity = 1.0

    if images is None:
        images = np.zeros_like(logits)

    cols = np.array(cols)

    logits = np.clip(logits, 0, 1)

    cols = np.expand_dims(cols, 0)
    cols = np.expand_dims(cols, 3)
    cols = np.expand_dims(cols, 4)

    alpha = np.expand_dims(logits, 2) * cols
    alpha = np.sum(alpha, axis=1)
    alpha = np.clip(alpha, 0, 1)

    out = overlay_intensity * alpha + images * (1.0 - alpha)

    return out