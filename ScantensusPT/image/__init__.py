import torch


def ensure_bw_t(img: torch.Tensor):
    if img.shape[-3] == 3:
        return img[..., [0], :, :]
    elif img.shape[-3] == 1:
        return img
    else:
        raise Exception("Expected 3 or 1 channel image")


def ensure_3channels_t(img: torch.Tensor):
    if img.shape[-3] == 3:
        return img
    elif img.shape[-3] == 1:
        return torch.tile(img, (3, 1, 1))
    else:
        raise Exception("Expected 3 or 1 channel image")


def image_logit_overlay_alpha_t(logits: torch.Tensor, images=None, cols=None, make_bg_bw=False):
    # NCHW

    device = logits.device

    overlay_intensity = 1.0

    if images is None:
        images = torch.zeros((logits.shape[0], 3, logits.shape[2], logits.shape[3]), device=device)

    cols = torch.tensor(cols, device=device, dtype=torch.float)

    logits = torch.clamp(logits, min=0, max=1)

    #This keeps the dims
    if make_bg_bw == True:
        out = images[:, [0], :, :]
    else:
        out = images

    #BC(3)HW
    cols = cols.unsqueeze(0).unsqueeze(3).unsqueeze(4)

    #BC(1 for cols)HW
    alpha = logits.unsqueeze(2) * cols

    #And now RGB takes the channels position.
    alpha = torch.sum(alpha, dim=1)
    alpha = torch.clamp(alpha, 0, 1)

    out = overlay_intensity * alpha + out * (1.0 - alpha)

    return out


