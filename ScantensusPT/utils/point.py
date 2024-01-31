import torch
import torch.nn
import torch.nn.functional

def get_point_a(logits: torch.Tensor):
    device = logits.device
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

    y = y + 0.25 * (y2-y)/lens
    x = x + 0.25 * (x2-x)/lens

    out = torch.stack((y, x, val[..., 0]), dim=2)

    if input_dim == 4 or input_dim == 3:
        out = torch.reshape(out, shape=(batch_size, channels, 3))
    else:
        out = torch.reshape(out, shape=[3])

    return out


def get_point_b(logits: torch.Tensor):
    device = logits.device
    clip_constant = 1e-10
    input_dim = logits.dim()

    logits = torch.nn.functional.pad(logits, pad=[2, 2, 2, 2], value=0)
    logits = torch.clip(logits, min=clip_constant)
    logits = torch.log(logits)

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

    bc = batch_size * channels
    hw = height * width

    logits = logits.reshape(bc, hw)

    idx = torch.argmax(logits, 1)

    idx_wm1 = idx - 1
    idx_wm2 = idx - 2
    idx_wp1 = idx + 1
    idx_wp2 = idx + 2
    idx_hm1 = idx - width
    idx_hp1 = idx + width
    idx_hm2 = idx - 2*width
    idx_hp2 = idx + 2*width

    idx_br = idx + 1 + width
    idx_bl = idx - 1 + width
    idx_tr = idx + 1 - width
    idx_tl = idx - 1 - width

    idx_range = torch.arange(bc)

    val = logits[idx_range, idx]
    val = val.unsqueeze(-1)
    val = torch.exp(val)

    dx = 0.5 * (logits[idx_range, idx_wp1] - logits[idx_range, idx_wm1])
    dy = 0.5 * (logits[idx_range, idx_hp1] - logits[idx_range, idx_hm1])

    dxx = 0.25 * (logits[idx_range, idx_wp2] - 2 * logits[idx_range, idx] + logits[idx_range, idx_wm2])
    dyy = 0.25 * (logits[idx_range, idx_hp2] - 2 * logits[idx_range, idx] + logits[idx_range, idx_hm2])

    dxy = 0.25 * (logits[idx_range, idx_br] - logits[idx_range, idx_tr] - logits[idx_range, idx_bl] + logits[idx_range, idx_tl])

    derivative = torch.stack([dx, dy], 1).unsqueeze(-1)
    hessian = torch.stack((torch.stack((dxx, dxy), -1), torch.stack((dxy, dyy), -1)), -1)

    det_mask = torch.abs(torch.det(hessian)) > 0.0

    derivative_m = derivative[det_mask]
    hessian_m = hessian[det_mask]

    hessianinv_m = torch.inverse(hessian_m)
    offset_m = torch.bmm(-hessianinv_m, derivative_m)
    offset_m = torch.squeeze(offset_m, dim=-1)

    offset = torch.zeros((bc, 2), device=device)
    offset[det_mask] = offset_m

    coord = torch.stack(((idx // height) - 2, (idx % height) - 2), -1)
    coord = coord + offset

    out = torch.cat((coord, val), dim=-1)

    if input_dim == 4 or input_dim == 3:
        out = torch.reshape(out, shape=(batch_size, channels, 3))
    else:
        out = torch.reshape(out, shape=[3])

    return out


get_point = get_point_a
