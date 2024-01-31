import torch


def center_crop_or_pad_t(image: torch.tensor, output_size=(608, 608), cval=0, return_shift=True, device=None):
    out_h, out_w = output_size

    ndim = image.ndim
    if ndim == 2:
        in_h, in_w = image.shape
        out_image = torch.ones((out_h, out_w), dtype=image.dtype) * cval
    elif ndim == 3:
        in_c, in_h, in_w = image.shape
        out_image = torch.ones((in_c, out_h, out_w), dtype=image.dtype) * cval
    elif ndim == 4:
        in_n, in_c, in_h, in_w = image.shape
        out_image = torch.ones((in_n, in_c, out_h, out_w), dtype=image.dtype) * cval
    else:
        raise Exception(f"Expected tensor to have ndim 2, 3, or 4")

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
        label_height_shift = out_s_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h
        label_height_shift = -in_s_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
        label_width_shift = out_s_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w
        label_width_shift = -in_s_w

    out_image[..., out_s_h:out_e_h, out_s_w:out_e_w] = image[..., in_s_h:in_e_h, in_s_w:in_e_w]

    if return_shift:
        return out_image, label_height_shift, label_width_shift

    return out_image


def patch_centre_t(image: torch.tensor, patch: torch.tensor, in_range, out_range):

    in_n, in_c, in_h, in_w = image.shape
    out_n, out_c, out_h, out_w = patch.shape

    if in_c != out_c:
        raise Exception

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w

    image[out_range, :, in_s_h:in_e_h, in_s_w:in_e_w] = patch[in_range, :, out_s_h:out_e_h, out_s_w:out_e_w,]

    #Modify in place, so no return.
    return

