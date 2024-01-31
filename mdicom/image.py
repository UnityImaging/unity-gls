import logging

import pydicom
import pydicom.pixel_data_handlers

import numpy as np


def make_image_4d(img: np.array, NumberOfFrames="unknown"):
    if img.ndim == 4:
        return img
    elif img.ndim == 2:
        return img[None, ..., None]
    elif img.ndim == 3:
        pass
    else:
        raise Exception(f"Error, img ndim {img.ndim} not understood in makeing 4d array")

    if NumberOfFrames == "unknown":
        if img.shape[-1] == 1 or img.shape[-1] == 3:
            return img[None, ...]
        else:
            return img[..., None]

    if NumberOfFrames is None or NumberOfFrames == 1:
        if img.shape[0] == 1:
            return img[..., None]
        else:
            return img[None, ...]

    if type(NumberOfFrames) is int or type(NumberOfFrames) is float:
        if img.shape[0] == int(NumberOfFrames):
            return img[..., None]
        else:
            raise Exception(f"Number of frames does not match first dimensions")


def make_image_3d(img: np.array):
    if img.ndim == 3:
        return img
    elif img.ndim == 2:
        return img[..., None]
    else:
        raise Exception(f"Error, img ndim {img.ndim} not understood in makeing 3d array")


def ensure_3channels(img):
    if img.shape[-1] == 3:
        return img
    elif img.shape[-1] == 1:
        return np.tile(img, 3)
    else:
        raise Exception("Expected 3 or 1 channel image")


def ensure_bw(img):
    if img.shape[-1] == 3:
        return img[..., [0]]
    elif img.shape[-1] == 1:
        return img
    else:
        raise Exception("Expected 3 or 1 channel image")


def center_crop_or_pad(image: np.ndarray, output_size=(608, 608), cval=0):
    #NHWC

    in_c = image.shape[-1]
    in_w = image.shape[-2]
    in_h = image.shape[-3]
    if image.ndim == 4:
        in_n = image.shape[-4]
    else:
        in_n = None

    out_h, out_w = output_size

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

    if in_n is not None:
        out_image = np.ones((in_n, out_h, out_w, in_c), dtype=image.dtype) * cval
    else:
        out_image = np.ones((out_h, out_w, in_c), dtype=image.dtype) * cval
    out_image[..., out_s_h:out_e_h, out_s_w:out_e_w, :] = image[..., in_s_h:in_e_h, in_s_w:in_e_w, :]

    return out_image, label_height_shift, label_width_shift


def apply_frame_conversion(raw_frame: np.array, metadata: pydicom.Dataset):
    in_PhotometricInterpretation = metadata.PhotometricInterpretation

    try:

        if in_PhotometricInterpretation == 'PALETTE COLOR':
            return pydicom.pixel_data_handlers.util.apply_color_lut(raw_frame, metadata)

        elif in_PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']:
            logging.info(f"Converting YBR")
            return pydicom.pixel_data_handlers.util.convert_color_space(raw_frame, in_PhotometricInterpretation, "RGB")

        elif in_PhotometricInterpretation in ['INSANE']:
            ### This is insane
            return pydicom.pixel_data_handlers.util.convert_color_space(raw_frame, "RGB", "YBR_FULL")

        elif in_PhotometricInterpretation == 'MONOCHROME1':
            return raw_frame

        elif in_PhotometricInterpretation == 'MONOCHROME2':
            if metadata.get('Modality', None) in ['IVOCT', 'OCT']:
                return pydicom.pixel_data_handlers.util.apply_color_lut(raw_frame, metadata)
            else:
                return raw_frame

        elif in_PhotometricInterpretation == "RGB":
            logging.info(f"RGB - returning as is")
            return raw_frame

        else:
            logging.warning(f"in_PhotometricInterpretation not understood: {in_PhotometricInterpretation}")
            return raw_frame

    except Exception:
        logging.exception(f"in_PhotometricInterpretation exception")
        return raw_frame
