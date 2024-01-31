import io
import logging
from typing import NamedTuple, Optional
from requests_cache import CachedSession

import numpy as np

import torch
import torch.nn
import torch.nn.functional

from mdicom.image import center_crop_or_pad, ensure_bw, ensure_3channels

from ScantensusPT.utils import load_checkpoint, fix_state_dict

from .utils import extract_tensors, replace_tensors


USE_CUDA = True

if USE_CUDA:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

CURVE_SD = 2
DOT_SD = 4


class LoadedModel(NamedTuple):
    """
    Represents a model that has been fetched and pre-configured for inference.
    """

    model: torch.nn.Module
    """
    The model.
    """

    tensors: dict
    """
    The extracted tensors - useful for hot-loading the model under memory-efficient conditions.
    """

    config: dict
    """
    The configuration associated with the model.
    """


def bmode_seg_init(checkpoint_weights_location,
                   logging_level: int = logging.INFO,
                   cache_backend='sqlite',
                   cache_name='unity'):

    logging.basicConfig(level=logging_level)

    session = CachedSession(cache_name=cache_name, backend=cache_backend, use_cache_dir=True)

    logging.info("Downloading weights file")
    if checkpoint_weights_location[0:4] == "http":
        weights_file = io.BytesIO(session.get(checkpoint_weights_location).content)
    else:
        with open(checkpoint_weights_location, 'rb') as w_f:
            weights_file = io.BytesIO(w_f.read())
    logging.info("Weights file downloaded")

    checkpoint_dict_cpu = load_checkpoint(checkpoint_path=weights_file, device="cpu")

    if checkpoint_dict_cpu.get('model_state_dict', None):
        state_dict_cpu = fix_state_dict(state_dict=checkpoint_dict_cpu['model_state_dict'], remove_module_prefix=True)
        net_cfg = checkpoint_dict_cpu['net_cfg']
        PRE_POST = checkpoint_dict_cpu['PRE_POST']
        PRE_POST_LIST = checkpoint_dict_cpu['PRE_POST_LIST']
        BW_IMAGES = checkpoint_dict_cpu['BW_IMAGES']
        HEATMAP_SCALE_FACTORS = checkpoint_dict_cpu['HEATMAP_SCALE_FACTORS']
        MODEL_CODE_NAME = checkpoint_dict_cpu['MODEL_CODE_NAME']
        keypoint_names = checkpoint_dict_cpu['keypoint_names']
    else:
        state_dict_cpu = fix_state_dict(state_dict=checkpoint_dict_cpu, remove_module_prefix=True)


    keypoint_cols = [[1, 0, 0] for _ in keypoint_names]

    if MODEL_CODE_NAME is None:
        raise Exception
    elif MODEL_CODE_NAME == "HRNetV2M7":
        from ScantensusPT.nets.HRNetV2M7 import get_seg_model
    elif MODEL_CODE_NAME == "HRNetV2M8":
        from ScantensusPT.nets.HRNetV2M8 import get_seg_model
    elif MODEL_CODE_NAME == "HRNetV2M9":
        from ScantensusPT.nets.HRNetV2M8 import get_seg_model
    elif MODEL_CODE_NAME == "HRNetV2M10":
        from ScantensusPT.nets.HRNetV2M10 import get_seg_model
    elif MODEL_CODE_NAME == "HRNetV2M11":
        from ScantensusPT.nets.HRNetV2M11 import get_seg_model
    else:
        raise Exception

    keypoint_sd = [CURVE_SD if 'curve' in keypoint_name else DOT_SD for keypoint_name in keypoint_names]
    keypoint_sd = torch.tensor(keypoint_sd, dtype=torch.float, device="cpu")
    keypoint_sd = keypoint_sd.unsqueeze(1).expand(-1, 2)

    model_cpu = get_seg_model(cfg=net_cfg)
    model_cpu.load_state_dict(state_dict_cpu)
    model_cpu.eval()

    num_input_channels = net_cfg['DATASET']['NUM_INPUT_CHANNELS']
    keypoint_names = keypoint_names
    default_output_layer_list = keypoint_names

    cfg = {}
    cfg['pre_post'] = PRE_POST
    cfg['bw_images'] = BW_IMAGES
    if PRE_POST:
        cfg['pre_post_list'] = PRE_POST_LIST
    else:
        cfg['pre_post_list'] = [0]
    cfg['keypoint_cols'] = keypoint_cols
    cfg['heatmap_scale_factors'] = HEATMAP_SCALE_FACTORS
    cfg['keypoint_names'] = keypoint_names
    cfg['keypoint_cols'] = keypoint_cols
    cfg['default_output_layer_list'] = default_output_layer_list
    cfg['num_input_channels'] = num_input_channels

    logging.info("Finished bmode seg model")

    m, t = extract_tensors(model_cpu)

    return LoadedModel(model=m, tensors=t, config=cfg)


def bmode_seg_process(
        bmode_seg_model: tuple[torch.nn.Module, torch.Tensor, dict],
        img: np.ndarray,
        FileHash: str = "99-Frame",
        output_layer_list: Optional[list[str]] = None,
        ignore_right_half: bool = False,
        logging_level: int = logging.INFO,
        debug_mode: bool = False
    ) -> tuple[np.ndarray, float, float]:
    """
    Performs pre-processing and inference on a whole series of echo B-mode images.

    :param bmode_seg_model: tuple of model, tensors, and config
    :param img: numpy array of shape (num_frames, height, width, channels). The channels are either 1 or 3.
    :param FileHash: string to identify the file
    :param output_layer_list: list of strings of the layers to output. This is used to lookup the position of output heatmaps pertaining to specific points or curves. 
    :param ignore_right_half: boolean to ignore the right half of the image.
    :param logging_level: logging level
    :param debug_mode: if true, skips memory optimizations that allow for the image to be returned. This is useful for debugging heatmaps.

    :return: tuple of numpy array of shape (num_frames, num_keypoints, height, width), float of height shift, float of width shift
    """

    logging.basicConfig(level=logging_level)

    model, t, cfg = bmode_seg_model
    replace_tensors(m=model, tensors=t)

    model = model.to("cuda")

    bw_images = cfg['bw_images']
    pre_post_list = cfg['pre_post_list']

    if not output_layer_list:
        output_layer_list = cfg['default_output_layer_list']

    output_layer_list_position = [cfg['keypoint_names'].index(x) for x in output_layer_list]

    if bw_images:
        img = ensure_bw(img)
    else:
        img = ensure_3channels(img)

    _, th, tw, _ = img.shape

    if th == 416 and tw == 240:
        vscan_mod = False
    else:
        vscan_mod = False

    if vscan_mod:
        img = img[:, :304, :, :]
        img = torch.from_numpy(img)
        img = torch.nn.functional.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
        img = img.numpy()
        img, label_height_shift, label_width_shift = center_crop_or_pad(img, output_size=(608, 608), cval=0)
    else:
        img, label_height_shift, label_width_shift = center_crop_or_pad(img, output_size=(608, 608), cval=0)

    if ignore_right_half:
        img[:, :, 304:, :] = 0

    if bw_images:
        img = np.concatenate((img, np.zeros((1, 608, 608, 1), dtype=np.uint8)), axis=0)
    else:
        img = np.concatenate((img, np.zeros((1, 608, 608, 3), dtype=np.uint8)), axis=0)

    ##########

    with torch.no_grad():
        # the numpy array has a blank last image to make masking easier below

        num_frames = img.shape[0] - 1

        if bw_images:
            channels = 1
        else:
            channels = 3

        pre_post_torch = np.array(pre_post_list, dtype=np.int64)
        output_layer_list_position_t = torch.tensor(output_layer_list_position, device="cuda", dtype=torch.int64)

        out_y_pred = torch.zeros((num_frames, len(output_layer_list), 608, 608), dtype=torch.float32, device="cuda")

        unity_f_codes = []

        for i in range(num_frames):
            select_frames = pre_post_torch + i

            # uses the last frame as blank
            select_frames[select_frames >= num_frames] = num_frames
            select_frames[select_frames < 0] = num_frames
            image_c = img[select_frames, :, :, :]
            image_t = torch.tensor(image_c, dtype=torch.float32, device='cuda')
            del image_c

            image_t = image_t.permute((0, 3, 1, 2)).reshape((len(pre_post_list) * channels, 608, 608)).unsqueeze(0).div(
                255.0)

            y_pred_25, y_pred_50 = model(image_t.add(-0.5))
            del image_t

            y_pred_25 = torch.nn.functional.interpolate(y_pred_25, scale_factor=4, mode='bilinear',
                                                        align_corners=True)
            y_pred_50 = torch.nn.functional.interpolate(y_pred_50, scale_factor=2, mode='bilinear',
                                                        align_corners=True)

            y_pred = (y_pred_25 + y_pred_50) / 2.0
            del y_pred_25, y_pred_50

            y_pred = torch.clamp(y_pred, 0, 1)
            out_y_pred[i, :, :, :] = y_pred[0, output_layer_list_position_t, :, :]
            del y_pred

            unity_f_codes.append(f"{FileHash}-{i:04}")

    if vscan_mod:
        out_y_pred = torch.nn.functional.interpolate(out_y_pred, scale_factor=0.5, mode='bilinear', align_corners=True)
        label_width_shift = label_width_shift / 2
        label_height_shift = label_height_shift / 2

    out_y_pred = out_y_pred.cpu().numpy()

    if not debug_mode:
        del img

        return out_y_pred, label_height_shift, label_width_shift
    else:
        # remove the blank last image
        img = img[:-1, :]

        return out_y_pred, label_height_shift, label_width_shift, img

def main():

    #####
    BMODE_SEG_ENDPOINT = "https://matt-minio.ts.zomirion.com"
    BMODE_SEG_BUCKET = "matt-output"
    BMODE_SEG_PROJECT = "unity"
    BMODE_SEG_EXPERIMENT = "unity-218"
    BMODE_SEG_EPOCH = 400
    BMODE_SEG_WEIGHTS_ADDRESS = f"{BMODE_SEG_ENDPOINT}/{BMODE_SEG_BUCKET}/checkpoints/{BMODE_SEG_PROJECT}/{BMODE_SEG_EXPERIMENT}/weights-{BMODE_SEG_EPOCH}.pt"
    #####

    ###
    output_layer_list = [
        "lv-apex-trabec",
        "lv-apex-endo",
        "lv-apex-midmyo",
        "mv-ant-wall-hinge",
        "mv-ant-wall-hinge-midmyo",
        "mv-inf-wall-hinge",
        "mv-inf-wall-hinge-midmyo",
        "curve-mv-hinge-connect",
        "curve-mv-ant-apex-connect",
        "curve-mv-post-apex-connect",
        "curve-lv-trabec",
        "curve-lv-endo",
        "curve-lv-endo-jump",
        "curve-lv-midmyo",
        "rv-apex-endo",
        "tv-ant-hinge",
        "tv-sep-hinge",
        "curve-rv-endo",
        "la-roof",
        "curve-la-endo",
        "ra-roof",
        "curve-ra-endo"
    ]
    ###

    bmode_seg_model = bmode_seg_init(checkpoint_weights_location=BMODE_SEG_WEIGHTS_ADDRESS)

    img = np.zeros((1, 600, 800, 1), dtype=np.uint8)
    out_y_pred, label_height_shift, label_width_shift = bmode_seg_process(
        bmode_seg_model,
        img)


if __name__ == "__main__":
    main()

