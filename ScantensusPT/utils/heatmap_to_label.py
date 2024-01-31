from typing import List

import torch

from .trace import *


def heatmap_to_label(y_pred: torch.Tensor,
                     label: str,
                     keypoint_names: List[str]):

    if label == "curve-lv-endo":
        return get_lv_endo_path(y_pred, keypoint_names)
    if label == "curve-lv-trabec":
        return get_lv_trabec_path(y_pred, keypoint_names)
    if label == "curve-lv-midmyo":
        return get_lv_midmyo_path(y_pred, keypoint_names)
    elif label == "curve-rv-endo":
        return get_rv_endo_path(y_pred, keypoint_names)

    elif label == "curve-la-endo":
        return get_la_endo_path(y_pred, keypoint_names)
    elif label == "curve-ra-endo":
        return get_ra_endo_path(y_pred, keypoint_names)

    elif label == "curve-lv-post-endo":
        return get_lv_post_endo_path(y_pred, keypoint_names)
    elif label == "curve-lv-post-epi":
        return get_lv_post_epi_path(y_pred, keypoint_names)
    elif label == "curve-lv-antsep-rv":
        return get_lv_antsep_rv_path(y_pred, keypoint_names)
    elif label == "curve-lv-antsep-endo":
        return get_lv_antsep_endo_path(y_pred, keypoint_names)

    elif label == "curve-flow":
        return get_flow_path(y_pred, keypoint_names)

    elif label == "curve-effusion-lv-visceral":
        return get_effusion_path(y_pred=y_pred, keypoint_names=keypoint_names)

    elif label == "curve-effusion-lv-parietal":
        return get_effusion_path(y_pred=y_pred, keypoint_names=keypoint_names)

    elif label == "curve-effusion-rv-visceral":
        return get_effusion_path(y_pred=y_pred, keypoint_names=keypoint_names)

    elif label == "curve-effusion-rv-parietal":
        return get_effusion_path(y_pred=y_pred, keypoint_names=keypoint_names)

    elif label == "curve-lv-endo-jump":
        return [float('nan')], [float('nan')], [float('nan')]

    elif label == "systolic-peak":
        point = get_point(y_pred[keypoint_names.index(label), :, :])
        y, x, conf = point.flatten().tolist()
        if conf < 0.05:
            return [float('nan')], [float('nan')], [float('nan')]
        else:
            return [y], [x], [conf]

    elif not label.startswith("curve"):
        point = get_point(y_pred[keypoint_names.index(label), :, :])
        y, x, conf = point.flatten().tolist()
        if conf < 0.05:
            return [float('nan')], [float('nan')], [float('nan')]
        else:
            return [y], [x], [conf]
    else:
        logging.exception(label)
        raise Exception
