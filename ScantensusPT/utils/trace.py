import logging
import math
import torch
import torch.nn
import torch.nn.functional

import scipy.interpolate
import scipy.signal

from typing import List

from .point import get_point
from .path import get_path_via_2d, get_path_2d

LOGGER_NAME = "scantensus"


def get_lv_endo_path(y_pred: torch.Tensor, keypoint_names):
    # c, h, w
    if keypoint_names.count("mv-ant-hinge"):
        mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    elif keypoint_names.count("mv-ant-wall-hinge"):
        mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    else:
        raise Exception

    if keypoint_names.count("mv-post-hinge"):
        mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    elif keypoint_names.count("mv-inf-wall-hinge"):
        mv_post_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")
    else:
        raise Exception

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-endo")
    lv_endo_jump_idx = keypoint_names.index("curve-lv-endo-jump")

    lv_endo = torch.max(y_pred[lv_endo_idx, :, :], y_pred[lv_endo_jump_idx, :, :] ** 0.25)

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_post_hinge = y_pred[mv_post_hinge_idx, :, :]
    lv_apex_endo = y_pred[lv_apex_endo_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_post_hinge_point = get_point(mv_post_hinge)
    lv_apex_endo_point = get_point(lv_apex_endo)

    out = get_path_via_2d(logits=lv_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=lv_apex_endo_point,
                          p3s=mv_post_hinge_point,
                          num_knots=15,
                          description="LV Endo")

    return out

def get_lv_trabec_path(y_pred: torch.Tensor, keypoint_names):
    mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    mv_post_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")

    lv_endo_idx = keypoint_names.index("curve-lv-trabec")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-trabec")

    lv_endo = y_pred[lv_endo_idx, :, :]

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_post_hinge = y_pred[mv_post_hinge_idx, :, :]
    lv_apex_endo = y_pred[lv_apex_endo_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_post_hinge_point = get_point(mv_post_hinge)
    lv_apex_endo_point = get_point(lv_apex_endo)

    out = get_path_via_2d(logits=lv_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=lv_apex_endo_point,
                          p3s=mv_post_hinge_point,
                          num_knots=15,
                          description="LV trabec")

    return out

def get_lv_midmyo_path(y_pred: torch.Tensor, keypoint_names):
    # c, h, w

    mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge-midmyo")
    mv_post_hinge_idx = keypoint_names.index("mv-inf-wall-hinge-midmyo")

    lv_endo_idx = keypoint_names.index("curve-lv-midmyo")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-midmyo")

    lv_endo = y_pred[lv_endo_idx, :, :]

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_post_hinge = y_pred[mv_post_hinge_idx, :, :]
    lv_apex_endo = y_pred[lv_apex_endo_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_post_hinge_point = get_point(mv_post_hinge)
    lv_apex_endo_point = get_point(lv_apex_endo)

    out = get_path_via_2d(logits=lv_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=lv_apex_endo_point,
                          p3s=mv_post_hinge_point,
                          num_knots=15,
                          description="LV Midmyo")

    return out


def get_lv_endo_path_2(y_pred: torch.Tensor, keypoint_names):

    if keypoint_names.count("mv-ant-hinge"):
        mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    elif keypoint_names.count("mv-ant-wall-hinge"):
        mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    else:
        raise Exception

    if keypoint_names.count("mv-post-hinge"):
        mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    elif keypoint_names.count("mv-inf-wall-hinge"):
        mv_post_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")
    else:
        raise Exception

    lv_endo_idx = keypoint_names.index("curve-lv-endo")
    lv_apex_endo_idx = keypoint_names.index("lv-apex-endo")

    lv_endo_jump_idx = keypoint_names.index("curve-lv-endo-jump")

    lv_endo = torch.max(y_pred[lv_endo_idx, :, :], y_pred[lv_endo_jump_idx, :, :] ** 0.25)

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_post_hinge = y_pred[mv_post_hinge_idx, :, :]
    lv_apex_endo = y_pred[lv_apex_endo_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_post_hinge_point = get_point(mv_post_hinge)
    lv_apex_endo_point = get_point(lv_apex_endo)

    out_left = get_path_2d(logits=lv_endo,
                           p1s=mv_ant_hinge_point,
                           p2s=lv_apex_endo_point)

    out_right = get_path_2d(logits=lv_endo,
                            p1s=lv_apex_endo_point,
                            p2s=mv_post_hinge_point)

    out_y = out_left[0].extend(out_right[0])
    out_x = out_left[1].extend(out_right[1])
    out_conf = out_left[2].extend(out_right[2])

    return out_y, out_x, out_conf

def get_rv_endo_path(y_pred: torch.Tensor, keypoint_names):
    # c, h, w

    rv_endo_idx = keypoint_names.index("curve-rv-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")
    rv_apex_endo_idx = keypoint_names.index("rv-apex-endo")

    rv_endo = y_pred[rv_endo_idx, :, :]
    tv_ant_hinge = y_pred[tv_ant_hinge_idx, :, :]
    tv_sep_hinge = y_pred[tv_sep_hinge_idx, :, :]
    rv_apex_endo = y_pred[rv_apex_endo_idx, :, :]

    tv_ant_hinge_point = get_point(tv_ant_hinge)
    rv_apex_endo_point = get_point(rv_apex_endo)
    tv_sep_hinge_point = get_point(tv_sep_hinge)

    out = get_path_via_2d(logits=rv_endo,
                          p1s=tv_ant_hinge_point,
                          p2s=rv_apex_endo_point,
                          p3s=tv_sep_hinge_point,
                          num_knots=11)

    return out



def get_la_endo_path(y_pred: torch.Tensor, keypoint_names: List):

    la_endo_idx = keypoint_names.index("curve-la-endo")

    if keypoint_names.count("mv-ant-hinge"):
        mv_ant_hinge_idx = keypoint_names.index("mv-ant-hinge")
    elif keypoint_names.count("mv-ant-wall-hinge"):
        mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    else:
        raise Exception

    if keypoint_names.count("mv-post-hinge"):
        mv_post_hinge_idx = keypoint_names.index("mv-post-hinge")
    elif keypoint_names.count("mv-inf-wall-hinge"):
        mv_post_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")
    else:
        raise Exception

    la_roof_idx = keypoint_names.index("la-roof")

    la_endo = y_pred[la_endo_idx, :, :]
    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_post_hinge = y_pred[mv_post_hinge_idx, :, :]
    la_roof = y_pred[la_roof_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_post_hinge_point = get_point(mv_post_hinge)
    la_roof_point = get_point(la_roof)

    out = get_path_via_2d(logits=la_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=la_roof_point,
                          p3s=mv_post_hinge_point,
                          cut_offs=(0.1, 0.2, 0.2, 0.2),
                          num_knots=11,
                          description="la")

    return out


def get_ra_endo_path(y_pred: torch.Tensor, keypoint_names):

    ra_endo_idx = keypoint_names.index("curve-ra-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")

    ra_endo = y_pred[ra_endo_idx, :, :]
    tv_ant_hinge = y_pred[tv_ant_hinge_idx, :, :]
    tv_sep_hinge = y_pred[tv_sep_hinge_idx, :, :]

    tv_ant_hinge_point = get_point(tv_ant_hinge)
    tv_sep_hinge_point = get_point(tv_sep_hinge)

    out = get_path_2d(logits=ra_endo,
                      p1s=tv_ant_hinge_point,
                      p2s=tv_sep_hinge_point,
                      num_knots=21)

    return out


def get_effusion_path(y_pred: torch.Tensor, keypoint_names):
    return [float('NaN')], [float('NaN')], [float('NaN')]


def get_flow_path(y_pred: torch.Tensor, keypoint_names):
    logger = logging.getLogger(LOGGER_NAME)
    channels, height, width = y_pred.shape

    flow_idx = keypoint_names.index("curve-flow")

    flow = y_pred[flow_idx, ...]

    flow_max, flow_argmax = torch.max(flow, dim=0, keepdim=False)

    try:

        xs = torch.arange(width, device=y_pred.device)
        ys = flow_argmax
        conf = flow_max

        conf_good = conf > 0.1
        start_x = conf_good.max(dim=0)[1]
        end_x = width - conf_good.flip(dims=[0]).max(dim=0)[1] - 1

        mask = torch.arange(width, device=y_pred.device)
        mask = (mask >= start_x) & (mask <= end_x) & conf_good

        new_xs = xs[mask]
        new_ys = ys[mask]
        new_conf = conf[mask]

        mean_score = torch.mean(new_conf)

        if mean_score < 0.01:
            raise Exception

        if True:
            new_xs_np = new_xs.detach().cpu().numpy()
            new_ys_np = new_ys.detach().cpu().numpy()
            new_conf_np = new_conf.detach().cpu().numpy()

            new_ys_np = scipy.signal.savgol_filter(new_ys_np, 11, 3)

            out_ys = new_ys_np.tolist()
            out_xs = new_xs_np.tolist()
            out_confs = new_conf_np.tolist()

        return out_ys, out_xs, out_confs

    except Exception as e:
        logger.warning("failed get_flow_path")
        return [float('NaN')], [float('NaN')], [float('NaN')]


def get_lv_antsep_endo_path(y_pred: torch.Tensor, keypoint_names):

    curve_idx = keypoint_names.index("curve-lv-antsep-endo")
    start_idx = keypoint_names.index("lv-antsep-endo-apex")
    end_idx = keypoint_names.index("ao-valve-top-inner")

    curve_map = y_pred[curve_idx, :, :]
    start_map = y_pred[start_idx, :, :]
    end_map = y_pred[end_idx, :, :]

    start_point = get_point(start_map)
    end_point = get_point(end_map)

    out = get_path_2d(logits=curve_map,
                      p1s=start_point,
                      p2s=end_point,
                      num_knots=11)

    return out


def get_lv_antsep_rv_path(y_pred: torch.Tensor, keypoint_names):

    curve_idx = keypoint_names.index("curve-lv-antsep-rv")
    start_idx = keypoint_names.index("lv-antsep-rv-apex")
    end_idx = keypoint_names.index("rv-bottom-inner")

    curve_map = y_pred[curve_idx, :, :]
    start_map = y_pred[start_idx, :, :]
    end_map = y_pred[end_idx, :, :]

    start_point = get_point(start_map)
    end_point = get_point(end_map)

    out = get_path_2d(logits=curve_map,
                      p1s=start_point,
                      p2s=end_point,
                      num_knots=11)

    return out


def get_lv_post_endo_path(y_pred: torch.Tensor, keypoint_names):

    curve_idx = keypoint_names.index("curve-lv-post-endo")
    start_idx = keypoint_names.index("lv-post-endo-apex")
    end_idx = keypoint_names.index("mv-inf-wall-hinge")

    curve_map = y_pred[curve_idx, :, :]
    start_map = y_pred[start_idx, :, :]
    end_map = y_pred[end_idx, :, :]

    start_point = get_point(start_map)
    end_point = get_point(end_map)

    out = get_path_2d(logits=curve_map,
                      p1s=start_point,
                      p2s=end_point,
                      num_knots=11)

    return out


def get_lv_post_epi_path(y_pred: torch.Tensor, keypoint_names):

    curve_idx = keypoint_names.index("curve-lv-post-epi")
    start_idx = keypoint_names.index("lv-post-epi-apex")
    end_idx = keypoint_names.index("mv-inf-wall-hinge")

    curve_map = y_pred[curve_idx, :, :]
    start_map = y_pred[start_idx, :, :]
    end_map = y_pred[end_idx, :, :]

    start_point = get_point(start_map)
    end_point = get_point(end_map)

    out = get_path_2d(logits=curve_map,
                      p1s=start_point,
                      p2s=end_point,
                      num_knots=11)

    return out
