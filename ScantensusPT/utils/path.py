import logging

import numpy as np

import scipy.interpolate
import skimage.graph

import torch
import torch.nn
import torch.nn.functional

import scipy.signal

from Scantensus.utils.geometry import interpolate_curveline, line_len
from ScantensusPT.utils.heatmaps import render_gaussian_curve


def get_path_len(points):
    # in is [batch, channels, points, [y, x, conf]]
    # no neet to supply conf
    # out is [batch, channels]
    distance = torch.cumsum(torch.sqrt(torch.sum((points[..., 1:, 0:2] - points[..., :-1, 0:2]) ** 2, dim=-1)), dim=-1)
    return distance[..., -1]


def get_path_len_np(points):
    # in is [batch, channels, points, [y, x, conf]]
    # no neet to supply conf
    # out is [batch, channels]
    distance = np.cumsum(np.sqrt(np.sum((points[..., 1:, 0:2] - points[..., :-1, 0:2]) ** 2, axis=-1)), axis=-1)
    return distance[..., -1]


def get_path_via_2d(logits: torch.Tensor,
                    p1s: torch.Tensor,
                    p2s: torch.Tensor,
                    p3s: torch.Tensor,
                    cost_exp=4,
                    cut_offs=(0.2, 0.05, 0.05, 0.05),
                    smooth_factor=None,
                    num_knots=None,
                    description=""):

    logger = logging.getLogger()
    device = logits.device

    if True:
        p1_p3 = p3s[..., 0:2] - p1s[..., 0:2]

        p1x = p1s[..., 0:2] + 0.1 * p1_p3
        p3x = p3s[..., 0:2] - 0.1 * p1_p3
        p13_mid = p1s[..., 0:2] + 0.5 * p1_p3

        p13_p2 = p13_mid[..., 0:2] - p2s[..., 0:2]

        p2x = p2s[..., 0:2] + 0.2 * p13_p2
        p4x = p13_mid + 0.2 * p13_p2

        block_points = torch.stack([p2x, p4x, p1x, p3x], dim=-2)


    costs = (1.0 - logits) ** cost_exp

    logit = logits.detach().cpu().numpy()

    cost = costs.detach().cpu().numpy()
    p1 = p1s.detach().cpu().numpy()
    p2 = p2s.detach().cpu().numpy()
    p3 = p3s.detach().cpu().numpy()
    block_point = block_points.detach().cpu().numpy()

    try:
        block_point_len = line_len(block_point)
        block_curve_points = block_point_len * 1
        block_ys = block_point[:, 0].tolist()
        block_xs = block_point[:, 1].tolist()
        block_curve_ys, block_curve_xs = interpolate_curveline(ys=block_ys, xs=block_xs, straight_segments=[1]*len(block_ys),
                                                               total_points_out=block_curve_points)

        block_curve_t = torch.tensor((block_curve_ys, block_curve_xs), dtype=torch.float32, device=device).T

        cost_mask = render_gaussian_curve(mean=block_curve_t,
                                          std=torch.tensor([2.0], dtype=torch.float32, device=device),
                                          size=cost.shape)

        cost_mask = cost_mask.mul_(10).detach().cpu().numpy()

        cost = cost + cost_mask

        if p1[..., 2] < cut_offs[1]:
            logger.warning(f"{description} P1: Low confidence {p1[..., 2]}")
            raise Exception

        if p2[..., 2] < cut_offs[2]:
            logger.warning(f"{description} P2: Low confidence {p2[..., 2]}")
            raise Exception

        if p3[..., 2] < cut_offs[3]:
            logger.warning(f"{description} P3: Low Confidence {p3[..., 2]}")
            raise Exception

        path_a, total_cost_a = skimage.graph.route_through_array(cost, p1[..., 0:2].astype(np.int32),
                                                                 p2[..., 0:2].astype(np.int32))

        path_b, total_cost_b = skimage.graph.route_through_array(cost, p2[..., 0:2].astype(np.int32),
                                                                 p3[..., 0:2].astype(np.int32))

        mean_score = 1 - ((total_cost_a + total_cost_b) / (len(path_a) + len(path_b))) ** (1 / 4)
        mean_score = float(mean_score)

        if mean_score < cut_offs[0]:
            logger.warning(f"{description} Low mean_score: {mean_score} < {cut_offs[0]}")
            raise Exception

        path_a.extend(path_b[1:])

        path = np.array(path_a).T

        path_weight = logit[path[0], path[1]]
        path_weight = path_weight ** 2
        path_weight[0] = 10
        path_weight[-1] = 10

        if smooth_factor is not None:
            (t, c, k), u = scipy.interpolate.splprep(x=path, w=path_weight, s=smooth_factor, k=3)
        elif num_knots is not None:
            (t, c, k), u = scipy.interpolate.splprep(x=path, w=path_weight, t=np.linspace(0, 1, num_knots), task=-1, k=3)
        else:
            (t, c, k), u = scipy.interpolate.splprep(x=path, w=path_weight, k=3)

        c = scipy.interpolate.splev(np.linspace(0, 1, num_knots), (t, c, k))

        out_y = c[0].tolist()
        out_x = c[1].tolist()

        # Patch the first and last points back
        out_y[0] = p1[0]
        out_x[0] = p1[1]

        out_y[-1] = p3[0]
        out_x[-1] = p3[1]

        out_conf = [mean_score] * len(out_y)

        return out_y, out_x, out_conf
    except Exception:
        logger.exception(f"{description}: Failed get_path_via_2d")
        return [float('NaN')], [float('NaN')], [float('NaN')]


def get_path_2d(logits: torch.Tensor,
                p1s: torch.Tensor,
                p2s: torch.Tensor,
                cost_exp=4,
                smooth_factor=None,
                num_knots=None,
                description=""):

    logger = logging.getLogger('scantensus')
    device = logits.device

    costs = (1 - logits) ** cost_exp

    logit = logits.cpu().detach().numpy()
    cost = costs.detach().cpu().numpy()
    p1 = p1s.detach().cpu().numpy()
    p2 = p2s.detach().cpu().numpy()

    try:
        if p1[..., 2] < 0.05:
            logger.warning(f"{description} P1: Low confidence {p1[..., 2]}")
            raise Exception

        if p2[..., 2] < 0.05:
            logger.warning(f"{description} P1: Low confidence {p2[..., 2]}")
            raise Exception

        path, total_cost = skimage.graph.route_through_array(cost,
                                                             p1[..., 0:2].astype(np.int32),
                                                             p2[..., 0:2].astype(np.int32),
                                                             fully_connected=True,
                                                             geometric=False)

        mean_score = 1 - (total_cost/len(path))**(1/4)

        if mean_score < 0.1:
            logger.warning(f"Low mean_score: {mean_score} < 0.1")
            raise Exception

        path = np.array(path).T

        path_weight = logit[path[0], path[1]]
        path_weight = path_weight ** 2
        path_weight[0] = 10
        path_weight[-1] = 10

        if smooth_factor is not None:
            (t, c, k), u = scipy.interpolate.splprep(x=path, w=path_weight, s=smooth_factor, k=3)
        elif num_knots is not None:
            (t, c, k), u = scipy.interpolate.splprep(x=path, w=path_weight, t=np.linspace(0, 1, num_knots), task=-1, k=3)
        else:
            (t, c, k), u = scipy.interpolate.splprep(x=path, w=path_weight, k=3)

        c = scipy.interpolate.splev(np.linspace(0, 1, num_knots), (t,c,k))

        out_y = c[0].tolist()
        out_x = c[1].tolist()

        # Patch the first and last points back
        out_y[0] = p1[0]
        out_x[0] = p1[1]

        out_y[-1] = p2[0]
        out_x[-1] = p2[1]

        out_conf = [mean_score] * len(out_y)

        return out_y, out_x, out_conf

    except Exception as e:
        logger.exception("Failed get_path_2d")
        return [float('NaN')], [float('NaN')], [float('NaN')]
