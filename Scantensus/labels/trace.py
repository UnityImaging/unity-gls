import logging

import numpy as np

import scipy.interpolate

import skimage.draw
import skimage.graph


def get_path_via_2d(logits: np.ndarray,
                    p1s: np.ndarray,
                    p2s: np.ndarray,
                    p3s: np.ndarray,
                    cost_exp=4,
                    cut_offs=(0.2, 0.05, 0.05, 0.05),
                    smooth_factor=None,
                    num_knots=None,
                    description=""):

    failed_out = [float('NaN')], [float('NaN')], [float('NaN')]

    size_y, size_x = logits.shape

    if True:
        p1_p3 = p3s[..., 0:2] - p1s[..., 0:2]

        p1x = p1s[..., 0:2] + 0.1 * p1_p3
        p3x = p3s[..., 0:2] - 0.1 * p1_p3
        p13_mid = p1s[..., 0:2] + 0.5 * p1_p3

        p13_p2 = p13_mid[..., 0:2] - p2s[..., 0:2]

        p2x = p2s[..., 0:2] + 0.2 * p13_p2
        p4x = p13_mid + 0.2 * p13_p2

        block_points = np.stack([p2x, p4x, p1x, p3x], axis=-2)

    cost = (1.0 - logits) ** cost_exp

    logit = logits

    p1 = p1s
    p2 = p2s
    p3 = p3s
    block_point = block_points

    try:
        block_ys = block_point[:, 0].tolist()
        block_xs = block_point[:, 1].tolist()

        cost_mask = np.zeros_like(cost)
        last_y = None
        last_x = None

        for cur_y, cur_x in zip(block_ys, block_xs):
            if last_y is not None:
                rr, cc, val = skimage.draw.line_aa(int(last_y), int(last_x),  int(cur_y), int(cur_x)) #r0, r1, c0, c1
                line_mask = (rr < size_y) & (cc < size_x) & (rr >= 0) & (rr >= 0)
                cost_mask[rr[line_mask], cc[line_mask]] = 10 # A large value
            last_y = cur_y
            last_x = cur_x

        cost = cost + cost_mask

        if p1[..., 2] < cut_offs[1]:
            logging.debug(f"{description} P1: Low confidence {p1[..., 2]}")
            return failed_out

        if p2[..., 2] < cut_offs[2]:
            logging.debug(f"{description} P2: Low confidence {p2[..., 2]}")
            return failed_out

        if p3[..., 2] < cut_offs[3]:
            logging.debug(f"{description} P3: Low Confidence {p3[..., 2]}")
            return failed_out

        path_a, total_cost_a = skimage.graph.route_through_array(cost, p1[..., 0:2].astype(np.int32),
                                                                 p2[..., 0:2].astype(np.int32))

        path_b, total_cost_b = skimage.graph.route_through_array(cost, p2[..., 0:2].astype(np.int32),
                                                                 p3[..., 0:2].astype(np.int32))

        mean_score = 1 - ((total_cost_a + total_cost_b) / (len(path_a) + len(path_b))) ** (1 / 4)
        mean_score = float(mean_score)

        if mean_score < cut_offs[0]:
            logging.debug(f"{description} Low mean_score: {mean_score} < {cut_offs[0]}")
            return failed_out

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
        logging.exception(f"{description}: Failed get_path_via_2d")
        return failed_out
