from skimage.draw import line_aa

import torch

from ScantensusPT.utils.heatmaps import render_gaussian_dot_u, render_gaussian_curve_u
from Scantensus.utils.geometry import line_len, line_len_t, interpolate_curveline


def draw_predictions(size, label_dict, keypoint_names, use_polyline=False, device="cpu"):

    num_keypoints = len(keypoint_names)

    point_std = torch.tensor([4], dtype=torch.float32, device=device)

    out = torch.empty((num_keypoints, size[0], size[1]), dtype=torch.uint8, device=device)

    for keypoint_idx, keypoint in enumerate(keypoint_names):

        keypoint_data_list = label_dict.get(keypoint, None)

        if keypoint_data_list is None:
            out[keypoint_idx, ...] = 0
            continue

        if type(keypoint_data_list) is list:
            keypoint_data = keypoint_data_list[0]
        else:
            keypoint_data = keypoint_data_list

        label_type = keypoint_data['type']

        if label_type is None:
            out[keypoint_idx, ...] = 0
            continue

        if label_type == "off":
            out[keypoint_idx, ...] = 0
            continue

        if label_type == "blurred":
            out[keypoint_idx, ...] = 0
            continue

        y_data = keypoint_data["y"]
        x_data = keypoint_data["x"]

        if len(y_data) > 1 and use_polyline:
            out[keypoint_idx, ...] = draw_polyline(size=size, ys=y_data, xs=x_data)
        elif len(y_data) > 1 and not use_polyline:
            points_t = torch.tensor((y_data, x_data)).T
            curve_points_len = line_len_t(points_t)
            curve_points_len = int(curve_points_len)
            curve_points_len = max(curve_points_len, len(y_data))
            total_points_out = curve_points_len * 2
            out_curve_y, out_curve_x = interpolate_curveline(ys=y_data, xs=x_data, straight_segments=None, total_points_out=total_points_out)

            curve_points = torch.tensor([out_curve_y, out_curve_x],
                                        dtype=torch.float,
                                        device=device).T

            out[keypoint_idx, ...] = render_gaussian_curve_u(points=curve_points,
                                                  std=torch.tensor([1.0], device=device),
                                                  size=size,
                                                  mul=255).to(device)

            #out[keypoint_idx, ...] = draw_polyline(size=size, ys=out_curve_y, xs=out_curve_x)

        else:
            y = y_data[0]
            x = x_data[0]
            out[keypoint_idx, ...] = render_gaussian_dot_u(point=torch.tensor([y, x], device=device),
                                                           std=point_std,
                                                           size=size,
                                                           mul=255)

    return out


def draw_polyline(size, ys, xs, device='cpu'):

    height, width = size

    num_points = len(ys)

    ys = [int(y) for y in ys]
    xs = [int(x) for x in xs]

    final_rows = []
    final_cols = []
    final_weights = []

    for i in range(num_points-1):
        rows, cols, weights = line_aa(ys[i], xs[i], ys[i+1], xs[i+1])
        final_rows.extend(rows)
        final_cols.extend(cols)
        final_weights.extend(weights)

    sort_index = sorted(range(len(final_weights)), key=lambda k: final_weights[k])   #sort weights from lowest to highest

    index_rows = [final_rows[x] for x in sort_index]
    index_cols = [final_cols[x] for x in sort_index]
    index_weights = [int(final_weights[x]*255) for x in sort_index]

    good_rows = [y >= 0 and y < height for y in index_rows]
    good_cols = [x >= 0 and x < width for x in index_cols]

    index_rows = [row for row, a, b in zip(index_rows, good_rows, good_cols) if a and b]
    index_cols = [col for col, a, b in zip(index_cols, good_rows, good_cols) if a and b]
    index_weights = [weight for weight, a, b in zip(index_weights, good_rows, good_cols) if a and b]

    heatmap = torch.zeros((height, width), dtype=torch.uint8, device=device)
    heatmap[index_rows, index_cols] = torch.tensor(index_weights, dtype=torch.uint8)

    return heatmap
