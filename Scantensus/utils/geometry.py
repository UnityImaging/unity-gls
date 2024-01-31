from typing import Sequence, Union, Optional
import numpy as np
import torch
import scipy.interpolate


def dedupe_points(points: np.array):
    last = None
    temp = []
    for point in points:
        if not (point == last).all():
            temp.append(point)
        last = point

    points = np.array(temp)

    return points


def line_len(points: np.array):
    line_len = np.sum(np.sqrt(np.sum((points[1:, 0:2] - points[:-1, 0:2]) ** 2, axis=-1)), axis=-1)
    return line_len


def line_len_t(points: torch.tensor):
    line_len = torch.sum(torch.sqrt(torch.sum((points[1:, 0:2] - points[:-1, 0:2]) ** 2, dim=-1)), dim=-1)
    return line_len


def interpolate_line(points: np.array, num_points: int, even_spacing=True):

    ##REMEMBER AXIS = 0 (by default is -1)

    if even_spacing:
        n = points.shape[0]
        distances = np.arange(n)
        curve_distance_max = n - 1
    else:
        # Uneven spacing not implemented
        raise Exception

    line = scipy.interpolate.interp1d(x=distances, y=points, axis=0, kind='linear')

    new_distances = np.linspace(0, curve_distance_max, num_points)

    return line(new_distances)


def interpolate_curve(points: np.array, num_points: int, even_spacing=True):

    num_points = int(num_points)

    if even_spacing:
        n = points.shape[0]
        distances = np.arange(n)
        curve_distance_max = n - 1
    else:
        raise Exception
        #distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        #num_curve_points = distance[-1]
        #distance = np.insert(distance, 0, 0)
        #distance = distance / distance[-1]


    cs = scipy.interpolate.CubicSpline(distances, points, bc_type='natural')

    new_distances = np.linspace(0, curve_distance_max, num_points)

    return cs(new_distances)


def interpolate_curveline(ys: Sequence[float], xs: Sequence[float], straight_segments: Optional[Sequence[int]] = None, total_points_out: int = 200):
    #straight_segments - straight is true, curve is false

    num_points_in = len(ys)

    if total_points_out < num_points_in:
        raise Exception(f"xs: {xs}, ys: {ys}, ss: {straight_segments}, npi: {num_points_in}")

    if straight_segments is None:
        straight_segments = [0] * num_points_in

    num_segments_in = num_points_in - 1
    points_per_segment_out = total_points_out // num_segments_in
    extra_points_last_segment = total_points_out - (points_per_segment_out * num_segments_in)

    out_curve_y = []
    out_curve_x = []

    temp_curve_y = []
    temp_curve_x = []

    last_straight_marker = False

    # This works like a state machine.
    # Iterate through the points
    # Add them to a temporary list
    # Every time you come across a straight segment
    # Interpolate the points that you have so far.
    # Add that point again to the list and keep on.
    # Trigger on next loop or if next point is last.

    for i in range(num_points_in):
        temp_curve_y.append(ys[i])
        temp_curve_x.append(xs[i])

        straight_segment = straight_segments[i]
        is_last_segment = (i == (num_points_in - 1))

        if straight_segment or last_straight_marker or is_last_segment:

            partial_curve_points = len(temp_curve_x)
            partial_curve_segments = partial_curve_points - 1

            if partial_curve_points == 0:
                raise Exception  # this shouldn't happen
            elif partial_curve_points == 1:
                pass  # The first point is a straight segment
            elif partial_curve_points == 2:
                if is_last_segment:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(partial_curve_segments*points_per_segment_out)+extra_points_last_segment)
                    out_curve_y.extend(temp[:, 0].tolist())
                    out_curve_x.extend(temp[:, 1].tolist())
                else:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(partial_curve_segments*points_per_segment_out)+1)
                    out_curve_y.extend(temp[:-1, 0].tolist())
                    out_curve_x.extend(temp[:-1, 1].tolist())
            elif partial_curve_points > 2:
                if is_last_segment:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(points_per_segment_out*partial_curve_segments)+extra_points_last_segment)
                    out_curve_y.extend(temp[:, 0].tolist())
                    out_curve_x.extend(temp[:, 1].tolist())
                else:
                    temp = interpolate_curve(np.stack((temp_curve_y, temp_curve_x), axis=1),
                                             num_points=(points_per_segment_out*partial_curve_segments)+1)
                    out_curve_y.extend(temp[:-1, 0].tolist())
                    out_curve_x.extend(temp[:-1, 1].tolist())

            temp_curve_x = []
            temp_curve_y = []

            temp_curve_y.append(ys[i])
            temp_curve_x.append(xs[i])

            if straight_segment:
                last_straight_marker = True
            else:
                last_straight_marker = False

        else:
            continue

    return out_curve_y, out_curve_x
