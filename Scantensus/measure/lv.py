import logging
import scipy.signal
import scipy.interpolate

import numpy as np

from .utils import frame_key_to_num


def fix_lv_endo_points(labels, start_name = 'mv-ant-wall-hinge', end_name='mv-inf-wall-hinge', curve_name='curve-lv-endo', threshold=0.1):
    try:
        ordered_frames = sorted(list(labels.keys()))
        num_frames = frame_key_to_num(ordered_frames[-1]) + 1

        start_np = np.zeros((num_frames, 2)) * np.nan
        end_np = np.zeros((num_frames, 2)) * np.nan

        for dest_np, label_name in zip([start_np, end_np], [start_name, end_name]):
            for k in ordered_frames:
                frame_num = frame_key_to_num(k)

                if labels[k]['labels'][label_name][0]['type'] == 'point':
                    if labels[k]['labels'][label_name][0]['conf'][0] >= threshold:
                        dest_np[frame_num, 0] = labels[k]['labels'][label_name][0]['y'][0]
                        dest_np[frame_num, 1] = labels[k]['labels'][label_name][0]['x'][0]
    except Exception:
        logging.exception(f"Failed to extract start and end points to fix curve-lv-endo")
        return labels

    try:
        for k in ordered_frames:
            frame_num = frame_key_to_num(k)

            if labels[k]['labels'][curve_name][0]['type'] == 'curve':
                labels[k]['labels'][curve_name][0]['y'][0] = start_np[frame_num, 0]
                labels[k]['labels'][curve_name][0]['x'][0] = start_np[frame_num, 1]
                labels[k]['labels'][curve_name][0]['y'][-1] = end_np[frame_num, 0]
                labels[k]['labels'][curve_name][0]['x'][-1] = end_np[frame_num, 1]
    except Exception:
        logging.exception(f"Failed to fix curve-lv-endo start and end points")
        return labels

    return labels


def calc_lv_tlen(labels, anterior_name, posterior_name, apical_name, threshold=0.1):
    ordered_frames = sorted(list(labels.keys()))
    num_frames = frame_key_to_num(ordered_frames[-1]) + 1

    anterior_np = np.zeros((num_frames, 2)) * np.nan
    posterior_np = np.zeros((num_frames, 2)) * np.nan
    apical_np = np.zeros((num_frames, 2)) * np.nan

    source_names = [anterior_name, posterior_name, apical_name]
    dest_nps = [anterior_np, posterior_np, apical_np]

    for dest_np, label_name in zip(dest_nps, source_names):
        for k in ordered_frames:
            frame_num = frame_key_to_num(k)

            label_data = labels[k]['labels'][label_name]

            if type(label_data) is list:
                inst = label_data[0]
            elif type(label_data) is dict:
                inst = label_data
            else:
                raise Exception(f"Unrecognised label data {label_data}")

            if inst['type'] == 'point':
                if inst['conf'][0] >= threshold:
                    dest_np[frame_num, 0] = inst['y'][0]
                    dest_np[frame_num, 1] = inst['x'][0]

    xs = np.arange(num_frames)

    try:
        for dest_np in dest_nps:
            valid_xs_mask = np.logical_not(np.isnan(dest_np.sum(axis=-1)))
            valid_xs = xs[valid_xs_mask]
            valid_ys = dest_np[valid_xs_mask]

            f = scipy.interpolate.interp1d(valid_xs, valid_ys, kind='cubic', axis=0)
            dest_np[:, :] = f(xs)
            dest_np[:, :] = scipy.signal.savgol_filter(dest_np, 7, 3, axis=0)
    except:
        logging.error(f"Interpolation failed - valid points: {np.sum(valid_xs_mask) / len(valid_xs_mask)}")
        return None

    mid_point_np = (anterior_np + posterior_np) / 2
    apical_np = np.mean(apical_np, axis=0)

    dist = np.abs(mid_point_np - apical_np) ** 2
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)
    dist = scipy.signal.savgol_filter(dist, 7, 3, axis=0)

    return dist


def smooth_and_interpolate_label(labels, label_name, threshold=0.1):

    try:
        ordered_frames = sorted(list(labels.keys()))
        num_frames = frame_key_to_num(ordered_frames[-1]) + 1
        dest_np = np.zeros((num_frames, 3)) * np.nan
    except Exception:
        logging.exception(f"Error in label smoothing - initialisation - returning original labels")
        return labels

    for k in ordered_frames:
        frame_num = frame_key_to_num(k)

        try:
            label_instances = labels[k]['labels'][label_name]
            if type(label_instances) is not list:
                logging.warning(f"Please convert label input to be in instance format")
                logging.warning(f"{label_instances}")
                continue

            if len(label_instances) > 1:
                logging.warning(f"Expecting only one instance - smothing doesn't make sense with more than one - using first")

            inst = label_instances[0]

            if inst['type'] == 'curve':
                logging.warning("Expection point for smoothing not curve")
                continue
            elif inst['type'] == 'point':
                if inst['conf'][0] > threshold:
                    dest_np[frame_num, 0] = labels[k]['labels'][label_name]['y'][0]
                    dest_np[frame_num, 1] = labels[k]['labels'][label_name]['x'][0]
                    dest_np[frame_num, 2] = labels[k]['labels'][label_name]['conf'][0]
            else:
                continue
        except Exception:
            continue

    try:
        xs = np.arange(num_frames)
        valid_xs_mask = np.logical_not(np.isnan(dest_np.sum(axis=-1)))

        first_valid_x_index = np.argmax(np.isfinite(valid_xs_mask))
        first_valid_y = dest_np[first_valid_x_index, :]
        last_valid_x_index = num_frames - np.argmax(np.isfinite(valid_xs_mask[::-1])) - 1
        last_valid_y = dest_np[last_valid_x_index]

        valid_xs = xs[valid_xs_mask]
        valid_ys = dest_np[valid_xs_mask]
    except Exception:
        logging.exception(f"Error in label smoothing - returning original labels")
        return labels

    try:
        f = scipy.interpolate.interp1d(valid_xs,
                                       valid_ys,
                                       kind='cubic',
                                       axis=0,
                                       bounds_error=False,
                                       fill_value=(first_valid_y, last_valid_y))
        dest_np[:, :] = f(xs)
        dest_np[:, :] = scipy.signal.savgol_filter(dest_np, 7, 3, axis=0)
    except Exception:
        logging.error(f"Smoothing failed {label_name} - valid points: {np.sum(valid_xs_mask)} / {len(valid_xs_mask)}")
        return labels

    for k in ordered_frames:
        frame_num = frame_key_to_num(k)
        dest_y = dest_np[frame_num, 0]
        dest_x = dest_np[frame_num, 1]
        dest_conf = dest_np[frame_num, 2]

        labels[k]['labels'][label_name] = []

        if np.isnan(dest_y) or np.isnan(dest_x):
            out = {
                'type': 'blurred',
                'y': [],
                'x': [],
                'conf': []
            }
        else:
            out = {
                'type': 'point',
                'y': [dest_y],
                'x': [dest_x],
                'conf': [0.5]
            }
        labels[k]['labels'][label_name].append(out)
    logging.info(f"Smoothing Success {label_name} - valid points: {np.sum(valid_xs_mask)} / {len(valid_xs_mask)}")
    return labels


def calc_dist_between_2_points(labels, labels_names, threshold=0.1):
    ordered_frames = sorted(list(labels.keys()))
    num_frames = frame_key_to_num(ordered_frames[-1]) + 1

    top_np = np.zeros((num_frames, 2)) * np.nan
    bottom_np = np.zeros((num_frames, 2)) * np.nan

    dest_nps = [top_np, bottom_np]

    for dest_np, label_name in zip(dest_nps, labels_names):
        for k in ordered_frames:
            frame_num = frame_key_to_num(k)

            label_data = labels[k]['labels'][label_name]

            if type(label_data) is list:
                inst = label_data[0]
            elif type(label_data) is dict:
                inst = label_data
            else:
                raise Exception(f"Unrecognised label data {label_data}")

            if inst['type'] == 'point':
                if inst['conf'][0] > threshold:
                    dest_np[frame_num, 0] = inst['y'][0]
                    dest_np[frame_num, 1] = inst['x'][0]

    dist = np.abs(top_np - bottom_np) ** 2
    dist = np.sum(dist, axis=-1)
    dist = np.sqrt(dist)
    #dist = scipy.signal.savgol_filter(dist, 7, 3, axis=0)

    return dist


def find_peaks_cm(dist, min_name: str, max_name: str):

    max_peaks = scipy.signal.find_peaks(dist, distance=10)[0]
    min_peaks = scipy.signal.find_peaks(-dist, distance=10)[0]

    max_90 = np.nanquantile(dist, 0.9)
    min_10 = np.nanquantile(dist, 0.1)

    min_list = []
    max_list = []


    ## make sure you return native float and int, not numpy

    for frame in max_peaks:
        value = dist[frame]
        if value > max_90:
            max_list.append({
                'value': float(value),
                'frame': int(frame),
                'unit': 'cm'
            })

    for frame in min_peaks:
        value = dist[frame]
        if value < min_10:
            min_list.append({
                'value': float(value),
                'frame': int(frame),
                'unit': 'cm'
            })

    out = {
        max_name: max_list,
        min_name: min_list
    }

    return out
