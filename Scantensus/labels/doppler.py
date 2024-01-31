import math
import numpy as np
import logging

import skimage.feature

from .utils import subtract_shift, add_shift
from .trace import get_path_via_2d


def calc_cw_systolic_labels(heatmaps, output_layer_list, unity_f_codes, just_peaks=True):
    y_pred = heatmaps[0]
    label_height_shift = heatmaps[1]
    label_width_shift = heatmaps[2]

    blurred_node = {
        'type': "blurred",
        'y': [],
        'x': [],
        'conf': []
    }

    out_labels_dict = {}

    if just_peaks:
        peaks = skimage.feature.peak_local_max(y_pred[0, output_layer_list.index('systolic-peak'), ...],
                                               min_distance=20, threshold_rel=0.5)

        peaks_list = [("peak", float(y), float(x), y_pred[0, output_layer_list.index('systolic-peak'), y, x]) for y, x in [a for a in peaks]]

        for i, unity_f_code in enumerate(unity_f_codes):
            out_labels_dict[unity_f_code] = {}
            out_labels_dict[unity_f_code]['labels'] = {}
            out_labels_dict[unity_f_code]['labels']['systolic-curve-start'] = []
            out_labels_dict[unity_f_code]['labels']['systolic-peak'] = []
            out_labels_dict[unity_f_code]['labels']['systolic-curve-end'] = []
            out_labels_dict[unity_f_code]['labels']['curve-systolic'] = []

            out_labels_dict[unity_f_code]['labels']['systolic-curve-start'].append(blurred_node)
            out_labels_dict[unity_f_code]['labels']['systolic-curve-end'].append(blurred_node)
            out_labels_dict[unity_f_code]['labels']['curve-systolic'].append(blurred_node)

            if len(peaks_list) == 0:
                out_labels_dict[unity_f_code]['labels']['systolic-peak'].append(blurred_node)

            for inst in peaks_list:

                _, ys, xs, conf = inst

                peak_out = {
                    'type': "point",
                    'y': subtract_shift([ys], label_height_shift),
                    'x': subtract_shift([xs], label_width_shift),
                    'conf': [conf]
                }

                out_labels_dict[unity_f_code]['labels']['systolic-peak'].append(peak_out)


    if not just_peaks:
        peaks = skimage.feature.peak_local_max(y_pred[0, output_layer_list.index('systolic-peak'), ...],
                                               min_distance=20, threshold_rel=0.5)

        starts = skimage.feature.peak_local_max(y_pred[0, output_layer_list.index('systolic-curve-start'), ...],
                                                min_distance=20, threshold_rel=0.5)

        ends = skimage.feature.peak_local_max(y_pred[0, output_layer_list.index('systolic-curve-end'), ...],
                                              min_distance=20, threshold_rel=0.5)

        peaks_list = [("peak", float(y), float(x), y_pred[0, output_layer_list.index('systolic-peak'), y, x]) for y, x in [a for a in peaks]]
        starts_list = [("start", float(y), float(x), y_pred[0, output_layer_list.index('systolic-curve-start'), y, x]) for y, x in [a for a in starts]]
        ends_list = [("end", float(y), float(x), y_pred[0, output_layer_list.index('systolic-curve-end'), y, x]) for y, x in [a for a in ends]]

        all = list()
        all.extend(peaks_list)
        all.extend(starts_list)
        all.extend(ends_list)

        all = sorted(all, key=lambda tup: tup[2])

        logging.warning(f"all: {all}")

        cur = []
        instance_list = []
        for x in all:
            if x[0] == "start":
                cur = []
                cur.append(x)
                continue
            elif x[0] == "peak":
                if len(cur) == 1:
                    cur.append(x)
                    continue
                else:
                    cur = []
                    continue
            elif x[0] == "end":
                if len(cur) == 2:
                    cur.append(x)
                    instance_list.append(cur.copy())
                    cur = []
                    continue
                else:
                    cur = []
                    continue

        print(instance_list)

        for i, unity_f_code in enumerate(unity_f_codes):
            out_labels_dict[unity_f_code] = {}
            out_labels_dict[unity_f_code]['labels'] = {}
            out_labels_dict[unity_f_code]['labels']['systolic-curve-start'] = []
            out_labels_dict[unity_f_code]['labels']['systolic-peak'] = []
            out_labels_dict[unity_f_code]['labels']['systolic-curve-end'] = []
            out_labels_dict[unity_f_code]['labels']['curve-systolic'] = []

            if len(instance_list) == 0:
                out_labels_dict[unity_f_code]['labels']['systolic-curve-start'].append(blurred_node)
                out_labels_dict[unity_f_code]['labels']['systolic-peak'].append(blurred_node)
                out_labels_dict[unity_f_code]['labels']['systolic-curve-end'].append(blurred_node)
                out_labels_dict[unity_f_code]['labels']['curve-systolic'].append(blurred_node)

            for inst in instance_list:
                start_tup = inst[0]
                peak_tup = inst[1]
                end_tup = inst[2]

                out = get_path_via_2d(logits=y_pred[0, 1, ...],
                                      p1s=np.array(start_tup[1:]),
                                      p2s=np.array(peak_tup[1:]),
                                      p3s=np.array(end_tup[1:]),
                                      num_knots=15,
                                      description="systolic-curve")

                ys, xs, conf = out

                if any([not math.isfinite(y) for y in ys]) or any([not math.isfinite(x) for x in xs]):
                    type = "blurred"
                else:
                    type = "curve"

                if type == "blurred":
                    curve_out = blurred_node
                else:
                    curve_out = {
                        'type': type,
                        'y': subtract_shift(ys, label_height_shift),
                        'x': subtract_shift(xs, label_width_shift),
                        'conf': conf
                    }

                start_out = {
                    'type': "point",
                    'y': subtract_shift([start_tup[1]], label_height_shift),
                    'x': subtract_shift([start_tup[2]], label_width_shift),
                    'conf': [start_tup[3]]
                }

                peak_out = {
                    'type': "point",
                    'y': subtract_shift([peak_tup[1]], label_height_shift),
                    'x': subtract_shift([peak_tup[2]], label_width_shift),
                    'conf': [peak_tup[3]]
                }

                end_out = {
                    'type': "point",
                    'y': subtract_shift([end_tup[1]], label_height_shift),
                    'x': subtract_shift([end_tup[2]], label_width_shift),
                    'conf': [end_tup[3]]
                }

                out_labels_dict[unity_f_code]['labels']['systolic-curve-start'].append(start_out)
                out_labels_dict[unity_f_code]['labels']['systolic-peak'].append(peak_out)
                out_labels_dict[unity_f_code]['labels']['systolic-curve-end'].append(end_out)
                out_labels_dict[unity_f_code]['labels']['curve-systolic'].append(curve_out)

    return out_labels_dict


def calc_mv_pw_labels(heatmaps, output_layer_list, unity_f_codes, just_peaks=True):
    y_pred = heatmaps[0]
    label_height_shift = heatmaps[1]
    label_width_shift = heatmaps[2]

    e_peaks_layer = y_pred[0, output_layer_list.index('e-peak'), ...]
    e_peaks = skimage.feature.peak_local_max(e_peaks_layer, min_distance=20, threshold_rel=0.5)

    a_peaks_layer = y_pred[0, output_layer_list.index('a-peak'), ...]
    a_peaks = skimage.feature.peak_local_max(a_peaks_layer, min_distance=20, threshold_rel=0.5)

    e_peaks_list = [("e-peak", float(y), float(x), e_peaks_layer[y, x]) for y, x in [a for a in e_peaks]]
    a_peaks_list = [("a-peak", float(y), float(x), a_peaks_layer[y, x]) for y, x in [a for a in a_peaks]]

    all = list()
    all.extend(e_peaks_list)
    all.extend(a_peaks_list)
    all = sorted(all, key=lambda tup: tup[2])

    logging.warning(f"all: {all}")

    cur = []
    instance_list = []
    for x in all:
        if x[0] == "e-peak":
            if len(cur) == 1:
                instance_list.append(cur.copy())
            cur = []
            cur.append(x)
            continue
        elif x[0] == "a-peak":
            if len(cur) == 1:
                cur.append(x)
                instance_list.append(cur.copy())
                cur = []
                continue
            else:
                cur = []
                continue
        else:
            raise Exception(f"Expecting either e-peak or a-peak")

    if len(cur) > 0:
        instance_list.append(cur.copy())
        cur = []

    logging.info(instance_list)

    blurred_node = {
        'type': "blurred",
        'y': [],
        'x': [],
        'conf': []
    }

    out_labels_dict = {}

    for i, unity_f_code in enumerate(unity_f_codes):
        out_labels_dict[unity_f_code] = {}
        out_labels_dict[unity_f_code]['labels'] = {}
        out_labels_dict[unity_f_code]['labels']['e-peak'] = []
        out_labels_dict[unity_f_code]['labels']['a-peak'] = []

        for inst in instance_list:
            if len(inst) > 0:
                e_peak = inst[0]

                e_peak_out = {
                    'type': "point",
                    'y': subtract_shift([e_peak[1]], label_height_shift),
                    'x': subtract_shift([e_peak[2]], label_width_shift),
                    'conf': [e_peak[3]]
                }


                out_labels_dict[unity_f_code]['labels']['e-peak'].append(e_peak_out)

            if len(inst) == 2:
                a_peak = inst[1]

                a_peak_out = {
                    'type': "point",
                    'y': subtract_shift([a_peak[1]], label_height_shift),
                    'x': subtract_shift([a_peak[2]], label_width_shift),
                    'conf': [a_peak[3]]
                }

                out_labels_dict[unity_f_code]['labels']['a-peak'].append(a_peak_out)

        if len(out_labels_dict[unity_f_code]['labels']['e-peak']) == 0:
            out_labels_dict[unity_f_code]['labels']['e-peak'].append(blurred_node)

        if len(out_labels_dict[unity_f_code]['labels']['a-peak']) == 0:
            out_labels_dict[unity_f_code]['labels']['a-peak'].append(blurred_node)
    logging.warning("doppler-mv-pw")
    logging.warning(out_labels_dict)
    return out_labels_dict

