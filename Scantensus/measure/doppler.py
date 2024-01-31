import logging

import numpy as np

from .utils import frame_key_to_num


def calc_peak_systole_cm_s(labels, sour):
    ordered_frames = sorted(list(labels.keys()))
    num_frames = frame_key_to_num(ordered_frames[-1]) + 1

    first_frame = ordered_frames[0]

    label_name = 'systolic-peak'

    label_data = labels[first_frame]['labels'][label_name]

    out = []

    if type(label_data) is list:
        label_data_list = label_data
    elif type(label_data) is dict:
        label_data_list = [label_data]
    else:
        raise Exception(f"Unrecognised label data {label_data}")

    for data in label_data_list:
        if data['type'] != 'point':
            pass
        else:
            y_loc = data['y'][0]
            y_dist_cm_s = (y_loc - (sour['ReferencePixelY0'] + sour['RegionLocationMinY0'])) * sour['PhysicalDeltaY']
            meas = {
                'value': float(y_dist_cm_s),
                'frame': first_frame,
                'unit': 'cm/s'
            }
            out.append(meas)

    return out
