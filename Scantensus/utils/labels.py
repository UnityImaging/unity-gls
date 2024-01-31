import logging
logger = logging.getLogger()

import math
import copy
import distutils.util
from typing import Sequence, Dict
from pathlib import Path

from ScantensusPT.utils.path import get_path_len_np

import numpy as np

from Scantensus.utils.geometry import interpolate_curveline, line_len


def curve_list_to_str(curve_list: Sequence[float], round_digits=1):
    out = " ".join([str(round(value, round_digits)) for value in curve_list])
    return out


def curve_str_to_list(curve_str: str):
    out = [(float(value)) for value in curve_str.split()]
    return out


def curve_str_to_list_int(curve_str: str):
    out = [(int(value)) for value in curve_str.split()]
    return out


def bool_list_to_str(curve_list: Sequence[bool]):
    out = " ".join([str(value) for value in curve_list])
    return out


def bool_str_to_list(curve_str: str):
    out = [(bool(distutils.util.strtobool(value))) for value in curve_str.split()]
    return out


def labels_convert_str_to_list(labels_dict: Dict):

    labels_dict = copy.deepcopy(labels_dict)

    for unity_f_code, data in labels_dict.items():
        for label in data['labels'].keys():

            curve_y_str = data['labels'][label]['y']
            curve_y = curve_str_to_list(curve_y_str)
            data['labels'][label]['y'] = curve_y

            curve_x_str = data['labels'][label]['x']
            curve_x = curve_str_to_list(curve_x_str)
            data['labels'][label]['x'] = curve_x

            try:
                curve_conf_str = data['labels'][label]['conf']
                curve_conf = curve_str_to_list(curve_conf_str)
                data['labels'][label]['conf'] = curve_conf
            except Exception as e:
                pass

            try:
                curve_straight_segment_str = data['labels'][label]['straight_segment']
                curve_straight_segment = bool_str_to_list(curve_straight_segment_str)
                data['labels'][label]['straight_segment'] = curve_straight_segment
            except Exception as e:
                pass

    return labels_dict


def labels_convert_list_to_str(labels_dict: Dict):

    labels_dict = copy.deepcopy(labels_dict)

    for unity_f_code, data in labels_dict.items():
        for label in data['labels'].keys():
            for i in range(len(data['labels'][label])):
                curve_y = data['labels'][label][i]['y']
                curve_y_str = curve_list_to_str(curve_y)
                data['labels'][label][i]['y'] = curve_y_str

                curve_x = data['labels'][label][i]['x']
                curve_x_str = curve_list_to_str(curve_x)
                data['labels'][label][i]['x'] = curve_x_str

                try:
                    curve_conf = data['labels'][label]['conf']
                    curve_conf_str = curve_list_to_str(curve_conf)
                    data['labels'][label]['conf'] = curve_conf_str
                except Exception as e:
                    pass

                try:
                    curve_straight_segment = data['labels'][label]['straight_segment']
                    curve_straight_segment_str = curve_list_to_str(curve_straight_segment)
                    data['labels'][label]['straight_segment'] = curve_straight_segment_str
                except Exception as e:
                    pass

    return labels_dict


def csv_convert_list_to_str(label_list: Sequence):

    out = []

    for data in label_list:
        out_data = copy.deepcopy(data)

        try:
            curve_y = data['y']
            curve_y_str = curve_list_to_str(curve_y)
            data['y'] = curve_y_str
        except Exception as e:
            pass

        try:
            curve_x = data['x']
            curve_x_str = curve_list_to_str(curve_x)
            data['x'] = curve_x_str
        except Exception as e:
            pass

        try:
            curve_conf = data['conf']
            curve_conf_str = curve_list_to_str(curve_conf)
            data['conf'] = curve_conf_str
        except Exception as e:
            pass

        try:
            curve_straight_segment = data['straight_segment']
            curve_straight_segment_str = curve_list_to_str(curve_straight_segment)
            data['straight_segment'] = curve_straight_segment_str
        except Exception as e:
            pass

        out.append(out_data)

    return out


def csv_convert_str_to_list(label_list: Sequence):

    out = []

    for data in label_list:
        out_data = copy.deepcopy(data)

        try:
            curve_y_str = data['y']
            curve_y = curve_str_to_list(curve_y_str)
            data['y'] = curve_y
        except Exception as e:
            pass

        try:
            curve_x_str = data['x']
            curve_x = curve_str_to_list(curve_x_str)
            data['x'] = curve_x
        except Exception as e:
            pass

        try:
            curve_conf_str = data['conf']
            curve_conf = curve_str_to_list(curve_conf_str)
            data['conf'] = curve_conf
        except Exception as e:
            pass

        try:
            curve_straight_segment_str = data['straight_segment']
            curve_straight_segment = bool_str_to_list(curve_straight_segment_str)
            data['straight_segment'] = curve_straight_segment
        except Exception as e:
            pass

        out.append(out_data)

    return out



def labels_upsample(labels_dict: Dict,
                    labels_upsample_dict: Dict,
                    upsample_method: str,
                    is_label_dict_single: bool = False):

    if is_label_dict_single:
        temp = {}
        temp['fake_unity_code'] = {}
        temp['fake_unity_code']['labels'] = labels_dict
        labels_dict = temp

    labels_dict = copy.deepcopy(labels_dict)

    for unity_f_code, data in labels_dict.items():

        for label, upsample_number in labels_upsample_dict.items():
            if upsample_number == 0:
                continue
            try:
                curve_y = data['labels'][label]['y']
                curve_x = data['labels'][label]['x']
                curve_conf = data['labels'][label].get('conf', None)

                if len(curve_x) > 1:

                    if upsample_method == 'points':
                        target_points = upsample_number
                    elif upsample_method == 'ratio':
                        target_points = math.ceil(upsample_number * len(curve_x))
                    elif upsample_method =='len_ratio':
                        curve = np.stack((curve_y, curve_x))
                        in_curve_len = line_len(curve)
                        target_points = math.ceil(in_curve_len * upsample_number)
                    else:
                        logging.error(f"Something has gone wrong in labels_upsample")
                        raise Exception

                    out_curve_ys, out_curve_xs = interpolate_curveline(ys=curve_y, xs=curve_x, straight_segments=None,
                                                                       total_points_out=target_points)

                    data['labels'][label]['y'] = out_curve_ys
                    data['labels'][label]['x'] = out_curve_xs
                    data['labels'][label]['straight_segment'] = [1] * len(out_curve_ys)

                    if curve_conf is not None:
                        out_curve_confs, out_curve_xs = interpolate_curveline(ys=curve_conf, xs=curve_x,
                                                                              straight_segments=None,
                                                                              total_points_out=target_points)

                        data['labels'][label]['conf'] = out_curve_confs

            except Exception as e:
                continue

    if is_label_dict_single:
        labels_dict = labels_dict['fake_unity_code']['labels']

    return labels_dict


def labels_calc_len(labels_dict: Dict):

    labels_dict = copy.deepcopy(labels_dict)

    for unity_f_code, data in labels_dict.items():
        for label in data['labels'].keys():
            for i in range(len(data['labels'][label])):
                curve_y = data['labels'][label][i]['y']
                curve_x = data['labels'][label][i]['x']

                if len(curve_x) >= 2:
                    curve = np.stack((curve_y, curve_x), axis=-1)
                    out_curve_len = line_len(curve)
                    out_curve_len = round(float(out_curve_len), 2)

                else:
                    out_curve_len = float('nan')

                data['labels'][label][i]['curve_len'] = out_curve_len

    return labels_dict


def convert_labels_to_firebase(out_labels_dict: Dict,
                               labels_to_process: Sequence[str],
                               firebase_reverse_dict: Dict,
                               user_name: str = "thready_prediction"):

    # Labels with lists

    out_firebase_dict = {}

    for unity_code, unity_data in out_labels_dict.items():

        data = unity_data['labels']

        out_unity_code = Path(unity_code).stem + ":png"

        out_firebase_dict[out_unity_code] = {}
        out_firebase_dict[out_unity_code]['curves'] = {}
        out_firebase_dict[out_unity_code]['nodes'] = {}

        if labels_to_process:
            label_names = labels_to_process
        else:
            label_names = data.keys()

        for label_name in label_names:
            try:
                firebase_label_name = firebase_reverse_dict[label_name]
            except Exception:
                logging.warning(f"{label_name} not in Firebase reverse dict, using original")
                firebase_label_name = label_name

            try:
                label_data = data[label_name]
            except KeyError:
                logging.debug(f"Converting labels to firebase: {label_name} not found in data - perhaps not generated - skipping")
                continue

            if type(label_data) is list:
                label_format = "v2"
            elif type(label_data) is dict:
                label_format = "v1"
                label_data = [label_data]
            else:
                logging.error("Unrecognised label format")
                continue

            try:
                if label_data[0]['type'] == 'curve':
                    firebase_type = 'curves'
                elif label_data[0]['type'] == 'point':
                    firebase_type = 'nodes'
                else: #blurred or off
                    continue
            except Exception:
                logging.exception(f"Error in firebase convert {out_unity_code} {label_data}")
                continue

            out_firebase_dict[out_unity_code][firebase_type][firebase_label_name] = {}
            out_firebase_dict[out_unity_code][firebase_type][firebase_label_name][user_name] = {}

            working_dict = out_firebase_dict[out_unity_code][firebase_type][firebase_label_name][user_name]

            working_dict['format'] = 2
            working_dict['n'] = 1
            working_dict['vis'] = 'blurred' # overwrite below
            working_dict['instances'] = []

            for inst in label_data:
                path_type = inst['type']
                path_y = inst['y']
                path_x = inst['x']
                path_conf = inst.get('conf', None)

                if path_type == 'curve':
                    firebase_type = 'curves'
                    firebase_type_node = 'nodes'
                    working_dict['vis'] = 'seen'
                elif path_type == 'point':
                    firebase_type = 'nodes'
                    firebase_type_node = 'node'
                    working_dict['vis'] = 'seen'
                else: #blurred or off
                    continue

                if path_type == 'curve':

                    node_keys = []

                    num_nodes = len(path_x)

                    for i in range(num_nodes):
                        temp = {}
                        temp['i2'] = 0.01
                        temp['x'] = round(float(path_x[i]), 1)
                        temp['y'] = round(float(path_y[i]), 1)
                        if path_conf is not None:
                            temp['z'] = round(float(path_conf[i]), 2)

                        if label_name == "curve-lv-trabec":
                            if i == 0:
                                temp['nodeKey'] = 'mv-ant-wall-hinge'
                            elif i == (num_nodes // 2):
                                temp['nodeKey'] = 'lv-apex-trabec'
                            elif i == (num_nodes - 1):
                                temp['nodeKey'] = 'mv-inf-wall-hinge'

                        if label_name == "curve-lv-endo":
                            if i == 0:
                                temp['nodeKey'] = 'mv-ant-wall-hinge'
                            elif i == (num_nodes // 2):
                                temp['nodeKey'] = 'lv-apex-endo'
                            elif i == (num_nodes - 1):
                                temp['nodeKey'] = 'mv-inf-wall-hinge'

                        if label_name == "curve-lv-midmyo":
                            if i == 0:
                                temp['nodeKey'] = 'mv-ant-wall-hinge-midmyo'
                            elif i == (num_nodes // 2):
                                temp['nodeKey'] = 'lv-apex-midmyo'
                            elif i == (num_nodes - 1):
                                temp['nodeKey'] = 'mv-inf-wall-hinge-midmyo'

                        if label_name == "curve-systolic":
                            if i == 0:
                                temp['nodeKey'] = 'systolic-curve-start'
                            elif i == (num_nodes // 2):
                                temp['nodeKey'] = 'systolic-peak'
                            elif i == (num_nodes - 1):
                                temp['nodeKey'] = 'systolic-curve-end'

                        if label_name == "la-endo":
                            if i == 0:
                                temp['nodeKey'] = 'mv-ant-wall-hinge'
                            elif i == (num_nodes - 1):
                                temp['nodeKey'] = 'mv-inf-wall-hinge'

                        if label_name == "ra-endo":
                            if i == 0:
                                temp['nodeKey'] = 'tv-ant-hinge'
                            elif i == (num_nodes - 1):
                                temp['nodeKey'] = 'tv-sep-hinge'

                        node_keys.append(temp)

                    working_dict['instances'].append({firebase_type_node: node_keys})

                if path_type == 'point':
                    node_key = {}
                    node_key['i2'] = 0.01
                    node_key['x'] = round(float(path_x[0]), 1)
                    node_key['y'] = round(float(path_y[0]), 1)
                    if path_conf is not None:
                        node_key['z'] = round(float(path_conf[0]), 2)
                    node_key['nodeKey'] = label_name

                    working_dict['instances'].append(node_key)

    return out_firebase_dict


def convert_labels_to_csv(labels_dict: dict,
                          labels_to_process: Sequence[str],
                          project: str,
                          user: str = 'scantensus-echo'):

    out_csv_list = []

    for unity_f_code, data in labels_dict.items():
        for label in labels_to_process:
            for i in range(len(data['labels'][label])):
                curve_y = data['labels'][label][i]['y']
                curve_x = data['labels'][label][i]['x']
                type = data['labels'][label][i]['type']
                curve_conf = data['labels'][label][i].get('conf', [])
                curve_len = data['labels'][label][i].get('curve_len', float("nan"))

                out_csv = {"file": unity_f_code + ".png",
                           "label": label,
                           "user": user,
                           "time": "",
                           "project": project,
                           "type": type,
                           "instance_num": i,
                           "value_y": curve_list_to_str(curve_y, 1),
                           "value_x": curve_list_to_str(curve_x, 1),
                           "conf": curve_list_to_str(curve_conf, 2),
                           "curve_len": str(curve_len)}

                out_csv_list.append(out_csv)

    return out_csv_list


def label_shift(label_dict: dict, label_height_shift, label_width_shift):

    label_dict = copy.deepcopy(label_dict)
    for label, inst_list in label_dict.items():
        for inst in inst_list:
            curve_y = inst['y']
            curve_x = inst['x']

            curve_y = [y + label_height_shift for y in curve_y]
            curve_x = [x + label_width_shift for x in curve_x]

            inst['y'] = curve_y
            inst['x'] = curve_x

    return label_dict

