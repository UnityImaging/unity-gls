import logging
import json
import datetime

from pathlib import Path

from typing import List, Dict


def curve_list_to_str(curve_list: List, round_digits=1):
    out = " ".join([str(round(value, round_digits)) for value in curve_list])
    return out


def curve_str_to_list(curve_str: str):
    out = [(float(value)) for value in curve_str.split()]
    return out


def get_keypoint_names_and_colors_from_json(json_path):
    if isinstance(json_path, (str, Path)):
        with open(json_path, "r") as read_file:
            data = json.load(read_file)

    elif isinstance(json_path, (bytes,)):
        data = json.loads(json_path)

    else:
        raise Exception("Unknown type passed to get_keypoint_names_and_colors_from_json")

    keypoint_names = list(data.keys())
    keypoint_cols = []

    for keypoint in keypoint_names:
        r, g, b = data[keypoint]['rgb'].split()
        keypoint_cols.append([float(r), float(g), float(b)])

    return keypoint_names, keypoint_cols


def get_cols_from_firebase_project_config(config_data: Dict):
    ## pass it this firebase_data['fiducial'][project]['config']

    col_data = config_data

    col_dict = {}

    try:
        for item in col_data['nodes'].items():
            item_name = item[0]
            col = item[1]['color'][1:]
            r = int(col[:2], 16) / 255
            g = int(col[2:4], 16) / 255
            b = int(col[4:6], 16) / 255
            col_dict[item_name] = [r, g, b]
    except Exception as e:
        logging.info(f'get_cols_from_firebase_project_config: no nodes')

    try:
        for item in col_data['curves'].items():
            item_name = item[0]
            col = item[1]['color'][1:]
            r = int(col[:2], 16) / 255
            g = int(col[2:4], 16) / 255
            b = int(col[4:6], 16) / 255
            col_dict[item_name] = [r, g, b]
    except Exception as e:
        logging.info(f'get_cols_from_firebase_project_config: no curves')

    return col_dict


def get_labels_from_firebase_project_data(project_data, project, mapping_data, EXPORT_FORMAT='backup'):
    logging.info(f'Project: {project}')
    # project_data = firebase_data['fiducial'][project]['labels']

    ACCEPT_TYPE_PROJECT = False

    db = []
    i = 0
    for image_name, image_data in project_data.items():
        i = i + 1
        if i % 100 == 0:
            logging.info(f"Parsing label {i}")

        if image_name.startswith('01-') or image_name.startswith('02-') or image_name.startswith('32-'):
            file = image_name.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            file = file.replace(":", ".")
        else:
            file_name_mangled_split = image_name.split('~')
            project_id = file_name_mangled_split[0]
            naming_scheme = file_name_mangled_split[1]

            if naming_scheme == 'unique':
                file = file_name_mangled_split[2].replace(":", ".")
                file = file.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            elif naming_scheme == 'clusters':
                file = image_name.replace(":", ".")
            else:
                logging.warning("Unrecognised file naming scheme")

        if project_id != project:
            logging.warning(f"project_id / project mismatch {project_id} / {project}")
            continue

        user_dict = {}

        try:
            for user_item in image_data['events'].items():
                try:
                    user_code = user_item[0]
                    try:
                        user = user_item[1]['user']
                    except Exception as e:
                        user = "unknown"
                    time_stamp = user_item[1]['t']

                    user_data = {}
                    user_data['user'] = user
                    user_data['time_stamp'] = time_stamp

                    user_dict[user_code] = user_data
                except Exception:
                    logging.exception("Error in User events")
                    print(f"user_item: {user_item}")
        except Exception:
            ACCEPT_TYPE_PROJECT = True

        nodes = image_data.get('nodes', {})
        curves = image_data.get('curves', {})

        labels = {**nodes, **curves}

        for label_name_old, label_data_all in labels.items():

            label_name = mapping_data.get(label_name_old, None)

            if label_name is None:
                logging.warning(f"Missing mapping for {label_name_old}")
                continue

            for user_code, node_data in label_data_all.items():
                if not ACCEPT_TYPE_PROJECT:
                    true_user = user_dict[user_code]['user']
                    true_time_stamp = user_dict[user_code]['time_stamp']
                else:
                    true_user = user_code
                    true_time_stamp = datetime.datetime.utcnow().isoformat() + 'Z'

                if 'format' in node_data:
                    DATABASE_FORMAT = node_data['format']
                else:
                    DATABASE_FORMAT = 1

                vis = node_data.get('vis', 'seen')

                if vis == "unasked":
                    continue

                if vis == "blurred":
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'vis': 'blurred',
                           'value_x': '',
                           'value_y': '',
                           'straight_segment': '',
                           }

                    db.append(out)
                    continue

                if vis == 'off':
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'vis': 'off',
                           'value_x': '',
                           'value_y': '',
                           'straight_segment': '',
                           }
                    db.append(out)
                    continue

                try:
                    if DATABASE_FORMAT == 2:
                        freehand_node = False
                        node_data_instances = node_data.get('instances')

                        if EXPORT_FORMAT == 'export':
                            node_instance = node_data_instances[0]

                        if EXPORT_FORMAT == 'backup':
                            node_instance = node_data_instances['0']

                        if node_instance.get('isFreehand'):
                            freehand_node = True
                            single_node = False
                            try:
                                node_list = node_instance['freehandPoints']
                            except:
                                logging.exception(f"No freehandPoints despite freehand")
                                continue

                        elif 'node' in node_instance:
                            single_node = True
                            node_list = node_instance['node']

                        elif 'nodes' in node_instance:
                            single_node = False
                            node_list = node_instance['nodes']

                        else:
                            single_node = True
                            node_list = node_instance


                    elif DATABASE_FORMAT == 1:
                        vis = 'seen'
                        freehand_node = False
                        if type(node_data) is list:
                            EXPORT_FORMAT = 'export'
                            single_node = False
                        elif type(node_data) is dict:
                            if node_data.get('0', None):
                                single_node = False
                            if node_data.get('x', None):
                                single_node = True
                        else:
                            print("error")

                        node_list = node_data

                    else:
                        raise Exception("Unrecognised format")

                except Exception as e:
                    logging.exception(f"Error in database format, vis {vis}")
                    continue

                curve_x = []
                curve_y = []
                curve_next_straight = []

                if not single_node:
                    if EXPORT_FORMAT == 'export':
                        for node in node_list:
                            curve_x.append(node['x'])
                            curve_y.append(node['y'])
                            if not freehand_node:
                                curve_next_straight.append(node.get("straightToNext", False))
                            else:
                                curve_next_straight.append(node.get("straightToNext", False))
                    elif EXPORT_FORMAT == 'backup':
                        keys = sorted([int(x) for x in node_list.keys()])
                        for key in keys:
                            node = node_list[str(key)]
                            curve_x.append(node['x'])
                            curve_y.append(node['y'])
                            if not freehand_node:
                                curve_next_straight.append(node.get("straightToNext", False))
                            else:
                                curve_next_straight.append(node.get("straightToNext", False))
                    else:
                        logging.error("error, unknown node-list type")

                elif single_node:
                    try:
                        curve_x.append(node_list['x'])
                        curve_y.append(node_list['y'])
                        curve_next_straight.append(False)
                    except Exception as e:
                        logging.exception("Error in single node")

                curve_next_straight = [1 if x else 0 for x in curve_next_straight]

                if freehand_node:
                    target_num_points = 50
                    num_points = len(curve_x)

                    if num_points > target_num_points:
                        select_idx = list(range(target_num_points))
                        select_idx = [((x * num_points) - 1) // (target_num_points - 1) for x in select_idx]
                        curve_x = [curve_x[i] for i in select_idx]
                        curve_y = [curve_y[i] for i in select_idx]
                        curve_next_straight = [curve_next_straight[i] for i in select_idx]

                if len(curve_x) == 0 or len(curve_y) == 0:
                    logging.error(f"Unexpected empty curve_x for seen")
                    out_vis = "blurred"
                else:
                    out_vis = "seen"

                out = {'project': project,
                       'file': file,
                       'user': true_user,
                       'time': true_time_stamp,
                       'label': label_name,
                       'vis': out_vis,
                       'value_x': curve_list_to_str(curve_x, round_digits=3),
                       'value_y': curve_list_to_str(curve_y, round_digits=3),
                       'straight_segment': curve_list_to_str(curve_next_straight),
                       }

                db.append(out)

    return db


def get_labels_from_firebase_project_data_new(project_data, project, mapping_data, EXPORT_FORMAT='backup'):
    logging.info(f'Project: {project}')
    # project_data = firebase_data['fiducial'][project]['labels']

    ACCEPT_TYPE_PROJECT = False

    db = []
    logging.info(f"Project {project} has {len(project_data)} labels")
    for image_name, image_data in project_data.items():

        if image_name[0].isnumeric() and image_name[1].isnumeric() and image_name[2] == "-":
            file = image_name.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            file = file.replace(":", ".")
        else:
            file_name_mangled_split = image_name.split('~')
            project_id = file_name_mangled_split[0]
            naming_scheme = file_name_mangled_split[1]

            if naming_scheme == 'unique':
                file = file_name_mangled_split[2].replace(":", ".")
                file = file.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")
            elif naming_scheme == 'clusters':
                file = image_name.replace(":", ".")
            else:
                logging.warning("Unrecognised file naming scheme")

        user_dict = {}

        try:
            for user_item in image_data['events'].items():
                try:
                    user_code = user_item[0]
                    try:
                        user = user_item[1]['user']
                    except Exception as e:
                        user = "unknown"
                    time_stamp = user_item[1]['t']

                    user_data = {}
                    user_data['user'] = user
                    user_data['time_stamp'] = time_stamp

                    user_dict[user_code] = user_data
                except Exception:
                    logging.exception("Error in User events")
                    print(f"user_item: {user_item}")
        except Exception:
            ACCEPT_TYPE_PROJECT = True

        nodes = image_data.get('nodes', {})
        curves = image_data.get('curves', {})

        labels = {**nodes, **curves}

        for label_name_old, label_data_all in labels.items():

            label_name = mapping_data.get(label_name_old, None)

            if label_name is None:
                logging.warning(f"Missing mapping for {label_name_old}")
                continue

            for user_code, node_data in label_data_all.items():
                if not ACCEPT_TYPE_PROJECT:
                    true_user = user_dict[user_code]['user']
                    true_time_stamp = user_dict[user_code]['time_stamp']
                else:
                    true_user = user_code
                    true_time_stamp = datetime.datetime.utcnow().isoformat() + 'Z'

                if 'format' in node_data:
                    DATABASE_FORMAT = node_data['format']
                else:
                    DATABASE_FORMAT = 1
                    logging.warning(f"Database format is old: {DATABASE_FORMAT}, {label_name}, converting")
                    fake_node_data = {}
                    if type(node_data) is list:
                        if len(node_data) > 0:
                            fake_node_data['vis'] = 'seen'
                        else:
                            fake_node_data['vis'] = 'blurred'
                    elif type(node_data) is dict:
                        if node_data.get('0'):
                            fake_node_data['vis'] = 'seen'
                        else:
                            fake_node_data['vis'] = 'blurred'

                    fake_node_data['instances'] = {}
                    fake_node_data['instances']['0'] = {}

                    if type(node_data) is dict:
                        if node_data.get('0'):
                            logging.info("nodes")
                            fake_node_data['instances']['0']['nodes'] = node_data.copy()
                        elif node_data.get('x'):
                            logging.info("single node")
                            fake_node_data['instances']['0']['node'] = node_data.copy()
                        else:
                            raise Exception(f"error in converting from type 1 to type 2 database format")
                    elif type(node_data) is list:
                        if len(node_data) > 1:
                            logging.info("nodes")
                            fake_node_data['instances']['0']['nodes'] = node_data.copy()
                        elif len(node_data) == 1:
                            logging.info("single node")
                            fake_node_data['instances']['0']['node'] = node_data.copy()
                        else:
                            raise Exception(f"error in converting from type 1 to type 2 database format")

                    node_data = fake_node_data
                    DATABASE_FORMAT = 2

                vis = node_data.get('vis', 'seen')

                if vis == "unasked":
                    continue

                if vis == "blurred":
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'instance_num': 0,
                           'vis': 'blurred',
                           'value_x': '',
                           'value_y': '',
                           'straight_segment': '',
                           }

                    db.append(out)
                    continue

                if vis == 'off':
                    out = {'project': project,
                           'file': file,
                           'user': true_user,
                           'time': true_time_stamp,
                           'label': label_name,
                           'instance_num': 0,
                           'vis': 'off',
                           'value_x': '',
                           'value_y': '',
                           'straight_segment': '',
                           }
                    db.append(out)
                    continue

                try:
                    if DATABASE_FORMAT == 2:
                        freehand_node = False
                        node_data_instances = node_data.get('instances')

                        if type(node_data_instances) is list:
                            node_data_instances_dict = {str(a): b for a, b in enumerate(node_data_instances)}
                        elif type(node_data_instances) is dict:
                            node_data_instances_dict = node_data_instances

                        for node_instance_num, node_instance in node_data_instances_dict.items():

                            if node_instance.get('isFreehand'):
                                freehand_node = True
                                single_node = False
                                try:
                                    node_list = node_instance['freehandPoints']
                                except:
                                    logging.exception(f"No freehandPoints despite freehand")
                                    continue

                            elif 'node' in node_instance:
                                single_node = True
                                node_list = node_instance['node']

                            elif 'nodes' in node_instance:
                                single_node = False
                                node_list = node_instance['nodes']

                            else:
                                single_node = True
                                node_list = node_instance

                            curve_x = []
                            curve_y = []
                            curve_next_straight = []

                            if not single_node:

                                if type(node_list) is list:
                                    node_list_dict = {str(a): b for a, b in enumerate(node_list)}
                                elif type(node_list) is dict:
                                    node_list_dict = node_list

                                keys = sorted([int(x) for x in node_list_dict.keys()])
                                for key in keys:
                                    node = node_list_dict[str(key)]
                                    curve_x.append(node['x'])
                                    curve_y.append(node['y'])
                                    if not freehand_node:
                                        curve_next_straight.append(node.get("straightToNext", False))
                                    else:
                                        curve_next_straight.append(node.get("straightToNext", False))

                            elif single_node:
                                try:
                                    curve_x.append(node_list['x'])
                                    curve_y.append(node_list['y'])
                                    curve_next_straight.append(False)
                                except Exception as e:
                                    logging.exception("Error in single node")

                            curve_next_straight = [1 if x else 0 for x in curve_next_straight]

                            if freehand_node:
                                target_num_points = 50
                                num_points = len(curve_x)

                                if num_points > target_num_points:
                                    step = (num_points - 1.0) / (target_num_points - 1.0)
                                    select_idx = [round(i*step) for i in list(range(target_num_points))]

                                    curve_x = [curve_x[i] for i in select_idx]
                                    curve_y = [curve_y[i] for i in select_idx]
                                    curve_next_straight = [curve_next_straight[i] for i in select_idx]

                            if len(curve_x) == 0 or len(curve_y) == 0:
                                out_vis = "blurred"
                                logging.error(f"Unexpected empty curve_x when seen {project} {file} {true_user} {label_name} {node_instance_num}")
                            else:
                                out_vis = "seen"

                            out = {'project': project,
                                   'file': file,
                                   'user': true_user,
                                   'time': true_time_stamp,
                                   'label': label_name,
                                   'vis': out_vis,
                                   'instance_num': int(node_instance_num),
                                   'value_x': curve_list_to_str(curve_x, round_digits=2),
                                   'value_y': curve_list_to_str(curve_y, round_digits=2),
                                   'straight_segment': curve_list_to_str(curve_next_straight),
                                   }

                            db.append(out)

                except Exception:
                    print("d")
                    logging.exception(f"Failure in project data")

    return db


def fix_unity_labels(json_out: dict):
    for file, node in json_out.items():

        # RV Apex
        try:
            if node['labels']['curve-rv-endo'][0]['type'] == 'curve' and node['labels']['rv-apex-endo'][0][
                'type'] == 'blurred':
                x = node['labels']['curve-rv-endo'][0]['x']
                y = node['labels']['curve-rv-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                min_y = min(y)
                min_y_x = x[y.index(min_y)]

                node['labels']['rv-apex-endo'] = [
                    {
                        'type': 'point',
                        'x': str(min_y_x),
                        'y': str(min_y),
                        'straight_segment': '0'
                    }
                ]

            if node['labels']['curve-rv-endo'][0]['type'] == 'off' and node['labels']['rv-apex-endo'][0][
                'type'] == 'blurred':
                node['labels']['rv-apex-endo'][0]['type'] = "off"
        except KeyError:
            logging.exception(f"Exception in RV Apex")

        # LV antsep-endo apex
        try:
            if node['labels']['curve-lv-antsep-endo'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-endo'][0]['x']
                y = node['labels']['curve-lv-antsep-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-antsep-endo-apex'] = {}
                node['labels']['lv-antsep-endo-apex'][0]['type'] = 'point'
                node['labels']['lv-antsep-endo-apex'][0]['x'] = str(find_x)
                node['labels']['lv-antsep-endo-apex'][0]['y'] = str(find_x_y)
                node['labels']['lv-antsep-endo-apex'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-antsep-endo-apex'] = {}
                node['labels']['lv-antsep-endo-apex'][0]['type'] = node['labels']['curve-lv-antsep-endo'][0]['type']
                node['labels']['lv-antsep-endo-apex'][0]['x'] = ""
                node['labels']['lv-antsep-endo-apex'][0]['y'] = ""
                node['labels']['lv-antsep-endo-apex'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"Exception in ant-sept endo apex")

        # LV antsep-endo base
        try:
            if node['labels']['curve-lv-antsep-endo'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-endo'][0]['x']
                y = node['labels']['curve-lv-antsep-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-antsep-endo-base'] = {}
                node['labels']['lv-antsep-endo-base'][0]['type'] = 'point'
                node['labels']['lv-antsep-endo-base'][0]['x'] = str(find_x)
                node['labels']['lv-antsep-endo-base'][0]['y'] = str(find_x_y)
                node['labels']['lv-antsep-endo-base'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-antsep-endo-base'] = {}
                node['labels']['lv-antsep-endo-base'][0]['type'] = node['labels']['curve-lv-antsep-endo'][0]['type']
                node['labels']['lv-antsep-endo-base'][0]['x'] = ""
                node['labels']['lv-antsep-endo-base'][0]['y'] = ""
                node['labels']['lv-antsep-endo-base'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"No lv-antsept-endo-base")

        # LV antsep-rv apex
        try:
            if node['labels']['curve-lv-antsep-rv'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-rv'][0]['x']
                y = node['labels']['curve-lv-antsep-rv'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-antsep-rv-apex'] = {}
                node['labels']['lv-antsep-rv-apex'][0]['type'] = 'point'
                node['labels']['lv-antsep-rv-apex'][0]['x'] = str(find_x)
                node['labels']['lv-antsep-rv-apex'][0]['y'] = str(find_x_y)
                node['labels']['lv-antsep-rv-apex'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-antsep-rv-apex'] = {}
                node['labels']['lv-antsep-rv-apex'][0]['type'] = node['labels']['curve-lv-antsep-rv'][0]['type']
                node['labels']['lv-antsep-rv-apex'][0]['y'] = ""
                node['labels']['lv-antsep-rv-apex'][0]['x'] = ""
                node['labels']['lv-antsep-rv-apex'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"No lv-antsep-rv-apex")

        # LV antsep-rv base
        try:
            if node['labels']['curve-lv-antsep-rv'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-antsep-rv'][0]['x']
                y = node['labels']['curve-lv-antsep-rv'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-antsep-rv-base'] = {}
                node['labels']['lv-antsep-rv-base'][0]['type'] = 'point'
                node['labels']['lv-antsep-rv-base'][0]['x'] = str(find_x)
                node['labels']['lv-antsep-rv-base'][0]['y'] = str(find_x_y)
                node['labels']['lv-antsep-rv-base'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-antsep-rv-base'] = {}
                node['labels']['lv-antsep-rv-base'][0]['type'] = node['labels']['curve-lv-antsep-rv'][0]['type']
                node['labels']['lv-antsep-rv-base'][0]['y'] = ""
                node['labels']['lv-antsep-rv-base'][0]['x'] = ""
                node['labels']['lv-antsep-rv-base'][0]['straight_segment'] = ""
        except KeyError:
            logging.warning(f"no lv-antsep-rv-base")

        # LV post-endo apex
        try:
            if node['labels']['curve-lv-post-endo'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-post-endo'][0]['x']
                y = node['labels']['curve-lv-post-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                if len(x) == 0:
                    logging.error("WTF len=0")
                    continue

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-post-endo-apex'] = {}
                node['labels']['lv-post-endo-apex'][0]['type'] = 'point'
                node['labels']['lv-post-endo-apex'][0]['x'] = str(find_x)
                node['labels']['lv-post-endo-apex'][0]['y'] = str(find_x_y)
                node['labels']['lv-post-endo-apex'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-post-endo-apex'] = {}
                node['labels']['lv-post-endo-apex'][0]['type'] = node['labels']['curve-lv-post-endo'][0]['type']
                node['labels']['lv-post-endo-apex'][0]['y'] = ""
                node['labels']['lv-post-endo-apex'][0]['x'] = ""
                node['labels']['lv-post-endo-apex'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"no lv-post-endo-apex")

        # LV post-endo base
        try:
            if node['labels']['curve-lv-post-endo'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-post-endo'][0]['x']
                y = node['labels']['curve-lv-post-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-post-endo-base'] = {}
                node['labels']['lv-post-endo-base'][0]['type'] = 'point'
                node['labels']['lv-post-endo-base'][0]['x'] = str(find_x)
                node['labels']['lv-post-endo-base'][0]['y'] = str(find_x_y)
                node['labels']['lv-post-endo-base'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-post-endo-base'] = {}
                node['labels']['lv-post-endo-base'][0]['type'] = node['labels']['curve-lv-post-endo'][0]['type']
                node['labels']['lv-post-endo-base'][0]['y'] = ""
                node['labels']['lv-post-endo-base'][0]['x'] = ""
                node['labels']['lv-post-endo-base'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"no lv-post-endo-base")

        # LV post-epi apex
        try:
            if node['labels']['curve-lv-post-epi'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-post-epi'][0]['x']
                y = node['labels']['curve-lv-post-epi'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = min(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-post-epi-apex'] = {}
                node['labels']['lv-post-epi-apex'][0]['type'] = 'point'
                node['labels']['lv-post-epi-apex'][0]['x'] = str(find_x)
                node['labels']['lv-post-epi-apex'][0]['y'] = str(find_x_y)
                node['labels']['lv-post-epi-apex'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-post-epi-apex'] = {}
                node['labels']['lv-post-epi-apex'][0]['type'] = node['labels']['curve-lv-post-epi'][0]['type']
                node['labels']['lv-post-epi-apex'][0]['y'] = ""
                node['labels']['lv-post-epi-apex'][0]['x'] = ""
                node['labels']['lv-post-epi-apex'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"no-lv-post-epi-apex")

        # LV post-epi base
        try:
            if node['labels']['curve-lv-post-epi'][0]['type'] == 'curve':
                x = node['labels']['curve-lv-post-epi'][0]['x']
                y = node['labels']['curve-lv-post-epi'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                find_x = max(x)
                find_x_y = y[x.index(find_x)]

                # node['labels']['lv-post-epi-base'] = {}
                node['labels']['lv-post-epi-base'][0]['type'] = 'point'
                node['labels']['lv-post-epi-base'][0]['x'] = str(find_x)
                node['labels']['lv-post-epi-base'][0]['y'] = str(find_x_y)
                node['labels']['lv-post-epi-base'][0]['straight_segment'] = "0"
            else:
                # node['labels']['lv-post-epi-base'] = {}
                node['labels']['lv-post-epi-base'][0]['type'] = node['labels']['curve-lv-post-epi'][0]['type']
                node['labels']['lv-post-epi-base'][0]['y'] = ""
                node['labels']['lv-post-epi-base'][0]['x'] = ""
                node['labels']['lv-post-epi-base'][0]['straight_segment'] = ""
        except KeyError:
            logging.warning(f"no lv-post-epi-base")

        # curve-lv-ed-connect
        try:
            top = node['labels']['lv-ivs-top'][0]
            bottom = node['labels']['lv-pw-bottom'][0]

            if top['type'] == bottom['type'] == "point":
                x_top = float(top['x'])
                y_top = float(top['y'])
                x_bot = float(bottom['x'])
                y_bot = float(bottom['y'])

                x = [x_top, x_bot]
                y = [y_top, y_bot]
                straight_segment = [1, 1]

                # node['labels']['curve-lv-ed-connect'] = {}
                node['labels']['curve-lv-ed-connect'][0]['type'] = 'curve'
                node['labels']['curve-lv-ed-connect'][0]['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels']['curve-lv-ed-connect'][0]['y'] = " ".join([str(round(value, 1)) for value in y])
                node['labels']['curve-lv-ed-connect'][0]['straight_segment'] = " ".join(
                    [str(round(value, 1)) for value in straight_segment])

                if False:
                    unity_o = SecurionSource(unity_code=file, png_cache_dir=Path(PNG_CACHE_DIR))
                    img = imageio.imread(unity_o.get_frame_path())

                    x_dir = np.sign(x_bot - x_top)
                    y_dir = np.sign(y_bot - y_top)

                    x_step = (x_bot - x_top) / np.abs(y_bot - y_top)
                    y_step = (y_bot - y_top) / np.abs(x_bot - x_top)

                    if x_step > y_step:
                        test_y = (np.arange(0, 7) * y_step) + y_bot
                        test_x = (np.arange(0, 7) * x_dir) + x_bot
                    else:
                        test_y = (np.arange(0, 7) * y_dir) + y_bot
                        test_x = (np.arange(0, 7) * x_step) + x_bot

                    print(test_y, test_x)

                    values = img[np.round(test_y).astype(np.int), np.round(test_x).astype(np.int)]

                    values = values.astype(np.float)
                    if values.ndim == 2:
                        values = np.sum(values, axis=-1)

                    if np.argmax(values) == 0:
                        shift = 0
                    else:
                        shift = np.argmax(np.diff(values)) + 1

                    print(shift)

                    new_y_bot = test_y[shift]
                    new_x_bot = test_x[shift]

                    bottom['y'] = str(round(new_y_bot, 1))
                    bottom['x'] = str(round(new_x_bot, 1))

            elif top['type'] == bottom['type'] == 'off':
                # node['labels']['curve-lv-ed-connect'] = {}
                node['labels']['curve-lv-ed-connect'][0]['type'] = 'off'
                node['labels']['curve-lv-ed-connect'][0]['y'] = ""
                node['labels']['curve-lv-ed-connect'][0]['x'] = ""
                node['labels']['curve-lv-ed-connect'][0]['straight_segment'] = ""

            else:
                # node['labels']['curve-lv-ed-connect'] = {}
                node['labels']['curve-lv-ed-connect'][0]['type'] = 'blurred'
                node['labels']['curve-lv-ed-connect'][0]['y'] = ""
                node['labels']['curve-lv-ed-connect'][0]['x'] = ""
                node['labels']['curve-lv-ed-connect'][0]['straight_segment'] = ""
        except KeyError:
            logging.exception(f"error in curve-lv-ed-connect")

        # Three-way
        try:
            ant = node['labels']['mv-ant-wall-hinge'][0]
            post = node['labels']['mv-inf-wall-hinge'][0]
            apex = node['labels']['lv-apex-endo'][0]

            curve_name = "curve-mv-hinge-connect"
            if ant['type'] == post['type'] == 'point':
                start_x = float(ant['x'])
                start_y = float(ant['y'])
                end_x = float(post['x'])
                end_y = float(post['y'])

                x = [start_x, end_x]
                y = [start_y, end_y]
                straight_segment = [1, 1]

                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'curve'
                node['labels'][curve_name][0]['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels'][curve_name][0]['y'] = " ".join([str(round(value, 1)) for value in y])
                node['labels'][curve_name][0]['straight_segment'] = " ".join(
                    [str(round(value, 1)) for value in straight_segment])

            elif ant['type'] == post['type'] == 'off':
                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'off'
                node['labels'][curve_name][0]['y'] = ""
                node['labels'][curve_name][0]['x'] = ""
                node['labels'][curve_name][0]['straight_segment'] = ""

            else:
                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'blurred'
                node['labels'][curve_name][0]['y'] = ""
                node['labels'][curve_name][0]['x'] = ""
                node['labels'][curve_name][0]['straight_segment'] = ""

            curve_name = "curve-mv-ant-apex-connect"
            if ant['type'] == apex['type'] == 'point':
                start_x = float(ant['x'])
                start_y = float(ant['y'])
                end_x = float(apex['x'])
                end_y = float(apex['y'])

                x = [start_x, end_x]
                y = [start_y, end_y]
                straight_segment = [1, 1]

                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'curve'
                node['labels'][curve_name][0]['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels'][curve_name][0]['y'] = " ".join([str(round(value, 1)) for value in y])
                node['labels'][curve_name][0]['straight_segment'] = " ".join(
                    [str(round(value, 1)) for value in straight_segment])

            elif ant['type'] == apex['type'] == 'off':
                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'off'
                node['labels'][curve_name][0]['y'] = ""
                node['labels'][curve_name][0]['x'] = ""
                node['labels'][curve_name][0]['straight_segment'] = ""

            else:
                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'blurred'
                node['labels'][curve_name][0]['y'] = ""
                node['labels'][curve_name][0]['x'] = ""
                node['labels'][curve_name][0]['straight_segment'] = ""

            curve_name = "curve-mv-post-apex-connect"
            if post['type'] == apex['type'] == 'point':
                start_x = float(post['x'])
                start_y = float(post['y'])
                end_x = float(apex['x'])
                end_y = float(apex['y'])

                x = [start_x, end_x]
                y = [start_y, end_y]
                straight_segment = [1, 1]

                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'curve'
                node['labels'][curve_name][0]['y'] = " ".join([str(round(value, 1)) for value in y])
                node['labels'][curve_name][0]['x'] = " ".join([str(round(value, 1)) for value in x])
                node['labels'][curve_name][0]['straight_segment'] = " ".join(
                    [str(round(value, 1)) for value in straight_segment])

            elif post['type'] == apex['type'] == 'off':
                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'off'
                node['labels'][curve_name][0]['y'] = ""
                node['labels'][curve_name][0]['x'] = ""
                node['labels'][curve_name][0]['straight_segment'] = ""
            else:
                # node['labels'][curve_name] = {}
                node['labels'][curve_name][0]['type'] = 'blurred'
                node['labels'][curve_name][0]['y'] = ""
                node['labels'][curve_name][0]['x'] = ""
                node['labels'][curve_name][0]['straight_segment'] = ""

        except KeyError:
            logging.exception(f"error on conect")

        # LA Roof
        try:
            if node['labels']['curve-la-endo'][0]['type'] == 'curve' and node['labels']['la-roof'][0][
                'type'] == 'blurred':
                x = node['labels']['curve-la-endo'][0]['x']
                y = node['labels']['curve-la-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                max_y = max(y)
                max_y_x = x[y.index(max_y)]

                node['labels']['la-roof'][0]['type'] = 'point'
                node['labels']['la-roof'][0]['x'] = str(max_y_x)
                node['labels']['la-roof'][0]['y'] = str(max_y)
                node['labels']['la-roof'][0]["straight_segment"] = "0"

            if node['labels']['curve-la-endo'][0]['type'] == 'off' and node['labels']['la-roof'][0][
                'type'] == 'blurred':
                node['labels']['la-roof'][0]['type'] = 'off'
        except Exception:
            logging.exception("Exception in LA Roof")

        # RA Roof
        try:
            if node['labels']['curve-ra-endo'][0]['type'] == 'curve' and node['labels']['ra-roof'][0]['type'] == 'blurred':
                x = node['labels']['curve-ra-endo'][0]['x']
                y = node['labels']['curve-ra-endo'][0]['y']

                x = [float(value) for value in x.split()]
                y = [float(value) for value in y.split()]

                max_y = max(y)
                max_y_x = x[y.index(max_y)]

                node['labels']['ra-roof'][0]['type'] = 'point'
                node['labels']['ra-roof'][0]['x'] = str(max_y_x)
                node['labels']['ra-roof'][0]['y'] = str(max_y)
                node['labels']['ra-roof'][0]["straight_segment"] = "0"

            if node['labels']['curve-ra-endo'][0]['type'] == 'off' and node['labels']['ra-roof'][0][
                'type'] == 'blurred':
                node['labels']['ra-roof'][0]['type'] = 'off'
        except Exception:
            logging.exception("Exception in RA Roof")

    return json_out
