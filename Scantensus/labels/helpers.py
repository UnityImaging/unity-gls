import math

from .utils import subtract_shift


def add_instance_to_labels_dict(ys: list,
                                xs: list,
                                confs: list,
                                label: str,
                                labels_dict: dict,
                                label_height_shift=None,
                                label_width_shift=None):

    if label in labels_dict:
        if type(labels_dict[label]) is list:
            pass
        else:
            raise Exception("Label present, but not a list")
    else:
        labels_dict[label] = []

    if any([not math.isfinite(y) for y in ys]) or any([not math.isfinite(x) for x in xs]):
        node_type = "blurred"
    elif label.startswith('curve'):
        node_type = "curve"
    else:
        node_type = "point"

    out = {}

    if label_height_shift:
        ys = subtract_shift(ys, label_height_shift)

    if label_width_shift:
        xs = subtract_shift(xs, label_width_shift)

    if type == "blurred":
        out['type'] = "blurred"
        out['y'] = []
        out['x'] = []
        out['conf'] = []
    else:
        out['type'] = node_type
        out['y'] = ys
        out['x'] = xs
        out['conf'] = confs

    labels_dict[label].append(out)
    return labels_dict
