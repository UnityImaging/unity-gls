import copy


def subtract_shift(values: list, shift: float):
    return [a - shift for a in values]


def add_shift(values: list, shift: float):
    return [a + shift for a in values]


def label_add_shift(label_dict, label_height_shift, label_width_shift):

    label_dict = copy.deepcopy(label_dict)

    for label in label_dict.keys():

        curve_y = label_dict[label]['y']
        curve_x = label_dict[label]['x']

        curve_y = [y + label_height_shift for y in curve_y]
        curve_x = [x + label_width_shift for x in curve_x]

        label_dict[label]['y'] = curve_y
        label_dict[label]['x'] = curve_x

    return label_dict