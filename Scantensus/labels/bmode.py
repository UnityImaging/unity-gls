import logging

import numpy as np

from .point import get_point
from .trace import get_path_via_2d
from .helpers import add_instance_to_labels_dict


def calc_bmode_labels(heatmaps,
                      output_layer_list,
                      unity_f_codes,
                      calc_lv_endo=False,
                      calc_lv_points=False,
                      calc_lv_gls=False,
                      calc_rv_points=False,
                      calc_plax_points=False,
                      calc_la_endo=False,
                      calc_ra_endo=False,
                      calc_aortic_points=False):

    y_pred = heatmaps[0]
    label_height_shift = heatmaps[1]
    label_width_shift = heatmaps[2]

    out_labels_dict = {}

    point_labels = []

    if calc_lv_gls:
        point_labels.extend([
            'lv-apex-endo',
            'mv-ant-wall-hinge',
            'mv-inf-wall-hinge',
        ])

    if calc_lv_points:
        point_labels.extend([
            'lv-apex-trabec',
            'lv-apex-endo',
            'lv-apex-midmyo',
            'mv-ant-wall-hinge',
            'mv-ant-wall-hinge-midmyo',
            'mv-inf-wall-hinge',
            'mv-inf-wall-hinge-midmyo',
        ])

    if calc_rv_points:
        point_labels.extend([
            'tv-sep-hinge',
            'tv-ant-hinge',
            'rv-apex-endo'
        ])

    if calc_plax_points:
        point_labels.extend([
            "lv-pw-top",
            "lv-pw-bottom",
            "lv-ivs-top",
            "lv-ivs-bottom",
            "la-bottom-inner",
            "la-top-inner",
            "rv-top-inner",
            "rv-bottom-inner"
        ])

    if calc_aortic_points:
        point_labels.extend([
            "ao-sinus-bottom-inner",
            "ao-sinus-top-inner",
            "ao-sinus-top-outer",
            "ao-stj-bottom-inner",
            "ao-stj-top-inner",
            "ao-stj-top-outer",
            "lvot-top-inner",
            "lvot-bottom-inner",
            "ao-valve-bottom-inner",
            "ao-valve-top-inner",
            "ao-asc-bottom-inner",
            "ao-asc-top-inner",
            "ao-asc-top-outer"
        ])

    for i, unity_f_code in enumerate(unity_f_codes):
        out_labels_dict[unity_f_code] = {}
        out_labels_dict[unity_f_code]['labels'] = {}

    for i, unity_f_code in enumerate(unity_f_codes):
        labels_dict = out_labels_dict[unity_f_code]['labels']

        y_pred_frame = y_pred[i, ...]
        # y_pred_frame_np = np.array(y_pred_frame)

        for label in point_labels:
            try:
                point_idx = output_layer_list.index(label)
            except Exception:
                logging.warning(f"{unity_f_code} Failed {label} as not in output_layer_list (this may be expected)")
                continue

            try:
                point = get_point(logits=y_pred_frame[point_idx, ...])
                point = point.tolist()
                ys = [point[0]]
                xs = [point[1]]
                confs = [point[2]]

                labels_dict = add_instance_to_labels_dict(ys=ys,
                                                          xs=xs,
                                                          confs=confs,
                                                          label=label,
                                                          labels_dict=labels_dict,
                                                          label_height_shift=label_height_shift,
                                                          label_width_shift=label_width_shift)
            except Exception:
                logging.exception(f"{unity_f_code} Failed {label}")

    for i, unity_f_code in enumerate(unity_f_codes):
        labels_dict = out_labels_dict[unity_f_code]['labels']

        y_pred_frame = y_pred[i, ...]
        y_pred_frame_np = np.array(y_pred_frame)

        if calc_lv_endo or calc_lv_gls:
            try:
                ys, xs, confs = get_lv_endo_path(y_pred=y_pred_frame_np, keypoint_names=output_layer_list)
                labels_dict = add_instance_to_labels_dict(ys=ys,
                                                          xs=xs,
                                                          confs=confs,
                                                          label='curve-lv-endo',
                                                          labels_dict=labels_dict,
                                                          label_height_shift=label_height_shift,
                                                          label_width_shift=label_width_shift)
            except Exception:
                logging.exception(f"{unity_f_code} Failed curve-lv-endo")

        if calc_lv_endo:
            try:
                ys, xs, confs = get_lv_midmyo_path(y_pred=y_pred_frame_np, keypoint_names=output_layer_list)
                labels_dict = add_instance_to_labels_dict(ys=ys,
                                                          xs=xs,
                                                          confs=confs,
                                                          label='curve-lv-midmyo',
                                                          labels_dict=labels_dict,
                                                          label_height_shift=label_height_shift,
                                                          label_width_shift=label_width_shift)
            except Exception:
                logging.exception(f"{unity_f_code} Failed curve-lv-midmyo")

        if calc_lv_endo:
            try:
                ys, xs, confs = get_lv_trabec_path(y_pred=y_pred_frame_np, keypoint_names=output_layer_list)
                labels_dict = add_instance_to_labels_dict(ys=ys,
                                                          xs=xs,
                                                          confs=confs,
                                                          label='curve-lv-trabec',
                                                          labels_dict=labels_dict,
                                                          label_height_shift=label_height_shift,
                                                          label_width_shift=label_width_shift)
            except Exception:
                logging.exception(f"{unity_f_code} Failed curve-lv-trabec")

        if calc_la_endo:
            try:
                ys, xs, confs = get_la_endo_path(y_pred=y_pred_frame_np, keypoint_names=output_layer_list)
                labels_dict = add_instance_to_labels_dict(ys=ys,
                                                          xs=xs,
                                                          confs=confs,
                                                          label='curve-la-endo',
                                                          labels_dict=labels_dict,
                                                          label_height_shift=label_height_shift,
                                                          label_width_shift=label_width_shift)
            except Exception:
                logging.exception(f"{unity_f_code} Failed curve-la-endo")

        if calc_ra_endo:
            try:
                ys, xs, confs = get_ra_endo_path(y_pred=y_pred_frame_np, keypoint_names=output_layer_list)
                labels_dict = add_instance_to_labels_dict(ys=ys,
                                                          xs=xs,
                                                          confs=confs,
                                                          label='curve-ra-endo',
                                                          labels_dict=labels_dict,
                                                          label_height_shift=label_height_shift,
                                                          label_width_shift=label_width_shift)
            except Exception:
                logging.exception(f"{unity_f_code} Failed curve-ra-endo")



    logging.info(f"returning out_labels_dict")
    return out_labels_dict


def get_lv_endo_path(y_pred: np.ndarray, keypoint_names: list):
    # c, h, w

    mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    mv_inf_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")
    lv_apex_idx = keypoint_names.index("lv-apex-endo")

    lv_endo_idx = keypoint_names.index("curve-lv-endo")

    try:
        lv_endo_jump_idx = keypoint_names.index("curve-lv-endo-jump")
        lv_endo = np.maximum(y_pred[lv_endo_idx, :, :], y_pred[lv_endo_jump_idx, :, :] ** 0.25)
    except Exception:
        lv_endo = y_pred[lv_endo_idx, :, :]

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_inf_hinge = y_pred[mv_inf_hinge_idx, :, :]
    lv_apex = y_pred[lv_apex_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_inf_hinge_point = get_point(mv_inf_hinge)
    lv_apex_point = get_point(lv_apex)

    out = get_path_via_2d(logits=lv_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=lv_apex_point,
                          p3s=mv_inf_hinge_point,
                          cut_offs=(0.1, 0.2, 0.1, 0.2),
                          num_knots=15,
                          description="curve-lv-endo")

    return out


def get_lv_midmyo_path(y_pred: np.ndarray, keypoint_names: list):
    # c, h, w

    mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge-midmyo")
    mv_inf_hinge_idx = keypoint_names.index("mv-inf-wall-hinge-midmyo")
    lv_apex_idx = keypoint_names.index("lv-apex-midmyo")

    lv_endo_idx = keypoint_names.index("curve-lv-midmyo")

    lv_endo = y_pred[lv_endo_idx, :, :]

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_inf_hinge = y_pred[mv_inf_hinge_idx, :, :]
    lv_apex = y_pred[lv_apex_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_inf_hinge_point = get_point(mv_inf_hinge)
    lv_apex_point = get_point(lv_apex)

    out = get_path_via_2d(logits=lv_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=lv_apex_point,
                          p3s=mv_inf_hinge_point,
                          cut_offs=(0.1, 0.2, 0.1, 0.2),
                          num_knots=15,
                          description="curve-lv-midmyo")

    return out


def get_lv_trabec_path(y_pred: np.ndarray, keypoint_names: list):
    # c, h, w

    mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    mv_inf_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")
    lv_apex_idx = keypoint_names.index("lv-apex-trabec")

    lv_endo_idx = keypoint_names.index("curve-lv-trabec")

    lv_endo = y_pred[lv_endo_idx, :, :]

    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_inf_hinge = y_pred[mv_inf_hinge_idx, :, :]
    lv_apex = y_pred[lv_apex_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_inf_hinge_point = get_point(mv_inf_hinge)
    lv_apex_point = get_point(lv_apex)

    out = get_path_via_2d(logits=lv_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=lv_apex_point,
                          p3s=mv_inf_hinge_point,
                          cut_offs=(0.1, 0.2, 0.1, 0.2),
                          num_knots=15,
                          description="curve-lv-trabec")

    return out


def get_la_endo_path(y_pred: np.ndarray, keypoint_names: list):

    la_endo_idx = keypoint_names.index("curve-la-endo")

    mv_ant_hinge_idx = keypoint_names.index("mv-ant-wall-hinge")
    mv_inf_hinge_idx = keypoint_names.index("mv-inf-wall-hinge")
    la_roof_idx = keypoint_names.index("la-roof")

    la_endo = y_pred[la_endo_idx, :, :]
    mv_ant_hinge = y_pred[mv_ant_hinge_idx, :, :]
    mv_inf_hinge = y_pred[mv_inf_hinge_idx, :, :]
    la_roof = y_pred[la_roof_idx, :, :]

    mv_ant_hinge_point = get_point(mv_ant_hinge)
    mv_inf_hinge_point = get_point(mv_inf_hinge)
    la_roof_point = get_point(la_roof)

    out = get_path_via_2d(logits=la_endo,
                          p1s=mv_ant_hinge_point,
                          p2s=la_roof_point,
                          p3s=mv_inf_hinge_point,
                          cut_offs=(0.1, 0.1, 0.1, 0.1),
                          num_knots=11,
                          description="curve-la-endo")

    return out


def get_ra_endo_path(y_pred: np.ndarray, keypoint_names: list):

    ra_endo_idx = keypoint_names.index("curve-ra-endo")
    tv_ant_hinge_idx = keypoint_names.index("tv-ant-hinge")
    tv_sep_hinge_idx = keypoint_names.index("tv-sep-hinge")
    ra_roof_idx = keypoint_names.index("ra-roof")

    ra_endo = y_pred[ra_endo_idx, :, :]
    ra_roof = y_pred[ra_roof_idx, :, :]
    tv_ant_hinge = y_pred[tv_ant_hinge_idx, :, :]
    tv_sep_hinge = y_pred[tv_sep_hinge_idx, :, :]

    tv_ant_hinge_point = get_point(tv_ant_hinge)
    tv_sep_hinge_point = get_point(tv_sep_hinge)
    ra_roof_point = get_point(ra_roof)


    out = get_path_via_2d(logits=ra_endo,
                          p1s=tv_ant_hinge_point,
                          p2s=ra_roof_point,
                          p3s=tv_sep_hinge_point,
                          cut_offs=(0.1, 0.2, 0.1, 0.2),
                          num_knots=11,
                          description="curve-ra-endo")

    return out