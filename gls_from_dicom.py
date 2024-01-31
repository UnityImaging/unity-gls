import os
import argparse
import time
import logging

from pathlib import Path

import numpy as np

import pydicom
import scipy.signal

from mdicom.utils import fix_issues_in_pydicom
from mdicom.sour import get_2d_area_from_sour

from scanpilot.models.BMode import bmode_seg_init, bmode_seg_process
from Scantensus.labels.bmode import calc_bmode_labels
from Scantensus.measure.lv import smooth_and_interpolate_label, fix_lv_endo_points, find_peaks_cm
from Scantensus.utils.labels import labels_calc_len
from Scantensus.utils.image import image_logit_overlay_alpha

import imageio.v3

parser = argparse.ArgumentParser(
                    prog='DICOM to Longitudinal Strain',
                    description='Pass in a dicom file and it will output longitudinal strain. It will download the weights file and cache it in the output directory. It will also generate an mp4 file of the heatmaps',
                    epilog='(c) Matthew Shun-Shin')


parser.add_argument('--file', help='dicom file location', type=str, required=True)
parser.add_argument('--output_dir', help='Output Directory (and cache location)', type=str, required=True)
parser.add_argument('--weight_server', type=str, default="https://data.unityimaging.net/buckets")
args = parser.parse_args()


#####
BMODE_SEG_ENDPOINT = args.weight_server
BMODE_SEG_BUCKET = "matt-output"
BMODE_SEG_PROJECT = "unity"
BMODE_SEG_EXPERIMENT = "unity-211"
BMODE_SEG_EPOCH = 401
BMODE_SEG_WEIGHTS_ADDRESS = f"{BMODE_SEG_ENDPOINT}/{BMODE_SEG_BUCKET}/checkpoints/{BMODE_SEG_PROJECT}/{BMODE_SEG_EXPERIMENT}/weights-{BMODE_SEG_EPOCH}.pt"
#####


def main():

    logging.info("Making output dir")
    try:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        logging.exception("Failed to make output directory")
    logging.info("Made output dir")

    logging.info('Loading DICOM')
    try:
        dcm_fn = Path(args.file)
    except Exception:
        logging.exception(f"Failed to parse filename: {args.file}")
    logging.info("Loaded DICOM")

    output_layer_list = ['curve-lv-endo',
                         'lv-apex-endo',
                         'mv-ant-wall-hinge',
                         'mv-inf-wall-hinge']

    output_cols = [[1,1,0], [1,0,0], [0,1,0], [0,0,1]]

    dcm = pydicom.read_file(dcm_fn)
    dcm, needed_fix = fix_issues_in_pydicom(dcm)

    SequenceOfUltrasoundRegions = getattr(dcm, 'SequenceOfUltrasoundRegions', None)
    ultrasound_region_2d = get_2d_area_from_sour(SequenceOfUltrasoundRegions)

    if needed_fix:
        logging.info('Fixed DICOM')

    img_raw = dcm.pixel_array

    frames_to_process = list(range(img_raw.shape[0]))
    unity_f_codes = []
    for frame_num in frames_to_process:
        unity_f_codes.append(f"frame-{frame_num:04}")

    bmode_seg_model, weights, cfg = bmode_seg_init(checkpoint_weights_location=BMODE_SEG_WEIGHTS_ADDRESS,
                                                   cache_name=str(output_dir / "cache.sqlite"),
                                                   cache_backend='sqlite')

    logging.info(f"Starting inference")
    start = time.time()
    heatmaps, height_shift, width_shift, img = bmode_seg_process(
        bmode_seg_model=(bmode_seg_model, weights, cfg),
        img=img_raw,
        output_layer_list=output_layer_list,
        debug_mode=True
    )
    end = time.time()

    logging.info(f"Inference took {end - start} seconds")

    logging.info(f"Making mp4")
    y_pred_mix = image_logit_overlay_alpha(logits=np.transpose(heatmaps, axes=(0, 2, 3, 1)),
                                           images=img/255.0,
                                           cols=output_cols)
    y_pred_mix = y_pred_mix * 255
    y_pred_mix = y_pred_mix.astype(np.uint8)
    imageio.v3.imwrite(uri=output_dir / "video.mp4", image=y_pred_mix)
    logging.info(f"Finished making mp4")

    unity_seg_labels = calc_bmode_labels(heatmaps=(heatmaps, height_shift, width_shift),
                                         output_layer_list=output_layer_list,
                                         unity_f_codes=unity_f_codes,
                                         calc_lv_endo=False,
                                         calc_lv_points=False,
                                         calc_lv_gls=True,
                                         calc_rv_points=False,
                                         calc_plax_points=False,
                                         calc_la_endo=False,
                                         calc_ra_endo=False,
                                         calc_aortic_points=False
                                         )

    try:
        unity_seg_labels = fix_lv_endo_points(unity_seg_labels)
    except Exception:
        logging.exception(f"Failed to fix LV endo-points")

    try:
        unity_seg_labels = labels_calc_len(unity_seg_labels)
    except Exception:
        logging.exception(f"Failed to calc label length")

    lv_len = []
    for frame_num in frames_to_process:
        lv_len.append(unity_seg_labels[f"frame-{frame_num:04}"]['labels']['curve-lv-endo'][0]['curve_len'])

    lv_len = np.array(lv_len)

    lv_len_cm = lv_len * ultrasound_region_2d['PhysicalDeltaY']
    lv_len_smooth_cm = scipy.signal.savgol_filter(lv_len_cm, 7, 3)

    peaks = find_peaks_cm(lv_len_smooth_cm, min_name="lv_len_es", max_name="lv_len_ed")

    print(peaks)

    lv_len_ed_mean = np.mean([x['value'] for x in peaks['lv_len_ed']])
    lv_len_es_mean = np.mean([x['value'] for x in peaks['lv_len_es']])

    lv_len_gls = 100 * (lv_len_es_mean - lv_len_ed_mean) / lv_len_ed_mean

    print(f"LV ED Mean Len: {lv_len_ed_mean} cm")
    print(f"LV ES Mean Len: {lv_len_es_mean} cm")
    print(f"LV Longitudinal Strain: {round(lv_len_gls, 1)}%")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
