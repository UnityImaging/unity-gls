from dataclasses import dataclass
import sys
from typing import Optional, Literal
import numpy as np

from pydicom import FileDataset
from loguru import logger
import torch
from Scantensus.labels.bmode import calc_bmode_labels
from Scantensus.measure.lv import calc_lv_tlen, find_peaks_cm, fix_lv_endo_points, smooth_and_interpolate_label
from Scantensus.measure.utils import convert_dist_pixel_to_cm
from ScantensusPT.image import image_logit_overlay_alpha_t
from mdicom.sour import get_2d_area_from_sour

from scanpilot.models.BMode import bmode_seg_init, bmode_seg_process, LoadedModel

from scipy.interpolate import BSpline, splrep, splev, splprep

from PIL import Image, ImageDraw

LV_POINTS = ['lv-apex-trabec',
            'lv-apex-endo',
            'lv-apex-midmyo',
            'mv-ant-wall-hinge',
            'mv-ant-wall-hinge-midmyo',
            'mv-inf-wall-hinge',
            'mv-inf-wall-hinge-midmyo']

@dataclass
class CurveValidationPipelineConfig:
    """
    Holds configuration for the curve validation pipeline
    """

    curve_name: Literal['lv-endo', 'la-endo', 'rv-endo', 'ra-endo']
    """
    The name of the curve (such as `lv-endo`).
    """

    checkpoint_path: str
    """
    The path to the checkpoint file.
    """


class CurveValidationPipeline:
    """
    Performs the validation for a specific curve.
    """

    config: CurveValidationPipelineConfig
    """
    The configuration for the current pipeline.
    """

    loaded_model: LoadedModel = None
    """
    The loaded model object.
    """
    
    def __init__(self, config: CurveValidationPipelineConfig):
        self.config = config
        

    def validate(self, dicom: FileDataset) -> Optional[dict]:
        """
        Validates the model on the DICOM data and returns the result.

        Arguments:
            dicom (pydicom.FileDataset): The DICOM data to validate.

        Returns:
            dict: The validation result, or `None` if the validation failed.
        """

        if self.loaded_model is None:
            self.loaded_model = bmode_seg_init(self.config.checkpoint_path)

            logger.success("Loaded model.")

        logger.info("Running inference...")

        unity_f_codes = [f'99-Frame-{frame_num:04}' for frame_num in range(len(dicom.pixel_array))]

        heatmaps, height_shift, width_shift, img = bmode_seg_process(
            bmode_seg_model=self.loaded_model, 
            img=dicom.pixel_array,
            debug_mode=True
        )

        logger.success("Inference done.")

        logger.info('Computing labels...')

        unity_seg = calc_bmode_labels(
            heatmaps=(heatmaps, height_shift, width_shift),
            output_layer_list=self.loaded_model.config['default_output_layer_list'],
            unity_f_codes=unity_f_codes,
            calc_lv_endo=self.config.curve_name == 'lv-endo',
            calc_lv_points=self.config.curve_name == 'lv-endo',
            calc_rv_points=False,
            calc_plax_points=False,
            calc_la_endo=self.config.curve_name == 'la-endo',
            calc_ra_endo=self.config.curve_name == 'ra-endo',
            calc_aortic_points=False                  
        )

        #unity_seg = smooth_and_interpolate_label(unity_seg, f'curve-{self.config.curve_name}')
        #unity_seg = fix_lv_endo_points(unity_seg)

        dotted_images = self._render_images_with_curve_points(img, unity_seg, height_shift, width_shift)
        dotted_images[0].save(f'./out/dotted-{self.config.curve_name}.gif', save_all=True, append_images=dotted_images[1:], duration=5, loop=0)

        logger.success('Labels computed.')

        logger.info('Computing measures...')

        lv_tlen = calc_lv_tlen(unity_seg, anterior_name='mv-ant-wall-hinge', posterior_name='mv-inf-wall-hinge', apical_name='lv-apex-endo')

        SequenceOfUltrasoundRegions = getattr(dicom, 'SequenceOfUltrasoundRegions', None)
        ultrasound_region_2d = get_2d_area_from_sour(SequenceOfUltrasoundRegions)

        lv_tlen_cm = convert_dist_pixel_to_cm(lv_tlen, sour=ultrasound_region_2d)
        lv_tlen_measurements = find_peaks_cm(dist=lv_tlen_cm, min_name='lv_length_a4c_systole', max_name='lv_length_a4c_diastole')

        splines = self._create_splines(unity_seg)

        logger.success('Measures computed.')

        pass


    def _render_images_with_curve_points(self, images: np.ndarray, unity_seg: dict, height_shift: float, width_shift: float) -> np.ndarray:
        """
        Renders the images with the curve points.

        Arguments:
            images (np.ndarray): The images to render.
            unity_seg (dict): The segmentation to use.

        Returns:
            np.ndarray: The rendered images.
        """

        output: list[Image.Image] = []

        for idx, (frame_name, data) in enumerate(unity_seg.items()):
            xs = data['labels'][f'curve-{self.config.curve_name}'][0]['x']
            ys = data['labels'][f'curve-{self.config.curve_name}'][0]['y']

            image = images[idx, :]
            
            # convert HWC to WHC
            #image = np.transpose(image, (1, 0, 2))

            img = Image.fromarray(np.squeeze(image)).convert('RGB')

            draw = ImageDraw.Draw(img)

            # draw circles with radius 4 around the points
            for x, y in zip(xs, ys):
                draw.ellipse((x - 4 + width_shift, 
                              y - 4 + height_shift, 
                              x + 4 + width_shift, 
                              y + 4 + height_shift), fill='red', outline='red')

            output.append(img)    

        return output


    def _create_splines(self, unity_seg: dict) -> dict[str, BSpline]:
        """
        Creates a spline from the given segmentation.

        Arguments:
            unity_seg (dict): The segmentation to use.

        Returns:
            dict[str, scipy.interpolate.BSpline]: The splines per frame.
        """

        splines = {}

        for image, data in unity_seg.items():
            xs = data['labels'][f'curve-{self.config.curve_name}'][0]['x']
            ys = data['labels'][f'curve-{self.config.curve_name}'][0]['y']

            spl = splrep(xs, ys, k=3)
            b_spline = BSpline(spl[0], spl[1], spl[2], extrapolate=False)

            splines[image] = b_spline

        return splines

        

        