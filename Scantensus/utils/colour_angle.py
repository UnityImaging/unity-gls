import logging
import math
import numpy as np


def colour_angle_calc(img, ultrasound_region_2d=None):
    logger = logging.getLogger("unity")
    frames, height, width, channels = img.shape
    if channels != 3:
        raise Exception("Expected colour image")

    image_x_reference = width / 2
    image_y_reference = 0

    img_mask = np.zeros((1, height, width, 1), dtype=np.uint8)

    try:
        if ultrasound_region_2d:
            RegionLocationMinX0 = ultrasound_region_2d['RegionLocationMinX0']
            RegionLocationMaxX1 = ultrasound_region_2d['RegionLocationMaxX1']
            RegionLocationMinY0 = ultrasound_region_2d['RegionLocationMinY0']
            RegionLocationMaxY1 = ultrasound_region_2d['RegionLocationMaxY1']

            try:
                ReferencePixelX0 = ultrasound_region_2d['ReferencePixelX0']
                ReferencePixelY0 = ultrasound_region_2d['ReferencePixelY0']
                logger.info(f"Using encoded reference pixels x {ReferencePixelX0}, y {ReferencePixelY0}")
            except Exception:
                image_x_reference = (RegionLocationMinX0 + RegionLocationMaxX1) / 2
                image_y_reference = RegionLocationMinY0

            img_mask[0, RegionLocationMinY0:RegionLocationMaxY1, RegionLocationMinX0:RegionLocationMaxX1, 0] = 1
            img = img * img_mask
        else:
            img_mask[0, 15:, :, 0] = 1
    except Exception:
        img_mask[0, 15:, :, 0] = 1

    img_col = img * img_mask
    img_col = img_col.astype(np.float32)
    img_col = np.log(img_col + 10)
    img_col = np.std(img_col, axis=-1) > 0.1  #Decimal percent
    img_col = np.max(img_col, axis=0)

    yv, xv = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    mean_y = np.mean(yv[img_col])
    mean_x = np.mean(xv[img_col])

    angle = (180 / math.pi) * math.atan2(mean_x - image_x_reference, mean_y - image_y_reference)

    colour_zone = {
        'angle_degrees_anticlockwise_from_down': angle,
        'mean_x': mean_x,
        'mean_y': mean_y
    }

    return colour_zone
