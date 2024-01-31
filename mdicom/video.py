import io
import logging

import imageio.v3

import numpy as np

from skimage.transform import resize
from mdicom.image import center_crop_or_pad
from mdicom.image import make_image_4d
from mdicom.image import ensure_3channels


def make_gif_from_img(img,
                      frame_time,
                      output_size=(200, 200),
                      max_crop_dimension=608,
                      still_crop_method="shrink",
                      video_crop_method="shrink",
                      output_fps=25,
                      still_frame_num=1,
                      video_repeats=1):

    NumberOfFrames = img.shape[0]
    num_channels = img.shape[3]

    img = make_image_4d(img)

    out_img_list = []

    if NumberOfFrames is None or NumberOfFrames == 1:

        if still_crop_method == "full":
            max_size_y = max_size_x = max(img.shape[1], img.shape[2])
        elif still_crop_method == "shrink":
            max_size_y = max_size_x = min(img.shape[1], img.shape[2])
        elif still_crop_method == "crop":
            max_size_y, max_size_x = max_crop_dimension
        else:
            raise Exception(f"still_crop_method should be full or shrink")

        out_pixel_array, _, _ = center_crop_or_pad(img, (max_size_y, max_size_x), cval=0)

        for i in range(still_frame_num):
            out_data = resize(out_pixel_array[0, ...], output_size)
            out_data = (out_data * 255).astype(np.uint8)
            if num_channels == 3:
                out_img_list.append(out_data)
            elif num_channels == 1:
                out_img_list.append(out_data[..., 0])

    elif NumberOfFrames > 2:

        if video_crop_method == "full":
            max_size = max(img.shape[1], img.shape[2])
        elif video_crop_method == "shrink":
            max_size = min(img.shape[1], img.shape[2])
        else:
            raise Exception(f"video_crop_method should be full or shrink")

        out_pixel_array, _, _ = center_crop_or_pad(img, (max_size, max_size), cval=0)

        for rep in range(video_repeats):
            true_time = 0
            video_time = 0

            if frame_time is None:
                frame_time = 1000 / output_fps
            input_frame_time = round(frame_time, 3)
            output_frame_time = round(1000/output_fps, 3)
            i = 0

            while i < NumberOfFrames:
                logging.info(f"Gif frame {i} total {NumberOfFrames}")
                if true_time == video_time:
                    # If you give a uint it will convert to float from 0..1 so both need * 255 below
                    out_data = resize(out_pixel_array[i, ...], output_size)
                    out_data = (out_data * 255).astype(np.uint8)
                    if num_channels == 3:
                        out_img_list.append(out_data)
                    elif num_channels == 1:
                        out_img_list.append(out_data[..., 0])

                    true_time = true_time + input_frame_time
                    video_time = video_time + output_frame_time
                    i = i + 1
                    continue

                if true_time > video_time:
                    out_data = resize(out_pixel_array[i, ...], output_size)
                    out_data = (out_data * 255).astype(np.uint8)
                    if num_channels == 3:
                        out_img_list.append(out_data)
                    elif num_channels == 1:
                        out_img_list.append(out_data[..., 0])

                    video_time = video_time + output_frame_time
                    continue

                if true_time < video_time:
                    true_time = true_time + input_frame_time
                    i = i + 1
                    continue

    out_gif_bytesio = io.BytesIO()
    out_duration = 1000 / output_fps

    imageio.v3.imwrite(uri=out_gif_bytesio,
                       image=out_img_list,
                       extension=".gif",
                       format_hint='.gif',
                       duration=out_duration,
                       loop=0)


    out_gif_bytesio_len = out_gif_bytesio.tell()
    out_gif_bytesio.seek(0)

    return out_gif_bytesio, out_gif_bytesio_len


def make_mp4_from_img(img,
                      frame_time
                      ):

    NumberOfFrames = img.shape[0]

    out_bytesio = io.BytesIO()

    fps = 1000/frame_time

    imageio.v3.imwrite(out_bytesio, image=img, format_hint='.mp4', output_params=["-f", "mp4"], fps=fps)

    out_bytesio_len = out_bytesio.tell()
    out_bytesio.seek(0)

    return out_bytesio, out_bytesio_len
