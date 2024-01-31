import logging

import io

import json
import math
import random
import logging
import datetime

import requests

import torch
import numpy as np

import requests_cache

from collections import namedtuple

from torch.utils.data import Dataset

from kornia.augmentation import RandomErasing

from Scantensus.sources.securion import SecurionSource
from Scantensus.sources.clusters import ClusterSource

from Scantensus.utils.labels import bool_str_to_list, curve_str_to_list
from Scantensus.utils.heatmaps import make_curve_labels
from Scantensus.utils.geometry import line_len, interpolate_curveline

from ScantensusPT.image.read import read_image_into_t
from ScantensusPT.transforms.helpers import get_random_transform_parm, get_affine_matrix
from ScantensusPT.utils.heatmaps import render_gaussian_dot_u, render_gaussian_curve_u
from ScantensusPT.image.crop import center_crop_or_pad_t

CLUSTER_STORE_URL = "https://storage.googleapis.com/scantensus/fiducial"
SECURION_STORE_URL = "http://cardiac5.ts.zomirion.com:50601/scantensus-database-png-flat"

UnityInferSetT = namedtuple("UnityInferSetT", "image unity_f_code label_height_shift label_width_shift")
UnityTrainSetT = namedtuple("UnityTrainSetT", "image unity_f_code label_data label_height_shift label_width_shift transform_matrix")


class UnityRawInferSet(Dataset):

    def __init__(self,
                 image_t,
                 FileHash,
                 image_crop_size=(640, 640),
                 pre_post=False,
                 pre_post_list=None,
                 bw_images=False,
                 device="cpu",
                 name=None):

        super().__init__()

        self.frames = image_t.shape[0]

        self.unity_f_codes = [f"{FileHash}-{i:04}" for i in range(self.frames)]

        self.image_crop_size = image_crop_size
        self.pre_post = pre_post
        self.relative_frames_to_include = torch.as_tensor(pre_post_list, dtype=torch.int64)
        self.bw_images = bw_images
        self.name = name
        self.device = device

        image_t, self.label_height_shift, self.label_width_shift = center_crop_or_pad_t(image_t, output_size=image_crop_size, return_shift=True)
        if self.bw_images:
            image_t = image_t[:, 0, :, :]
        else:
            raise Exception("Only BW")

        self.image_t = image_t

    def __len__(self):
        return self.frames

    def __getitem__(self, idx):
        actual_frames_to_include = self.relative_frames_to_include + idx
        mask_frames_to_include = torch.bitwise_and(actual_frames_to_include >= 0, actual_frames_to_include < self.frames)
        clamp_actual_frames_to_include = torch.clamp(actual_frames_to_include, min=0, max=self.frames - 1)
        out_t = self.image_t[clamp_actual_frames_to_include, ...]
        out_t[torch.bitwise_not(mask_frames_to_include), ...] = 0

        return UnityInferSetT(image=out_t,
                              unity_f_code=self.unity_f_codes[idx],
                              label_height_shift=self.label_height_shift,
                              label_width_shift=self.label_width_shift)



class UnityInferSet(Dataset):

    def __init__(self,
                 image_fn_list,
                 png_cache_dir,
                 image_crop_size=(640, 640),
                 pre_post=False,
                 pre_post_list=None,
                 bw_images=False,
                 device="cpu",
                 name=None):

        super().__init__()

        self.image_fn_list = image_fn_list

        self.png_cache_dir = png_cache_dir
        self.image_crop_size = image_crop_size
        self.pre_post = pre_post
        self.pre_post_list = pre_post_list
        self.bw_images = bw_images
        self.name = name
        self.device = device
        self.png_session = requests_cache.CachedSession(cache_name='png_cache',
                                                        cache_control=False,
                                                        expire_after=datetime.timedelta(days=100),
                                                        backend='sqlite',
                                                        use_cache_dir=True)

    def __len__(self):
        return len(self.image_fn_list)

    def __getitem__(self, idx):

        unity_code = self.image_fn_list[idx]

        image_crop_size = self.image_crop_size
        device = self.device

        if "clusters" in unity_code:
            unity_o = ClusterSource(unity_code=unity_code,
                                    png_cache_dir=self.png_cache_dir,
                                    server_url=CLUSTER_STORE_URL)
        else:
            unity_o = SecurionSource(unity_code=unity_code,
                                     png_cache_dir=self.png_cache_dir,
                                     server_url=SECURION_STORE_URL)

        if self.pre_post:
            pre_post_list = self.pre_post_list
        else:
            pre_post_list = [0]

        out_image = []

        out_height_shift = None
        out_width_shift = None

        for offset in pre_post_list:
            image_path = unity_o.get_frame_url(frame_offset=offset)
            try:
                image = read_image_into_t(image_path=image_path,
                                          png_session=self.png_session,
                                          device=self.device)

                if image_crop_size:
                    image, height_shift, width_shift = center_crop_or_pad_t(image=image,
                                                                            output_size=image_crop_size,
                                                                            cval=0,
                                                                            device=device)
                else:
                    height_shift = 0
                    width_shift = 0
            except Exception:
                if offset == 0:
                    logging.exception(f"failed to load {unity_code} - {image_path} at offset {offset}")

                image = torch.zeros((1, image_crop_size[0], image_crop_size[1]), device=device, dtype=torch.uint8)

                height_shift = None
                width_shift = None

            if height_shift is not None and width_shift is not None:
                out_height_shift = height_shift
                out_width_shift = width_shift

            if self.bw_images:
                image = image[[0], ...]
            else:
                if image.shape[0] == 1:
                    image = torch.vstack((image, image, image))

            out_image.append(image)

        image = torch.cat(out_image)

        if out_height_shift == None or out_width_shift == None:
            raise Exception(f"No images were able to be loaded")

        return UnityInferSetT(image=image,
                              unity_f_code=unity_o.unity_f_code,
                              label_height_shift=out_height_shift,
                              label_width_shift=out_width_shift)


class UnityDataset(Dataset):

    def __init__(self,
                 database_url,
                 keypoint_names,
                 transform=False,
                 transform_translate=True,
                 transform_scale=True,
                 transform_rotate=True,
                 transform_shear=True,
                 image_crop_size=(608, 608),
                 image_out_size=(512, 512),
                 pre_post=False,
                 pre_post_list=None,
                 bw_images=False,
                 device="cpu",
                 name=None):

        super().__init__()

        self.logger = logging.getLogger()
        self.keypoint_names = keypoint_names
        self.transform = transform
        self.transform_translate = transform_translate
        self.transform_rotate = transform_rotate
        self.transform_scale = transform_scale
        self.transform_shear = transform_shear
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.pre_post = pre_post
        self.pre_post_list = pre_post_list
        self.bw_images = bw_images
        self.device = device
        self.name = name

        r = requests.get(database_url)
        if r.status_code != 200:
            raise Exception(f"Failed to load database url {database_url}")

        self.db_raw = json.loads(r.content)

        self.image_fn_list = list(self.db_raw.keys())
        logging.info(f"Number of cases {len(self.image_fn_list)}")
        hard_list = []

        if False:
            for key, data in self.db_raw.items():
                try:
                    if data['labels']['av-centre']['type'] == 'point':
                        hard_list.append(key)
                except Exception:
                    pass

            logging.info(f"Hard list has {len(hard_list)} items")

            self.image_fn_list.extend(hard_list * 6)

        self.aug_random_erase = RandomErasing(scale=(0.05, 0.15),
                                              ratio=(0.3, 3.3),
                                              value=0.0,
                                              same_on_batch=False,
                                              p=1,
                                              keepdim=False)

        self.png_session = None

    def __len__(self):
        return len(self.image_fn_list)

    def __getitem__(self, idx):

        if not self.png_session:
            self.png_session = requests_cache.CachedSession(cache_name='png_cache',
                                                            use_cache_dir=True,
                                                            cache_control=False,
                                                            expire_after=datetime.timedelta(days=300),
                                                            backend='sqlite',
                                                            stale_if_error=True,
                                                            wal=True,
                                                            timeout=30)

        image_crop_size = self.image_crop_size
        image_out_size = self.image_out_size
        transform = self.transform

        unity_code = self.image_fn_list[idx]

        image_crop_size = self.image_crop_size
        device = self.device

        transform_rand_num = random.random()

        if "clusters" in unity_code:
            unity_o = ClusterSource(unity_code=unity_code,
                                    png_cache_dir=None,
                                    server_url=CLUSTER_STORE_URL)
        else:
            unity_o = SecurionSource(unity_code=unity_code,
                                     png_cache_dir=None,
                                     server_url=SECURION_STORE_URL)

        if self.pre_post:
            pre_post_list = self.pre_post_list
        else:
            pre_post_list = [0]

        out_image = []

        out_height_shift = None
        out_width_shift = None

        for offset in pre_post_list:
            image_path = unity_o.get_frame_url(frame_offset=offset)
            try:
                image = read_image_into_t(image_path=image_path,
                                          png_session=self.png_session,
                                          device=self.device)
                image, height_shift, width_shift = center_crop_or_pad_t(image=image,
                                                                        output_size=image_crop_size,
                                                                        cval=0,
                                                                        device=device)
            except Exception:
                if offset == 0:
                    logging.exception(f"failed to load {unity_code} at offset {offset}")

                image = torch.zeros((1, image_crop_size[0], image_crop_size[1]), device=device, dtype=torch.uint8)

                height_shift = None
                width_shift = None

            if height_shift is not None and width_shift is not None:
                out_height_shift = height_shift
                out_width_shift = width_shift

            if self.bw_images:
                image = image[[0], ...]
            else:
                if image.shape[0] == 1:
                    image = torch.vstack((image, image, image))

            if transform:
                if self.pre_post:
                    if offset != 0:
                        if transform_rand_num <= 0.2:
                            image = torch.zeros_like(image)
                    elif offset == 0:
                        if 0.2 < transform_rand_num <= 0.25:
                            image = torch.zeros_like(image)

            out_image.append(image)

        image = torch.cat(out_image)

        label_data = self.db_raw[unity_code]['labels']

        if out_height_shift is None and out_width_shift is None:
            out_height_shift = 0
            out_width_shift = 0
            # Note that when label_data goes through json.dumps() it becomes the string 'null'
            label_data = None

        in_out_height_ratio = image_crop_size[0] / image_out_size[0]
        in_out_width_ratio = image_crop_size[1] / image_out_size[1]

        if transform:
            translate_h, translate_w, scale_h, scale_w, rotation_theta, shear_theta = get_random_transform_parm(translate=self.transform_translate,
                                                                                                                scale=self.transform_scale,
                                                                                                                rotate=self.transform_rotate,
                                                                                                                shear=self.transform_shear)

            transform_matrix = get_affine_matrix(tx=translate_w,
                                                 ty=translate_h,
                                                 sx=scale_w * in_out_width_ratio,
                                                 sy=scale_h * in_out_height_ratio,
                                                 rotation_theta=rotation_theta,
                                                 shear_theta=shear_theta,
                                                 device=device)

            transform_matrix_inv = transform_matrix.inverse()

        else:
            transform_matrix = get_affine_matrix(tx=0,
                                                 ty=0,
                                                 sx=in_out_width_ratio,
                                                 sy=in_out_height_ratio,
                                                 rotation_theta=0,
                                                 shear_theta=0,
                                                 device=device)

            transform_matrix_inv = transform_matrix.inverse()

        image = image.float().div(255)

        if transform:
            if 0.2 < transform_rand_num <= 0.4:
                image = self.aug_random_erase(image)

        image = transform_image(image=image,
                                transform_matrix=transform_matrix_inv,
                                out_image_size=self.image_out_size)

        if transform:
            random_gamma = math.exp(random.triangular(-0.8, 0.8))
            image = image.pow(random_gamma)

        image = image.mul(255).to(torch.uint8)

        return UnityTrainSetT(image=image,
                              unity_f_code=unity_code,
                              label_data=json.dumps(label_data),
                              label_height_shift=out_height_shift,
                              label_width_shift=out_width_shift,
                              transform_matrix=transform_matrix)


def apply_matrix_to_coords(transform_matrix: torch.Tensor, coord: torch.Tensor):

    if coord.dim() == 2:
        coord = coord.unsqueeze(0)

    batch_size = coord.shape[0]

    if transform_matrix.dim() == 2:
        transform_matrix = transform_matrix.unsqueeze(0)

    if transform_matrix.size()[1:] == (3, 3):
        transform_matrix = transform_matrix[:, :2, :]

    A_batch = transform_matrix[:, :, :2]
    if A_batch.size(0) != batch_size:
        A_batch = A_batch.repeat(batch_size, 1, 1)

    B_batch = transform_matrix[:, :, 2].unsqueeze(1)

    coord = coord.bmm(A_batch.transpose(1, 2)) + B_batch.expand(coord.shape)

    return coord


def transform_image(image: torch.Tensor, transform_matrix: torch.Tensor, out_image_size=(512,512)):

    device = image.device

    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]

    out_image_h = out_image_size[0]
    out_image_w = out_image_size[1]

    identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=device)
    intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

    grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w], align_corners=False)
    grid = grid.reshape(intermediate_grid_shape)

    # For some reason it gives you w, h at the output of affine_grid. So switch here.
    grid = grid[..., [1, 0]]
    grid = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=grid)
    grid = grid[..., [1, 0]]

    grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

    # There is no constant selection for padding mode - so border will have to do to weights.
    image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0)

    return image


def normalize_coord(coord: torch.Tensor, image_size: torch.Tensor):

    coord = (coord * 2 / image_size) - 1

    return coord


def unnormalize_coord(coord: torch.Tensor, image_size: torch.tensor):

    coord = (coord + 1) * image_size / 2

    return coord

class UnityMakeHeatmaps(torch.nn.Module):

    def __init__(self,
                 keypoint_names,
                 image_crop_size,
                 image_out_size,
                 heatmap_scale_factors=(2, 4),
                 dot_sd=4,
                 curve_sd=2,
                 dot_weight_sd=40,
                 curve_weight_sd=10,
                 dot_weight=40,
                 curve_weight=10,
                 sub_pixel=True,
                 single_weight=True,
                 device="cpu"):

        super().__init__()

        self.keypoint_names = keypoint_names
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.heatmap_scale_factors = heatmap_scale_factors
        self.dot_sd = dot_sd
        self.curve_sd = curve_sd
        self.dot_weight_sd = dot_weight_sd
        self.curve_weight_sd = curve_weight_sd
        self.dot_weight = dot_weight
        self.curve_weight = curve_weight
        self.sub_pixel = sub_pixel
        self.single_weight = single_weight
        self.device = device

    def forward(self,
                label_data,
                label_height_shift,
                label_width_shift,
                transform_matrix):

        batch_size = len(transform_matrix)

        out_heatmaps = []
        out_weights = []
        for scale_factor in self.heatmap_scale_factors:
            heatmaps_batch = []
            weights_batch = []
            for i in range(batch_size):
                heatmaps, weights = make_labels_and_masks(image_in_size=self.image_crop_size,
                                                          image_out_size=self.image_out_size,
                                                          label_data=label_data[i],
                                                          keypoint_names=self.keypoint_names,
                                                          label_height_shift=label_height_shift[i],
                                                          label_width_shift=label_width_shift[i],
                                                          heatmap_scale_factor=scale_factor,
                                                          transform_matrix=transform_matrix[i],
                                                          dot_sd=self.dot_sd,
                                                          curve_sd=self.curve_sd,
                                                          dot_weight_sd=self.dot_weight_sd,
                                                          curve_weight_sd=self.curve_weight_sd,
                                                          dot_weight=self.dot_weight,
                                                          curve_weight=self.curve_weight,
                                                          sub_pixel=self.sub_pixel,
                                                          single_weight=self.single_weight,
                                                          device=self.device)
                heatmaps_batch.append(heatmaps)
                weights_batch.append(weights)
            out_heatmaps.append(torch.stack(heatmaps_batch))
            out_weights.append(torch.stack(weights_batch))

        return out_heatmaps, out_weights


def make_labels_and_masks(image_in_size,
                          image_out_size,
                          keypoint_names,
                          label_data,
                          label_height_shift=0,
                          label_width_shift=0,
                          transform_matrix=None,
                          heatmap_scale_factor=1,
                          dot_sd=4,
                          curve_sd=2,
                          dot_weight_sd=40,
                          curve_weight_sd=10,
                          dot_weight=40,
                          curve_weight=10,
                          sub_pixel=True,
                          single_weight=True,
                          device="cpu"):

    # if you are using in a different thread, e.g. a dataloader and device must be cpu.

    device = torch.device(device)

    num_keypoints = len(keypoint_names)

    if transform_matrix is not None:
        transform_matrix = transform_matrix.to(device)
        target_out_height = image_out_size[0] // heatmap_scale_factor
        target_out_width = image_out_size[1] // heatmap_scale_factor
        image_out_size_t = torch.tensor(image_out_size, dtype=torch.float, device=device)
    else:
        target_out_height = image_in_size[0] // heatmap_scale_factor
        target_out_width = image_in_size[1] // heatmap_scale_factor
        image_out_size_t = None

    image_in_size_t = torch.tensor(image_in_size, dtype=torch.float, device=device)

    heatmaps = torch.zeros((num_keypoints, target_out_height, target_out_width), device=device, dtype=torch.uint8)

    if not single_weight:
        weights = torch.zeros((num_keypoints, target_out_height, target_out_width), device=device, dtype=torch.uint8)
    else:
        weights = torch.zeros((num_keypoints, 1, 1), device=device, dtype=torch.uint8)

    if type(label_data) is str:
        label_data = json.loads(label_data)

    if label_data is None:
        logging.error(f"No label data supplied")
        return heatmaps, weights

    for keypoint_idx, keypoint in enumerate(keypoint_names):
        try:
            keypoint_data = label_data.get(keypoint, None)
        except Exception:
            logging.exception(f"{keypoint} - {label_data}")

        if keypoint_data is None:
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        label_type = keypoint_data[0]['type']

        if label_type is None:
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        if label_type == "off":
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 1
            continue

        if label_type == "blurred":
            heatmaps[keypoint_idx, ...] = 0
            weights[keypoint_idx, ...] = 0
            continue

        heatmaps[keypoint_idx, ...] = 0
        weights[keypoint_idx, ...] = 0

        for instance_num, instance_data in enumerate(keypoint_data):

            y_data = instance_data["y"]
            x_data = instance_data["x"]
            straight_segment = instance_data.get("straight_segment", None)
            if straight_segment is not None:
                straight_segment = straight_segment

            if type(y_data) is str:
                ys = curve_str_to_list(y_data)
            else:
                ys = y_data

            if type(x_data) is str:
                xs = curve_str_to_list(x_data)
            else:
                xs = x_data

            if type(straight_segment) is str:
                straight_segments = bool_str_to_list(straight_segment)
            else:
                straight_segments = straight_segment

            if len(ys) != len(xs) or not all(np.isfinite(ys)) or not all(np.isfinite(xs)) or len(ys) == 0:
                print(f"problem with data {keypoint}, {ys}, {xs}")
                heatmaps[keypoint_idx, ...] = 0
                weights[keypoint_idx, ...] = 0
                continue

            coord = torch.tensor([ys, xs], device=device).transpose(0, 1)
            label_shift = torch.tensor([label_height_shift, label_width_shift], device=device)
            coord = coord + label_shift

            if transform_matrix is not None:
                coord = normalize_coord(coord=coord, image_size=image_in_size_t)
                coord = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=coord)
                coord = unnormalize_coord(coord=coord, image_size=image_out_size_t)
            else:
                coord = coord.unsqueeze(0)

            coord = coord / heatmap_scale_factor

            dot_sd_t = torch.tensor([dot_sd, dot_sd], dtype=torch.float, device=device)
            dot_weight_sd_t = torch.tensor([dot_weight_sd, dot_weight_sd], dtype=torch.float, device=device)

            if "flow" not in keypoint:
                curve_sd_t = torch.tensor([curve_sd, curve_sd], dtype=torch.float, device=device)
                curve_weight_sd_t = torch.tensor([curve_weight_sd, curve_weight_sd], dtype=torch.float, device=device)
            else:
                curve_sd_t = torch.tensor([curve_sd*2, 0.5], dtype=torch.float, device=device)
                curve_weight_sd_t = torch.tensor([curve_weight_sd*2, 0.5], dtype=torch.float, device=device)

            if len(ys) == 1:
                out_heatmap = render_gaussian_dot_u(point=coord[0, 0, :],
                                                    std=dot_sd_t,
                                                    size=(target_out_height, target_out_width),
                                                    mul=255)

                if not single_weight:
                    out_weight = render_gaussian_dot_u(point=coord[0, 0, :],
                                                       std=dot_weight_sd_t,
                                                       size=(target_out_height, target_out_width),
                                                       mul=(dot_weight-1)).add(1)
                else:
                    out_weight = torch.tensor([dot_weight], device=device, dtype=torch.uint8)

            elif len(ys) >= 2:
                points_np = coord[0, :, :].cpu().numpy()
                ys = points_np[:, 0].tolist()
                xs = points_np[:, 1].tolist()

                curve_points_len = line_len(points_np)
                curve_points_len = int(curve_points_len)
                curve_points_len = max(curve_points_len, len(ys))
                out_curve_y, out_curve_x = interpolate_curveline(ys=ys, xs=xs, straight_segments=straight_segments, total_points_out=curve_points_len * 2)

                curve_points = torch.tensor([out_curve_y, out_curve_x],
                                            dtype=torch.float,
                                            device=device).T

                if sub_pixel:
                    out_heatmap = render_gaussian_curve_u(points=curve_points,
                                                          std=curve_sd_t,
                                                          size=(target_out_height, target_out_width),
                                                          mul=255).to(device)

                    if not single_weight:
                        out_weight = render_gaussian_curve_u(points=curve_points,
                                                             std=curve_weight_sd_t,
                                                             size=(target_out_height, target_out_width),
                                                             mul=(curve_weight-1)).add(1).to(device)
                    else:
                        out_weight = torch.tensor([curve_weight], device=device, dtype=torch.uint8)

                else:
                    curve_kernel_size = 2 * ((math.ceil(curve_sd) * 5) // 2) + 1
                    curve_weight_kernel_size = 2 * ((math.ceil(curve_weight_sd) * 5) // 2) + 1

                    out_heatmap = make_curve_labels(points=curve_points,
                                                    image_size=(target_out_height, target_out_width),
                                                    kernel_sd=curve_sd,
                                                    kernel_size=curve_kernel_size)

                    out_heatmap = torch.tensor(out_heatmap, device=device)
                    out_heatmap = out_heatmap.mul(255).to(torch.uint8)

                    if not single_weight:
                        out_weight = make_curve_labels(points=curve_points,
                                                       image_size=(target_out_height, target_out_width),
                                                       kernel_sd=curve_weight_sd,
                                                       kernel_size=curve_weight_kernel_size)

                        out_weight = torch.tensor(out_weight, device=device)
                        out_weight = out_weight.mul(curve_weight-1).add(1).to(torch.uint8)
                    else:
                        out_weight = torch.tensor([curve_weight], device=device, dtype=torch.uint8)

            else:
                print(f"Error - no idea what problem with data was: {ys}, {xs}")
                continue

            heatmaps[keypoint_idx, ...] = torch.max(out_heatmap, heatmaps[keypoint_idx, ...])
            weights[keypoint_idx, ...] = torch.max(out_weight, weights[keypoint_idx, ...])

    return heatmaps, weights
