from typing import List

import torch

from .point import get_point

def get_label_keypoint(y_pred: torch.Tensor, keypoint_names: List[str], label: str):
    point = get_point(y_pred[:, [keypoint_names.index(label)], :, :])
    return point.unsqueeze(2)

