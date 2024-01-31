import torch
import torch.nn


class MSEClampLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_weights: torch.Tensor):

        out = ((y_pred - y_true) ** 2) * torch.clamp(y_weights, 0, 1)

        return out


class MSESumLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_weights: torch.Tensor):

        # The weights are 1 to 255, with 0 used for those that are missing. So this turns it into a check.
        y_weights_mask = torch.clamp(y_weights, 0, 1)

        out = (y_pred - y_true) ** 2.
        out = out * torch.max(y_weights, dim=1, keepdim=True)[0] * y_weights_mask

        return out
