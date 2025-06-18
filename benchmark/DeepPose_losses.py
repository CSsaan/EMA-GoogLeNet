import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            mask [N, K]
        """
        losses = F.l1_loss(pred, label, reduction="none")
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class SmoothL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            mask [N, K]
        """
        losses = F.smooth_l1_loss(pred, label, reduction="none")
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class L2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor, mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            mask [N, K]
        """
        losses = F.mse_loss(pred, label, reduction="none")
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class WingLoss(nn.Module):
    """refer https://github.com/TropComplique/wing-loss/blob/master/loss.py
    """
    def __init__(self, w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w * (1.0 - math.log(1.0 + w / epsilon))

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                wh_tensor: torch.Tensor,
                mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            wh_tensor [1, 1, 2]
            label [N, K, 2]
            mask [N, K]
        """
        delta = (pred - label).abs() * wh_tensor  # rel to abs
        losses = torch.where(condition=self.w > delta,
                             input=self.w * torch.log(1.0 + delta / self.epsilon),
                             other=delta - self.C)
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        return torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)


class SoftWingLoss(nn.Module):
    """refer mmpose/models/losses/regression_loss.py
    """
    def __init__(self, omega1: float = 2.0, omega2: float = 20.0, epsilon: float = 0.5) -> None:
        super().__init__()
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon
        self.B = omega1 - omega2 * math.log(1.0 + omega1 / epsilon)

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                wh_tensor: torch.Tensor,
                mask: torch = None) -> torch.Tensor:
        """
        Args:
            pred [N, K, 2]
            label [N, K, 2]
            wh_tensor [1, 1, 2]
            mask [N, K]
        """
        delta = (pred - label).abs() * wh_tensor  # rel to abs
        losses = torch.where(condition=delta < self.omega1,
                             input=delta,
                             other=self.omega2 * torch.log(1.0 + delta / self.epsilon) + self.B)
        if mask is not None:
            # filter invalid keypoints(e.g. out of range)
            losses = losses * mask.unsqueeze(2)

        loss = torch.mean(torch.sum(losses, dim=(1, 2)), dim=0)
        return loss
    


class NMEMetric:
    def __init__(self, device: torch.device) -> None:
        # 两眼外角点对应keypoint索引
        self.keypoint_idxs = [60, 72]
        self.nme_accumulator: float = 0.
        self.counter: float = 0.
        self.device = device

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred (shape [N, K, 2]): pred keypoints
            gt (shape [N, K, 2]): gt keypoints
            mask (shape [N, K]): valid keypoints mask
        """
        # ion: inter-ocular distance normalized error
        ion = torch.linalg.norm(gt[:, self.keypoint_idxs[0]] - gt[:, self.keypoint_idxs[1]], dim=1)

        valid_ion_mask = ion > 0
        if mask is None:
            mask = valid_ion_mask
        else:
            mask = torch.logical_and(mask, valid_ion_mask.unsqueeze_(dim=1)).sum(dim=1) > 0
        num_valid = mask.sum().item()

        # equal: (pred - gt).pow(2).sum(dim=2).pow(0.5).mean(dim=1)
        l2_dis = torch.linalg.norm(pred - gt, dim=2)[mask].mean(dim=1)  # [N]

        # avoid divide by zero
        ion = ion[mask]  # [N]

        self.nme_accumulator += l2_dis.div(ion).sum().item()
        self.counter += num_valid

    def evaluate(self):
        return self.nme_accumulator / self.counter
    

if __name__ == '__main__':
    metric = NMEMetric(device=torch.device("cpu"))
    metric.update(pred=torch.randn(32, 98, 2),
                  gt=torch.randn(32, 98, 2),
                  mask=torch.randn(32, 98))
    print(metric.evaluate())