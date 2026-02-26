import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GradientLoss3D(nn.Module):
    """L1 loss between finite-difference gradient magnitudes of pred and target."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
        if pred.ndim not in (4, 5):
            raise ValueError(f"expected 4D or 5D tensors (B,C,H,W or B,C,D,H,W), got ndim={pred.ndim}")

        losses = []
        for axis in range(2, pred.ndim):
            if pred.shape[axis] <= 1:
                continue
            pred_grad = torch.abs(torch.diff(pred, dim=axis))
            target_grad = torch.abs(torch.diff(target, dim=axis))
            losses.append(F.l1_loss(pred_grad, target_grad))

        if len(losses) == 0:
            return pred.new_zeros(())
        return torch.stack(losses).sum()


class MAEAndGradientLoss3D(nn.Module):
    def __init__(self, lambda_l1: float = 1.0, lambda_grad: float = 0.1):
        super().__init__()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_grad = float(lambda_grad)
        self.grad_loss = GradientLoss3D()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss_l1 = F.l1_loss(pred, target)
        loss_grad = self.grad_loss(pred, target)
        return self.lambda_l1 * loss_l1 + self.lambda_grad * loss_grad
