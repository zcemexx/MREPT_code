import torch
from torch import autocast

from nnunetv2.training.loss.mae_grad3d import MAEAndGradientLoss3D
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_mae import (
    nnUNetTrainerMRCT_mae,
)
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerMRCT_mae_grad(nnUNetTrainerMRCT_mae):
    """MRCT regression trainer using L1 + 0.1 * 3D gradient loss."""

    def _build_loss(self):
        return MAEAndGradientLoss3D(lambda_l1=1.0, lambda_grad=0.1)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = torch.sigmoid(self.network(data))
            del data
            l = self.loss(output, self._normalize_kernel_radius(target))

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}
