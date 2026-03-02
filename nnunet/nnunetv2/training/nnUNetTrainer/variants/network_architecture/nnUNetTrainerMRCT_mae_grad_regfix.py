from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_regressionBase import (
    nnUNetTrainerMRCT_regressionBase,
)
from nnunetv2.utilities.regression import MaskedL1AndGradientLoss


class nnUNetTrainerMRCT_mae_grad_regfix(nnUNetTrainerMRCT_regressionBase):
    def _build_loss(self):
        return MaskedL1AndGradientLoss(lambda_l1=1.0, lambda_grad=0.1)
