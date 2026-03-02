from __future__ import annotations

from time import time
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.utilities.file_and_folder_operations import join
from torch import autocast, distributed as dist, nn

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D_MRCT
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D_MRCT
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.regression import (
    MaskedL1Loss,
    extract_valid_mask_from_input,
    normalize_radius,
)


class nnUNetTrainerMRCT_regressionBase(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
        self.logger.my_fantastic_logging["online_masked_mae"] = []
        self.logger.my_fantastic_logging["online_valid_voxels"] = []

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            1,
            allow_init=True,
            deep_supervision=False,
        )

    def _build_loss(self):
        return MaskedL1Loss()

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        return nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = np.array(self.configuration_manager.patch_size).astype(int)
        rotation_for_DA = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        do_dummy_2d_data_aug = False
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        self.print_to_log_file("Regression trainer uses deterministic train transforms with mirroring disabled")
        return rotation_for_DA, do_dummy_2d_data_aug, patch_size, mirror_axes

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D_MRCT(
                dataset_tr,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
            )
            dl_val = nnUNetDataLoader2D_MRCT(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
            )
        else:
            dl_tr = nnUNetDataLoader3D_MRCT(
                dataset_tr,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
            )
            dl_val = nnUNetDataLoader3D_MRCT(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
            )
        return dl_tr, dl_val

    @staticmethod
    def _unwrap_target(target):
        if isinstance(target, list):
            if len(target) != 1:
                raise ValueError(f"Regression trainer does not support deep supervision targets, got {len(target)}")
            return target[0]
        return target

    def _compute_masked_mae(self, pred_norm: torch.Tensor, target_norm: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return MaskedL1Loss()(pred_norm, target_norm, valid_mask)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._unwrap_target(batch["target"]).to(self.device, non_blocking=True)
        valid_mask = extract_valid_mask_from_input(data, self.dataset_json)
        target_norm = normalize_radius(target, self.dataset_json)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            pred_logits = self.network(data)
            pred_norm = torch.sigmoid(pred_logits)
            loss = self.loss(pred_norm, target_norm, valid_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._unwrap_target(batch["target"]).to(self.device, non_blocking=True)
        valid_mask = extract_valid_mask_from_input(data, self.dataset_json)
        target_norm = normalize_radius(target, self.dataset_json)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            pred_logits = self.network(data)
            pred_norm = torch.sigmoid(pred_logits)
            loss = self.loss(pred_norm, target_norm, valid_mask)
            masked_mae = self._compute_masked_mae(pred_norm, target_norm, valid_mask)

        return {
            "loss": loss.detach().cpu().numpy(),
            "masked_mae": masked_mae.detach().cpu().numpy(),
            "valid_voxels": int(valid_mask.sum().detach().cpu().item()),
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        if self.is_ddp:
            world_size = dist.get_world_size()

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()

            masked_maes = [None for _ in range(world_size)]
            dist.all_gather_object(masked_maes, outputs_collated["masked_mae"])
            masked_mae_here = np.vstack(masked_maes).mean()

            valid_voxels = [None for _ in range(world_size)]
            dist.all_gather_object(valid_voxels, outputs_collated["valid_voxels"])
            valid_voxels_here = int(np.sum([np.sum(v) for v in valid_voxels]))
        else:
            loss_here = np.mean(outputs_collated["loss"])
            masked_mae_here = np.mean(outputs_collated["masked_mae"])
            valid_voxels_here = int(np.sum(outputs_collated["valid_voxels"]))

        self.logger.log("val_losses", loss_here, self.current_epoch)
        self.logger.log("online_masked_mae", masked_mae_here, self.current_epoch)
        self.logger.log("online_valid_voxels", valid_voxels_here, self.current_epoch)

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file("train_loss", np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4))
        self.print_to_log_file("val_loss", np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4))
        self.print_to_log_file(
            "online_masked_mae",
            np.round(self.logger.my_fantastic_logging["online_masked_mae"][-1], decimals=4),
        )
        self.print_to_log_file(
            "online_valid_voxels",
            int(self.logger.my_fantastic_logging["online_valid_voxels"][-1]),
        )
        epoch_duration = (
            self.logger.my_fantastic_logging["epoch_end_timestamps"][-1]
            - self.logger.my_fantastic_logging["epoch_start_timestamps"][-1]
        )
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        if self._best_ema is None or self.logger.my_fantastic_logging["val_losses"][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging["val_losses"][-1]
            self.print_to_log_file(f"New best validation loss: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1
