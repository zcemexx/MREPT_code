import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.evaluation.evaluate_predictions import compute_regression_metrics_on_folder
from nnunetv2.inference.export_prediction import convert_predicted_regression_to_original_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.regression import masked_gradient_loss, masked_l1_loss


class FakeRegressionReaderWriter:
    def read_seg(self, filename):
        return np.load(filename), {}


class DummyNetwork(torch.nn.Module):
    def forward(self, x):
        return x


def identity_resample(array, new_shape, current_spacing, target_spacing):
    return array


class TestRegressionPipeline(unittest.TestCase):
    def test_masked_l1_loss_empty_mask_keeps_graph_connected(self):
        pred = torch.randn((1, 1, 2, 2), requires_grad=True)
        target = torch.zeros_like(pred)
        valid_mask = torch.zeros_like(pred, dtype=torch.bool)

        loss = masked_l1_loss(pred, target, valid_mask)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_masked_gradient_loss_empty_pairs_keeps_graph_connected(self):
        pred = torch.randn((1, 1, 1, 1, 1), requires_grad=True)
        target = torch.zeros_like(pred)
        valid_mask = torch.ones_like(pred, dtype=torch.bool)

        loss = masked_gradient_loss(pred, target, valid_mask)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_masked_gradient_loss_single_axis_valid_is_finite(self):
        pred = torch.tensor([[[[[0.0], [1.0], [2.0]]]]], requires_grad=True)
        target = torch.tensor([[[[[0.0], [1.5], [2.0]]]]])
        valid_mask = torch.ones_like(pred, dtype=torch.bool)

        loss = masked_gradient_loss(pred, target, valid_mask)

        self.assertTrue(torch.isfinite(loss))

    def test_predict_logits_from_preprocessed_data_keeps_raw_logits_for_regression(self):
        predictor = nnUNetPredictor(
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.network = DummyNetwork()
        predictor.list_of_parameters = [predictor.network.state_dict()]
        predictor.dataset_json = {'regression_task': True}

        with patch.object(
            predictor,
            'predict_sliding_window_return_logits',
            return_value=torch.full((1, 4, 4, 4), 2.0, dtype=torch.float32),
        ):
            result = predictor.predict_logits_from_preprocessed_data(
                torch.zeros((1, 4, 4, 4), dtype=torch.float32),
                reconstruction_mode='gaussian',
            )

        self.assertTrue(torch.allclose(result, torch.full_like(result, 2.0)))

    def test_rec_gaussian_supports_output_channels_not_matching_input_channels(self):
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.configuration_manager = SimpleNamespace(patch_size=(4, 4, 4))
        predictor.allowed_mirroring_axes = None
        predictor.network = DummyNetwork()
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=31)

        def predict_one_channel(x):
            return torch.ones((x.shape[0], 1, *x.shape[2:]), dtype=torch.float32, device=x.device)

        predictor._internal_maybe_mirror_and_predict = predict_one_channel
        data = torch.zeros((2, 6, 6, 6), dtype=torch.float32)
        slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])

        result = predictor.rec_gaussian(slicers, data, results_device=torch.device('cpu'))

        self.assertEqual(tuple(result.shape), (1, 6, 6, 6))

    def test_convert_predicted_regression_to_original_shape_zeros_outside_mask_and_clips(self):
        predicted_logits = torch.full((1, 2, 2, 2), 10.0, dtype=torch.float32)
        valid_mask_preprocessed = np.array(
            [[
                [[1, 0], [0, 1]],
                [[0, 0], [1, 1]],
            ]],
            dtype=np.uint8,
        )
        plans_manager = SimpleNamespace(transpose_backward=[0, 1, 2])
        configuration_manager = SimpleNamespace(
            spacing=(1.0, 1.0, 1.0),
            resampling_fn_data=identity_resample,
            resampling_fn_seg=identity_resample,
        )
        properties = {
            'shape_after_cropping_and_before_resampling': (2, 2, 2),
            'shape_before_cropping': (2, 2, 2),
            'bbox_used_for_cropping': [[0, 2], [0, 2], [0, 2]],
            'spacing': (1.0, 1.0, 1.0),
        }
        dataset_json = {
            'regression_task': True,
            'kernel_radius_min': 1,
            'kernel_radius_max': 30,
        }

        radius_map, valid_mask = convert_predicted_regression_to_original_shape(
            predicted_logits,
            valid_mask_preprocessed,
            plans_manager,
            configuration_manager,
            properties,
            dataset_json,
        )

        self.assertTrue(np.all(radius_map[valid_mask] >= 1))
        self.assertTrue(np.all(radius_map[valid_mask] <= 30))
        self.assertTrue(np.all(radius_map[~valid_mask] == 0))

    def test_convert_predicted_regression_to_original_shape_rejects_multi_channel_input(self):
        plans_manager = SimpleNamespace(transpose_backward=[0, 1, 2])
        configuration_manager = SimpleNamespace(
            spacing=(1.0, 1.0, 1.0),
            resampling_fn_data=identity_resample,
            resampling_fn_seg=identity_resample,
        )
        properties = {
            'shape_after_cropping_and_before_resampling': (2, 2, 2),
            'shape_before_cropping': (2, 2, 2),
            'bbox_used_for_cropping': [[0, 2], [0, 2], [0, 2]],
            'spacing': (1.0, 1.0, 1.0),
        }

        with self.assertRaises(ValueError):
            convert_predicted_regression_to_original_shape(
                torch.zeros((2, 2, 2, 2), dtype=torch.float32),
                np.ones((1, 2, 2, 2), dtype=np.uint8),
                plans_manager,
                configuration_manager,
                properties,
                {'regression_task': True, 'kernel_radius_min': 1, 'kernel_radius_max': 30},
            )

    def test_compute_regression_metrics_on_folder_missing_case_raises_without_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            gt_dir.mkdir()
            pred_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))
            np.save(gt_dir / 'case2.npy', np.array([[[2, 2], [1, 0]]], dtype=np.uint8))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))
            output_file = pred_dir / 'summary.json'

            with self.assertRaises(RuntimeError):
                compute_regression_metrics_on_folder(
                    str(gt_dir),
                    str(pred_dir),
                    str(output_file),
                    FakeRegressionReaderWriter(),
                    '.npy',
                    num_processes=1,
                )

            self.assertFalse(output_file.exists())

    def test_compute_regression_metrics_on_folder_writes_regression_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            gt_dir.mkdir()
            pred_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))
            np.save(gt_dir / 'case2.npy', np.array([[[2, 2], [1, 0]]], dtype=np.uint8))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 3], [0, 4]]], dtype=np.uint8))
            np.save(pred_dir / 'case2.npy', np.array([[[2, 1], [2, 0]]], dtype=np.uint8))
            output_file = pred_dir / 'summary.json'

            summary = compute_regression_metrics_on_folder(
                str(gt_dir),
                str(pred_dir),
                str(output_file),
                FakeRegressionReaderWriter(),
                '.npy',
                num_processes=1,
            )

            self.assertEqual(summary['task_type'], 'regression')
            self.assertEqual(summary['case_counts']['expected_cases'], 2)
            self.assertEqual(summary['case_counts']['predicted_cases'], 2)
            self.assertEqual(summary['case_counts']['missing_cases'], [])
            self.assertEqual(summary['selection_metric']['name'], 'MAE')
            self.assertEqual(summary['selection_metric']['mode'], 'min')
            for key in ('MAE', 'MSE', 'RMSE', 'Global_RMSE', 'Acc_1', 'Acc_3', 'Acc_5', 'ValidVoxelCount'):
                self.assertIn(key, summary['foreground_mean'])
            self.assertTrue(output_file.exists())


if __name__ == '__main__':
    unittest.main()
