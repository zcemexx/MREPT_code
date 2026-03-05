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
from nnunetv2.utilities.regression import (
    compute_masked_gradient_mae,
    compute_masked_pearson_r,
    compute_regression_case_metrics,
    masked_gradient_loss,
    masked_l1_loss,
)


class FakeRegressionReaderWriter:
    def read_seg(self, filename):
        return np.load(filename), {'spacing': (2.0, 1.0, 1.0)}


class DummyNetwork(torch.nn.Module):
    def forward(self, x):
        return x


def identity_resample(array, new_shape, current_spacing, target_spacing):
    return array


class TestRegressionPipeline(unittest.TestCase):
    def test_compute_masked_pearson_r_returns_expected_value(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        gt = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        valid_mask = np.array([True, True, True, False])

        pearson_r = compute_masked_pearson_r(pred, gt, valid_mask)

        self.assertAlmostEqual(pearson_r, 1.0, places=6)

    def test_compute_masked_pearson_r_constant_input_returns_nan(self):
        pred = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        gt = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        valid_mask = np.array([True, True, True])

        pearson_r = compute_masked_pearson_r(pred, gt, valid_mask)

        self.assertTrue(np.isnan(pearson_r))

    def test_compute_masked_gradient_mae_returns_expected_value(self):
        pred = np.array([[[0.0, 1.0, 3.0]]], dtype=np.float32)
        gt = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32)
        valid_mask = np.ones_like(pred, dtype=bool)

        gradient_mae = compute_masked_gradient_mae(pred, gt, valid_mask)

        self.assertAlmostEqual(gradient_mae, 0.5, places=6)

    def test_compute_masked_gradient_mae_respects_spacing(self):
        pred = np.array([[[0.0, 2.0, 4.0]]], dtype=np.float32)
        gt = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32)
        valid_mask = np.ones_like(pred, dtype=bool)

        gradient_mae = compute_masked_gradient_mae(pred, gt, valid_mask, spacing=(1.0, 1.0, 2.0))

        self.assertAlmostEqual(gradient_mae, 0.5, places=6)

    def test_compute_regression_case_metrics_includes_extended_metrics(self):
        pred = np.array([[[1.0, 3.0], [0.0, 4.0]]], dtype=np.float32)
        gt = np.array([[[1.0, 2.0], [0.0, 3.0]]], dtype=np.float32)
        valid_mask = gt > 0
        tissue_mask = np.array([[[1, 2], [0, 3]]], dtype=np.uint8)

        metrics = compute_regression_case_metrics(
            pred,
            gt,
            valid_mask,
            tissue_mask=tissue_mask,
            spacing=(1.0, 1.0, 1.0),
        )

        for key in (
            'Pearson_r',
            'Gradient_MAE',
            'WM_MAE',
            'GM_MAE',
            'CSF_MAE',
            'WM_ValidVoxelCount',
            'GM_ValidVoxelCount',
            'CSF_ValidVoxelCount',
        ):
            self.assertIn(key, metrics)

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

    def test_compute_regression_metrics_on_folder_missing_case_writes_summary(self):
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

            summary = compute_regression_metrics_on_folder(
                str(gt_dir),
                str(pred_dir),
                str(output_file),
                FakeRegressionReaderWriter(),
                '.npy',
                num_processes=1,
            )

            self.assertTrue(output_file.exists())
            self.assertEqual(summary['case_counts']['included_cases'], 1)
            self.assertEqual(summary['case_counts']['missing_cases'], 1)
            self.assertEqual(summary['case_counts']['included_case_ids'], ['case1'])
            self.assertEqual(summary['case_counts']['missing_case_ids'], ['case2'])

    def test_compute_regression_metrics_on_folder_zero_overlap_still_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            gt_dir.mkdir()
            pred_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))
            np.save(pred_dir / 'case9.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))

            with self.assertRaises(RuntimeError):
                compute_regression_metrics_on_folder(
                    str(gt_dir),
                    str(pred_dir),
                    None,
                    FakeRegressionReaderWriter(),
                    '.npy',
                    num_processes=1,
                )

    def test_compute_regression_metrics_on_folder_writes_regression_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            raw_images_dir = base / 'imagesTr'
            gt_dir.mkdir()
            pred_dir.mkdir()
            raw_images_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(gt_dir / 'case2.npy', np.array([[[2, 2], [1, 0]]], dtype=np.float32))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 3], [0, 4]]], dtype=np.float32))
            np.save(pred_dir / 'case2.npy', np.array([[[2, 1], [2, 0]]], dtype=np.float32))
            np.save(raw_images_dir / 'case1_0001.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))
            np.save(raw_images_dir / 'case2_0001.npy', np.array([[[2, 1], [3, 0]]], dtype=np.uint8))
            output_file = pred_dir / 'summary.json'

            summary = compute_regression_metrics_on_folder(
                str(gt_dir),
                str(pred_dir),
                str(output_file),
                FakeRegressionReaderWriter(),
                '.npy',
                num_processes=1,
                raw_images_folder=str(raw_images_dir),
            )

            self.assertEqual(summary['task_type'], 'regression')
            self.assertEqual(summary['case_counts']['expected_cases'], 2)
            self.assertEqual(summary['case_counts']['predicted_cases'], 2)
            self.assertEqual(summary['case_counts']['included_cases'], 2)
            self.assertEqual(summary['case_counts']['missing_cases'], 0)
            self.assertEqual(summary['case_counts']['missing_case_ids'], [])
            self.assertEqual(summary['selection_metric']['name'], 'MAE')
            self.assertEqual(summary['selection_metric']['mode'], 'min')
            for key in ('MAE', 'MSE', 'RMSE', 'Global_RMSE', 'Acc_1', 'Acc_3', 'Acc_5', 'ValidVoxelCount', 'Pearson_r', 'Gradient_MAE'):
                self.assertIn(key, summary['foreground_mean'])
            self.assertIn('tissue_mean', summary)
            self.assertIn('WM', summary['tissue_mean'])
            self.assertIn('GM', summary['tissue_mean'])
            self.assertIn('CSF', summary['tissue_mean'])
            self.assertTrue(output_file.exists())

    def test_compute_regression_metrics_on_folder_missing_tissue_file_skips_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            raw_images_dir = base / 'imagesTr'
            gt_dir.mkdir()
            pred_dir.mkdir()
            raw_images_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(gt_dir / 'case2.npy', np.array([[[2, 2], [1, 0]]], dtype=np.float32))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(pred_dir / 'case2.npy', np.array([[[2, 2], [1, 0]]], dtype=np.float32))
            np.save(raw_images_dir / 'case2_0001.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))

            summary = compute_regression_metrics_on_folder(
                str(gt_dir),
                str(pred_dir),
                None,
                FakeRegressionReaderWriter(),
                '.npy',
                num_processes=1,
                raw_images_folder=str(raw_images_dir),
            )
            self.assertEqual(summary['case_counts']['included_cases'], 1)
            self.assertEqual(summary['case_counts']['included_case_ids'], ['case2'])
            self.assertEqual(summary['case_counts']['skipped_missing_tissue_cases'], ['case1'])

    def test_compute_regression_metrics_on_folder_supports_custom_tissue_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            raw_images_dir = base / 'imagesTr'
            gt_dir.mkdir()
            pred_dir.mkdir()
            raw_images_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(raw_images_dir / 'case1_0002.npy', np.array([[[1, 2], [0, 3]]], dtype=np.uint8))

            summary = compute_regression_metrics_on_folder(
                str(gt_dir),
                str(pred_dir),
                None,
                FakeRegressionReaderWriter(),
                '.npy',
                num_processes=1,
                raw_images_folder=str(raw_images_dir),
                tissue_channel_suffix='_0002',
            )

            self.assertIn('tissue_mean', summary)

    def test_compute_regression_metrics_on_folder_rejects_tissue_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            raw_images_dir = base / 'imagesTr'
            gt_dir.mkdir()
            pred_dir.mkdir()
            raw_images_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(raw_images_dir / 'case1_0001.npy', np.zeros((2, 2, 2), dtype=np.uint8))

            with self.assertRaises(ValueError):
                compute_regression_metrics_on_folder(
                    str(gt_dir),
                    str(pred_dir),
                    None,
                    FakeRegressionReaderWriter(),
                    '.npy',
                    num_processes=1,
                    raw_images_folder=str(raw_images_dir),
                )

    def test_compute_regression_metrics_on_folder_reports_nan_for_missing_tissue_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            gt_dir = base / 'gt'
            pred_dir = base / 'pred'
            raw_images_dir = base / 'imagesTr'
            gt_dir.mkdir()
            pred_dir.mkdir()
            raw_images_dir.mkdir()
            np.save(gt_dir / 'case1.npy', np.array([[[1, 2], [0, 3]]], dtype=np.float32))
            np.save(pred_dir / 'case1.npy', np.array([[[1, 2], [0, 4]]], dtype=np.float32))
            np.save(raw_images_dir / 'case1_0001.npy', np.array([[[1, 2], [0, 2]]], dtype=np.uint8))

            summary = compute_regression_metrics_on_folder(
                str(gt_dir),
                str(pred_dir),
                None,
                FakeRegressionReaderWriter(),
                '.npy',
                num_processes=1,
                raw_images_folder=str(raw_images_dir),
            )

            self.assertTrue(np.isnan(summary['tissue_mean']['CSF']['MAE']))


if __name__ == '__main__':
    unittest.main()
