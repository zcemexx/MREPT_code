from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nnunetv2.evaluation.regression_tissue_metrics import (
    align_prediction_to_gt,
    build_tissue_masks,
    compute_boundary_mask,
    compute_masked_gradient_mae,
    compute_masked_pearson_r,
    compute_regression_metrics,
    compute_regression_metrics_on_folder_with_tissues,
    compute_residual_histogram,
    find_optimal_slice,
)
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class FakeReaderWriter(BaseReaderWriter):
    def read_images(self, image_fnames):
        arrays = []
        properties = None
        for fname in image_fnames:
            with open(fname, "rb") as f:
                payload = pickle.load(f)
            arrays.append(np.asarray(payload["data"], dtype=np.float32))
            if properties is None:
                properties = payload["properties"]
        return np.vstack(arrays), properties

    def read_seg(self, seg_fname):
        return self.read_images((seg_fname,))

    def write_seg(self, seg, output_fname, properties):
        with open(output_fname, "wb") as f:
            pickle.dump({"data": np.asarray(seg)[None], "properties": properties}, f)


def _write_fake_case(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 1.0), affine=None):
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    payload = {
        "data": np.asarray(array, dtype=np.float32)[None],
        "properties": {
            "spacing": tuple(float(x) for x in spacing),
            "nibabel_stuff": {"original_affine": np.asarray(affine, dtype=np.float32)},
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


class RegressionTissueMetricsTests(unittest.TestCase):
    def test_align_prediction_to_gt_identity(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        out = align_prediction_to_gt(arr, (2, 3, 4))
        self.assertEqual(out.shape, (2, 3, 4))

    def test_align_prediction_to_gt_reversed_shape_transpose(self):
        arr = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        out = align_prediction_to_gt(arr, (2, 3, 4))
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertTrue(np.array_equal(out, np.transpose(arr)))

    def test_align_prediction_to_gt_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            align_prediction_to_gt(np.zeros((2, 2, 2), dtype=np.float32), (2, 3, 4))

    def test_build_tissue_masks(self):
        tissue = np.array(
            [
                [[0, 1], [2, 3]],
                [[1, 1], [0, 2]],
            ],
            dtype=np.int16,
        )
        masks = build_tissue_masks(tissue)
        self.assertEqual(int(masks["Global"].sum()), 6)
        self.assertEqual(int(masks["WM"].sum()), 3)
        self.assertEqual(int(masks["GM"].sum()), 2)
        self.assertEqual(int(masks["CSF"].sum()), 1)

    def test_compute_masked_gradient_mae(self):
        gt = np.zeros((2, 2, 2), dtype=np.float32)
        pred = np.zeros((2, 2, 2), dtype=np.float32)
        pred[1, :, :] = 2.0
        valid = np.ones((2, 2, 2), dtype=bool)
        value = compute_masked_gradient_mae(pred, gt, valid, spacing=(2.0, 1.0, 1.0))
        self.assertAlmostEqual(value, 1.0 / 3.0, places=6)

    def test_compute_masked_pearson_r_constant_returns_nan(self):
        pred = np.ones((2, 2, 2), dtype=np.float32)
        gt = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        valid = np.ones((2, 2, 2), dtype=bool)
        self.assertTrue(np.isnan(compute_masked_pearson_r(pred, gt, valid)))

    def test_compute_residual_histogram_counts_match_valid_voxels(self):
        eres = np.array([-10, -5, 0, 5, 10], dtype=np.float32)
        valid = np.array([1, 1, 1, 1, 1], dtype=bool)
        counts, edges, stats = compute_residual_histogram(eres, valid, limit=5)
        self.assertEqual(int(counts.sum()), 5)
        self.assertEqual(len(edges), 12)
        self.assertAlmostEqual(stats["Residual_Mean"], 0.0, places=6)

    def test_find_optimal_slice(self):
        mask = np.zeros((4, 5, 6), dtype=bool)
        mask[:, :, 4] = True
        idx = find_optimal_slice(mask, axis=2)
        self.assertEqual(idx, 4)

    def test_compute_boundary_mask_subset(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        boundary = compute_boundary_mask(mask, width=1)
        self.assertTrue(np.all(boundary <= mask))
        self.assertGreater(int(boundary.sum()), 0)

    def test_compute_regression_metrics_returns_nan_for_empty_tissue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref = tmp / "labelsTr" / "CaseA.nii.gz"
            pred = tmp / "pred" / "CaseA.nii.gz"
            tissue = tmp / "imagesTr" / "CaseA_0001.nii.gz"
            ref.parent.mkdir()
            pred.parent.mkdir()
            tissue.parent.mkdir()

            gt = np.ones((3, 3, 3), dtype=np.float32)
            pred_arr = np.ones((3, 3, 3), dtype=np.float32)
            tissue_arr = np.zeros((3, 3, 3), dtype=np.float32)
            tissue_arr[0:2, 0:2, 0:2] = 1

            _write_fake_case(ref, gt)
            _write_fake_case(pred, pred_arr)
            _write_fake_case(tissue, tissue_arr)

            result = compute_regression_metrics(str(ref), str(pred), FakeReaderWriter(), tissue_mask_file=str(tissue))
            self.assertTrue(np.isnan(result["tissue_metrics"]["GM"]["MAE"]))
            self.assertEqual(result["tissue_metrics"]["WM"]["ValidVoxelCount"], 8)

    def test_folder_evaluation_writes_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            folder_ref = tmp / "labelsTr"
            folder_pred = tmp / "pred"
            folder_images = tmp / "imagesTr"
            folder_ref.mkdir()
            folder_pred.mkdir()
            folder_images.mkdir()

            gt = np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                ],
                dtype=np.float32,
            )
            pred = gt.copy()
            pred[0, 0, 0] = 2
            tissue = np.array(
                [
                    [[1, 1], [2, 2]],
                    [[3, 3], [0, 0]],
                ],
                dtype=np.float32,
            )

            _write_fake_case(folder_ref / "CaseA.nii.gz", gt)
            _write_fake_case(folder_pred / "CaseA.nii.gz", pred)
            _write_fake_case(folder_images / "CaseA_0001.nii.gz", tissue)

            out_json = tmp / "out" / "summary.json"
            summary = compute_regression_metrics_on_folder_with_tissues(
                folder_ref=str(folder_ref),
                folder_pred=str(folder_pred),
                folder_images=str(folder_images),
                output_file=str(out_json),
                image_reader_writer=FakeReaderWriter(),
                file_ending=".nii.gz",
                num_processes=1,
                include_gradient_mae=True,
                include_pearson_r=True,
                plots_dir=None,
                boundary_width=1,
                residual_limit=5,
            )

            self.assertEqual(summary["task_type"], "regression_tissue_eval")
            self.assertTrue(out_json.exists())
            self.assertTrue((out_json.parent / "metrics_long.csv").exists())
            self.assertTrue((out_json.parent / "boundary_metrics.csv").exists())
            self.assertTrue((out_json.parent / "residual_distribution.csv").exists())
            self.assertTrue((out_json.parent / "cases" / "CaseA" / "case_report.json").exists())

            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["selection_metric"]["name"], "Global.MAE")
            self.assertEqual(len(saved["metric_per_case"]), 1)
            self.assertEqual(saved["case_counts"]["expected_cases"], 1)
            self.assertEqual(saved["case_counts"]["predicted_cases"], 1)
            self.assertEqual(saved["case_counts"]["evaluated_cases"], 1)
            self.assertEqual(saved["case_counts"]["ref_only_cases"], [])
            self.assertEqual(saved["case_counts"]["pred_only_cases"], [])
            self.assertEqual(saved["case_counts"]["missing_tissue_cases"], [])

    def test_folder_evaluation_only_reports_pred_ref_intersection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            folder_ref = tmp / "labelsTr"
            folder_pred = tmp / "pred"
            folder_images = tmp / "imagesTr"
            folder_ref.mkdir()
            folder_pred.mkdir()
            folder_images.mkdir()

            arr = np.ones((2, 2, 2), dtype=np.float32)
            _write_fake_case(folder_ref / "CaseA.nii.gz", arr)
            _write_fake_case(folder_ref / "CaseB.nii.gz", arr)
            _write_fake_case(folder_pred / "CaseA.nii.gz", arr)
            _write_fake_case(folder_pred / "CaseC.nii.gz", arr)
            # Intentionally omit CaseA tissue mask to force missing_tissue_mask

            out_json = tmp / "out" / "summary.json"
            summary = compute_regression_metrics_on_folder_with_tissues(
                folder_ref=str(folder_ref),
                folder_pred=str(folder_pred),
                folder_images=str(folder_images),
                output_file=str(out_json),
                image_reader_writer=FakeReaderWriter(),
                file_ending=".nii.gz",
                num_processes=1,
                include_gradient_mae=True,
                include_pearson_r=True,
                plots_dir=None,
                boundary_width=1,
                residual_limit=5,
            )

            self.assertEqual(summary["case_counts"]["expected_cases"], 1)
            self.assertEqual(summary["case_counts"]["predicted_cases"], 2)
            self.assertEqual(summary["case_counts"]["evaluated_cases"], 1)
            self.assertEqual(summary["case_counts"]["ref_only_cases"], ["CaseB"])
            self.assertEqual(summary["case_counts"]["pred_only_cases"], ["CaseC"])
            self.assertEqual(summary["case_counts"]["missing_tissue_cases"], ["CaseA"])
            self.assertEqual(summary["case_counts"]["status_counts"].get("missing_tissue_mask"), 1)
            self.assertEqual([c["case_id"] for c in summary["metric_per_case"]], ["CaseA"])
            self.assertEqual(summary["metric_per_case"][0]["status"], "missing_tissue_mask")

            self.assertTrue((out_json.parent / "cases" / "CaseA" / "case_report.json").exists())
            self.assertFalse((out_json.parent / "cases" / "CaseB" / "case_report.json").exists())
            self.assertFalse((out_json.parent / "cases" / "CaseC" / "case_report.json").exists())


if __name__ == "__main__":
    unittest.main()
