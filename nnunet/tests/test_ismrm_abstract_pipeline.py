from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from nnunetv2.utilities.ismrm_abstract_pipeline import (
    crop_2d,
    is_npz_compressed,
    masked_slicewise_ssim,
    save_residual_archive,
    safe_crop_bounds_2d,
)


class ISMRMAbstractPipelineTests(unittest.TestCase):
    def test_masked_slice_ssim_differs_from_zero_padded_3d(self):
        gt = np.zeros((7, 7, 7), dtype=np.float32)
        pred = np.zeros((7, 7, 7), dtype=np.float32)
        gt[0:4, 0:4, :] = 1.0
        pred[0:4, 0:4, :] = 0.8
        global_mask = np.zeros_like(gt, dtype=bool)
        global_mask[0:4, 0:4, :] = True

        masked_score = masked_slicewise_ssim(pred, gt, global_mask, global_mask)

        gt_zero = gt.copy()
        pred_zero = pred.copy()
        gt_zero[~global_mask] = 0.0
        pred_zero[~global_mask] = 0.0
        zero_padded_score = structural_similarity(pred_zero, gt_zero, data_range=1.0, channel_axis=None)

        self.assertFalse(np.isclose(masked_score, zero_padded_score, atol=1e-5))

    def test_masked_slice_ssim_reduces_window_for_small_bbox(self):
        gt = np.zeros((5, 5, 2), dtype=np.float32)
        gt[0:3, 0:3, 0] = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]], dtype=np.float32)
        gt[0:3, 0:3, 1] = np.array([[0.2, 0.3, 0.4], [0.3, 0.5, 0.7], [0.4, 0.7, 1.0]], dtype=np.float32)
        pred = gt.copy()
        pred[0:3, 0:3, :] *= 0.95
        mask = np.zeros_like(gt, dtype=bool)
        mask[0:3, 0:3, :] = True

        score = masked_slicewise_ssim(pred, gt, mask, mask)
        self.assertTrue(np.isfinite(score))

    def test_masked_slice_ssim_uses_global_bbox_range_not_eval_region_range(self):
        gt = np.zeros((5, 5, 1), dtype=np.float32)
        pred = np.zeros((5, 5, 1), dtype=np.float32)
        global_mask = np.zeros_like(gt, dtype=bool)
        eval_mask = np.zeros_like(gt, dtype=bool)

        gt[0:3, 0:3, 0] = np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]],
            dtype=np.float32,
        )
        pred[0:3, 0:3, 0] = gt[0:3, 0:3, 0] * 0.9
        global_mask[0:3, 0:3, 0] = True
        eval_mask[1:3, 1:3, 0] = True

        score = masked_slicewise_ssim(pred, gt, eval_mask, global_mask)
        self.assertTrue(np.isfinite(score))

    def test_masked_slice_ssim_ignores_invalid_pixels_outside_eval_mean(self):
        gt = np.zeros((7, 7, 1), dtype=np.float32)
        pred = np.zeros((7, 7, 1), dtype=np.float32)
        global_mask = np.zeros_like(gt, dtype=bool)
        eval_mask = np.zeros_like(gt, dtype=bool)

        gt[1:6, 1:6, 0] = np.linspace(0.1, 1.0, 25, dtype=np.float32).reshape(5, 5)
        pred[1:6, 1:6, 0] = gt[1:6, 1:6, 0] * 0.95
        pred[1, 1, 0] = np.nan
        global_mask[1:6, 1:6, 0] = True
        eval_mask[2:6, 2:6, 0] = True

        score = masked_slicewise_ssim(pred, gt, eval_mask, global_mask)
        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, 0.0)

    def test_safe_crop_bounds_clips_to_image_edges(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[1:3, 0:2] = True
        bounds = safe_crop_bounds_2d(mask, margin=4)
        self.assertEqual(bounds, (0, 7, 0, 6))

        array = np.arange(120, dtype=np.float32).reshape(10, 12)
        cropped = crop_2d(array, bounds)
        self.assertEqual(cropped.shape, (7, 6))

    def test_safe_crop_bounds_raises_for_empty_mask(self):
        with self.assertRaises(ValueError):
            safe_crop_bounds_2d(np.zeros((4, 4), dtype=bool), margin=4)

    def test_save_residual_archive_enforces_float32_and_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "residuals_dist.npz"
            payload = {"SNR_010_CNN": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
            save_residual_archive(output, payload)

            loaded = np.load(output)
            self.assertEqual(loaded["SNR_010_CNN"].dtype, np.float32)
            self.assertTrue(is_npz_compressed(output))


if __name__ == "__main__":
    unittest.main()
