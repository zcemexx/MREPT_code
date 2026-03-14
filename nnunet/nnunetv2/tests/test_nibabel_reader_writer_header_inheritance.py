import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from nnunetv2.inference import export_prediction as export_prediction_module


def _make_affine(scale_xyz=(1.0, 1.0, 1.0), translation_xyz=(0.0, 0.0, 0.0)):
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = float(scale_xyz[0])
    affine[1, 1] = float(scale_xyz[1])
    affine[2, 2] = float(scale_xyz[2])
    affine[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return affine


def _write_nifti(path: Path,
                 affine: np.ndarray,
                 qform_affine: np.ndarray,
                 qform_code: int,
                 sform_affine: np.ndarray,
                 sform_code: int,
                 qfac: float = 1.0,
                 shape=(3, 4, 5)) -> Path:
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    img = nib.Nifti1Image(data, affine=affine)
    img.set_qform(qform_affine, code=int(qform_code))
    img.set_sform(sform_affine, code=int(sform_code))
    img.header["pixdim"][0] = np.float32(qfac)
    nib.save(img, str(path))
    return path


class _DummyPlansManager:
    @staticmethod
    def image_reader_writer_class():
        return NibabelIO()


class TestNibabelHeaderInheritance(unittest.TestCase):
    def test_read_images_returns_reference_form_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "case_0000.nii.gz"
            qaff = _make_affine((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
            saff = _make_affine((2.0, 3.0, 4.0), (1.0, 2.0, 3.0))
            _write_nifti(path, qaff, qaff, 1, saff, 2, qfac=-1.0)
            loaded = nib.load(str(path))
            expected_original_affine = loaded.affine
            expected_qform, _ = loaded.get_qform(coded=True)
            expected_sform, _ = loaded.get_sform(coded=True)

            _, props = NibabelIO().read_images((str(path),))
            meta = props["nibabel_stuff"]

            self.assertEqual(meta["qform_code"], 1)
            self.assertEqual(meta["sform_code"], 2)
            self.assertTrue(np.allclose(meta["original_affine"], expected_original_affine))
            self.assertTrue(np.allclose(meta["qform_affine"], expected_qform))
            self.assertTrue(np.allclose(meta["sform_affine"], expected_sform))
            self.assertEqual(float(meta["reference_qfac"]), -1.0)

    def test_read_images_warns_on_multichannel_header_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "case_0000.nii.gz"
            path_b = Path(tmpdir) / "case_0001.nii.gz"
            affine = _make_affine()
            _write_nifti(path_a, affine, affine, 1, affine, 1, qfac=1.0)
            _write_nifti(path_b, affine, affine, 0, affine, 2, qfac=-1.0)

            with patch("builtins.print") as mocked_print:
                _, props = NibabelIO().read_images((str(path_a), str(path_b)))

            printed = "\n".join(" ".join(str(arg) for arg in call.args) for call in mocked_print.call_args_list)
            self.assertIn("qform_code", printed)
            self.assertIn("sform_code", printed)
            self.assertIn("reference_qfac", printed)
            self.assertEqual(props["nibabel_stuff"]["qform_code"], 1)
            self.assertEqual(props["nibabel_stuff"]["sform_code"], 1)

    def test_write_seg_inherits_reference_forms_and_qfac(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "case_0000.nii.gz"
            out = Path(tmpdir) / "pred.nii.gz"
            qaff = _make_affine((1.0, 1.0, -2.0), (10.0, 20.0, 30.0))
            saff = _make_affine((2.0, 3.0, 4.0), (1.0, 2.0, 3.0))
            _write_nifti(ref, qaff, qaff, 0, saff, 2, qfac=-1.0)

            images, props = NibabelIO().read_images((str(ref),))
            seg = np.asfortranarray(np.zeros_like(images[0], dtype=np.uint8))
            NibabelIO().write_seg(seg, str(out), props)

            saved = nib.load(str(out))
            qform, qcode = saved.get_qform(coded=True)
            sform, scode = saved.get_sform(coded=True)

            self.assertEqual(int(qcode), 0)
            self.assertEqual(int(scode), 2)
            self.assertTrue(np.allclose(sform, saff))
            self.assertEqual(float(saved.header["pixdim"][0]), -1.0)

    def test_write_seg_keeps_legacy_properties_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "legacy.nii.gz"
            affine = _make_affine((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
            seg = np.zeros((5, 4, 3), dtype=np.uint8)
            props = {
                "nibabel_stuff": {
                    "original_affine": affine,
                },
                "spacing": (3.0, 2.0, 1.0),
            }

            NibabelIO().write_seg(seg, str(out), props)
            saved = nib.load(str(out))

            self.assertTrue(np.allclose(saved.affine, affine))

    def test_write_seg_tolerates_partial_form_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "partial.nii.gz"
            affine = _make_affine((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
            seg = np.zeros((5, 4, 3), dtype=np.uint8)
            props = {
                "nibabel_stuff": {
                    "original_affine": affine,
                    "qform_code": 1,
                },
                "spacing": (3.0, 2.0, 1.0),
            }

            NibabelIO().write_seg(seg, str(out), props)
            saved = nib.load(str(out))

            self.assertTrue(np.allclose(saved.affine, affine))

    def test_reorient_writer_inherits_original_reference_forms_and_qfac(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "case_0000.nii.gz"
            out = Path(tmpdir) / "pred_reorient.nii.gz"
            qaff = np.array(
                [
                    [-2.0, 0.0, 0.0, 10.0],
                    [0.0, 3.0, 0.0, 20.0],
                    [0.0, 0.0, 4.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            saff = np.array(
                [
                    [-2.0, 0.0, 0.0, 11.0],
                    [0.0, 3.0, 0.0, 21.0],
                    [0.0, 0.0, 4.0, 31.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            _write_nifti(ref, qaff, qaff, 1, saff, 2, qfac=-1.0)
            expected_original_affine = nib.load(str(ref)).affine

            images, props = NibabelIOWithReorient().read_images((str(ref),))
            seg = np.zeros_like(images[0], dtype=np.uint8)[:, :, ::-1]
            NibabelIOWithReorient().write_seg(seg, str(out), props)

            saved = nib.load(str(out))
            qform, qcode = saved.get_qform(coded=True)
            sform, scode = saved.get_sform(coded=True)

            self.assertTrue(np.allclose(saved.affine, expected_original_affine))
            self.assertEqual(int(qcode), 1)
            self.assertEqual(int(scode), 2)
            self.assertTrue(np.allclose(qform, qaff))
            self.assertTrue(np.allclose(sform, saff))
            self.assertEqual(float(saved.header["pixdim"][0]), -1.0)

    def test_export_regression_prediction_uses_writer_header_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "case_0000.nii.gz"
            out_truncated = Path(tmpdir) / "regression_pred"
            qaff = _make_affine((1.0, 1.0, 2.0), (3.0, 4.0, 5.0))
            saff = _make_affine((2.0, 2.0, 2.0), (6.0, 7.0, 8.0))
            _write_nifti(ref, qaff, qaff, 1, saff, 1, qfac=-1.0)
            loaded = nib.load(str(ref))
            expected_qform, _ = loaded.get_qform(coded=True)
            expected_sform, _ = loaded.get_sform(coded=True)

            _, props = NibabelIO().read_images((str(ref),))
            radius_map = np.ones((5, 4, 3), dtype=np.uint8)
            valid_mask = np.ones((5, 4, 3), dtype=bool)

            with patch.object(
                export_prediction_module,
                "convert_predicted_regression_to_original_shape",
                return_value=(radius_map, valid_mask),
            ):
                export_prediction_module.export_regression_prediction(
                    predicted_logits=np.zeros((1, 5, 4, 3), dtype=np.float32),
                    valid_mask_preprocessed=np.ones((1, 5, 4, 3), dtype=np.uint8),
                    properties_dict=props,
                    configuration_manager=object(),
                    plans_manager=_DummyPlansManager(),
                    dataset_json={"file_ending": ".nii.gz"},
                    output_file_truncated=str(out_truncated),
                    save_probabilities=False,
                )

            saved = nib.load(str(out_truncated) + ".nii.gz")
            qform, qcode = saved.get_qform(coded=True)
            sform, scode = saved.get_sform(coded=True)

            self.assertEqual(int(qcode), 1)
            self.assertEqual(int(scode), 1)
            self.assertTrue(np.allclose(qform, expected_qform))
            self.assertTrue(np.allclose(sform, expected_sform))
            self.assertEqual(float(saved.header["pixdim"][0]), -1.0)


if __name__ == "__main__":
    unittest.main()
