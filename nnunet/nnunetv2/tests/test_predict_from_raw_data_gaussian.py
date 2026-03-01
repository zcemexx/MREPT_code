import argparse
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nnunetv2.inference import predict_from_raw_data as predict_module
from nnunetv2.inference.predict_from_raw_data import (
    DEFAULT_RECONSTRUCTION_MODE,
    RECONSTRUCTION_MODES,
    nnUNetPredictor,
)


def identity_tqdm(iterable, *args, **kwargs):
    return iterable


class ParserCaptured(RuntimeError):
    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__('captured parser')
        self.parser = parser


class FakePreprocessAdapterFromNpy:
    def __init__(self, *args, **kwargs):
        self._used = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._used:
            raise StopIteration
        self._used = True
        return {
            'data': torch.zeros((1, 4, 4, 4), dtype=torch.float32),
            'data_properties': {'spacing': (1.0, 1.0, 1.0)},
        }


class DummyNetwork(torch.nn.Module):
    def forward(self, x):
        return x


class TestGaussianReconstruction(unittest.TestCase):
    def _make_predictor(self, patch_size=(4, 4, 4), tile_step_size=0.5, use_gaussian=True):
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device('cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.configuration_manager = SimpleNamespace(patch_size=patch_size)
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=1)
        predictor.allowed_mirroring_axes = None
        predictor.network = DummyNetwork()
        return predictor

    def _constant_predict(self, value):
        def _predict(x):
            return torch.full(
                (x.shape[0], 1, *x.shape[2:]),
                value,
                dtype=torch.float32,
                device=x.device,
            )

        return _predict

    def _varying_predict(self):
        counter = {'value': 0}

        def _predict(x):
            counter['value'] += 1
            return torch.full(
                (x.shape[0], 1, *x.shape[2:]),
                float(counter['value']),
                dtype=torch.float32,
                device=x.device,
            )

        return _predict

    def test_dispatch_accepts_gaussian(self):
        predictor = self._make_predictor(use_gaussian=True)
        data = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        slicers = [(slice(None), slice(0, 4), slice(0, 4), slice(0, 4))]

        with patch.object(predictor, 'rec_gaussian', return_value=torch.zeros((1, 4, 4, 4))) as rec_gaussian:
            result = predictor._internal_predict_sliding_window_return_logits(
                data,
                slicers,
                do_on_device=False,
                reconstruction_mode='gaussian',
            )

        rec_gaussian.assert_called_once()
        self.assertEqual(tuple(result.shape), (1, 4, 4, 4))

    def test_predict_single_npy_array_forwards_reconstruction_mode(self):
        predictor = self._make_predictor()
        predictor.plans_manager = object()
        predictor.dataset_json = {}
        predictor.label_manager = object()

        with patch.object(predict_module, 'PreprocessAdapterFromNpy', FakePreprocessAdapterFromNpy), \
             patch.object(predictor, 'predict_logits_from_preprocessed_data',
                          return_value=torch.zeros((1, 4, 4, 4), dtype=torch.float32)) as predict_logits, \
             patch.object(predict_module, 'convert_predicted_logits_to_segmentation_with_correct_shape',
                          return_value='segmentation'):
            result = predictor.predict_single_npy_array(
                input_image=torch.zeros((1, 4, 4, 4), dtype=torch.float32).numpy(),
                image_properties={'spacing': (1.0, 1.0, 1.0)},
                reconstruction_mode='gaussian',
            )

        predict_logits.assert_called_once()
        self.assertEqual(predict_logits.call_args.kwargs['reconstruction_mode'], 'gaussian')
        self.assertEqual(result, 'segmentation')

    def test_predict_from_list_of_npy_arrays_forwards_reconstruction_mode(self):
        predictor = self._make_predictor()
        iterator = object()

        with patch.object(predictor, 'get_data_iterator_from_raw_npy_data', return_value=iterator), \
             patch.object(predictor, 'predict_from_data_iterator', return_value='done') as predict_from_iterator:
            result = predictor.predict_from_list_of_npy_arrays(
                image_or_list_of_images=[torch.zeros((1, 4, 4, 4), dtype=torch.float32).numpy()],
                segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
                properties_or_list_of_properties=[{'spacing': (1.0, 1.0, 1.0)}],
                truncated_ofname=None,
                reconstruction_mode='gaussian',
            )

        predict_from_iterator.assert_called_once_with(
            iterator,
            False,
            predict_module.default_num_processes,
            reconstruction_mode='gaussian',
        )
        self.assertEqual(result, 'done')

    def test_gaussian_mode_ignores_use_gaussian_false(self):
        predictor = self._make_predictor(use_gaussian=False)
        data = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        slicers = [(slice(None), slice(0, 4), slice(0, 4), slice(0, 4))]

        with patch.object(predictor, 'rec_gaussian', return_value=torch.zeros((1, 4, 4, 4))) as rec_gaussian:
            predictor._internal_predict_sliding_window_return_logits(
                data,
                slicers,
                do_on_device=False,
                reconstruction_mode='gaussian',
            )

        rec_gaussian.assert_called_once()

    def test_mean_mode_does_not_use_gaussian_when_flag_true(self):
        predictor = self._make_predictor(use_gaussian=True)
        data = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        slicers = [(slice(None), slice(0, 4), slice(0, 4), slice(0, 4))]

        with patch.object(predictor, 'rec_gaussian', side_effect=AssertionError('gaussian path should not run')), \
             patch.object(predictor, 'rec_mean', return_value=torch.ones((1, 4, 4, 4), dtype=torch.float32)) as rec_mean:
            result = predictor._internal_predict_sliding_window_return_logits(
                data,
                slicers,
                do_on_device=False,
                reconstruction_mode='mean',
            )

        rec_mean.assert_called_once()
        self.assertTrue(torch.allclose(result, torch.ones_like(result)))

    def test_gaussian_preserves_constant_output_under_overlap(self):
        predictor = self._make_predictor(tile_step_size=0.5)
        predictor._internal_maybe_mirror_and_predict = self._constant_predict(1.0)
        data = torch.zeros((1, 6, 6, 6), dtype=torch.float32)
        slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])

        with patch.object(predict_module, 'tqdm', identity_tqdm):
            result = predictor.rec_gaussian(slicers, data, results_device=torch.device('cpu'))

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.ones_like(result), atol=1e-5))

    def test_gaussian_differs_from_mean_for_nonidentical_overlaps(self):
        data = torch.zeros((1, 6, 6, 6), dtype=torch.float32)

        mean_predictor = self._make_predictor(tile_step_size=0.5)
        mean_predictor._internal_maybe_mirror_and_predict = self._varying_predict()
        slicers = mean_predictor._internal_get_sliding_window_slicers(data.shape[1:])

        with patch.object(predict_module, 'tqdm', identity_tqdm):
            mean_result = mean_predictor.rec_mean(slicers, data).float()

        gaussian_predictor = self._make_predictor(tile_step_size=0.5)
        gaussian_predictor._internal_maybe_mirror_and_predict = self._varying_predict()

        with patch.object(predict_module, 'tqdm', identity_tqdm):
            gaussian_result = gaussian_predictor.rec_gaussian(slicers, data, results_device=torch.device('cpu'))

        self.assertFalse(torch.isclose(mean_result[0, 2, 2, 2], gaussian_result[0, 2, 2, 2]))

    def test_gaussian_high_overlap_stays_finite(self):
        predictor = self._make_predictor(tile_step_size=0.25)
        predictor._internal_maybe_mirror_and_predict = self._constant_predict(1.0)
        data = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])

        with patch.object(predict_module, 'tqdm', identity_tqdm):
            result = predictor.rec_gaussian(slicers, data, results_device=torch.device('cpu'))

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())

    def test_gaussian_supports_patch_dims_one_shorter_than_image_dims(self):
        predictor = self._make_predictor(patch_size=(4, 4), tile_step_size=0.5)
        predictor._internal_maybe_mirror_and_predict = self._constant_predict(1.0)
        data = torch.zeros((1, 3, 6, 6), dtype=torch.float32)
        slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])

        with patch.object(predict_module, 'tqdm', identity_tqdm):
            result = predictor.rec_gaussian(slicers, data, results_device=torch.device('cpu'))

        self.assertEqual(tuple(result.shape), (1, 3, 6, 6))
        self.assertTrue(torch.allclose(result, torch.ones_like(result), atol=1e-5))

    def test_api_defaults_are_gaussian(self):
        method_names = (
            'predict_from_files',
            'predict_from_list_of_npy_arrays',
            'predict_from_data_iterator',
            'predict_single_npy_array',
            'predict_logits_from_preprocessed_data',
            '_internal_predict_sliding_window_return_logits',
            'predict_sliding_window_return_logits',
        )
        for method_name in method_names:
            with self.subTest(method=method_name):
                signature = inspect.signature(getattr(nnUNetPredictor, method_name))
                self.assertEqual(signature.parameters['reconstruction_mode'].default, DEFAULT_RECONSTRUCTION_MODE)

    def test_cli_parsers_accept_gaussian_and_default_to_it(self):
        def capture_parser(self, *args, **kwargs):
            raise ParserCaptured(self)

        entry_points = (
            predict_module.predict_entry_point_modelfolder,
            predict_module.predict_entry_point,
        )
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point.__name__):
                with patch.object(argparse.ArgumentParser, 'parse_args', capture_parser):
                    with self.assertRaises(ParserCaptured) as captured:
                        entry_point()
                parser = captured.exception.parser
                rec_action = next(action for action in parser._actions if action.dest == 'rec')
                self.assertEqual(rec_action.default, DEFAULT_RECONSTRUCTION_MODE)
                self.assertEqual(tuple(rec_action.choices), RECONSTRUCTION_MODES)


if __name__ == '__main__':
    unittest.main()
