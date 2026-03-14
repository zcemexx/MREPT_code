#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from nibabel import io_orientation

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import nibabel

FORM_METADATA_ATOL = 1e-5
FORM_METADATA_RTOL = 1e-6


def _optional_affine_to_float64(affine: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if affine is None:
        return None
    return np.asarray(affine, dtype=np.float64)


def _extract_nibabel_spatial_metadata(nib_image: nibabel.spatialimages.SpatialImage) -> Dict[str, Any]:
    qaff, qcode = nib_image.get_qform(coded=True)
    saff, scode = nib_image.get_sform(coded=True)
    return {
        'original_affine': np.asarray(nib_image.affine, dtype=np.float64),
        'qform_affine': _optional_affine_to_float64(qaff),
        'qform_code': int(qcode),
        'sform_affine': _optional_affine_to_float64(saff),
        'sform_code': int(scode),
        'reference_qfac': float(nib_image.header['pixdim'][0]),
    }


def _same_optional_affine(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return np.allclose(left, right, atol=FORM_METADATA_ATOL, rtol=FORM_METADATA_RTOL)


def _warn_if_metadata_field_mismatch(field_name: str,
                                     values: List[Any],
                                     image_fnames: Union[List[str], Tuple[str, ...]]) -> None:
    print(f'WARNING! Not all input images have matching {field_name}!')
    print(f'{field_name}:')
    print(values)
    print('Image files:')
    print(image_fnames)
    print('The first image metadata will be used for export.')


def _warn_if_nifti_header_mismatch(metadata_list: List[Dict[str, Any]],
                                   image_fnames: Union[List[str], Tuple[str, ...]]) -> None:
    if len(metadata_list) <= 1:
        return

    ref = metadata_list[0]

    qcodes = [int(m['qform_code']) for m in metadata_list]
    if any(v != qcodes[0] for v in qcodes[1:]):
        _warn_if_metadata_field_mismatch('qform_code', qcodes, image_fnames)

    scodes = [int(m['sform_code']) for m in metadata_list]
    if any(v != scodes[0] for v in scodes[1:]):
        _warn_if_metadata_field_mismatch('sform_code', scodes, image_fnames)

    if any(not _same_optional_affine(ref['qform_affine'], m['qform_affine']) for m in metadata_list[1:]):
        _warn_if_metadata_field_mismatch(
            'qform_affine',
            [m['qform_affine'] for m in metadata_list],
            image_fnames,
        )

    if any(not _same_optional_affine(ref['sform_affine'], m['sform_affine']) for m in metadata_list[1:]):
        _warn_if_metadata_field_mismatch(
            'sform_affine',
            [m['sform_affine'] for m in metadata_list],
            image_fnames,
        )

    qfacs = [float(m['reference_qfac']) for m in metadata_list]
    if any(not np.allclose(qfacs[0], v, atol=FORM_METADATA_ATOL, rtol=FORM_METADATA_RTOL) for v in qfacs[1:]):
        _warn_if_metadata_field_mismatch('reference_qfac', qfacs, image_fnames)


def _maybe_apply_reference_form(out_img: nibabel.Nifti1Image,
                                form_name: str,
                                nibabel_stuff: Dict[str, Any]) -> None:
    code_key = f'{form_name}_code'
    affine_key = f'{form_name}_affine'

    code_val = nibabel_stuff.get(code_key, None)
    if code_val is None:
        return

    code_val = int(code_val)
    affine_val = nibabel_stuff.get(affine_key, None)
    if affine_val is not None:
        affine_val = np.asarray(affine_val, dtype=np.float64)
    elif code_val == 0:
        affine_val = np.asarray(out_img.affine, dtype=np.float64)
    else:
        return

    setter = out_img.set_qform if form_name == 'qform' else out_img.set_sform
    setter(affine_val, code=code_val)


def _apply_reference_nifti_forms(out_img: nibabel.Nifti1Image, nibabel_stuff: Dict[str, Any]) -> nibabel.Nifti1Image:
    if not isinstance(nibabel_stuff, dict):
        return out_img

    _maybe_apply_reference_form(out_img, 'qform', nibabel_stuff)
    _maybe_apply_reference_form(out_img, 'sform', nibabel_stuff)

    reference_qfac = nibabel_stuff.get('reference_qfac', None)
    if reference_qfac is not None:
        out_img.header['pixdim'][0] = np.float32(float(reference_qfac))
    return out_img


class NibabelIO(BaseReaderWriter):
    """
    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []
        metadata_list = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            metadata = _extract_nibabel_spatial_metadata(nib_image)
            original_affine = metadata['original_affine']

            original_affines.append(original_affine)
            metadata_list.append(metadata)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in nib_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(nib_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same_array(original_affines):
            print('WARNING! Not all input images have the same original_affines!')
            print('Affines:')
            print(original_affines)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not '
                  'having the same affine')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        _warn_if_nifti_header_mismatch(metadata_list, image_fnames)

        stacked_images = np.vstack(images)
        props = {
            'nibabel_stuff': dict(metadata_list[0]),
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), props

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)
        seg = np.ascontiguousarray(seg)
        nibabel_stuff = properties.get('nibabel_stuff', {})
        seg_nib = nibabel.Nifti1Image(seg, affine=nibabel_stuff['original_affine'])
        seg_nib = _apply_reference_nifti_forms(seg_nib, nibabel_stuff)
        nibabel.save(seg_nib, output_fname)


class NibabelIOWithReorient(BaseReaderWriter):
    """
    Reorients images to RAS

    Nibabel loads the images in a different order than sitk. We convert the axes to the sitk order to be
    consistent. This is of course considered properly in segmentation export as well.

    IMPORTANT: Run nnUNet_plot_dataset_pngs to verify that this did not destroy the alignment of data and seg!
    """
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        original_affines = []
        reoriented_affines = []
        metadata_list = []

        spacings_for_nnunet = []
        for f in image_fnames:
            nib_image = nibabel.load(f)
            assert nib_image.ndim == 3, 'only 3d images are supported by NibabelIO'
            metadata = _extract_nibabel_spatial_metadata(nib_image)
            original_affine = metadata['original_affine']
            reoriented_image = nib_image.as_reoriented(io_orientation(original_affine))
            reoriented_affine = reoriented_image.affine

            original_affines.append(original_affine)
            reoriented_affines.append(reoriented_affine)
            metadata_list.append(metadata)

            # spacing is taken in reverse order to be consistent with SimpleITK axis ordering (confusing, I know...)
            spacings_for_nnunet.append(
                    [float(i) for i in reoriented_image.header.get_zooms()[::-1]]
            )

            # transpose image to be consistent with the way SimpleITk reads images. Yeah. Annoying.
            images.append(reoriented_image.get_fdata().transpose((2, 1, 0))[None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same_array(reoriented_affines):
            print('WARNING! Not all input images have the same reoriented_affines!')
            print('Affines:')
            print(reoriented_affines)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNet_plot_dataset_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! This might be caused by them not '
                  'having the same affine')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        _warn_if_nifti_header_mismatch(metadata_list, image_fnames)

        stacked_images = np.vstack(images)
        nibabel_stuff = dict(metadata_list[0])
        nibabel_stuff['reoriented_affine'] = reoriented_affines[0]
        props = {
            'nibabel_stuff': nibabel_stuff,
            'spacing': spacings_for_nnunet[0]
        }
        return stacked_images.astype(np.float32), props

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # revert transpose
        seg = seg.transpose((2, 1, 0)).astype(np.uint8)
        seg = np.ascontiguousarray(seg)
        nibabel_stuff = properties.get('nibabel_stuff', {})

        seg_nib = nibabel.Nifti1Image(seg, affine=nibabel_stuff['reoriented_affine'])
        seg_nib_reoriented = seg_nib.as_reoriented(io_orientation(nibabel_stuff['original_affine']))
        assert np.allclose(nibabel_stuff['original_affine'], seg_nib_reoriented.affine), \
            'restored affine does not match original affine'
        seg_nib_reoriented = _apply_reference_nifti_forms(seg_nib_reoriented, nibabel_stuff)
        nibabel.save(seg_nib_reoriented, output_fname)


if __name__ == '__main__':
    img_file = 'patient028_frame01_0000.nii.gz'
    seg_file = 'patient028_frame01.nii.gz'

    nibio = NibabelIO()
    images, dct = nibio.read_images([img_file])
    seg, dctseg = nibio.read_seg(seg_file)

    nibio_r = NibabelIOWithReorient()
    images_r, dct_r = nibio_r.read_images([img_file])
    seg_r, dctseg_r = nibio_r.read_seg(seg_file)

    nibio.write_seg(seg[0], '/home/isensee/seg_nibio.nii.gz', dctseg)
    nibio_r.write_seg(seg_r[0], '/home/isensee/seg_nibio_r.nii.gz', dctseg_r)

    s_orig = nibabel.load(seg_file).get_fdata()
    s_nibio = nibabel.load('/home/isensee/seg_nibio.nii.gz').get_fdata()
    s_nibio_r = nibabel.load('/home/isensee/seg_nibio_r.nii.gz').get_fdata()
