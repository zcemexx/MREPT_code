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
import os.path
from functools import lru_cache
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re

from nnunetv2.paths import nnUNet_raw


def _split_identifier_and_channel(filename: str, file_ending: str):
    if not filename.endswith(file_ending):
        return None
    stem = filename[:-len(file_ending)]
    match = re.fullmatch(r"(.+)_([0-9]{4})", stem)
    if match is None:
        return None
    return match.group(1), match.group(2)


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    identifiers = []
    for i in files:
        parsed = _split_identifier_and_channel(i, file_ending)
        if parsed is None:
            continue
        identifiers.append(parsed[0])
    files = np.unique(identifiers)
    return files


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[
    List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    files = [i for i in files if _split_identifier_and_channel(i, file_ending) is not None]
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists
    # return [files] #arthur, fix for nnUNet Imene8


def get_filenames_of_train_images_and_targets(raw_dataset_folder: str, dataset_json: dict = None):
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, 'dataset.json'))

    if 'dataset' in dataset_json.keys():
        dataset = dataset_json['dataset']
        for k in dataset.keys():
            dataset[k]['label'] = os.path.abspath(join(raw_dataset_folder, dataset[k]['label'])) if not os.path.isabs(dataset[k]['label']) else dataset[k]['label']
            dataset[k]['images'] = [os.path.abspath(join(raw_dataset_folder, i)) if not os.path.isabs(i) else i for i in dataset[k]['images']]
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'])
        images = create_lists_from_splitted_dataset_folder(join(raw_dataset_folder, 'imagesTr'), dataset_json['file_ending'], identifiers)
        segs = [join(raw_dataset_folder, 'labelsTr', i + dataset_json['file_ending']) for i in identifiers]
        dataset = {i: {'images': im, 'label': se} for i, im, se in zip(identifiers, images, segs)}
    return dataset


if __name__ == '__main__':
    print(get_filenames_of_train_images_and_targets(join(nnUNet_raw, 'Dataset002_Heart')))
