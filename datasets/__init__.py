# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import numpy as np
from torch.utils.data import ConcatDataset

from datasets.base import BaseDataset


def build_train(config, device):
    data_list = []
    total_images = 0
    for dataset in config.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=False)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


def build_val(config, device):
    data_list = []
    total_images = 0
    for dataset in config.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=True)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images
