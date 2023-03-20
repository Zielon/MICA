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


from abc import ABC
from glob import glob
from pathlib import Path

from datasets.creation.instances.instance import Instance


class D3DFACS(Instance, ABC):
    def __init__(self):
        super(D3DFACS, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/D3DFACS/'
        self.src = '/home/wzielonka/datasets/D3DFACS/'

    def get_images(self):
        images = {}
        for file in sorted(glob(self.get_src() + 'processed/images/*')):
            actor = Path(file).stem
            images[actor] = glob(f'{file}/*.jpg')

        return images

    def get_flame_params(self):
        params = {}
        for file in sorted(glob(self.get_src() + 'processed/FLAME/*.npz')):
            actor = Path(file).stem
            params[actor] = [file]

        return params

    def get_registrations(self):
        registrations = {}
        for file in sorted(glob(self.get_src() + 'processed/registrations/*')):
            actor = Path(file).stem.split('_')[0]
            registrations[actor] = [file]

        return registrations
