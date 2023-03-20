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


class FaceWarehouse(Instance, ABC):
    def __init__(self):
        super(FaceWarehouse, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/FACEWAREHOUSE/'
        self.src = '/scratch/NFC/FaceWarehouse/'

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'Images/*')):
            images[Path(actor).stem] = glob(f'{actor}/*.png')

        return images

    def get_flame_params(self):
        params = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_fits/*')):
            params[Path(actor).stem] = [sorted(glob(f'{actor}/*.npz'))[0]]

        return params

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_fits/*')):
            registrations[Path(actor).stem] = [f'{actor}/tmp/pose_0__def_trafo_fit.obj']

        return registrations
