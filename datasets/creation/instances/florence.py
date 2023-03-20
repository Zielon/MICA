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

import numpy as np

from datasets.creation.instances.instance import Instance


class Florence(Instance, ABC):
    def __init__(self):
        super(Florence, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/FLORENCE/'
        self.src = '/scratch/NFC/MICC_Florence/'

    def get_min_det_score(self):
        return 0.85

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'images/*')):
            imgs = sorted(list(filter(lambda f: 'PTZ-Outdoor' not in f, glob(f'{actor}/*/*.jpg'))))
            indecies = np.random.choice(len(imgs), 1000, replace=False)
            images[Path(actor).stem] = [imgs[i] for i in indecies]

        return images

    def get_flame_params(self):
        params = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_parameters/iter1/*')):
            params[Path(actor).stem] = glob(f'{actor}/*.npz')

        return params

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'registrations/iter1/*')):
            if 'rendering' in actor:
                continue
            registrations[Path(actor).stem] = glob(f'{actor}/*.obj')

        return registrations
