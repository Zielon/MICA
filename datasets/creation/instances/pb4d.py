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
from pytorch3d.io import load_objs_as_meshes

from datasets.creation.instances.instance import Instance


class PB4D(Instance, ABC):
    def __init__(self):
        super(PB4D, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/BP4D/'
        self.src = '/scratch/NFC/BP4D/'

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src() + 'images/*')):
            imgs = sorted(glob(f'/{actor}/*.jpg'))
            indecies = np.random.choice(len(imgs), 100, replace=False)
            images[Path(actor).name] = [imgs[i] for i in indecies]

        return images

    def get_flame_params(self):
        prams = {}
        for file in sorted(glob(self.get_src() + 'FLAME_parameters/*.npz')):
            prams[Path(file).stem] = [file]

        return prams

    def get_registrations(self):
        registrations = {}
        for file in sorted(glob(self.get_src() + 'registrations/*')):
            registrations[Path(file).stem] = [file]

        return registrations

    def get_meshes(self):
        meshes = {}
        for file in sorted(glob(self.get_src() + 'scans/*.obj')):
            meshes[Path(file).stem] = [file]

        return meshes

    def transform_mesh(self, path):
        mesh = load_objs_as_meshes(path, device=self.device)
        mesh.scale_verts_(0.01)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]

        return mesh.clone()
