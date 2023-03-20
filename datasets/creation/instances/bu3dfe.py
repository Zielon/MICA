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

from pytorch3d.io import load_objs_as_meshes

from datasets.creation.instances.instance import Instance


class BU3DFE(Instance, ABC):
    def __init__(self):
        super(BU3DFE, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/BU3DFE/'
        self.src = '/scratch/NFC/BU-3DFE/'

    def get_images(self):
        images = {}
        for actor in sorted(glob(self.get_src().replace('BU-3DFE', 'BU-3DFE_clean') + 'images/*')):
            images[Path(actor).name] = glob(f'{actor}/*.jpg')

        return images

    def get_flame_params(self):
        prams = {}
        for actor in sorted(glob(self.get_src() + 'FLAME_parameters/iter2/*')):
            prams[Path(actor).name] = glob(f'{actor}/*.npz')

        return prams

    def get_registrations(self):
        registrations = {}
        for actor in sorted(glob(self.get_src() + 'registrations/iter2/neutral_align/*')):
            registrations[Path(actor).name] = glob(f'{actor}/*.obj')

        return registrations

    def get_meshes(self):
        meshes = {}
        files = sorted(glob(self.get_src() + 'raw_ne_data/*'))
        actors = set(map(lambda f: Path(f).name[0:5], files))
        for actor in actors:
            meshes[Path(actor).name] = next(filter(lambda f: actor in f and 'obj' in f, files))

        return meshes

    def transform_mesh(self, path):
        self.update_obj(path)
        mesh = load_objs_as_meshes([path], device=self.device)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]
        mesh.scale_verts_(0.01)

        return mesh.clone()
