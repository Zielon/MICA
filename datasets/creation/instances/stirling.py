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


class Stirling(Instance, ABC):
    def __init__(self):
        super(Stirling, self).__init__()
        self.dst = '/scratch/NFC/OnFlame/STIRLING/'
        self.src = '/scratch/NFC/Stirling/'

    def get_min_det_score(self):
        return 0.75

    def get_images(self):
        images = {}
        for file in sorted(glob(self.get_src() + 'images/Real_images__Subset_2D_FG2018/HQ/*')):
            actor = Path(file).stem.split('_')[0].upper()
            if actor not in images:
                images[actor] = []
            images[actor].append(file)

        return images

    def get_flame_params(self):
        prams = {}
        for file in sorted(glob(self.get_src() + 'FLAME_parameters/iter1/*/*.npz')):
            actor = Path(file).stem[0:5].upper()
            prams[Path(actor).name] = [file]

        return prams

    def get_registrations(self):
        registrations = {}
        for file in sorted(glob(self.get_src() + 'registrations/iter1/*/*')):
            if 'obj' not in file:
                continue
            actor = Path(file).stem[0:5].upper()
            registrations[Path(actor).name] = [file]

        return registrations

    def get_meshes(self):
        meshes = {}
        for file in sorted(glob(self.get_src() + 'scans/*/*.obj')):
            actor = Path(file).stem[0:5].upper()
            if 'obj' in file:
                meshes[actor] = file

        return meshes

    def transform_mesh(self, path):
        self.update_obj(path, fix_mtl=True)
        mesh = load_objs_as_meshes([path], device=self.device)
        vertices = mesh._verts_list[0]
        center = vertices.mean(0)
        mesh._verts_list = [vertices - center]
        mesh.scale_verts_(0.01)

        return mesh.clone()

    def transform_path(self, file):
        name = Path(file).name
        return name
