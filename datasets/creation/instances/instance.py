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


import os
from abc import abstractmethod
from pathlib import Path

from pytorch3d.transforms import RotateAxisAngle


class Instance:
    def __init__(self):
        self.mount = '/home/wzielonka/Cluster/lustre'
        self.dst = 'empty'
        self.src = 'empty'
        self.device = 'cuda:0'
        self.actors = []
        self.use_mount = os.path.exists(self.mount)

    def get_dst(self):
        return self.dst if not self.use_mount else self.mount + self.dst

    def get_src(self):
        return self.src if not self.use_mount else self.mount + self.src

    @abstractmethod
    def get_min_det_score(self):
        return 0

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def get_images(self):
        return {}

    @abstractmethod
    def get_flame_params(self):
        return {}

    @abstractmethod
    def get_registrations(self):
        return {}

    @abstractmethod
    def get_meshes(self):
        return {}

    @abstractmethod
    def transform_mesh(self, path):
        return None

    @abstractmethod
    def transform_image(self, img):
        return [img]

    @abstractmethod
    def transform_path(self, file):
        return Path(file).name

    @abstractmethod
    def get_rotations(self):
        rots = {}
        degree = 2.5
        step = int(15 / degree / 2)
        X = range(-step, step + 1)
        degree = 8.0
        step = int(144 / degree / 2)
        Y = range(-step, step + 1)
        for a, angles in [('X', X), ('Y', Y)]:
            r = []
            for i in angles:
                r.append((RotateAxisAngle(float(degree * i), axis=a, device=self.device), float(degree * i)))
            rots[a] = r
        return rots

    @abstractmethod
    def update_obj(self, path, fix_mtl=False):
        mesh = Path(path).stem
        with open(path, 'r') as file:
            filedata = file.readlines()

        input = []
        for line in filedata:
            if 'usemtl' in line or 'newmtl' in line:
                continue
            input.append(line)

        output = []
        for line in input:
            if 'mtllib' in line:
                mtl = line.split(' ')[-1].split('.')[0]
                line += f'usemtl {mtl}\n'
            output.append(line)
        with open(path, 'w') as file:
            file_lines = "".join(output)
            file.write(file_lines)

        if not fix_mtl:
            return

        with open(path.replace('obj', 'mtl'), 'r') as file:
            filedata = file.readlines()

        output = []
        for line in filedata:
            if 'newmtl' in line:
                line = 'newmtl ' + mesh + '\n'
            output.append(line)
        with open(path.replace('obj', 'mtl'), 'w') as file:
            file_lines = "".join(output)
            file.write(file_lines)
