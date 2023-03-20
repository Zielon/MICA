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
import pickle

import numpy as np
import torch
import torch.nn as nn
from trimesh import Trimesh


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class Masking(nn.Module):
    def __init__(self, config):
        super(Masking, self).__init__()
        ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        with open(f'{ROOT_DIR}/data/FLAME2020/FLAME_masks/FLAME_masks.pkl', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            self.masks = Struct(**ss)

        with open(f'{ROOT_DIR}/data/FLAME2020/generic_model.pkl', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.masked_faces = None

        self.cfg = config.mask_weights
        self.dtype = torch.float32
        self.register_buffer('faces', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('vertices', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        self.neighbours = {}
        for f in self.faces.numpy():
            for v in f:
                if str(v) not in self.neighbours:
                    self.neighbours[str(v)] = set()
                for a in list(filter(lambda i: i != v, f)):
                    self.neighbours[str(v)].add(a)

    def get_faces(self):
        return self.faces

    def get_mask_face(self):
        return self.masks.face

    def get_mask_eyes(self):
        left = self.masks.left_eyeball
        right = self.masks.right_eyeball

        return np.unique(np.concatenate((left, right)))

    def get_mask_forehead(self):
        return self.masks.forehead

    def get_mask_lips(self):
        return self.masks.lips

    def get_mask_eye_region(self):
        return self.masks.eye_region

    def get_mask_lr_eye_region(self):
        left = self.masks.left_eye_region
        right = self.masks.right_eye_region

        return np.unique(np.concatenate((left, right, self.get_mask_eyes())))

    def get_mask_nose(self):
        return self.masks.nose

    def get_mask_ears(self):
        left = self.masks.left_ear
        right = self.masks.right_ear

        return np.unique(np.concatenate((left, right)))

    def get_triangle_face_mask(self):
        m = self.masks.face
        return self.get_triangle_mask(m)

    def get_triangle_eyes_mask(self):
        m = self.get_mask_eyes()
        return self.get_triangle_mask(m)

    def get_triangle_whole_mask(self):
        m = self.get_whole_mask()
        return self.get_triangle_mask(m)

    def get_triangle_mask(self, m):
        f = self.faces.cpu().numpy()
        selected = []
        for i in range(f.shape[0]):
            l = f[i]
            valid = 0
            for j in range(3):
                if l[j] in m:
                    valid += 1
            if valid == 3:
                selected.append(i)

        return np.unique(selected)

    def make_soft(self, mask, value, degree=4):
        soft = []
        mask = set(mask)
        for ring in range(degree):
            soft_ring = []
            for v in mask.copy():
                for n in self.neighbours[str(v)]:
                    if n in mask:
                        continue
                    soft_ring.append(n)
                    mask.add(n)

            soft.append((soft_ring, value / (ring + 2)))

        return soft

    def get_binary_triangle_mask(self):
        mask = self.get_whole_mask()
        faces = self.faces.cpu().numpy()
        reduced_faces = []
        for f in faces:
            valid = 0
            for v in f:
                if v in mask:
                    valid += 1
            reduced_faces.append(True if valid == 3 else False)

        return reduced_faces

    def get_masked_faces(self):
        if self.masked_faces is None:
            faces = self.faces.cpu().numpy()
            vertices = self.vertices.cpu().numpy()
            m = Trimesh(vertices=vertices, faces=faces, process=False)
            m.update_faces(self.get_binary_triangle_mask())
            self.masked_faces = torch.from_numpy(np.array(m.faces)).cuda().long()[None]

        return self.masked_faces

    def get_weights_per_triangle(self):
        mask = torch.ones_like(self.get_faces()[None]).detach() * self.cfg.whole

        mask[:, self.get_triangle_eyes_mask(), :] = self.cfg.eyes
        mask[:, self.get_triangle_face_mask(), :] = self.cfg.face

        return mask[:, :, 0:1]

    def get_weights_per_vertex(self):
        mask = torch.ones_like(self.vertices[None]).detach() * self.cfg.whole

        mask[:, self.get_mask_eyes(), :] = self.cfg.eyes
        mask[:, self.get_mask_ears(), :] = self.cfg.ears
        mask[:, self.get_mask_face(), :] = self.cfg.face

        return mask

    def get_masked_mesh(self, vertices, triangle_mask):
        if len(vertices.shape) == 2:
            vertices = vertices[None]
        B, N, V = vertices.shape
        faces = self.faces.cpu().numpy()
        masked_vertices = torch.empty(0, 0, 3).cuda()
        masked_faces = torch.empty(0, 0, 3).cuda()
        for i in range(B):
            m = Trimesh(vertices=vertices[i].detach().cpu().numpy(), faces=faces, process=False)
            m.update_faces(triangle_mask)
            m.process()
            f = torch.from_numpy(np.array(m.faces)).cuda()[None]
            v = torch.from_numpy(np.array(m.vertices)).cuda()[None].float()
            if masked_vertices.shape[1] != v.shape[1]:
                masked_vertices = torch.empty(0, v.shape[1], 3).cuda()
            if masked_faces.shape[1] != f.shape[1]:
                masked_faces = torch.empty(0, f.shape[1], 3).cuda()
            masked_vertices = torch.cat([masked_vertices, v])
            masked_faces = torch.cat([masked_faces, f])

        return masked_vertices, masked_faces
