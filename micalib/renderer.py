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


import pytorch3d
import torch
import torch.nn as nn
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesVertex
)


class MeshShapeRenderer(nn.Module):
    def __init__(self, obj_filename):
        super().__init__()

        verts, faces, aux = load_obj(obj_filename)
        faces = faces.verts_idx[None, ...].cuda()
        self.register_buffer('faces', faces)

        R, T = look_at_view_transform(2.7, 10.0, 10.0)
        self.cameras = FoVPerspectiveCameras(device='cuda:0', R=R, T=T, fov=6)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True
        )

        lights = pytorch3d.renderer.DirectionalLights(
            device='cuda:0',
            direction=((0, 0, 1),),
            ambient_color=((0.4, 0.4, 0.4),),
            diffuse_color=((0.35, 0.35, 0.35),),
            specular_color=((0.05, 0.05, 0.05),))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device='cuda:0', cameras=self.cameras, lights=lights)
        )

    def render_mesh(self, vertices, faces=None, verts_rgb=None):
        B, N, V = vertices.shape
        if faces is None:
            faces = self.faces.repeat(B, 1, 1)
        else:
            faces = faces.repeat(B, 1, 1)

        if verts_rgb is None:
            verts_rgb = torch.ones_like(vertices)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

        rendering = self.renderer(meshes).permute(0, 3, 1, 2)
        color = rendering[:, 0:3, ...]

        return color
