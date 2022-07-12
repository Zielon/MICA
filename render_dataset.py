# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from configs.config import get_cfg_defaults
from micalib.renderer import MeshShapeRenderer
from models.flame import FLAME

np.random.seed(125)
random.seed(125)


def main():
    cfg = get_cfg_defaults()
    render = MeshShapeRenderer(obj_filename=cfg.model.topology_path)
    flame = FLAME(cfg.model).to('cuda:0')
    datasets = sorted(glob('/home/wzielonka/datasets/MICA/*'))
    for dataset in tqdm(datasets):
        meshes = sorted(glob(f'{dataset}/FLAME_parameters/*/*.npz'))
        sample_list = np.array(np.random.choice(range(len(meshes)), size=30 * 5))
        dst = Path('./output', Path(dataset).name)
        dst.mkdir(parents=True, exist_ok=True)
        j = 0
        k = 0
        images = np.zeros((512, 512 * 5, 3))
        for i in sample_list:
            params = np.load(meshes[i], allow_pickle=True)
            betas = torch.tensor(params['betas']).float().cuda()
            shape_params = betas[:300][None]
            v = flame(shape_params=shape_params)[0]
            rendering = render.render_mesh(v)
            image = (rendering[0].cpu().numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            image = np.minimum(np.maximum(image, 0), 255).astype(np.uint8)
            images[0:512, 512 * j:512 * (j + 1), :] = image
            j += 1

            if j % 5 == 0 and j > 0:
                dst.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{dst}/{str(k).zfill(4)}.png', images)
                images = np.zeros((512, 512 * 5, 3))
                j = 0
                k += 1

        os.system(f'ffmpeg -y -framerate 1 -pattern_type glob -i \'{dst}/*.png\' -c:v libx264 -pix_fmt yuv420p {dst}/video.mp4')
        os.system(f'gifski -o ./output/{Path(dataset).name}.gif {dst}/*.png --quality 100 --fps 1')


if __name__ == '__main__':
    main()
