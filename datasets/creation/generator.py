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
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from tqdm import tqdm

from datasets.creation.instances.instance import Instance
from datasets.creation.util import get_image, get_center, get_arcface_input


def _transfer(src, dst):
    src.parent.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.system(f'cp {str(src)} {str(dst)}')


def _copy(payload):
    instance, func, target, transform_path = payload
    files = func()
    for actor in files.keys():
        for file in files[actor]:
            _transfer(Path(file), Path(instance.get_dst(), target, actor, transform_path(file)))


class Generator:
    def __init__(self, instances):
        self.instances: List[Instance] = instances
        self.ARCFACE = 'arcface_input'

    def copy(self):
        logger.info('Start copying...')
        for instance in tqdm(self.instances):
            payloads = [(instance, instance.get_images, 'images', instance.transform_path)]
            with Pool(processes=len(payloads)) as pool:
                for _ in tqdm(pool.imap_unordered(_copy, payloads), total=len(payloads)):
                    pass

    def preprocess(self):
        logger.info('Start preprocessing...')
        for instance in tqdm(self.instances):
            instance.preprocess()

    def arcface(self):
        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224))

        logger.info('Start arcface...')
        for instance in tqdm(self.instances):
            src = instance.get_dst()
            for image_path in tqdm(sorted(glob(f'{src}/images/*/*'))):
                dst = image_path.replace('images', self.ARCFACE)
                Path(dst).parent.mkdir(exist_ok=True, parents=True)
                for img in instance.transform_image(get_image(image_path[0:-4])):
                    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
                    if bboxes.shape[0] == 0:
                        continue
                    i = get_center(bboxes, img)
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    if det_score < instance.get_min_det_score():
                        continue
                    kps = None
                    if kpss is not None:
                        kps = kpss[i]
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    blob, aimg = get_arcface_input(face, img)
                    np.save(dst[0:-4], blob)
                    cv2.imwrite(dst, face_align.norm_crop(img, landmark=face.kps, image_size=224))

    def run(self):
        self.copy()
        self.preprocess()
        self.arcface()
