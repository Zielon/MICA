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
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from insightface.utils import face_align
from numpy.lib import math

input_mean = 127.5
input_std = 127.5


def create_folders(folders):
    if not type(folders) is list:
        folders = folders.split('/')
    parents = '/'
    for folder in folders:
        parents = os.path.join(parents, folder)
        if os.path.exists(parents):
            continue
        Path(parents).mkdir(exist_ok=True)


def get_arcface_input(face, img):
    aimg = face_align.norm_crop(img, landmark=face.kps)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0], aimg


def get_image(name, to_rgb=False):
    images_dir = osp.join(Path(__file__).parent.absolute(), '../images')
    ext_names = ['.jpg', '.png', '.jpeg']
    image_file = None
    for ext_name in ext_names:
        _image_file = osp.join(images_dir, "%s%s" % (name, ext_name))
        if osp.exists(_image_file):
            image_file = _image_file
            break
    assert image_file is not None, '%s not found' % name
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:, :, ::-1]
    return img


# from the original insightface.app.face_analysis.py file
def draw_on(img, faces):
    import cv2
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(np.int)
            # print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                           2)
        if face.gender is not None and face.age is not None:
            cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

    return dimg


def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_center(bboxes, img):
    img_center = img.shape[1] // 2, img.shape[0] // 2
    size = bboxes.shape[0]
    distance = np.Inf
    j = 0
    for i in range(size):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx = abs(x2 - x1) / 2.0
        dy = abs(y2 - y1) / 2.0
        current = dist((x1 + dx, y1 + dy), img_center)
        if current < distance:
            distance = current
            j = i

    return j


def bbox2point(left, right, top, bottom, type='bbox'):
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    else:
        raise NotImplementedError
    return old_size, center


def get_bbox(image, lmks, bb_scale=1.0):
    h, w, c = image.shape
    bbox = []
    for i in range(lmks.shape[0]):
        lmks = lmks.astype(np.int32)
        x_min, x_max, y_min, y_max = np.min(lmks[i, :, 0]), np.max(lmks[i, :, 0]), np.min(lmks[i, :, 1]), np.max(lmks[i, :, 1])
        x_center, y_center = int((x_max + x_min) / 2.0), int((y_max + y_min) / 2.0)
        size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
        xb_min, xb_max, yb_min, yb_max = max(x_center - size // 2, 0), min(x_center + size // 2, w - 1), \
            max(y_center - size // 2, 0), min(y_center + size // 2, h - 1)

        yb_max = min(yb_max, h - 1)
        xb_max = min(xb_max, w - 1)
        yb_min = max(yb_min, 0)
        xb_min = max(xb_min, 0)

        if (xb_max - xb_min) % 2 != 0:
            xb_min += 1

        if (yb_max - yb_min) % 2 != 0:
            yb_min += 1

        # x1, y1, x2, y2
        bbox.append(np.array([xb_min, yb_min, xb_max, yb_max, 0]))

    return np.stack(bbox)
