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


import face_alignment
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger

from datasets.creation.util import get_bbox


class Detectors:
    def __init__(self):
        self.RETINAFACE = 'RETINAFACE'
        self.FAN = 'FAN'


detectors = Detectors()


class LandmarksDetector:
    def __init__(self, model='retinaface', device='cuda:0'):
        model = model.upper()
        self.predictor = model
        if model == detectors.RETINAFACE:
            self._face_detector = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
            self._face_detector.prepare(ctx_id=0, det_size=(224, 224))
        elif model == detectors.FAN:
            self._face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        else:
            logger.error(f'[ERROR] Landmark predictor not supported {model}')
            exit(-1)

        logger.info(f'[DETECTOR] Selected {model} as landmark detector.')

    def detect(self, img):
        if self.predictor == detectors.RETINAFACE:
            bboxes, kpss = self._face_detector.det_model.detect(img, max_num=0, metric='default')
            return bboxes, kpss

        if self.predictor == detectors.FAN:
            lmks, scores, detected_faces = self._face_detector.get_landmarks_from_image(img, return_landmark_score=True, return_bboxes=True)
            if detected_faces is None:
                return np.empty(0), np.empty(0)
            bboxes = np.stack(detected_faces)
            # bboxes = get_bbox(img, np.stack(lmks))
            # bboxes[:, 4] = detected_faces[:, 4]
            # https://github.com/Rubikplayer/flame-fitting/blob/master/data/landmarks_51_annotated.png
            lmk51 = np.stack(lmks)[:, 17:, :]
            kpss = lmk51[:, [20, 27, 13, 43, 47], :]  # left eye, right eye, nose, left mouth, right mouth
            kpss[:, 0, :] = lmk51[:, [21, 24], :].mean(1)  # center of eye
            kpss[:, 1, :] = lmk51[:, [27, 29], :].mean(1)
            return bboxes, kpss

        return None, None
