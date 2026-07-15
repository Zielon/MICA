# -*- coding: utf-8 -*-
# Single-image MICA inference: image -> FLAME facial mesh.
# Based on demo.py, but takes one image path and keeps preprocessing in memory.
#
# Usage (inside the MICA conda env):
#   python generate_mesh.py sample_face_image
#   python generate_mesh.py path/to/photo.jpg -o output/photo

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import trimesh
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger

from configs.config import get_cfg_defaults
from datasets.creation.util import get_arcface_input, get_center
from utils import util
from utils.landmark_detector import LandmarksDetector, detectors


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def preprocess(image_path, app, device, image_size=224):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    bboxes, kpss = app.detect(img)
    if bboxes.shape[0] == 0:
        raise RuntimeError(f'Face not detected in {image_path}')

    i = get_center(bboxes, img)
    face = Face(bbox=bboxes[i, 0:4], kps=kpss[i] if kpss is not None else None, det_score=bboxes[i, 4])
    blob, _ = get_arcface_input(face, img)

    aligned = face_align.norm_crop(img, landmark=face.kps, image_size=image_size)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB) / 255.
    image = torch.tensor(aligned.transpose(2, 0, 1), dtype=torch.float32, device=device)[None]
    arcface = torch.tensor(blob, dtype=torch.float32, device=device)[None]

    return image, arcface


def load_checkpoint(path, mica, device):
    checkpoint = torch.load(path, map_location=device)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])

def run_generator(image, cfg = get_cfg_defaults()):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint('data/pretrained/mica.tar', mica, device)
    mica.eval()

    faces = mica.flameModel.generator.faces_tensor.cpu()
    app = LandmarksDetector(model=detectors.RETINAFACE, device=device)

    with torch.no_grad():
        image, arcface = preprocess(image, app, device)

        codedict = mica.encode(image, arcface)
        opdict = mica.decode(codedict)
        meshes = opdict['pred_canonical_shape_vertices']
        code = opdict['pred_shape_code']
        lmk = mica.flame.compute_landmarks(meshes)

        mesh = meshes[0]
        landmark_51 = lmk[0, 17:]
        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

        # vertices exported in millimeters, matching demo.py
        return trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False)

def main(cfg, args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint(args.m, mica, device)
    mica.eval()

    faces = mica.flameModel.generator.faces_tensor.cpu()
    app = LandmarksDetector(model=detectors.RETINAFACE, device=device)

    with torch.no_grad():
        image, arcface = preprocess(args.image, app, device)

        codedict = mica.encode(image, arcface)
        opdict = mica.decode(codedict)
        meshes = opdict['pred_canonical_shape_vertices']
        code = opdict['pred_shape_code']
        lmk = mica.flame.compute_landmarks(meshes)

        mesh = meshes[0]
        landmark_51 = lmk[0, 17:]
        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

        dst = Path(args.o)
        dst.mkdir(parents=True, exist_ok=True)
        # vertices exported in millimeters, matching demo.py
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')
        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
        np.save(f'{dst}/identity', code[0].cpu().numpy())
        np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
        np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

        logger.info(f'Mesh saved to {dst}/mesh.ply and {dst}/mesh.obj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA single-image facial mesh generation')
    parser.add_argument('image', nargs='?', default='sample_face_image', type=str, help='Path to the input face image')
    parser.add_argument('-o', default='output/sample_face', type=str, help='Output folder')
    parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')

    args = parser.parse_args()
    cfg = get_cfg_defaults()

    deterministic(42)
    main(cfg, args)
