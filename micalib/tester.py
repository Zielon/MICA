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

import cv2
import numpy as np
import torch
import torch.distributed as dist
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from configs.config import cfg
from utils import util

input_mean = 127.5
input_std = 127.5

NOW_SCANS = '/home/wzielonka/datasets/NoWDataset/final_release_version/scans/'
NOW_PICTURES = '/home/wzielonka/datasets/NoWDataset/final_release_version/iphone_pictures/'
NOW_BBOX = '/home/wzielonka/datasets/NoWDataset/final_release_version/detected_face/'
STIRLING_PICTURES = '/home/wzielonka/datasets/Stirling/images/'


class Tester(object):
    def __init__(self, nfc_model, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.K = self.cfg.dataset.K
        self.render_mesh = True
        self.embeddings_lyhm = {}

        # deca model
        self.nfc = nfc_model.to(self.device)
        self.nfc.testing = True

        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')

    def load_checkpoint(self, model_path):
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}

        checkpoint = torch.load(model_path, map_location)

        if 'arcface' in checkpoint:
            self.nfc.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            self.nfc.flameModel.load_state_dict(checkpoint['flameModel'])

        logger.info(f"[TESTER] Resume from {model_path}")

    def load_model_dict(self, model_dict):
        dist.barrier()

        self.nfc.canonicalModel.load_state_dict(model_dict['canonicalModel'])
        self.nfc.arcface.load_state_dict(model_dict['arcface'])

    def process_image(self, img, app):
        images = []
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
        if bboxes.shape[0] != 1:
            logger.error('Face not detected!')
            return images
        i = 0
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        aimg = face_align.norm_crop(img, landmark=face.kps)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)

        images.append(torch.tensor(blob[0])[None])

        return images

    def process_folder(self, folder, app):
        images = []
        arcface = []
        files_actor = sorted(sorted(os.listdir(folder)))
        for file in files_actor:
            image_path = folder + '/' + file
            logger.info(image_path)

            ### NOW CROPPING
            scale = 1.6
            # scale = np.random.rand() * (1.8 - 1.1) + 1.1
            bbx_path = image_path.replace('.jpg', '.npy').replace('iphone_pictures', 'detected_face')
            bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
            left = bbx_data['left']
            right = bbx_data['right']
            top = bbx_data['top']
            bottom = bbx_data['bottom']

            image = imread(image_path)[:, :, :3]

            h, w, _ = image.shape
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            size = int(old_size * scale)

            crop_size = 224
            # crop image
            src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
            DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.
            dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))

            arcface += self.process_image(cv2.cvtColor(dst_image.astype(np.float32) * 255.0, cv2.COLOR_RGB2BGR), app)

            dst_image = dst_image.transpose(2, 0, 1)
            images.append(torch.tensor(dst_image)[None])

        images = torch.cat(images, dim=0).float()
        arcface = torch.cat(arcface, dim=0).float()

        return images, arcface

    def get_name(self, best_model, id):
        if '_' in best_model:
            name = id if id is not None else best_model.split('_')[-1][0:-4]
        else:
            name = id if id is not None else best_model.split('/')[-1][0:-4]
        return name

    def test_now(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.now(name)

    def test_stirling(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.stirling(name)

    def save_mesh(self, file, vertices):
        scaled = vertices * 1000.0
        save_ply(file, scaled.cpu(), self.nfc.render.faces[0].cpu())

        # mask = self.nfc.masking.get_triangle_whole_mask()
        # v, f = self.nfc.masking.get_masked_mesh(vertices, mask)
        # save_obj(file, v[0], f[0])

    def cache_to_cuda(self, cache):
        for key in cache.keys():
            i, a = cache[key]
            cache[key] = (i.to(self.device), a.to(self.device))
        return cache

    def create_now_cache(self):
        if os.path.exists('test_now_cache.pt'):
            cache = self.cache_to_cuda(torch.load('test_now_cache.pt'))
            return cache
        else:
            cache = {}

        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.4)

        for actor in tqdm(sorted(os.listdir(NOW_PICTURES))):
            image_paths = sorted(glob(NOW_PICTURES + actor + '/*'))
            for folder in image_paths:
                images, arcface = self.process_folder(folder, app)
                cache[folder] = (images, arcface)

        torch.save(cache, 'test_now_cache.pt')
        return self.cache_to_cuda(cache)

    def create_stirling_cache(self):
        if os.path.exists('test_stirling_cache.pt'):
            cache = torch.load('test_stirling_cache.pt')
            return cache
        else:
            cache = {}

        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224), det_thresh=0.1)

        cache['HQ'] = {}
        cache['LQ'] = {}

        for folder in ['Real_images__Subset_2D_FG2018']:
            for quality in ['HQ', 'LQ']:
                for path in tqdm(sorted(glob(STIRLING_PICTURES + folder + '/' + quality + '/*.jpg'))):
                    actor = path.split('/')[-1][:9].upper()
                    image = imread(path)[:, :, :3]
                    blobs = self.process_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), app)
                    if len(blobs) == 0:
                        continue
                    image = image / 255.
                    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
                    image = torch.tensor(image).cuda()[None]

                    if actor not in cache[quality]:
                        cache[quality][actor] = []
                    cache[quality][actor].append((image, blobs[0]))

        for q in cache.keys():
            for a in cache[q].keys():
                images, arcface = list(zip(*cache[q][a]))
                images = torch.cat(images, dim=0).float()
                arcface = torch.cat(arcface, dim=0).float()
                cache[q][a] = (images, arcface)

        torch.save(cache, 'test_stirling_cache.pt')
        return self.cache_to_cuda(cache)

    def update_embeddings(self, actor, arcface):
        if actor not in self.embeddings_lyhm:
            self.embeddings_lyhm[actor] = []
        self.embeddings_lyhm[actor] += [arcface[i].data.cpu().numpy() for i in range(arcface.shape[0])]

    def stirling(self, best_id):
        logger.info(f"[TESTER] Stirling testing has begun!")
        self.nfc.eval()
        cache = self.create_stirling_cache()
        for quality in cache.keys():
            images_processed = 0
            self.embeddings_lyhm = {}
            for actor in tqdm(cache[quality].keys()):
                images, arcface = cache[quality][actor]
                with torch.no_grad():
                    codedict = self.nfc.encode(images.cuda(), arcface.cuda())
                    opdict = self.nfc.decode(codedict, 0)

                self.update_embeddings(actor, codedict['arcface'])

                dst_actor = actor[:5]
                os.makedirs(os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality), exist_ok=True)
                dst_folder = os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor)
                os.makedirs(dst_folder, exist_ok=True)

                meshes = opdict['pred_canonical_shape_vertices']
                lmk = self.nfc.flame.compute_landmarks(meshes)

                for m in range(meshes.shape[0]):
                    v = torch.reshape(meshes[m], (-1, 3))
                    savepath = os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor, f'{actor}.ply')
                    self.save_mesh(savepath, v)
                    landmark_51 = lmk[m, 17:]
                    landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]
                    landmark_7 = landmark_7.cpu().numpy() * 1000.0
                    np.save(os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor, f'{actor}.npy'), landmark_7)
                    images_processed += 1

                    pred = self.nfc.render.render_mesh(meshes)
                    dict = {
                        'pred': pred,
                        'images': images
                    }

                    savepath = os.path.join(self.cfg.output_dir, f'stirling_test_{best_id}', 'predicted_meshes', quality, dst_actor, f'{actor}.jpg')
                    util.visualize_grid(dict, savepath, size=512)

            logger.info(f"[TESTER] Stirling dataset {quality} with {images_processed} processed!")

            # util.save_embedding_projection(self.embeddings_lyhm, f'{self.cfg.output_dir}/stirling_test_{best_id}/stirling_{quality}_arcface_embeds.pdf')

    def now(self, best_id):
        logger.info(f"[TESTER] NoW testing has begun!")
        self.nfc.eval()
        cache = self.create_now_cache()
        # for actor in tqdm(sorted(os.listdir(NOW_SCANS))): # only 20
        for actor in tqdm(sorted(os.listdir(NOW_PICTURES))):  # all 100
            image_paths = sorted(glob(NOW_PICTURES + actor + '/*'))
            for folder in image_paths:
                files_actor = sorted(os.listdir(folder))
                images, arcface = cache[folder]
                with torch.no_grad():
                    codedict = self.nfc.encode(images, arcface)
                    opdict = self.nfc.decode(codedict, 0)

                self.update_embeddings(actor.split('_')[2], codedict['arcface'])

                type = folder.split('/')[-1]
                os.makedirs(os.path.join(self.cfg.output_dir, f'now_test_{best_id}', 'predicted_meshes'), exist_ok=True)
                dst_folder = os.path.join(self.cfg.output_dir, f'now_test_{best_id}', 'predicted_meshes', actor, type)
                os.makedirs(dst_folder, exist_ok=True)

                meshes = opdict['pred_canonical_shape_vertices']
                lmk = self.nfc.flame.compute_landmarks(meshes)

                for m in range(meshes.shape[0]):
                    a = files_actor[m]
                    v = torch.reshape(meshes[m], (-1, 3))
                    savepath = os.path.join(self.cfg.output_dir, f'now_test_{best_id}', 'predicted_meshes', actor, type, a.replace('jpg', 'ply'))
                    self.save_mesh(savepath, v)
                    landmark_51 = lmk[m, 17:]
                    landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]
                    landmark_7 = landmark_7.cpu().numpy() * 1000.0
                    np.save(os.path.join(self.cfg.output_dir, f'now_test_{best_id}', 'predicted_meshes', actor, type, a.replace('jpg', 'npy')), landmark_7)

                if self.render_mesh:
                    pred = self.nfc.render.render_mesh(meshes)

                    dict = {
                        'pred': pred,
                        # 'deca': deca,
                        'images': images
                    }

                    savepath = os.path.join(dst_folder, 'render.jpg')
                    util.visualize_grid(dict, savepath, size=512)

        # util.save_embedding_projection(self.embeddings_lyhm, f'{self.cfg.output_dir}/now_test_{best_id}/now_arcface_embeds.pdf')
