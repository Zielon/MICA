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


import argparse
import os

from yacs.config import CfgNode as CN

cfg = CN()

abs_mica_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg.mica_dir = abs_mica_dir
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.pretrained_model_path = os.path.join(cfg.mica_dir, 'data/pretrained', 'mica.tar')
cfg.output_dir = ''

# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.testing = False
cfg.model.name = 'mica'

cfg.model.topology_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'head_template.obj')
cfg.model.flame_model_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'generic_model.pkl')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'landmark_embedding.npy')
cfg.model.n_shape = 300
cfg.model.layers = 8
cfg.model.hidden_layers_size = 256
cfg.model.mapping_layers = 3
cfg.model.use_pretrained = True
cfg.model.arcface_pretrained_model = '/scratch/is-rg-ncs/models_weights/arcface-torch/backbone100.pth'

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['LYHM']
cfg.dataset.eval_data = ['FLORENCE']
cfg.dataset.batch_size = 2
cfg.dataset.K = 4
cfg.dataset.n_train = 100000
cfg.dataset.num_workers = 4
cfg.dataset.root = '/datasets/MICA/'

# ---------------------------------------------------------------------------- #
# Mask weights
# ---------------------------------------------------------------------------- #
cfg.mask_weights = CN()
cfg.mask_weights.face = 150.0
cfg.mask_weights.nose = 50.0
cfg.mask_weights.lips = 50.0
cfg.mask_weights.forehead = 50.0
cfg.mask_weights.lr_eye_region = 50.0
cfg.mask_weights.eye_region = 50.0

cfg.mask_weights.whole = 1.0
cfg.mask_weights.ears = 0.01
cfg.mask_weights.eyes = 0.01

cfg.running_average = 7

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.use_mask = False
cfg.train.max_epochs = 50
cfg.train.max_steps = 100000
cfg.train.lr = 1e-4
cfg.train.arcface_lr = 1e-3
cfg.train.weight_decay = 0.0
cfg.train.lr_update_step = 100000000
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 200
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 1000
cfg.train.checkpoint_epochs_steps = 2
cfg.train.val_steps = 1000
cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 5000
cfg.train.reset_optimizer = False
cfg.train.val_save_img = 5000
cfg.test_dataset = 'now'


def get_cfg_defaults():
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path', required=True)
    parser.add_argument('--test_dataset', type=str, help='Test dataset type', default='')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load', default='')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg, args
